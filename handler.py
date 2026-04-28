"""RunPod serverless handler for Hallo2 audio-driven portrait animation.

Inference contract — base64-payload mode (no external URLs):
  Input:
    {
      "image_b64":  "<base64 PNG bytes>",
      "audio_b64":  "<base64 WAV bytes>",
      "pose_weight":  1.0,    # optional
      "face_weight":  1.0,    # optional
      "lip_weight":   1.0,    # optional
      "face_expand_ratio": 1.2,  # optional
      "output_format": "b64"   # "b64" | "volume" — default b64
    }
  Output:
    Either {"video_b64": "..."} (if small enough)
    Or {"volume_path": "/runpod-volume/outputs/<job-id>.mp4"} (if too big to embed)
    Or {"error": "..."}

URL-mode is also supported for backward compat:
  {"image_url": "...", "audio_url": "...", "output_upload_url": "..."}
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests
import runpod

sys.path.insert(0, "/opt")
from download_models import ensure_models  # noqa: E402

HALLO2_DIR = Path("/opt/hallo2")
INFERENCE_SCRIPT = HALLO2_DIR / "scripts" / "inference_long.py"
DEFAULT_CONFIG = HALLO2_DIR / "configs" / "inference" / "long.yaml"
VOLUME_OUT_DIR = Path("/runpod-volume/outputs")

# RunPod's response payload soft cap. Outputs larger than this are
# saved to the network volume and returned as a path instead.
MAX_INLINE_MB = 18


def _download_url(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def _put_url(url: str, src: Path) -> None:
    with src.open("rb") as f:
        r = requests.put(url, data=f, timeout=300,
                         headers={"Content-Type": "video/mp4"})
        r.raise_for_status()


def _ensure_pretrained_symlink() -> None:
    target = HALLO2_DIR / "pretrained_models"
    if target.exists() or target.is_symlink():
        return
    models_dir = ensure_models()
    target.symlink_to(models_dir, target_is_directory=True)


def handler(event: dict[str, Any]) -> dict[str, Any]:
    inp = event.get("input", {}) or {}
    job_id = event.get("id") or uuid.uuid4().hex[:12]

    image_b64 = inp.get("image_b64")
    audio_b64 = inp.get("audio_b64")
    image_url = inp.get("image_url")
    audio_url = inp.get("audio_url")
    upload_url = inp.get("output_upload_url")

    if not (image_b64 or image_url):
        return {"error": "must provide image_b64 or image_url"}
    if not (audio_b64 or audio_url):
        return {"error": "must provide audio_b64 or audio_url"}

    pose_w = float(inp.get("pose_weight", 1.0))
    face_w = float(inp.get("face_weight", 1.0))
    lip_w = float(inp.get("lip_weight", 1.0))
    face_expand = float(inp.get("face_expand_ratio", 1.2))
    output_format = inp.get("output_format", "b64")

    t0 = time.time()
    try:
        _ensure_pretrained_symlink()
    except Exception as e:
        return {"error": f"model setup failed: {e!r}"}

    base_dir = "/runpod-volume" if Path("/runpod-volume").exists() else None
    workdir = Path(tempfile.mkdtemp(prefix=f"hallo2_{job_id}_", dir=base_dir))
    image_path = workdir / "source.png"
    audio_path = workdir / "audio.wav"
    output_path = workdir / "out.mp4"

    try:
        # Materialize inputs
        if image_b64:
            image_path.write_bytes(base64.b64decode(image_b64))
        else:
            _download_url(image_url, image_path)
        if audio_b64:
            audio_path.write_bytes(base64.b64decode(audio_b64))
        else:
            _download_url(audio_url, audio_path)

        cmd = [
            "python", str(INFERENCE_SCRIPT),
            "--config", str(DEFAULT_CONFIG),
            "--source_image", str(image_path),
            "--driving_audio", str(audio_path),
            "--save_path", str(output_path),
            "--pose_weight", str(pose_w),
            "--face_weight", str(face_w),
            "--lip_weight", str(lip_w),
            "--face_expand_ratio", str(face_expand),
        ]
        print(f"[hallo2] {' '.join(cmd)}", flush=True)
        infer_t0 = time.time()
        proc = subprocess.run(
            cmd, cwd=str(HALLO2_DIR),
            capture_output=True, text=True, check=False, timeout=2400,
        )
        infer_secs = time.time() - infer_t0
        if proc.returncode != 0:
            return {
                "error": "hallo2 inference failed",
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-2000:],
                "stdout_tail": proc.stdout[-1000:],
            }
        if not output_path.exists():
            return {
                "error": "hallo2 produced no output file",
                "stderr_tail": proc.stderr[-2000:],
                "stdout_tail": proc.stdout[-1000:],
            }

        size_mb = output_path.stat().st_size / (1 << 20)
        result: dict[str, Any] = {
            "duration_s": round(time.time() - t0, 2),
            "model_seconds": round(infer_secs, 2),
            "video_size_mb": round(size_mb, 2),
        }

        # Backward-compat: explicit upload URL wins
        if upload_url:
            _put_url(upload_url, output_path)
            result["video_url"] = upload_url.split("?")[0]
            return result

        # Default: return inline base64 if small, otherwise drop to volume
        if size_mb <= MAX_INLINE_MB and output_format != "volume":
            with output_path.open("rb") as f:
                result["video_b64"] = base64.b64encode(f.read()).decode("ascii")
            return result

        # Too big for inline — store on the volume so caller can fetch
        # via RunPod's S3-compatible network-volume API.
        VOLUME_OUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = VOLUME_OUT_DIR / f"{job_id}.mp4"
        shutil.move(str(output_path), str(dest))
        result["volume_path"] = str(dest)
        result["note"] = (
            f"output too large for inline ({size_mb:.1f} MB > "
            f"{MAX_INLINE_MB} MB) — stored on /runpod-volume; fetch "
            f"via S3 API or follow-up retrieval job."
        )
        return result
    except Exception as e:
        return {"error": f"handler exception: {e!r}"}
    finally:
        # Don't clean workdir if we moved output to volume already
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
