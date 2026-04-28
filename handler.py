"""RunPod serverless handler for Hallo2 audio-driven portrait animation.

Inference contract:
  Input:
    {
      "image_url":  "<presigned URL to the still PNG>",
      "audio_url":  "<presigned URL to the WAV/MP3 chunk>",
      "pose_weight":  1.0,    # optional; controls head motion intensity
      "face_weight":  1.0,    # optional; expression intensity
      "lip_weight":   1.0,    # optional; lip-sync strength
      "face_expand_ratio": 1.2,  # optional; crop padding around face
      "output_upload_url":  "<presigned PUT URL>"  # where to deposit the mp4
    }
  Output:
    { "video_url": "<the same upload URL, now populated>",
      "duration_s": 30.5, "model_seconds": 18.4 }

The handler does:
  1. ensure model weights (download once into /runpod-volume on cold)
  2. fetch image + audio to local /tmp
  3. run Hallo2 inference_long.py via subprocess
  4. PUT the result back to output_upload_url
  5. return {"video_url": ...}

Errors return {"error": "...message..."} so the client can retry.
"""
from __future__ import annotations

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

# Make sure the model-download helper is importable.
sys.path.insert(0, "/opt")
from download_models import ensure_models  # noqa: E402

HALLO2_DIR = Path("/opt/hallo2")
INFERENCE_SCRIPT = HALLO2_DIR / "scripts" / "inference_long.py"
DEFAULT_CONFIG = HALLO2_DIR / "configs" / "inference" / "long.yaml"


def _download(url: str, dest: Path) -> None:
    """Stream a presigned URL to disk."""
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def _upload(url: str, src: Path) -> None:
    """PUT a local file to a presigned upload URL."""
    with src.open("rb") as f:
        r = requests.put(url, data=f, timeout=300,
                         headers={"Content-Type": "video/mp4"})
        r.raise_for_status()


def _ensure_pretrained_symlink() -> None:
    """Hallo2's scripts hardcode `./pretrained_models` relative to the
    inference cwd. We keep weights on the persistent volume; symlink
    them into the repo dir on first use of each cold start."""
    target = HALLO2_DIR / "pretrained_models"
    if target.exists() or target.is_symlink():
        return
    models_dir = ensure_models()
    target.symlink_to(models_dir, target_is_directory=True)


def handler(event: dict[str, Any]) -> dict[str, Any]:
    inp = event.get("input", {}) or {}
    image_url = inp.get("image_url")
    audio_url = inp.get("audio_url")
    upload_url = inp.get("output_upload_url")

    if not (image_url and audio_url and upload_url):
        return {"error": "image_url, audio_url, and output_upload_url required"}

    pose_w = float(inp.get("pose_weight", 1.0))
    face_w = float(inp.get("face_weight", 1.0))
    lip_w = float(inp.get("lip_weight", 1.0))
    face_expand = float(inp.get("face_expand_ratio", 1.2))

    t0 = time.time()
    try:
        _ensure_pretrained_symlink()
    except Exception as e:
        return {"error": f"model setup failed: {e!r}"}

    workdir = Path(tempfile.mkdtemp(prefix=f"hallo2_{uuid.uuid4().hex[:8]}_",
                                    dir="/runpod-volume" if Path("/runpod-volume").exists() else None))
    image_path = workdir / "source.png"
    audio_path = workdir / "audio.wav"
    output_path = workdir / "out.mp4"

    try:
        _download(image_url, image_path)
        _download(audio_url, audio_path)

        # Hallo2's inference_long.py expects to be invoked from its own
        # cwd so its `pretrained_models` and `configs` paths resolve.
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

        _upload(upload_url, output_path)
        return {
            "video_url": upload_url.split("?")[0],
            "duration_s": round(time.time() - t0, 2),
            "model_seconds": round(infer_secs, 2),
        }
    except Exception as e:
        return {"error": f"handler exception: {e!r}"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
