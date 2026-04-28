"""One-time model download into the persistent volume.

Run on first warmup. Subsequent cold starts skip if files exist.
Hallo2's pretrained_models bundle is fetched from the Fudan HF repo;
expected total ≈12-15 GB so this takes 3-8 min depending on the pod's
network. Kept idempotent so partial downloads can resume.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


HALLO2_REPO = "fudan-generative-ai/hallo2"
MODELS_DIR = Path(os.environ.get("HALLO2_MODELS", "/runpod-volume/models"))
SENTINEL = MODELS_DIR / ".hallo2_downloaded"


def ensure_models() -> Path:
    """Download Hallo2 pretrained_models if not already present.

    Returns the path that contains the unpacked weights — Hallo2's
    inference scripts expect this to be `./pretrained_models/` in the
    working dir, so the caller symlinks it into place.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if SENTINEL.exists():
        print(f"[hallo2-dl] models already present at {MODELS_DIR}, skipping",
              flush=True)
        return MODELS_DIR

    print(f"[hallo2-dl] downloading {HALLO2_REPO} → {MODELS_DIR}", flush=True)
    snapshot_download(
        repo_id=HALLO2_REPO,
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8,
    )
    SENTINEL.touch()
    print(f"[hallo2-dl] done", flush=True)
    return MODELS_DIR


if __name__ == "__main__":
    try:
        ensure_models()
    except Exception as e:
        print(f"[hallo2-dl] FAILED: {e!r}", file=sys.stderr, flush=True)
        sys.exit(1)
