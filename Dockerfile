# Hallo2 RunPod serverless worker.
#
# Builds a CUDA image containing Hallo2 (audio-driven portrait animation)
# wrapped in a RunPod handler. Inference contract:
#   input:  { image_url, audio_url, [pose_weight, face_weight, lip_weight] }
#   output: { video_url }            (uploaded back to a presigned URL)
#
# Model weights are NOT baked in (they're ~12-15 GB across 6 sub-models).
# On first warmup, the handler downloads them to /runpod-volume/models/
# so subsequent cold starts on the same worker reuse them. RunPod
# attaches a persistent network volume per endpoint when configured;
# without it, each cold start re-downloads (~3-5 min penalty).

FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/runpod-volume/hf_cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf_cache \
    HALLO2_MODELS=/runpod-volume/models

# System deps. ffmpeg is required by Hallo2 for video assembly; git-lfs
# for the huggingface model pull; libsndfile for soundfile loading.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    git-lfs \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Clone Hallo2 at a pinned commit. The README's `inference_long.py` is
# the long-form entry point we want.
RUN git clone https://github.com/fudan-generative-vision/hallo2.git && \
    cd hallo2 && \
    git checkout main

WORKDIR /opt/hallo2

# Hallo2 requirements. We install on top of the base image's torch so
# we don't downgrade. Pin to known-working versions where possible.
COPY requirements.txt /tmp/worker_requirements.txt
# Hallo2's gradio==4.36.1 tries to upgrade blinker, which is a
# distutils-installed package on the Ubuntu base — pip can't uninstall
# it. Force-overwrite via --ignore-installed first.
RUN pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r /tmp/worker_requirements.txt

# Worker handler.
COPY handler.py /opt/handler.py
COPY download_models.py /opt/download_models.py

# RunPod expects the handler at the path given in CMD.
CMD ["python", "-u", "/opt/handler.py"]
