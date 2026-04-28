# StoryForge Hallo2 RunPod worker

Serverless worker that runs Fudan's [Hallo2](https://github.com/fudan-generative-vision/hallo2)
model. Given a still portrait + a short audio clip, returns a lip-synced
talking-head video.

## Architecture

```
storyforge pipeline               RunPod serverless endpoint
─────────────────────             ──────────────────────────
generate TTS mp3 ──── chunks ──▶  handler.py
                                    ├── ensure_models() (cold start only)
                                    ├── fetch image_url + audio_url
                                    ├── subprocess: hallo2 inference_long.py
                                    └── PUT result to output_upload_url
                  ◀──── face mp4 ──┘
composite as PiP overlay
on existing chess scene render
```

Model weights (~12-15 GB across hallo2/audio_separator/face_analysis/
motion_module/stable-diffusion-v1-5/wav2vec) live on the endpoint's
persistent network volume so they're downloaded once and reused across
warm starts.

## Build & push

```sh
cd runpod_worker
docker build --platform linux/amd64 -t USER/storyforge-hallo2:0.1.0 .
docker push USER/storyforge-hallo2:0.1.0
```

The `--platform linux/amd64` flag is critical when building on Apple
Silicon — RunPod GPUs are x86 and silently fail on arm64 images.

## Deploy on RunPod

1. RunPod console → Serverless → New Endpoint → "Use your own Repository"
   → choose "Docker Image"
2. Image URL: `docker.io/USER/storyforge-hallo2:0.1.0`
3. Container disk: 30 GB (we extract weights to /runpod-volume so disk
   only needs the image footprint + scratch)
4. Network volume: attach a 50 GB persistent volume mounted at
   `/runpod-volume` (cuts cold-start to ~30s after first warmup)
5. GPU: A40 / RTX 4090 / A6000 — anything with ≥24 GB VRAM
6. Min workers: 0  ·  Max workers: 1 (raise as needed)
7. Idle timeout: 30s, execution timeout: 2400s
8. Save the endpoint ID — paste it as `RUNPOD_ENDPOINT_ID` in the
   storyforge `.env`.

## Test

```sh
curl -X POST \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run \
  -d '{
    "input": {
      "image_url": "https://example.com/nyx.png",
      "audio_url": "https://example.com/clip.wav",
      "output_upload_url": "https://s3.../upload.mp4?signed=..."
    }
  }'
```

The first run will trigger model download (~5 min). Subsequent runs on
the same worker should complete inference in ~1-2x audio duration.
