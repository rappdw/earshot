# Running offload in a container on the DGX Spark

This runs the remote transcribe/diarize steps inside a CUDA container instead of
a bare venv. It gives you a reproducible GPU environment (built on NVIDIA's NGC
PyTorch image, which already has a working aarch64/Blackwell CUDA torch) and is
the clean place to add a CUDA build of CTranslate2 later.

`earshot offload` is unchanged — you just point its **remote command** at the
`earshot-container` wrapper, which bind-mounts the meeting dir into the container.

## What runs on GPU

- **Diarization (pyannote)** → GPU, via the base image's CUDA torch.
- **Transcription (faster-whisper)** → CPU. `ctranslate2`'s aarch64 wheel is
  CPU-only; the Spark's 20-core CPU handles it fine. To move it to the GPU you'd
  build CTranslate2 from source with CUDA (see the bottom of this file).

`earshot offload` defaults to `--device auto`, so each step lands on the right
device without any flag.

## One-time setup on the Spark

1. **Prerequisites** (DGX OS ships these; verify):
   - Docker (or Podman) and the NVIDIA Container Toolkit, so `--gpus all` works:
     ```bash
     docker run --rm --gpus all nvcr.io/nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
     ```
   - Your user can run `docker` without sudo (in the `docker` group), since the
     offload job runs as you over SSH. Otherwise use `EARSHOT_CONTAINER_RUNTIME=podman`.
   - Access to `nvcr.io` for the base image: `docker login nvcr.io` with your NGC
     API key (DGX systems are usually already entitled).

2. **Clone the repo** (you may already have it from the venv install):
   ```bash
   git clone https://github.com/rappdw/earshot.git
   cd earshot
   ```

3. **Build the image.** Pick the NGC PyTorch tag NVIDIA recommends for your DGX
   OS / driver and pass it as `BASE`. Build from the repo root so the Dockerfile
   can copy `python/`:
   ```bash
   docker build -f docker/Dockerfile \
     --build-arg BASE=nvcr.io/nvidia/pytorch:25.06-py3 \
     -t earshot:latest .
   ```

4. **Make the wrapper executable:**
   ```bash
   chmod +x docker/earshot-container
   ```

## One-time setup on the Mac

Point the offload's remote command at the wrapper, in `~/.config/earshot/earshot.conf`:

```sh
EARSHOT_SPARK_HOST="dgx"
EARSHOT_SPARK_EARSHOT="~/earshot/docker/earshot-container"   # path to the wrapper on the Spark
```

Keep `HF_TOKEN` exported in your Mac shell (forwarded into the container for the
diarization model download).

## Run

Exactly the same commands as before — they now execute in the container:

```bash
earshot offload ~/Notes/meetings/business/SOMEDIR --diarize        # transcribe (CPU) + diarize (GPU)
earshot offload ~/Notes/meetings/business/SOMEDIR --no-watch       # no live mosh attach
earshot offload ~/Notes/meetings/business/SOMEDIR --diarize --rm-remote
```

The first diarize run downloads the pyannote model into the cache (needs
internet); it's persisted in `~/.cache/earshot` on the Spark and reused after.

## How the wrapper maps things

`earshot offload` runs, from inside the meeting's staging dir, the equivalent of:

```
earshot-container transcribe . --device auto
earshot-container diarize    . --device auto
```

The wrapper turns each into:

```
docker run --rm --gpus all --user <you> \
  -e HF_TOKEN -e EARSHOT_SPEAKERS \
  -v <staging dir>:/work -w /work \
  -v ~/.cache/earshot:/cache \
  earshot:latest python /opt/earshot/<step>.py . --device auto
```

So `.` is the meeting dir, `speakers.json` (synced by offload) is read from it,
outputs are written back to it owned by you, and models are cached in
`~/.cache/earshot`. Knobs: `EARSHOT_IMAGE`, `EARSHOT_CONTAINER_RUNTIME`
(docker|podman), `EARSHOT_CACHE_DIR`.

## Optional: GPU transcription (CUDA CTranslate2)

The stock image transcribes on CPU because there's no prebuilt CUDA `ctranslate2`
for aarch64. To move transcription onto the GPU, extend the Dockerfile to build
CTranslate2 from source with CUDA against the base image's toolkit:

- clone `OpenNMT/CTranslate2`, `cmake -DWITH_CUDA=ON -DWITH_CUDNN=ON` with the
  right `CMAKE_CUDA_ARCHITECTURES` for the Spark's GPU (check `nvidia-smi`), build,
  then `pip install` the `python/` bindings against the built library.
- rebuild the image and run offload with `--device cuda`.

This is involved and the CUDA-arch support must match Blackwell; do it only if
CPU transcription on the Spark is too slow for you. If you want, the alternative
is adding a `whisper.cpp` CUDA backend to earshot, which builds on ARM far more
easily — open an issue / ask.
