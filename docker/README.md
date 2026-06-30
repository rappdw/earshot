# Running offload in a container on the DGX Spark

This runs the remote transcribe/diarize steps inside a CUDA container instead of
a bare venv. It gives you a reproducible GPU environment (built on NVIDIA's NGC
PyTorch image, which already has a working aarch64/Blackwell CUDA torch) and is
the clean place to add a CUDA build of CTranslate2 later.

`earshot offload` is unchanged — you just point its **remote command** at the
`earshot-container` wrapper, which bind-mounts the meeting dir into the container.

## What runs on GPU

Both, by default:

- **Diarization (pyannote)** → GPU, via the base image's CUDA torch.
- **Transcription (faster-whisper)** → GPU, via a **CUDA build of CTranslate2**
  compiled from source in the image (PyPI ships no CUDA `ctranslate2` for
  aarch64). `install-dgx.sh` auto-detects your GPU's compute capability and
  passes it to the build. Pass `--no-ct2-cuda` to skip the (long) source build
  and transcribe on CPU instead.

`earshot offload` defaults to `--device auto`, so each step lands on the right
device without any flag.

## Quick setup on the Spark

After cloning the repo on the Spark, one command builds the image, fixes the
wrapper perms, and verifies CUDA:

```bash
cd ~/earshot
./docker/install-dgx.sh                                   # uses the default NGC base tag
./docker/install-dgx.sh --base nvcr.io/nvidia/pytorch:25.06-py3   # or pin a tag
```

It prints the exact `EARSHOT_SPARK_EARSHOT` line to set on the Mac when it's
done. The manual steps below are the same thing, broken out, if you'd rather run
them yourself or the script hits something specific to your box.

## One-time setup on the Spark (manual)

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
   OS / driver (`BASE`) and set `CUDA_ARCH` to your GPU's compute capability with
   no dot (`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` → `12.1` →
   `121`). Build from the repo root so the Dockerfile can copy `python/`:
   ```bash
   docker build -f docker/Dockerfile \
     --build-arg BASE=nvcr.io/nvidia/pytorch:25.06-py3 \
     --build-arg CUDA_ARCH=121 \
     -t earshot:latest .
   ```
   This compiles CUDA CTranslate2 from source (tens of minutes). Add
   `--build-arg CT2_CUDA=0` for the fast CPU-transcription image instead.
   `install-dgx.sh` does all of this (and auto-detects `CUDA_ARCH`) for you.

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
earshot offload ~/Notes/meetings/business/SOMEDIR --diarize        # transcribe + diarize (GPU)
earshot offload ~/Notes/meetings/business/SOMEDIR --no-watch       # no live mosh attach
earshot offload ~/Notes/meetings/business/SOMEDIR --diarize --rm-remote
```

## Offloading summarization too (Ollama on the Spark)

`--summarize` adds a notes step that runs in the container against an **Ollama
server on the Spark host** — so the Spark's GPU runs the summary LLM as well.

One-time on the Spark:
```bash
# install Ollama (https://ollama.com/download/linux), then pull a model:
ollama pull llama3.1
# verify it's serving (default 127.0.0.1:11434):
curl -s localhost:11434/api/tags >/dev/null && echo ok
```

Then the full pipeline in one command:
```bash
earshot offload ~/Notes/meetings/business/SOMEDIR --diarize --summarize
earshot offload DIR --diarize --summarize --sum-args "--model qwen2.5:14b"   # pick the Ollama model
```

`notes.md` is generated on the Spark and pulled back alongside the transcript.
The summarize step runs the container with `--network host` so it reaches Ollama
at `localhost:11434` (Ollama's default bind), and defaults to the `llama3.1`
model — pull that one or pass `--sum-args "--model <name>"` for another. To use
hosted Claude instead (sends text off-box), `--sum-args "--backend claude"`;
offload forwards your `ANTHROPIC_API_KEY` for it.

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

## GPU transcription internals & troubleshooting

The image builds `libctranslate2` + its Python bindings from source with
`-DWITH_CUDA=ON -DWITH_CUDNN=ON` against the base image's CUDA toolkit, pinned to
your GPU's `CMAKE_CUDA_ARCHITECTURES`. `torch` (and the built `ctranslate2`) are
version-pinned before installing faster-whisper/pyannote so PyPI can't swap in a
CPU wheel.

If the post-build check shows **transcription will use CPU** even though you ran
the CUDA build:

- The `CUDA_ARCH` likely doesn't match the GPU. Re-run with the exact value from
  `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` (drop the dot).
- The `CTranslate2` tag may predate your CUDA toolkit / Blackwell. Try a newer
  one: `./docker/install-dgx.sh --ct2-version vX.Y.Z`.

If the source build is more trouble than it's worth on your setup, run
`./docker/install-dgx.sh --no-ct2-cuda` for CPU transcription (diarization still
uses the GPU). A `whisper.cpp` CUDA backend (easier to build on ARM) is a
possible future alternative — ask if you want it.
