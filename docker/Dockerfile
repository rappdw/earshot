# syntax=docker/dockerfile:1
#
# earshot transcription + diarization image for the DGX Spark (aarch64 + CUDA).
#
# Base: an NVIDIA NGC PyTorch image, which already ships a CUDA-enabled torch
# built for the Spark's aarch64 Blackwell GPU. Pick the tag NVIDIA recommends
# for your DGX OS / driver and pass it as BASE, e.g.:
#   docker build -f docker/Dockerfile --build-arg BASE=nvcr.io/nvidia/pytorch:25.06-py3 -t earshot:latest .
# (run from the repo root so the COPY below can see python/)
#
# What runs on GPU vs CPU in this image:
#   - diarization (pyannote) uses the base image's CUDA torch -> GPU.
#   - transcription (faster-whisper) uses ctranslate2, whose aarch64 wheel is
#     CPU-only -> CPU. To get GPU transcription you must build CTranslate2 from
#     source with CUDA; see docker/README.md. Until then run with --device auto
#     (offload's default), which sends transcribe to CPU and diarize to GPU.
ARG BASE=nvcr.io/nvidia/pytorch:25.06-py3
FROM ${BASE}

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# torch is already present (CUDA) in the base image. Pin it so installing the
# earshot deps can't silently replace it with a CPU wheel from PyPI.
RUN TORCHVER="$(python -c 'import torch; print(torch.__version__)')" \
 && echo "torch==${TORCHVER}" > /tmp/constraints.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt \
      "faster-whisper>=1.0.3" "pyannote.audio>=3.1"

# earshot's Python steps. diarize.py imports speakers.py, so both must be here.
COPY python/transcribe.py python/diarize.py python/speakers.py /opt/earshot/

WORKDIR /work
