# syntax=docker/dockerfile:1
#
# earshot transcription + diarization image for the DGX Spark (aarch64 + CUDA),
# with a CUDA build of CTranslate2 so faster-whisper transcription runs on GPU.
#
# Base: an NVIDIA NGC PyTorch image (CUDA torch + full CUDA toolkit + cuDNN for
# aarch64 Blackwell). Pick the tag NVIDIA recommends for your DGX OS / driver.
#
# Build from the repo root (so COPY python/ works). Set CUDA_ARCH to your GPU's
# compute capability with no dot (nvidia-smi --query-gpu=compute_cap -> "12.1"
# -> 121). docker/install-dgx.sh auto-detects and passes it for you:
#   docker build -f docker/Dockerfile \
#     --build-arg BASE=nvcr.io/nvidia/pytorch:25.06-py3 \
#     --build-arg CUDA_ARCH=121 -t earshot:latest .
#
# Set CT2_CUDA=0 to skip the (long) source build and use the CPU-only ctranslate2
# wheel instead; transcription then runs on CPU, diarization still on GPU.
ARG BASE=nvcr.io/nvidia/pytorch:25.06-py3
FROM ${BASE}

ARG CT2_CUDA=1
ARG CT2_VERSION=v4.6.0
ARG CUDA_ARCH=121

ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# --- CUDA build of CTranslate2 (the GPU transcription path) ------------------
# Builds libctranslate2 with CUDA/cuDNN against the base image's toolkit, then
# the matching Python bindings. OpenBLAS + Ruy provide the CPU backend (MKL is
# x86-only). CUDA_DYNAMIC_LOADING dlopens cuBLAS/cuDNN at runtime.
RUN if [ "${CT2_CUDA}" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git libopenblas-dev && rm -rf /var/lib/apt/lists/* && \
      git clone --depth 1 --branch "${CT2_VERSION}" --recursive \
        https://github.com/OpenNMT/CTranslate2 /tmp/ct2 && \
      cmake -S /tmp/ct2 -B /tmp/ct2/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCUDA_DYNAMIC_LOADING=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DWITH_RUY=ON -DOPENMP_RUNTIME=COMP && \
      cmake --build /tmp/ct2/build -j"$(nproc)" --target install && \
      ldconfig && \
      CTRANSLATE2_ROOT=/usr/local pip install --no-cache-dir /tmp/ct2/python && \
      rm -rf /tmp/ct2 ; \
    fi

# --- earshot Python deps ----------------------------------------------------
# Pin torch (and the source-built ctranslate2, if present) so installing
# faster-whisper / pyannote can't replace them with CPU wheels from PyPI.
# pyannote pulls torchaudio. Off-the-shelf torchaudio wheels hard-pin an exact
# torch (e.g. torch==2.8.0), which conflicts with the NGC dev torch
# (2.8.0a0+...nv), and the PyPI aarch64 wheel is a CUDA-13 build that mismatches
# the CUDA-12 base (-> "libcudart.so.13 not found"). The only torchaudio that
# fits the NGC torch is the one already in the base image. So pin torch +
# torchaudio (+ the source-built ctranslate2) to the installed versions and let
# pip keep them. importlib.metadata reads versions without importing the native
# libs; `cat` echoes the pins into the build log for debugging.
RUN python -c "import importlib.metadata as m; print('torch=='+m.version('torch'))" > /tmp/constraints.txt && \
    python -c "import importlib.metadata as m; print('torchaudio=='+m.version('torchaudio'))" >> /tmp/constraints.txt && \
    (python -c "import importlib.metadata as m; print('ctranslate2=='+m.version('ctranslate2'))" >> /tmp/constraints.txt || true) && \
    cat /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt \
      "faster-whisper>=1.0.3" "pyannote.audio>=3.1"

# earshot's Python steps. diarize.py imports speakers.py, so both must be here.
COPY python/transcribe.py python/diarize.py python/speakers.py python/summarize.py /opt/earshot/

WORKDIR /work
