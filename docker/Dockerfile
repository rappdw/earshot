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
# pyannote needs torchaudio, but the NGC base doesn't ship one, and neither
# off-the-shelf option works against the NGC dev torch (2.8.0a0+...nv): the PyPI
# aarch64 wheel is CUDA-13 (-> "libcudart.so.13 not found" at runtime), and the
# stable wheel hard-pins torch==2.8.0 (resolver conflict).
#
# So: install pyannote/faster-whisper first (pip pulls the CUDA-13 torchaudio,
# but torch is pinned and stays the NGC build), THEN force-replace torchaudio
# with the CPU wheel matching torch's minor, using --no-deps so its torch== pin
# is ignored. The CPU torchaudio has no CUDA runtime dependency (always imports)
# and pyannote still runs its models on the GPU via the NGC torch. The final
# import is a build-time check that torchaudio loads.
RUN python -c "import torch; print('torch=='+torch.__version__)" > /tmp/constraints.txt && \
    (python -c "import ctranslate2; print('ctranslate2=='+ctranslate2.__version__)" >> /tmp/constraints.txt || true) && \
    pip install --no-cache-dir -c /tmp/constraints.txt \
      "huggingface_hub<1.0" "faster-whisper>=1.0.3" "pyannote.audio>=3.1" && \
    TMINOR="$(python -c 'import torch; v=torch.__version__.split("+")[0].split("."); print(v[0]+"."+v[1])')" && \
    echo "force-installing CPU torchaudio==${TMINOR}.* (no-deps) over the CUDA-13 wheel" && \
    pip install --no-cache-dir --no-deps --force-reinstall \
      --index-url https://download.pytorch.org/whl/cpu "torchaudio==${TMINOR}.*" && \
    python -c "import torchaudio; print('torchaudio OK', torchaudio.__version__)"

# earshot's Python steps. diarize.py imports speakers.py, so both must be here.
COPY python/transcribe.py python/diarize.py python/speakers.py python/summarize.py /opt/earshot/

WORKDIR /work
