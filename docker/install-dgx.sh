#!/usr/bin/env bash
# install-dgx.sh - build the earshot CUDA image on a DGX Spark and prepare the
# container-offload wrapper. Run this ON THE SPARK, from a clone of the repo:
#
#   git clone https://github.com/rappdw/earshot.git
#   cd earshot && ./docker/install-dgx.sh
#
# Then on the Mac set, in ~/.config/earshot/earshot.conf:
#   EARSHOT_SPARK_EARSHOT="~/earshot/docker/earshot-container"
#
# Options:
#   --base TAG     NGC PyTorch base image (default below). Match your DGX OS/driver.
#   --image NAME   image tag to build (default earshot:latest)
#   --runtime R    docker | podman (default docker)
#   --no-verify    skip the post-build CUDA check
#   -h, --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE="nvcr.io/nvidia/pytorch:25.06-py3"
IMAGE="earshot:latest"
RUNTIME="docker"
VERIFY=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --base)      BASE="$2"; shift 2 ;;
    --image)     IMAGE="$2"; shift 2 ;;
    --runtime)   RUNTIME="$2"; shift 2 ;;
    --no-verify) VERIFY=0; shift ;;
    -h|--help)   sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "install-dgx.sh: unknown argument: $1" >&2; exit 1 ;;
  esac
done

# --- preflight --------------------------------------------------------------
if ! command -v "${RUNTIME}" >/dev/null 2>&1; then
  echo "error: '${RUNTIME}' not found. Install Docker (or pass --runtime podman)." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found - is this the Spark with the GPU driver?" >&2
else
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
fi
if [ ! -f "${SCRIPT_DIR}/Dockerfile" ] || [ ! -d "${REPO_DIR}/python" ]; then
  echo "error: run this from a full clone of the repo (Dockerfile / python/ missing)" >&2
  exit 1
fi

# --- build ------------------------------------------------------------------
echo "==> building ${IMAGE} from ${BASE} (this pulls the NGC base; first time is slow)"
if ! "${RUNTIME}" build -f "${SCRIPT_DIR}/Dockerfile" \
       --build-arg BASE="${BASE}" -t "${IMAGE}" "${REPO_DIR}"; then
  echo >&2
  echo "build failed. Common causes:" >&2
  echo "  - not logged in to NGC:   ${RUNTIME} login nvcr.io   (use your NGC API key)" >&2
  echo "  - base tag unavailable for this system: override with --base <tag>" >&2
  exit 1
fi

chmod +x "${SCRIPT_DIR}/earshot-container"
echo "==> built ${IMAGE} and made docker/earshot-container executable"

# --- verify CUDA inside the image ------------------------------------------
if [ "${VERIFY}" -eq 1 ]; then
  echo "==> checking CUDA inside the image"
  if ! "${RUNTIME}" run --rm -i --gpus all "${IMAGE}" python - <<'PY'
import torch, ctranslate2
print("  torch CUDA:", torch.cuda.is_available())
print("  ctranslate2 CUDA devices:", ctranslate2.get_cuda_device_count())
print("  -> diarization will use", "GPU" if torch.cuda.is_available() else "CPU")
print("  -> transcription will use",
      "GPU" if ctranslate2.get_cuda_device_count() > 0 else "CPU (no CUDA ctranslate2 wheel for aarch64)")
PY
  then
    echo "WARNING: could not run the image with --gpus all." >&2
    echo "  Check the NVIDIA Container Toolkit:" >&2
    echo "  ${RUNTIME} run --rm --gpus all nvcr.io/nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi" >&2
  fi
fi

# --- next steps -------------------------------------------------------------
cat <<EOF

Done. On the Mac, set in ~/.config/earshot/earshot.conf:

  EARSHOT_SPARK_EARSHOT="${SCRIPT_DIR}/earshot-container"

(or the ~ form, e.g. "~/earshot/docker/earshot-container"), then run:

  earshot offload <meeting-dir> --diarize

See docker/README.md for details (GPU/CPU split, GPU transcription, podman).
EOF
