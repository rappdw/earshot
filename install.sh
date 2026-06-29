#!/usr/bin/env bash
# install.sh - install earshot for the current user.
#
#   bin/earshot      -> ~/.local/bin/earshot            (always updated)
#   python/*.py      -> ~/.local/share/earshot/*.py
#   etc/earshot.conf -> ~/.config/earshot/earshot.conf (preserved if it exists)
#
# The venv lives at ~/.local/share/earshot/venv. --with-python installs the
# transcription deps; --with-diarize adds the (heavier) pyannote diarization deps.
#
# Usage:
#   ./install.sh                 # install/update scripts; never clobbers config
#   ./install.sh --with-python   # create/refresh venv + transcription deps
#   ./install.sh --with-diarize  # also add diarization deps (pyannote/torch)
#   ./install.sh --force         # also overwrite an existing config (backs it up)
#   ./install.sh --uninstall

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DST="${HOME}/.local/bin/earshot"
CONF_DIR="${HOME}/.config/earshot"
CONF_DST="${CONF_DIR}/earshot.conf"
SHARE_DIR="${HOME}/.local/share/earshot"
VENV_DIR="${SHARE_DIR}/venv"

FORCE=0
UNINSTALL=0
WITH_PYTHON=0
WITH_DIARIZE=0
for arg in "$@"; do
  case "$arg" in
    --force)        FORCE=1 ;;
    --uninstall)    UNINSTALL=1 ;;
    --with-python)  WITH_PYTHON=1 ;;
    --with-diarize) WITH_DIARIZE=1 ;;
    -h|--help)   sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "install.sh: unknown argument: $arg" >&2; exit 1 ;;
  esac
done

if [ "${UNINSTALL}" -eq 1 ]; then
  rm -f "${BIN_DST}"
  echo "removed ${BIN_DST}"
  echo "left config in place: ${CONF_DST} (remove by hand if you want it gone)"
  echo "left venv/share in place: ${SHARE_DIR} (remove by hand if you want it gone)"
  exit 0
fi

ensure_venv() {
  PYBIN="${EARSHOT_PYTHON:-python3}"
  if ! command -v "${PYBIN}" >/dev/null 2>&1; then
    echo "install.sh: ${PYBIN} not found; cannot build venv" >&2; exit 1
  fi
  if [ ! -d "${VENV_DIR}" ]; then
    echo "creating venv at ${VENV_DIR} ..."
    "${PYBIN}" -m venv "${VENV_DIR}"
  fi
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip >/dev/null
}

# --- sanity: sources present ------------------------------------------------
for f in bin/earshot \
         python/transcribe.py python/diarize.py python/speakers.py python/summarize.py \
         python/requirements.txt python/requirements-diarize.txt \
         etc/earshot.conf; do
  if [ ! -f "${SRC_DIR}/${f}" ]; then
    echo "install.sh: ${SRC_DIR}/${f} not found" >&2; exit 1
  fi
done

# --- install the scripts (always update) ------------------------------------
mkdir -p "${HOME}/.local/bin" "${SHARE_DIR}"
install -m 0755 "${SRC_DIR}/bin/earshot" "${BIN_DST}"
echo "installed ${BIN_DST}"
install -m 0644 "${SRC_DIR}/python/transcribe.py" "${SHARE_DIR}/transcribe.py"
echo "installed ${SHARE_DIR}/transcribe.py"
install -m 0644 "${SRC_DIR}/python/diarize.py" "${SHARE_DIR}/diarize.py"
echo "installed ${SHARE_DIR}/diarize.py"
install -m 0644 "${SRC_DIR}/python/speakers.py" "${SHARE_DIR}/speakers.py"
echo "installed ${SHARE_DIR}/speakers.py"
install -m 0644 "${SRC_DIR}/python/summarize.py" "${SHARE_DIR}/summarize.py"
echo "installed ${SHARE_DIR}/summarize.py"

# --- install the config (preserve existing) ---------------------------------
mkdir -p "${CONF_DIR}"
if [ -f "${CONF_DST}" ] && [ "${FORCE}" -eq 0 ]; then
  echo "kept existing config: ${CONF_DST} (use --force to overwrite)"
else
  if [ -f "${CONF_DST}" ]; then
    BACKUP="${CONF_DST}.bak.$(date +%Y%m%d%H%M%S)"
    cp "${CONF_DST}" "${BACKUP}"
    echo "backed up old config to ${BACKUP}"
  fi
  install -m 0644 "${SRC_DIR}/etc/earshot.conf" "${CONF_DST}"
  echo "installed ${CONF_DST}"
fi

# --- optional: build/refresh the venv ---------------------------------------
if [ "${WITH_PYTHON}" -eq 1 ] || [ "${WITH_DIARIZE}" -eq 1 ]; then
  ensure_venv
fi
if [ "${WITH_PYTHON}" -eq 1 ]; then
  echo "installing transcription deps (faster-whisper + ctranslate2) ..."
  "${VENV_DIR}/bin/python" -m pip install -r "${SRC_DIR}/python/requirements.txt"
  echo "transcription deps ready."
fi
if [ "${WITH_DIARIZE}" -eq 1 ]; then
  echo "installing diarization deps (pyannote.audio + torch; this is large) ..."
  "${VENV_DIR}/bin/python" -m pip install -r "${SRC_DIR}/python/requirements-diarize.txt"
  echo "diarization deps ready."
fi

# --- environment checks ------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not found on PATH. Install it: brew install ffmpeg" >&2
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "NOTE: venv not built yet. Run: ./install.sh --with-python (and --with-diarize)" >&2
fi

case ":${PATH}:" in
  *":${HOME}/.local/bin:"*) ;;
  *)
    echo
    echo "NOTE: ${HOME}/.local/bin is not on your PATH. Add this to your shell rc:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    ;;
esac

echo
echo "Done. Try:  earshot devices"
