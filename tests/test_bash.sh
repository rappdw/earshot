#!/usr/bin/env bash
# Integration tests for bin/earshot, using stubbed ffmpeg/ssh/rsync/mosh.
# Run via tests/run.sh (or directly). Exits non-zero on any failure.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EARSHOT="${HERE}/../bin/earshot"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

PASS=0 FAIL=0
ok()   { PASS=$((PASS+1)); echo "  ok: $1"; }
fail() { FAIL=$((FAIL+1)); echo "  FAIL: $1"; }
check() { # check <desc> <expected-substr> <actual>
  case "$3" in *"$2"*) ok "$1" ;; *) fail "$1 (wanted '$2' in: $3)" ;; esac
}
check_not() {
  case "$3" in *"$2"*) fail "$1 (did NOT want '$2' in: $3)" ;; *) ok "$1" ;; esac
}

# --- stubs -------------------------------------------------------------------
STUB="${WORK}/stub"; mkdir -p "${STUB}"
cat > "${STUB}/ffmpeg" <<'EOF'
#!/usr/bin/env bash
# device-listing stub: emit a canned avfoundation device list on stderr
cat >&2 <<'DEV'
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] Aggregate Macbook Air
[AVFoundation indev @ 0x1] [1] Dan's iPhone Microphone
[AVFoundation indev @ 0x1] [2] BlackHole 2ch
[AVFoundation indev @ 0x1] [3] OBSBOT Meet Microphone
DEV
exit 1
EOF
cat > "${STUB}/ssh" <<'EOF'
#!/usr/bin/env bash
echo "SSH|$*" >> "${TEST_LOG}"
case "$*" in
  *"echo done"*)           echo done ;;
  *"cat '"*"/.status'"*)   echo 0 ;;
  *"echo yes || echo no"*) echo "${TEST_SESSION_EXISTS:-no}" ;;
esac
EOF
cat > "${STUB}/rsync" <<'EOF'
#!/usr/bin/env bash
echo "RSYNC|$*" >> "${TEST_LOG}"
dest="${@: -1}"; src="${@: -2:1}"
case "$dest" in *.run.sh) cp "$src" "${TEST_LOG}.run.sh" ;; esac
EOF
cat > "${STUB}/mosh" <<'EOF'
#!/usr/bin/env bash
echo "MOSH|$*" >> "${TEST_LOG}"
EOF
chmod +x "${STUB}"/*
export PATH="${STUB}:${PATH}"

# --- test conf with contexts -------------------------------------------------
CONF="${WORK}/earshot.conf"
cat > "${CONF}" <<EOF
EARSHOT_SPARK_HOST="dgx"
EARSHOT_SPARK_EARSHOT="~/earshot/docker/earshot-container"
EARSHOT_DEFAULT_CONTEXT="business"
EARSHOT_OUT_ROOT_BUSINESS="${WORK}/meet/business"
EARSHOT_OUT_ROOT_PERSONAL="${WORK}/meet/personal"
EARSHOT_SPEAKERS_BUSINESS="${WORK}/biz.json"
EARSHOT_SPEAKERS_PERSONAL="${WORK}/per.json"
EARSHOT_OUT_ROOT="${WORK}/meet"
earshot_load_context() { case "\$1" in business) CONTEXT_OUT="\$EARSHOT_OUT_ROOT_BUSINESS"; CONTEXT_SPEAKERS="\$EARSHOT_SPEAKERS_BUSINESS";; personal) CONTEXT_OUT="\$EARSHOT_OUT_ROOT_PERSONAL"; CONTEXT_SPEAKERS="\$EARSHOT_SPEAKERS_PERSONAL";; *) return 1;; esac; }
earshot_load_profile() { case "\$1" in obsbot) MIC_NAME="OBSBOT Meet Microphone";; *) return 1;; esac; }
EOF
export EARSHOT_CONF="${CONF}"

# meeting fixture (in the personal tree so context inference has something to find)
M="${WORK}/meet/personal/2026-07-01_120000_test"; mkdir -p "${M}"
: > "${M}/far_remote.wav"; : > "${M}/near_me.wav"
echo '{"speakers":{"Bob":{"samples":[[1]]}}}' > "${WORK}/per.json"

# ==========================================================================
echo "[devices] parses stubbed avfoundation list"
OUT="$("${EARSHOT}" devices 2>&1)"
check "lists BlackHole with index" "2   BlackHole 2ch" "$OUT"
check "lists OBSBOT" "3   OBSBOT Meet Microphone" "$OUT"

# ==========================================================================
echo "[offload] default steps, env forwarding, notes pull"
export TEST_LOG="${WORK}/t1.log"; : > "${TEST_LOG}"
HF_TOKEN=hf_TEST ANTHROPIC_API_KEY=sk_TEST "${EARSHOT}" offload "${M}" --poll 1 >/dev/null 2>&1
RUN="$(cat "${TEST_LOG}.run.sh" 2>/dev/null)"
check "default includes transcribe" "earshot-container transcribe . --device auto" "$RUN"
check "default includes diarize" "earshot-container diarize . --device auto" "$RUN"
check_not "default excludes summarize" "summarize" "$RUN"
check "HF_TOKEN forwarded" "export HF_TOKEN='hf_TEST'" "$RUN"
check "speaker lib env set for diarize" "export EARSHOT_SPEAKERS=speakers.json" "$RUN"
check "exit code captured" 'PIPESTATUS[0]' "$RUN"
LOG="$(cat "${TEST_LOG}")"
check "quiet -T ssh everywhere" "SSH|-T -o LogLevel=ERROR dgx" "$LOG"
check "personal lib pushed (context inferred from path)" "per.json" "$LOG"
check "pull includes notes.md" "notes.md" "$LOG"
check_not "no mosh by default" "MOSH|" "$LOG"

# ==========================================================================
echo "[offload] generated runner executes, self-deletes, records status"
RUNDIR="${WORK}/exec"; mkdir -p "${RUNDIR}"
# stand in for the remote earshot command so the real pipeline line runs
FAKE="${WORK}/fake-earshot"; printf '#!/usr/bin/env bash\nexit 0\n' > "${FAKE}"; chmod +x "${FAKE}"
sed "s|~/earshot/docker/earshot-container|${FAKE}|g" "${TEST_LOG}.run.sh" > "${RUNDIR}/.run.sh"
bash "${RUNDIR}/.run.sh" >/dev/null 2>&1
[ "$(cat "${RUNDIR}/.status" 2>/dev/null)" = "0" ] && ok "status recorded 0" || fail "status recorded 0"
[ ! -f "${RUNDIR}/.run.sh" ] && ok "runner self-deleted (secrets off disk)" || fail "runner self-deleted"
[ -f "${RUNDIR}/run.log" ] && ok "run.log kept for debugging" || fail "run.log kept"
check "runner had secrets before running" "hf_TEST" "$(cat "${TEST_LOG}.run.sh")"

# ==========================================================================
echo "[offload] --summarize appends step; session guard refuses without --force"
export TEST_LOG="${WORK}/t2.log"; : > "${TEST_LOG}"
"${EARSHOT}" offload "${M}" --summarize --sum-args "--backend vllm --model x" --poll 1 >/dev/null 2>&1
RUN="$(cat "${TEST_LOG}.run.sh" 2>/dev/null)"
check "summarize step present with args" "earshot-container summarize . --backend vllm --model x" "$RUN"
check_not "summarize-only omits transcribe" "transcribe" "$RUN"

export TEST_LOG="${WORK}/t3.log"; : > "${TEST_LOG}"
OUT="$(TEST_SESSION_EXISTS=yes "${EARSHOT}" offload "${M}" --poll 1 2>&1)"; RC=$?
check "existing session refused" "already exists" "$OUT"
[ "$RC" -ne 0 ] && ok "guard exits non-zero" || fail "guard exits non-zero"

# ==========================================================================
echo "[context routing] speakers list picks per-context library"
OUT="$("${EARSHOT}" speakers -c personal list 2>&1)"
check "personal lib selected" "per.json" "$OUT"
check "Bob enrolled there" "Bob" "$OUT"
OUT="$("${EARSHOT}" speakers -c business list 2>&1)"
check "business lib selected" "biz.json" "$OUT"

# ==========================================================================
echo "[old config] context-less conf degrades gracefully"
OLDCONF="${WORK}/old.conf"
cat > "${OLDCONF}" <<EOF
EARSHOT_SPARK_HOST="dgx"
EARSHOT_SPARK_EARSHOT="cmd"
earshot_load_profile() { MIC_NAME="x"; }
EOF
export TEST_LOG="${WORK}/t4.log"; : > "${TEST_LOG}"
OUT="$(EARSHOT_CONF="${OLDCONF}" "${EARSHOT}" offload "${M}" --poll 1 2>&1)"
check_not "no 'command not found'" "command not found" "$OUT"
check "offload still completes" "done ->" "$OUT"

# ==========================================================================
echo "[edges] unsafe meeting-dir name rejected; json_escape handles backslash"
BAD="${WORK}/meet/personal/bad name'"; mkdir -p "${BAD}"; : > "${BAD}/far_remote.wav"
export TEST_LOG="${WORK}/t5.log"; : > "${TEST_LOG}"
OUT="$("${EARSHOT}" offload "${BAD}" --poll 1 2>&1)"; RC=$?
check "unsafe name rejected" "unsafe" "$OUT"
[ "$RC" -ne 0 ] && ok "unsafe name exits non-zero" || fail "unsafe name exits non-zero"

# extract json_escape from the script and exercise it
eval "$(sed -n '/^json_escape()/,/^}/p' "${EARSHOT}")"
ESC="$(json_escape 'a "quote" and \ backslash')"
OUT="$(printf '{"t": "%s"}' "${ESC}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["t"])')"
check "json_escape round-trips through a JSON parser" 'a "quote" and \ backslash' "$OUT"

echo "[edges] rec -c with a pre-context config gives a clear error"
OUT="$(EARSHOT_CONF="${OLDCONF}" "${EARSHOT}" rec -c business -t x 2>&1 </dev/null)"; RC=$?
check "clear pre-context error" "predates contexts" "$OUT"
[ "$RC" -ne 0 ] && ok "rec -c exits non-zero on old conf" || fail "rec -c exits non-zero"

# ==========================================================================
echo
echo "bash tests: ${PASS} passed, ${FAIL} failed"
[ "${FAIL}" -eq 0 ]
