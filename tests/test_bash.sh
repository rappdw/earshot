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
# record stub: for a capture invocation, write bytes to the output wav and
# exit 0 so rec's happy path is exercisable
case "$*" in
  *avfoundation*pcm_s16le*)
    [ -n "${FFCAP_LOG:-}" ] && echo "$*" >> "${FFCAP_LOG}"
    out="${!#}"; head -c 4096 /dev/zero > "$out"; exit 0 ;;
  *volumedetect*)
    # emit a max_volume line only when the test sets FF_MAXVOL (far-silence check)
    [ -n "${FF_MAXVOL:-}" ] && echo "[Parsed_volumedetect_0 @ 0x0] max_volume: ${FF_MAXVOL} dB" >&2
    exit 0 ;;
esac
# otherwise: device-listing stub, canned avfoundation list on stderr
cat >&2 <<'DEV'
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] Aggregate Macbook Air
[AVFoundation indev @ 0x1] [1] Dan's iPhone Microphone
[AVFoundation indev @ 0x1] [2] BlackHole 2ch
[AVFoundation indev @ 0x1] [3] OBSBOT Meet Microphone
[AVFoundation indev @ 0x1] [4] OBSBOT Meet 4K Microphone
DEV
exit 1
EOF
cat > "${STUB}/ssh" <<'EOF'
#!/usr/bin/env bash
echo "SSH|$*" >> "${TEST_LOG}"
case "$*" in
  *"tmux new-session"*)    if [ "${TEST_FAIL_LAUNCH:-0}" = "1" ]; then exit 1; fi ;;
  *"echo done"*)           echo done ;;
  *"cat '"*"/.status'"*)   echo 0 ;;
  *"echo yes || echo no"*) echo "${TEST_SESSION_EXISTS:-no}" ;;
esac
EOF
cat > "${STUB}/rsync" <<'EOF'
#!/usr/bin/env bash
echo "RSYNC|$*" >> "${TEST_LOG}"
# bash-3.2-safe last / second-to-last positional (macOS runners)
dest="${!#}"; n=$(($# - 1)); src="${!n}"
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
earshot_load_profile() { case "\$1" in obsbot) MIC_NAME="OBSBOT Meet Microphone|OBSBOT Meet 4K Microphone";; only4k) MIC_NAME="Not Present Mic|OBSBOT Meet 4K Microphone";; ghost) MIC_NAME="Ghost Mic";; *) return 1;; esac; }
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
echo "[rec] profile mic candidates resolve in order (multi-model support)"
# stub ffmpeg exits immediately, so rec finishes fast with empty-wav warnings;
# we only assert the device resolution lines.
OUT="$("${EARSHOT}" rec -p obsbot -t t1 -o "${WORK}/rec1" 2>&1 </dev/null || true)"
check "first candidate wins when present" "near 'OBSBOT Meet Microphone' (index 3)" "$OUT"
OUT="$("${EARSHOT}" rec -p only4k -t t2 -o "${WORK}/rec2" 2>&1 </dev/null || true)"
check "falls through to next candidate" "near 'OBSBOT Meet 4K Microphone' (index 4)" "$OUT"
OUT="$(EARSHOT_CONF="${CONF}" "${EARSHOT}" rec -p obsbot -t t3 -o "${WORK}/rec3" 2>&1 </dev/null || true)"
OUT3="${OUT}"   # preserve rec3 output; later blocks reassign OUT
J="$(cat "${WORK}"/rec3/*/meeting.json 2>/dev/null)"
check "meeting.json records the resolved name, not the list" '"near_device": "OBSBOT Meet Microphone"' "$J"
check "meeting.json mode is call for two-channel" '"mode": "call"' "$J"
check "meeting.json records a stop_reason" '"stop_reason":' "$J"
check "meeting.json records duration_sec" '"duration_sec":' "$J"
check "recording succeeded via stub" "earshot: done." "$OUT"

echo "[rec] auto-stop watches BOTH channels (silencedetect on far and near)"
FFCAP="${WORK}/ffcap.log"; : > "${FFCAP}"
FFCAP_LOG="${FFCAP}" "${EARSHOT}" rec -p obsbot -t sil -o "${WORK}/recsil" 2>&1 </dev/null >/dev/null || true
NSIL="$(grep -c "silencedetect" "${FFCAP}" 2>/dev/null)" || NSIL=0
check "both capture processes carry silencedetect" "2" "${NSIL}"
FFCAP2="${WORK}/ffcap2.log"; : > "${FFCAP2}"
FFCAP_LOG="${FFCAP2}" "${EARSHOT}" rec --in-person -p obsbot -t silip -o "${WORK}/recsilip" 2>&1 </dev/null >/dev/null || true
NSIL2="$(grep -c "silencedetect" "${FFCAP2}" 2>/dev/null)" || NSIL2=0
check "in-person carries silencedetect on its single channel" "1" "${NSIL2}"

echo "[rec] warns when the far (call) channel captured no audio"
OUT="$(FF_MAXVOL="-91.0" "${EARSHOT}" rec -p obsbot -t dead -o "${WORK}/recdead" 2>&1 </dev/null || true)"
check "far-silence warning fires on a dead call channel" "call (far) channel is essentially silent" "$OUT"
OUT="$(FF_MAXVOL="-12.0" "${EARSHOT}" rec -p obsbot -t live -o "${WORK}/reclive" 2>&1 </dev/null || true)"
check_not "no far-silence warning when the call channel is audible" "essentially silent" "$OUT"
check "monitor is off by default (json null)" '"monitor_stereo": null' "$J"
M3D="$(find "${WORK}/rec3" -mindepth 1 -maxdepth 1 -type d | head -1)"
if [ -n "${M3D}" ] && [ ! -f "${M3D}/monitor_stereo.wav" ]; then
  ok "no monitor file by default"
else
  fail "no monitor file by default"
fi
check "stop prints the take-notes pairing pointer" "**Recording**: ${M3D}/" "$OUT3"

echo "[rec] --in-person records the mic into the far slot (no call-audio device)"
OUT="$(EARSHOT_BLACKHOLE_NAME="No Such Device" "${EARSHOT}" rec --in-person -p obsbot -t t9 -o "${WORK}/rec9" 2>&1 </dev/null || true)"
check "records without the call-audio device present" "earshot: done." "$OUT"
check "announces the room routing" "far_remote.wav (the whole room)" "$OUT"
J9="$(cat "${WORK}"/rec9/*/meeting.json 2>/dev/null)"
check "meeting.json mode is in-person" '"mode": "in-person"' "$J9"
check "far_device is the mic" '"far_device": "OBSBOT Meet Microphone"' "$J9"
check "near_device is null" '"near_device": null' "$J9"
M9D="$(find "${WORK}/rec9" -mindepth 1 -maxdepth 1 -type d | head -1)"
if [ -n "${M9D}" ] && [ -s "${M9D}/far_remote.wav" ] && [ ! -f "${M9D}/near_me.wav" ]; then
  ok "far audio only, no near_me.wav"
else
  fail "far audio only, no near_me.wav"
fi

echo "[rec] --then runs the follow-up earshot command on the meeting dir"
OUT="$("${EARSHOT}" rec -p obsbot -t t8 -o "${WORK}/rec8" --then "devices" 2>&1 </dev/null || true)"
check "announces follow-up" "follow-up -> earshot devices" "$OUT"
check "follow-up executed after recording" "Current audio input devices" "$OUT"

# ==========================================================================
echo "[rec] interactive device pick when the configured mic is absent"
printf '4\n' > "${WORK}/tty-pick"
OUT="$(EARSHOT_TTY="${WORK}/tty-pick" "${EARSHOT}" rec -p ghost -t t4 -o "${WORK}/rec4" 2>&1 </dev/null || true)"
check "prompted with the device list" "Select a device number" "$OUT"
check "picked device used for the run" "near 'OBSBOT Meet 4K Microphone' (index 4)" "$OUT"
check "permanence hint shows profile edit" 'MIC_NAME="Ghost Mic|OBSBOT Meet 4K Microphone"' "$OUT"

printf '\n' > "${WORK}/tty-pick"
OUT="$(EARSHOT_TTY="${WORK}/tty-pick" "${EARSHOT}" rec -p ghost -t t5 -o "${WORK}/rec5" 2>&1 </dev/null)"; RC=$?
check "empty input aborts to the clean error" "no mic from profile 'ghost'" "$OUT"
[ "$RC" -ne 0 ] && ok "abort exits non-zero" || fail "abort exits non-zero"

printf '99\n' > "${WORK}/tty-pick"
OUT="$(EARSHOT_TTY="${WORK}/tty-pick" "${EARSHOT}" rec -p ghost -t t6 -o "${WORK}/rec6" 2>&1 </dev/null)"; RC=$?
check "invalid index rejected" "not a listed device index" "$OUT"
[ "$RC" -ne 0 ] && ok "invalid index exits non-zero" || fail "invalid index exits non-zero"

# ==========================================================================
echo "[offload] default steps, env forwarding, notes pull"
export TEST_LOG="${WORK}/t1.log"; : > "${TEST_LOG}"
printf 'Satori Kong Twingate\n' > "${WORK}/glossary.txt"
HF_TOKEN=hf_TEST ANTHROPIC_API_KEY=sk_TEST EARSHOT_MY_NAME="Dan T" \
  EARSHOT_PROMPT_FILE="${WORK}/glossary.txt" \
  "${EARSHOT}" offload "${M}" --poll 1 >/dev/null 2>&1
RUN="$(cat "${TEST_LOG}.run.sh" 2>/dev/null)"
check "default includes transcribe" "earshot-container transcribe . --device auto" "$RUN"
check "default includes diarize" "earshot-container diarize . --device auto" "$RUN"
check_not "default excludes summarize" "summarize" "$RUN"
check "HF_TOKEN forwarded" "export HF_TOKEN='hf_TEST'" "$RUN"
check "MY_NAME forwarded" "export EARSHOT_MY_NAME='Dan T'" "$RUN"
check "glossary pushed + wired" "export EARSHOT_PROMPT_FILE=glossary.txt" "$RUN"
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
echo "[offload] launch failure cleans up the secret-bearing runner"
export TEST_LOG="${WORK}/t3b.log"; : > "${TEST_LOG}"
OUT="$(TEST_FAIL_LAUNCH=1 HF_TOKEN=hf_X "${EARSHOT}" offload "${M}" --poll 1 2>&1)"; RC=$?
check "clear launch error" "is tmux installed" "$OUT"
[ "$RC" -ne 0 ] && ok "launch failure exits non-zero" || fail "launch failure exits non-zero"
check "remote runner removed" "rm -f 'earshot-work" "$(cat "${TEST_LOG}")"

echo "[offload] --collect does not require local audio"
NOWAV="${WORK}/meet/personal/collect-only"; mkdir -p "${NOWAV}"
export TEST_LOG="${WORK}/t3c.log"; : > "${TEST_LOG}"
OUT="$("${EARSHOT}" offload "${NOWAV}" --collect --poll 1 2>&1)"; RC=$?
[ "$RC" -eq 0 ] && ok "collect works without far_remote.wav" || fail "collect works without far_remote.wav ($OUT)"

echo "[flags] -c without a value errors clearly"
OUT="$("${EARSHOT}" speakers -c 2>&1)"; RC=$?
check "clear missing-value error" "requires a value" "$OUT"
[ "$RC" -ne 0 ] && ok "missing value exits non-zero" || fail "missing value exits non-zero"

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

echo "[edges] rec -c works on a pre-context config via built-in defaults"
printf '\n' > "${WORK}/tty-empty"
OUT="$(EARSHOT_CONF="${OLDCONF}" EARSHOT_TTY="${WORK}/tty-empty" \
       "${EARSHOT}" rec -c business -t x 2>&1 </dev/null)"; RC=$?
check_not "no pre-context error anymore" "predates contexts" "$OUT"
# it gets PAST context resolution and fails only on the (fake) mic
check "reached mic resolution" "no mic from profile" "$OUT"

# ==========================================================================
echo "[no config] earshot is fully functional without any earshot.conf"
# EARSHOT_CONF pointed at a nonexistent file + no repo/etc conf reachable
OUT="$(EARSHOT_CONF="${WORK}/does-not-exist.conf" EARSHOT_TTY="${WORK}/tty-empty" \
       "${EARSHOT}" rec -p obsbot -t x -o "${WORK}/noconf" 2>&1 </dev/null || true)"
check_not "no conf-missing error" "no earshot.conf found" "$OUT"
check "built-in obsbot profile resolves" "near 'OBSBOT Meet Microphone'" "$OUT"
OUT="$(EARSHOT_CONF="${WORK}/does-not-exist.conf" "${EARSHOT}" speakers -c personal list 2>&1)"
check "built-in personal context library" "speakers-personal.json" "$OUT"

# ==========================================================================
echo "[process] local pipeline chains transcribe -> diarize -> summarize"
FV="${WORK}/fakevenv"; mkdir -p "${FV}/bin"
cat > "${FV}/bin/python" <<'EOF'
#!/bin/sh
# fake venv python: log tool invocations, succeed on import checks
for a in "$@"; do case "$a" in *.py) echo "PY:$(basename "$a") $*";; esac; done
exit 0
EOF
chmod +x "${FV}/bin/python"
OUT="$(EARSHOT_VENV="${FV}" "${EARSHOT}" process "${M}" --summarize \
      --tx-args "--model small" --sum-args "--backend vllm --model x" 2>&1)"; RC=$?
[ "$RC" -eq 0 ] && ok "process exits 0" || fail "process exits 0 ($OUT)"
check "transcribe ran with args" "PY:transcribe.py" "$OUT"
check "tx-args passed" "--model small" "$OUT"
check "diarize ran" "PY:diarize.py" "$OUT"
check "summarize ran with args" "--backend vllm" "$OUT"
# order: transcribe before diarize before summarize
T=$(printf '%s\n' "$OUT" | grep -n "PY:transcribe.py" | head -1 | cut -d: -f1)
D=$(printf '%s\n' "$OUT" | grep -n "PY:diarize.py" | head -1 | cut -d: -f1)
S=$(printf '%s\n' "$OUT" | grep -n "PY:summarize.py" | head -1 | cut -d: -f1)
[ "$T" -lt "$D" ] && [ "$D" -lt "$S" ] && ok "pipeline order correct" || fail "pipeline order correct"

# ==========================================================================
echo "[dispatch] every python subcommand reaches its tool (not bash breakage)"
OUT="$("${EARSHOT}" timing /nonexistent 2>&1)";    check "timing dispatches" "transcript.json not found" "$OUT"
OUT="$("${EARSHOT}" attribute /nonexistent 2>&1)"; check "attribute dispatches" "transcript.json not found" "$OUT"
OUT="$("${EARSHOT}" speakers list 2>&1)";          check "speakers dispatches" "speakers" "$OUT"
OUT="$("${EARSHOT}" summarize 2>&1)";              check "summarize dispatches" "meeting directory" "$OUT"
OUT="$("${EARSHOT}" list --all 2>&1)";             check "list dispatches over roots" "2026-07-01_120000_test" "$OUT"
OUT="$("${EARSHOT}" grep nothingmatches 2>&1)";    check "grep dispatches" "no matches" "$OUT"
OUT="$("${EARSHOT}" clean 2>&1)";                  check "clean is dry-run by default" "dry run" "$OUT"
OUT="$("${EARSHOT}" help 2>&1)"
check "help shows subcommands" "earshot offload DIR" "$OUT"
check_not "help stops at marker (no code leaks)" "set -euo" "$OUT"
check_not "help omits the marker itself" "==help-end==" "$OUT"

# ==========================================================================
echo
echo "bash tests: ${PASS} passed, ${FAIL} failed"
[ "${FAIL}" -eq 0 ]
