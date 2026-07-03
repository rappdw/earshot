# Hardening roadmap

From the v0.2.0 code review (2026-07-01). Each chunk runs Plan → Implement →
Verify and lands as its own commit. Order chosen so quick guaranteed wins land
first, the safety net (tests) lands before any refactor, and the one item
needing real-world data (timing) ships its tooling early.

| # | Chunk | Why | Verify |
|---|-------|-----|--------|
| A | Pin dependencies | `requirements-diarize.txt` lacks the `huggingface_hub<1.0` cap that the container needed — a fresh venv install of diarize is broken today | pip resolver dry-run |
| B | Committed test suite | All pure logic (identify, relabel, overlap, slug, silence-stop, context routing) verified only ad hoc; refactors have no net | `tests/run.sh` green |
| C | Timestamp-drift diagnostic | Whisper timestamps observed drifting from audio; that skews diarize speaker assignment, not just clips. Ship `earshot timing DIR` to measure segment-vs-turn alignment | **RESOLVED 2026-07-01**: both real meetings (incl. the one with the observed clip mismatch) measure aligned, −0.25s in every third, 4–6% residual. No transcription change needed. The heard mismatch was attribute sampling from Whisper's longest segment, which over-selects the ~6% of segments outside detected speech — already fixed by turn-based clip selection. |
| D | Secrets off remote disk | HF/Anthropic tokens persist in the remote `.run.sh` | runner self-deletes; simulated exec |
| E | Correctness edges | 7 small known bugs (embedding collision, REMOTE fallback, dim check, ollama num_predict + length warning, unsafe dir names, title escaping, rec -c guard) | tests extended per fix |
| F | Bash maintainability | 4 duplicate python-runners; line-number `usage()` broke 3× | suite green + dispatch smoke |
| G | Container build smoke | Catch dependency-API breaks at image build, not first diarize | Dockerfile check; proven on next Spark rebuild |

## Review 2 (2026-07-02) — all executed

| Chunk | Landed |
|---|---|
| Bugs 1–5 | attribute offers generic REMOTE; speakers.json atomic + 0600 + .bak; offload launch guard cleans secrets; manual-name drop warning; every nit (rec -y, usage fallback, -c guard, collect w/o audio, per-channel languages, Ctrl-C reason, single-enumeration picker, macOS gate on release workflow) |
| Config | defaults-in-code; conf is pure overrides; missing/stale conf fully functional |
| Contract | transcript.json schema:1 + languages; .wav/.flac fallback everywhere; EARSHOT_PROMPT_FILE glossary; EARSHOT_MY_NAME; diarize inline alignment check |
| Lifecycle | monitor opt-in; `earshot clean` (FLAC/purge, dry-run default); `earshot list`; `earshot grep` |
| Glue | `rec --then` / EARSHOT_AFTER_REC auto-pipeline; `earshot process`; macOS notifications |

Deferred (tracked, not scheduled): auto-detect call start (resident watcher —
the last Stage-6 item, big and standalone), voiceprint retention policy
(`--rm-remote` default?), offload poll timeout, whisper.cpp backend, splitting
cmd_record/cmd_offload out of bin/earshot when the next big feature lands.
