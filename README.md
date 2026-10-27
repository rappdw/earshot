# earshot

[![ci](https://github.com/rappdw/earshot/actions/workflows/ci.yml/badge.svg)](https://github.com/rappdw/earshot/actions/workflows/ci.yml)

A fully local meeting capture, transcription, diarization, and notes pipeline for macOS.

earshot records your meetings, transcribes them with Whisper, figures out who
said what, learns to attribute real names to the people you meet with regularly,
and writes structured notes. It exists to replace cloud notetakers (Granola,
Circleback, and the like) for meetings you would rather not send to a third party.

**The hard rule: audio and transcripts never leave your hardware.** Everything
runs on your Mac, or optionally on your own GPU box over your own network. The
only network access is a one-time download of model weights from Hugging Face,
and (if you opt in) shipping audio to a machine you control. No SaaS, no
account, no telemetry.

## How it works

Capture records two separate channels:

- `far_remote.wav` - the call audio (everyone but you), tapped from a BlackHole
  virtual output device.
- `near_me.wav` - your microphone (just you).

That two-channel split is the whole trick. Because your voice is already
isolated on its own channel, diarization only has to separate the *remote*
speakers from each other, which is a much easier problem than untangling
everyone from a single mixed recording.

```
Zoom/Meet/Teams ─► Multi-Output ─┬─► your headphones (you hear)
                                 └─► BlackHole ─► far_remote.wav
your mic ──────────────────────────────────────► near_me.wav
                                 │
                       transcribe (faster-whisper)
                                 │
                       diarize  (pyannote)  ─► REMOTE-1, REMOTE-2 ...
                                 │
                       identify (voice library) ─► real names
                                 │
                  transcript.json  +  transcript.md
                                 │
                       summarize (Ollama, local)  ─► notes.md
```

Transcription and diarization run on the Mac by default, or can be offloaded to
a CUDA box (e.g. a DGX Spark) over SSH. Summarization runs against a local Ollama
model by default; a hosted Claude model is available as an explicit opt-in.

## Requirements

- macOS (built and tested on Apple Silicon, macOS 14.4+ / 26).
- [`ffmpeg`](https://ffmpeg.org/) and [BlackHole 2ch](https://existential.audio/blackhole/):
  `brew install ffmpeg blackhole-2ch` (BlackHole is only needed for call
  capture; `--in-person` recordings skip it)
- Python 3.9+ for transcription/diarization.
- For diarization + speaker ID: a free [Hugging Face](https://huggingface.co/)
  token and acceptance of two gated (free) model licenses, see below.
- For summarization: a local [Ollama](https://ollama.com/) install with a model
  pulled (e.g. `ollama pull llama3.1`). Optional; only needed for `summarize`.
- Optional: a CUDA host with SSH access for offload.

## Install

```bash
git clone <your-repo-url> earshot
cd earshot
./install.sh --with-python      # core + transcription deps
./install.sh --with-diarize     # add diarization + speaker ID (pulls torch; large)
```

This installs:

- `bin/earshot` -> `~/.local/bin/earshot`
- the Python tools -> `~/.local/share/earshot/`
- a config file -> `~/.config/earshot/earshot.conf` (your edits are preserved on
  reinstall; `--force` overwrites with a backup)
- a virtualenv -> `~/.local/share/earshot/venv`

Make sure `~/.local/bin` is on your `PATH` (the installer tells you if it isn't).

## Audio setup (one time)

You need BlackHole wired so you can both *hear* a call and *capture* it.

1. **Multi-Output Device** (so call audio reaches both your ears and BlackHole):
   in Audio MIDI Setup, create a Multi-Output Device containing your output
   (headphones/speakers) plus `BlackHole 2ch`. Make the physical device the
   clock/primary and enable drift correction on BlackHole. Make one per output
   you use (e.g. `MultiOut-Dell`, `MultiOut-MBA`).
2. **Per meeting**: set macOS Output (and the meeting app's Speaker) to the
   matching Multi-Output; set the app's Microphone to your real mic.
3. **Headphones give the best speaker labels.** On speakers, remote audio leaks
   into your mic, so the same words land on both channels. Transcription
   harmonizes that bleed away (see below), which makes speaker-mode meetings
   work fine; headphones just avoid the problem at the source. Either way the
   call channel itself is unaffected: BlackHole is a digital tap, so
   `far_remote.wav` is identical whether you wear headphones or not.

No aggregate devices are needed; earshot records BlackHole and your mic as two
parallel processes, resolved by device name.

## Usage

```bash
earshot devices                       # list audio inputs and current indices
earshot rec                           # record (prompts for a title)
earshot rec -c personal -t "Mom"      # personal context, titled
earshot rec -t "Standup" -d 30        # stop after 30 minutes (prompted if omitted)
earshot rec --in-person -t "1:1 Sam"  # in-person meeting: mic only, no BlackHole
earshot transcribe DIR                # faster-whisper -> transcript.json + .md
earshot diarize DIR                   # split REMOTE into speakers, attribute names
earshot summarize DIR                 # write notes.md (local Ollama by default)
earshot process DIR --summarize       # transcribe -> diarize -> notes, locally
earshot offload DIR                   # the same pipeline on the Spark over SSH
earshot attribute DIR                 # name unknown voices by ear, learn them
earshot enroll Dan --from DIR --speaker REMOTE-1   # teach a voice from a meeting
earshot speakers list                 # who's enrolled
earshot list                          # pipeline status of recent meetings
earshot pending                       # meetings with audio but no transcript yet
earshot pending --offload             # offload that backlog one meeting at a time
earshot grep "kong migration"         # search every transcript
earshot clean --apply                 # FLAC-compress old audio (dry-run w/o --apply)
earshot timing DIR                    # transcript-vs-audio alignment check
earshot help
```

A recording stops on **Enter**, on **Ctrl-C**, or **automatically** once
**both** channels (the call and your mic) have been silent for the auto-stop
window, i.e. the meeting has actually gone quiet. Watching both sides means a
long monologue (you presenting while everyone else is muted) no longer cuts the
recording off early. `meeting.json` records the `stop_reason`
(`manual`/`silence`/`max-duration`/`interrupted`/`recorder-exited`) and
`duration_sec`, so an unexpectedly short recording is easy to diagnose after the
fact. See `earshot rec -h` and the config for the auto-stop knobs.

**Anticipated duration:** after the title, `rec` asks how long the meeting
should run and stops when that time is up (`stop_reason: max-duration`). Enter
means no limit; `-d N` answers up front. Meetings run over, so about two minutes
before the cutoff it warns you (and posts a notification): type `+15` and Enter
to give it fifteen more minutes. A plain Enter still stops immediately.

**Catching up on a backlog:** `earshot pending` scans your meeting folders
(base, business, and personal) for recordings that never got transcribed.
`earshot pending --offload` walks them one at a time and asks what to do with
each:

- **transcribe** (Enter) offloads it and waits for the job to finish before
  moving to the next;
- **delete recording** removes the meeting folder after a second confirmation;
- **ignore** leaves it alone for good: a `.earshot-ignore` marker keeps it out of
  `pending` (`earshot pending --all` still lists it; delete the marker to undo);
- **quit** stops the walk.

Extra arguments pass through to each offload (`--summarize`), `--process` does
the same locally, and `-y` transcribes everything unattended (it never deletes).
A folder that's still being recorded is left alone.

**In-person meetings:** `earshot rec --in-person` records only the mic, into
the far channel, because a room mic hears everyone (you included). No
BlackHole or Multi-Output device is involved, so it works on a bare Mac.
Diarization and attribution treat the room like any other meeting: you show
up as one more `REMOTE-N` until you attribute yourself once; after that the
voice library names you automatically.

**Auto-pipeline:** `earshot rec --then "offload --summarize"` (or
`EARSHOT_AFTER_REC` in the config) runs any earshot command on the meeting the
moment the recording stops — record, walk away, get notified when the notes
land.

**Accuracy and identity:** set `EARSHOT_PROMPT_FILE` to a glossary of your
product names, acronyms, and people (biggest accuracy lever for jargon), and
`EARSHOT_MY_NAME` to label your own channel with your name instead of `YOU`.
Both apply locally and through offload.

**Disk:** raw capture is ~700MB per meeting-hour. The stereo monitor copy is
off by default (`--monitor` to enable), and `earshot clean` FLAC-compresses or
purges audio older than N days (dry-run unless `--apply`; transcripts are
always kept, and every tool reads `.flac` transparently).

### Speaker identification

Diarization alone produces anonymous, per-meeting labels (`REMOTE-1`...). To get
real names across meetings, enroll your regulars once:

```bash
earshot diarize ~/Notes/meetings/business/2026-06-29_1430_standup
# see that REMOTE-1 is Dan, REMOTE-2 is Sara:
earshot enroll Dan  --from ~/Notes/meetings/business/2026-06-29_1430_standup --speaker REMOTE-1
earshot enroll Sara --from ~/Notes/meetings/business/2026-06-29_1430_standup --speaker REMOTE-2
```

Future meetings then attribute those voices automatically. Enrolling the same
person from several meetings improves accuracy. Each diarize run prints the
match confidence so you can tune `--id-threshold`.

Or do it by ear with **attribution**: for each still-anonymous speaker it plays
a sample and asks who it is, then both enrolls them (for future meetings) and
relabels this transcript:

```bash
earshot attribute ~/Notes/meetings/business/2026-06-29_1430_standup
# plays REMOTE-1's voice -> you type "Dan"; plays REMOTE-2 -> "Sara"; ...
```

For each unknown `REMOTE-N` it plays a clip from `far_remote.wav`, you type a
name (or Enter to skip, `r` to replay, `q` to quit). Named speakers are added to
the library and the meeting's `transcript.json`/`transcript.md` are rewritten
with the names. Needs `ffmpeg` and a player (`afplay` on macOS). Run it after
`diarize`.

### Contexts (business vs personal)

Contexts route recordings to separate directories and keep separate speaker
libraries, so work and personal voiceprints never mix:

```bash
earshot rec -c business     # -> EARSHOT_OUT_ROOT_BUSINESS
earshot rec -c personal     # -> EARSHOT_OUT_ROOT_PERSONAL
```

`diarize`, `enroll`, and `speakers` pick the right library by `-c`, by inferring
it from the meeting's path, or by the default context.

### Summarization

`earshot summarize DIR` turns the transcript into `notes.md` with Summary,
Decisions, Action Items, Open Questions, and Notable Details.

```bash
earshot summarize DIR                      # local Ollama (default), nothing leaves the box
earshot summarize DIR --model qwen2.5:14b  # any model you have pulled
earshot summarize DIR --backend claude     # opt-in; sends the transcript to Anthropic
```

The `ollama` backend is fully local; point `EARSHOT_OLLAMA_MODEL` at any model
you've `ollama pull`ed. The `claude` backend is the one step that sends text
off-box: it's off by default, prints a loud warning, and needs `ANTHROPIC_API_KEY`.
Don't use it for sensitive meetings.

### Offload to a GPU box

`earshot offload DIR` ships the audio to a CUDA host over SSH + rsync, runs the
work there inside a tmux session (so it survives a disconnect), and pulls the
results back. It **transcribes + diarizes by default**; it polls quietly (Ctrl-C
is safe, the job keeps running), with `--watch` to attach live via mosh and
`--collect` to reconnect to a running job. The host needs earshot installed
(`--with-python --with-diarize`), a CUDA-matched torch, and `tmux` (+
`mosh-server` for `--watch`). Configure the host in `earshot.conf`.

You can offload any combination of `--transcribe`, `--diarize`, and
`--summarize` (summarization runs against an Ollama/vLLM server on the host). To run
the remote steps in a **CUDA container** instead of a venv (reproducible GPU
environment on NVIDIA's NGC base, no torch fiddling), see
[`docker/README.md`](docker/README.md). It's a drop-in: build the image on the
host and point `EARSHOT_SPARK_EARSHOT` at the `earshot-container` wrapper.

**Channel handling.** Transcription knows what the two channels mean and
reconciles them:

- A channel that **recorded nothing** (a misrouted call channel, or a mic the
  meeting app held exclusively) is skipped, so it costs no GPU time and can't
  hallucinate text out of noise. The last remaining channel is always
  transcribed, quiet or not.
- **Speaker bleed is harmonized away.** Without headphones your mic also picks
  up the remote audio, so the same words land on both channels and the near copy
  gets attributed to you. Near segments that overlap far speech *and* match its
  text are dropped, keeping the far copy (the digital tap, always the better
  audio). Genuine simultaneous speech, like interrupting or a backchannel,
  survives because the text doesn't match. Disable with `--no-harmonize`.
- `--channels far|near|both` forces the choice when you know better, and
  `transcript.json` records what was transcribed, what was skipped and why, and
  how many bleed segments were dropped.

Headphones still give the cleanest result, since bleed never reaches the mic in
the first place. This makes speaker-mode meetings usable, not equivalent.

**Transcription speed (long meetings).** On CUDA, earshot uses faster-whisper's
batched pipeline automatically — it feeds many audio chunks through the GPU at
once instead of one sequential decode step at a time (the difference between the
GPU running busy-but-starved and running full). Two more levers for multi-hour
recordings:

- `EARSHOT_WHISPER_MODEL="large-v3-turbo"` — roughly 4x faster than `large-v3`
  for a small accuracy cost. Set it in the config; it travels with offload jobs.
- `--tx-args "--batch-size 16"` (via `offload`/`process`) raises the batch size
  from the default 8; bigger is faster on long audio until you run out of
  memory. `--batch-size 0` forces the old sequential path.

## Hugging Face setup (for diarization)

1. Create a free token: https://huggingface.co/settings/tokens
2. Accept the conditions (free) on both model pages while logged in:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Make the token available: `export HF_TOKEN=hf_...` (or pass `--hf-token`).

## Configuration

All settings live in `~/.config/earshot/earshot.conf` (see `etc/earshot.conf`
for the documented template): output roots and contexts, capture profiles (which
mic each maps to), per-context speaker libraries, auto-stop thresholds, and the
Spark offload host.

## Output layout

```
~/Notes/meetings/<context>/YYYY-MM-DD_HHMMSS[_title]/
  far_remote.wav        the call (everyone but you)
  near_me.wav           your mic
  monitor_stereo.wav    L=remote, R=you (quick listen)
  meeting.json          capture metadata
  transcript.json       segments, timestamps, speakers, embeddings
  transcript.md         readable transcript
  notes.md              structured meeting notes (after summarize)
```

## Honest limitations

- **Transcription speed on the Mac**: faster-whisper has no Metal backend, so
  large-v3 runs on CPU on Apple Silicon. Fine for batch, not real time. Use a
  smaller model while testing, or offload to a GPU.
- **Diarization** on a single mixed call channel degrades with crosstalk and
  with compressed VoIP audio. Telling it the speaker count (`--num-speakers`)
  helps. pyannote on Apple Silicon MPS can hit unsupported ops; use
  `--device cpu` if it errors.
- **Speaker ID** inherits those same audio weaknesses; it is an assist, not
  infallible. The printed confidence scores tell you when it is unsure.
- **Auto-stop** can trip early on a genuine multi-minute lull. Raise the
  threshold or use `--no-auto-stop` for meetings likely to go quiet.

## Claude Code plugin (skills)

The repo doubles as a Claude Code plugin carrying skills that work on earshot
output. Install it in Claude Code:

```
/plugin marketplace add rappdw/earshot
/plugin install earshot@earshot
```

Skills:

- **`/meeting-narrative <meeting-dir>`** — synthesize a diarized earshot
  meeting (plus any notes you took live with thinkkit's `/take-notes`) into
  `narrative.md`: a topic-organized story of the meeting with decisions,
  action items, significant realizations, and open questions.

More meeting-intelligence skills will land here as they're built.

## Roadmap

- Convenience: auto-detect call start, hotkey trigger, calendar-based titling.

## License

MIT. See [LICENSE](LICENSE).
