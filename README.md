# earshot

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
  `brew install ffmpeg blackhole-2ch`
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
3. **Wear headphones** when speaker labels matter. On speakers, remote audio
   leaks into your mic and muddies the clean "just you" channel.

No aggregate devices are needed; earshot records BlackHole and your mic as two
parallel processes, resolved by device name.

## Usage

```bash
earshot devices                       # list audio inputs and current indices
earshot rec                           # record (prompts for a title)
earshot rec -c personal -t "Mom"      # personal context, titled
earshot transcribe DIR                # faster-whisper -> transcript.json + .md
earshot diarize DIR                   # split REMOTE into speakers, attribute names
earshot summarize DIR                 # write notes.md (local Ollama by default)
earshot enroll Dan --from DIR --speaker REMOTE-1   # teach a voice from a meeting
earshot speakers list                 # who's enrolled
earshot offload DIR --diarize         # run it all on the Spark over SSH
earshot help
```

A recording stops on **Enter**, on **Ctrl-C**, or **automatically** after a
sustained silence on the call channel (a meeting that has ended). See
`earshot rec -h` and the config for the auto-stop knobs.

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
results back. By default it attaches live via mosh; `--no-watch` for
fire-and-forget, `--collect` to reconnect to a still-running job. The host needs
earshot installed (`--with-python --with-diarize`), a CUDA-matched torch, and
`tmux` (+ `mosh-server` for `--watch`). Configure the host in `earshot.conf`.

You can offload any combination of `--transcribe`, `--diarize`, and
`--summarize` (summarization runs against an Ollama server on the host). To run
the remote steps in a **CUDA container** instead of a venv (reproducible GPU
environment on NVIDIA's NGC base, no torch fiddling), see
[`docker/README.md`](docker/README.md). It's a drop-in: build the image on the
host and point `EARSHOT_SPARK_EARSHOT` at the `earshot-container` wrapper.

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

## Roadmap

- Convenience: auto-detect call start, hotkey trigger, calendar-based titling.

## License

MIT. See [LICENSE](LICENSE).
