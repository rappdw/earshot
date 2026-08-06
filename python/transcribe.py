#!/usr/bin/env python3
"""earshot transcription - batch transcribe a meeting's two channels locally.

Transcribes near_me.wav (labeled YOU) and far_remote.wav (labeled REMOTE) with
faster-whisper, merges the segments by time, and writes transcript.json and
transcript.md into the meeting folder.

The two-channel capture gives a free YOU-vs-REMOTE split with no diarization
model. Stage 4 subdivides REMOTE into individual speakers. In-person captures
(rec --in-person) have only far_remote: whichever channels are present get
transcribed, missing ones are skipped.

Local-only note: the *audio* never leaves the machine. The only network access
is a one-time download of the model weights from Hugging Face on first run;
after that everything is offline. Pre-seed with HF_HUB_OFFLINE=0 once, then you
can run with no network.
"""

import argparse
import json
import os
import sys
import datetime as dt


def pick_device(requested):
    """auto -> cuda if available else cpu."""
    if requested and requested != "auto":
        return requested
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def pick_compute_type(requested, device):
    if requested and requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"


def fmt_ts(seconds):
    """seconds -> HH:MM:SS."""
    if seconds is None:
        seconds = 0
    td = dt.timedelta(seconds=int(seconds))
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_prompt(args):
    """Build the initial_prompt that biases decoding toward your jargon."""
    parts = []
    if args.prompt:
        parts.append(args.prompt.strip())
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as fh:
            parts.append(fh.read().strip())
    text = " ".join(p for p in parts if p)
    return text or None


def resolve_batch_size(requested, device):
    """Batch size for batched inference. requested None (or <0) -> auto: batch
    on CUDA (fills the GPU, far faster on long audio), sequential on CPU. An
    explicit >= 0 wins, so --batch-size 0 forces the sequential path."""
    if requested is not None and requested >= 0:
        return requested
    return 8 if device == "cuda" else 0


def transcribe_channel(model, batched, batch_size, path, speaker, channel, args, prompt):
    """Run one channel; return a list of segment dicts tagged with speaker.

    With batch_size > 0 the batched pipeline runs many VAD chunks through the
    GPU at once - much higher throughput on multi-hour audio, where the
    sequential decode loop leaves the GPU starved. Falls back to the
    sequential path (which we still use on CPU) otherwise."""
    common = dict(
        language=args.language,            # None -> autodetect
        beam_size=args.beam_size,
        word_timestamps=True,              # words kept for Stage 4 alignment
        initial_prompt=prompt,
    )
    if batched is not None and batch_size > 0:
        # the batched pipeline runs VAD internally (required for chunking), so
        # vad_filter is implicit here; drops silence just like the seq path.
        segments, info = batched.transcribe(path, batch_size=batch_size, **common)
    else:
        segments, info = model.transcribe(
            path, vad_filter=True, **common)  # vad drops silence / spurious text
    out = []
    for seg in segments:
        words = None
        if seg.words:
            words = [
                {"start": w.start, "end": w.end, "word": w.word,
                 "prob": round(float(w.probability), 4) if w.probability is not None else None}
                for w in seg.words
            ]
        out.append({
            "speaker": speaker,
            "channel": channel,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": words,
        })
    return out, info


def find_audio(d, base):
    """Channel audio path: .wav, or .flac once `earshot clean` compressed it.
    (Same helper in diarize.py/attribute.py - keep in sync.)"""
    for ext in (".wav", ".flac"):
        p = os.path.join(d, base + ext)
        if os.path.isfile(p):
            return p
    return None


def resolve_inputs(args):
    """Return a list of (path, speaker, channel) to transcribe."""
    jobs = []
    if args.file:
        speaker = args.speaker or "SPEAKER"
        jobs.append((args.file, speaker, "single"))
        return jobs
    d = args.meeting_dir
    if not d or not os.path.isdir(d):
        sys.exit("error: pass a meeting directory or --file PATH")
    near = find_audio(d, "near_me")
    far = find_audio(d, "far_remote")
    me = os.environ.get("EARSHOT_MY_NAME", "").strip() or "YOU"
    if near:
        jobs.append((near, me, "near"))
    if far:
        jobs.append((far, "REMOTE", "far"))
    if not jobs:
        sys.exit(f"error: no near_me/far_remote audio (.wav/.flac) found in {d}")
    return jobs


def write_outputs(args, all_segments, meta):
    """Write transcript.json and transcript.md next to the audio."""
    outdir = args.meeting_dir or os.path.dirname(os.path.abspath(args.file))
    title = ""
    mj = os.path.join(outdir, "meeting.json")
    if os.path.isfile(mj):
        try:
            with open(mj, "r", encoding="utf-8") as fh:
                title = (json.load(fh).get("title") or "").strip()
        except Exception:
            pass

    js = {
        "schema": 1,
        "title": title,
        "model": args.model,
        "device": meta["device"],
        "compute_type": meta["compute_type"],
        "language": meta["language"],
        "languages": meta.get("languages", {}),
        "generated_at": meta["generated_at"],
        "segments": all_segments,
    }
    jpath = os.path.join(outdir, "transcript.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(js, fh, ensure_ascii=False, indent=2)

    mpath = os.path.join(outdir, "transcript.md")
    with open(mpath, "w", encoding="utf-8") as fh:
        head = title if title else os.path.basename(os.path.normpath(outdir))
        fh.write(f"# Transcript - {head}\n\n")
        fh.write(f"_model {args.model} - {meta['device']}/{meta['compute_type']} "
                 f"- lang {meta['language']} - generated {meta['generated_at']}_\n\n")
        for seg in all_segments:
            if not seg["text"]:
                continue
            fh.write(f"**[{fmt_ts(seg['start'])}] {seg['speaker']}:** {seg['text']}\n\n")

    return jpath, mpath


def main():
    ap = argparse.ArgumentParser(description="Batch transcribe a meeting locally with faster-whisper.")
    ap.add_argument("meeting_dir", nargs="?", help="meeting folder with near_me.wav / far_remote.wav")
    ap.add_argument("--file", help="transcribe a single arbitrary audio file instead")
    ap.add_argument("--speaker", help="speaker label for --file mode (default SPEAKER)")
    ap.add_argument("--model", default=os.environ.get("EARSHOT_WHISPER_MODEL") or "large-v3",
                    help="model size or path (default large-v3, or $EARSHOT_WHISPER_MODEL). "
                         "large-v3-turbo is ~4x faster with a small accuracy cost.")
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=None,
                    help="batched-inference batch size on CUDA (0 = sequential decode; "
                         "default auto: 8 on CUDA, 0 on CPU). Larger is faster on long "
                         "audio at the cost of memory.")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--compute-type", dest="compute_type", default="auto",
                    help="auto | int8 | int8_float16 | float16 | float32")
    ap.add_argument("--language", default=None, help="force a language code (e.g. en); default autodetect")
    ap.add_argument("--beam-size", dest="beam_size", type=int, default=5)
    ap.add_argument("--prompt", help="initial prompt to bias jargon/acronyms/names")
    ap.add_argument("--prompt-file", dest="prompt_file",
                    default=os.environ.get("EARSHOT_PROMPT_FILE") or None,
                    help="glossary file to bias decoding (default: $EARSHOT_PROMPT_FILE)")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    device = pick_device(args.device)
    compute_type = pick_compute_type(args.compute_type, device)
    prompt = load_prompt(args)

    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    # Batched inference (faster-whisper >= 1.1) fills the GPU; fall back to the
    # sequential path with a clear note if the installed version lacks it.
    batch_size = resolve_batch_size(args.batch_size, device)
    batched = None
    if batch_size > 0:
        try:
            from faster_whisper import BatchedInferencePipeline
            batched = BatchedInferencePipeline(model=model)
        except Exception as e:
            print(f"earshot-transcribe: batched inference unavailable ({e}); using "
                  f"sequential decode. Upgrade faster-whisper to >=1.1 for the speedup.",
                  file=sys.stderr)
            batch_size = 0
    mode = f"batched(bs={batch_size})" if batch_size > 0 else "sequential"
    print(f"earshot-transcribe: model={args.model} device={device} "
          f"compute={compute_type} {mode}", file=sys.stderr)

    jobs = resolve_inputs(args)
    all_segments = []
    languages = {}
    for path, speaker, channel in jobs:
        print(f"earshot-transcribe: {channel} -> {speaker}: {path}", file=sys.stderr)
        segs, info = transcribe_channel(model, batched, batch_size, path, speaker, channel, args, prompt)
        languages[channel] = info.language
        all_segments.extend(segs)
        print(f"  {len(segs)} segments (lang={info.language} "
              f"p={info.language_probability:.2f})", file=sys.stderr)

    # primary language: forced > the far channel (most content) > whatever ran
    detected_lang = args.language or languages.get("far") \
        or next(iter(languages.values()), "")

    all_segments.sort(key=lambda s: (s["start"] if s["start"] is not None else 0.0))

    meta = {
        "device": device,
        "compute_type": compute_type,
        "language": detected_lang,
        "languages": languages,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    jpath, mpath = write_outputs(args, all_segments, meta)
    print(f"earshot-transcribe: wrote\n  {jpath}\n  {mpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
