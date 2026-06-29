#!/usr/bin/env python3
"""earshot transcription - batch transcribe a meeting's two channels locally.

Transcribes near_me.wav (labeled YOU) and far_remote.wav (labeled REMOTE) with
faster-whisper, merges the segments by time, and writes transcript.json and
transcript.md into the meeting folder.

The two-channel capture gives a free YOU-vs-REMOTE split with no diarization
model. Stage 4 subdivides REMOTE into individual speakers.

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


def transcribe_channel(model, path, speaker, channel, args, prompt):
    """Run one channel; return a list of segment dicts tagged with speaker."""
    segments, info = model.transcribe(
        path,
        language=args.language,            # None -> autodetect
        beam_size=args.beam_size,
        vad_filter=True,                   # drop silence / reduce spurious text
        word_timestamps=True,              # words kept for Stage 4 alignment
        initial_prompt=prompt,
    )
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
    near = os.path.join(d, "near_me.wav")
    far = os.path.join(d, "far_remote.wav")
    if os.path.isfile(near):
        jobs.append((near, "YOU", "near"))
    if os.path.isfile(far):
        jobs.append((far, "REMOTE", "far"))
    if not jobs:
        sys.exit(f"error: no near_me.wav or far_remote.wav found in {d}")
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
        "title": title,
        "model": args.model,
        "device": meta["device"],
        "compute_type": meta["compute_type"],
        "language": meta["language"],
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
    ap.add_argument("--model", default="large-v3", help="model size or path (default large-v3)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--compute-type", dest="compute_type", default="auto",
                    help="auto | int8 | int8_float16 | float16 | float32")
    ap.add_argument("--language", default=None, help="force a language code (e.g. en); default autodetect")
    ap.add_argument("--beam-size", dest="beam_size", type=int, default=5)
    ap.add_argument("--prompt", help="initial prompt to bias jargon/acronyms/names")
    ap.add_argument("--prompt-file", dest="prompt_file", help="file whose contents bias decoding")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    device = pick_device(args.device)
    compute_type = pick_compute_type(args.compute_type, device)
    prompt = load_prompt(args)

    print(f"earshot-transcribe: model={args.model} device={device} compute={compute_type}",
          file=sys.stderr)
    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    jobs = resolve_inputs(args)
    all_segments = []
    detected_lang = args.language or ""
    for path, speaker, channel in jobs:
        print(f"earshot-transcribe: {channel} -> {speaker}: {path}", file=sys.stderr)
        segs, info = transcribe_channel(model, path, speaker, channel, args, prompt)
        if not detected_lang:
            detected_lang = info.language
        all_segments.extend(segs)
        print(f"  {len(segs)} segments (lang={info.language} "
              f"p={info.language_probability:.2f})", file=sys.stderr)

    all_segments.sort(key=lambda s: (s["start"] if s["start"] is not None else 0.0))

    meta = {
        "device": device,
        "compute_type": compute_type,
        "language": detected_lang,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    jpath, mpath = write_outputs(args, all_segments, meta)
    print(f"earshot-transcribe: wrote\n  {jpath}\n  {mpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
