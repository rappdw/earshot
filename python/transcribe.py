#!/usr/bin/env python3
"""earshot transcription - batch transcribe a meeting's two channels locally.

Transcribes near_me.wav (labeled YOU) and far_remote.wav (labeled REMOTE) with
faster-whisper, merges the segments by time, and writes transcript.json and
transcript.md into the meeting folder.

The two-channel capture gives a free YOU-vs-REMOTE split with no diarization
model. Stage 4 subdivides REMOTE into individual speakers. In-person captures
(rec --in-person) have only far_remote: whichever channels are present get
transcribed, missing ones are skipped.

Channel awareness (see resolve_inputs / harmonize_channels):
  * a channel that recorded but is effectively silent is skipped, so a
    misrouted call channel or a mic held by the meeting app costs no GPU time
    and can't hallucinate text out of noise;
  * without headphones your mic also picks up the remote audio from the
    speakers, so the same words land on BOTH channels and the near copy gets
    attributed to you. Harmonization drops near segments that duplicate an
    overlapping far segment. far is the digital tap and always wins.

Local-only note: the *audio* never leaves the machine. The only network access
is a one-time download of the model weights from Hugging Face on first run;
after that everything is offline. Pre-seed with HF_HUB_OFFLINE=0 once, then you
can run with no network.
"""

import argparse
import bisect
import difflib
import json
import os
import re
import subprocess
import sys
import datetime as dt

# A channel whose peak is at/below this (dBFS) counts as "recorded nothing".
try:
    SILENT_PEAK_DB = float(os.environ.get("EARSHOT_SILENT_DB") or -70.0)
except ValueError:
    SILENT_PEAK_DB = -70.0


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


def channel_peak_db(path, sample_sec=300):
    """Peak level (dBFS) of the first sample_sec of a channel, or None if it
    can't be measured (no ffmpeg, unreadable file). Sampling keeps this cheap
    on multi-hour meetings; a dead channel is dead from the first minute."""
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-t", str(sample_sec),
             "-i", path, "-af", "volumedetect", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    except Exception:
        return None
    m = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB",
                  p.stderr.decode("utf-8", "replace"))
    return float(m.group(1)) if m else None


def resolve_inputs(args):
    """Return (jobs, skipped): jobs is a list of (path, speaker, channel) to
    transcribe, skipped maps channel -> why it was left out."""
    jobs = []
    skipped = {}
    if args.file:
        speaker = args.speaker or "SPEAKER"
        jobs.append((args.file, speaker, "single"))
        return jobs, skipped
    d = args.meeting_dir
    if not d or not os.path.isdir(d):
        sys.exit("error: pass a meeting directory or --file PATH")
    near = find_audio(d, "near_me")
    far = find_audio(d, "far_remote")
    me = os.environ.get("EARSHOT_MY_NAME", "").strip() or "YOU"
    want = getattr(args, "channels", None) or "both"

    cands = []
    if near and want in ("both", "near"):
        cands.append((near, me, "near"))
    if far and want in ("both", "far"):
        cands.append((far, "REMOTE", "far"))
    if not cands:
        sys.exit(f"error: no {want} audio (.wav/.flac) found in {d}")

    # Only auto-skip when something else would still be transcribed: a single
    # quiet channel is all we have, so transcribe it rather than return nothing.
    for path, speaker, channel in cands:
        peak = channel_peak_db(path) if len(cands) > 1 else None
        if peak is not None and peak <= SILENT_PEAK_DB:
            skipped[channel] = "silent (peak %.1f dB)" % peak
            print(f"earshot-transcribe: skipping {channel} channel, it recorded "
                  f"nothing (peak {peak:.1f} dB)", file=sys.stderr)
            continue
        jobs.append((path, speaker, channel))
    if not jobs:  # every channel looked silent; transcribe anyway
        print("earshot-transcribe: every channel looks silent; transcribing all "
              "of them anyway", file=sys.stderr)
        skipped = {}
        jobs = list(cands)
    return jobs, skipped


def _norm_text(t):
    return re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower()).strip()


def harmonize_channels(segments, ratio=0.8, min_overlap=0.5):
    """Drop near-channel segments that duplicate overlapping far-channel text.

    Without headphones the mic picks up the call audio from the speakers, so the
    remote participants' words are transcribed twice: once cleanly on far, once
    on near where they are wrongly attributed to you. Same time window plus the
    same words means bleed, so the near copy goes and the far copy (the digital
    tap, always the better audio) stays.

    Conservative by construction: a near segment must overlap far speech for at
    least min_overlap of its length AND match the text, so genuine simultaneous
    speech (you talking over someone) survives. Returns (segments, dropped)."""
    far = sorted((s for s in segments
                  if s.get("channel") == "far" and s.get("start") is not None),
                 key=lambda s: float(s["start"]))
    if not far:
        return segments, 0
    starts = [float(s["start"]) for s in far]
    kept, dropped = [], 0
    for s in segments:
        if s.get("channel") != "near" or not (s.get("text") or "").strip():
            kept.append(s)
            continue
        ns = float(s.get("start") or 0.0)
        ne = float(s.get("end") or ns)
        dur = max(ne - ns, 1e-6)
        # far segments are short; a bounded look-back finds every overlap
        i = bisect.bisect_left(starts, ne) - 1
        texts, ov_total = [], 0.0
        while i >= 0 and starts[i] > ns - 60.0:
            f = far[i]
            fs = float(f["start"])
            fe = float(f.get("end") or fs)
            ov = min(ne, fe) - max(ns, fs)
            if ov > 0:
                ov_total += ov
                texts.append(f.get("text") or "")
            i -= 1
        if min(ov_total / dur, 1.0) < min_overlap:
            kept.append(s)
            continue
        a = _norm_text(s.get("text"))
        b = _norm_text(" ".join(reversed(texts)))
        if a and b and (a in b or difflib.SequenceMatcher(None, a, b).ratio() >= ratio):
            dropped += 1
            continue
        kept.append(s)
    return kept, dropped


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
        "channels": meta.get("channels", {}),
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
    ap.add_argument("--channels", choices=("both", "far", "near"), default="both",
                    help="which capture channels to transcribe (default both; a "
                         "channel that recorded nothing is skipped automatically)")
    ap.add_argument("--no-harmonize", dest="no_harmonize", action="store_true",
                    help="keep near-channel segments that duplicate the call audio "
                         "(speaker bleed); harmonization is on by default")
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

    jobs, skipped = resolve_inputs(args)
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

    # Harmonize the two channels: with both in hand, near segments that repeat
    # overlapping far speech are speaker bleed, not you.
    done = [c for _, _, c in jobs]
    drops = 0
    if not args.no_harmonize and "near" in done and "far" in done:
        all_segments, drops = harmonize_channels(all_segments)
        if drops:
            print(f"earshot-transcribe: harmonized {drops} near-channel segment(s) "
                  f"duplicating the call audio (speaker bleed)", file=sys.stderr)

    meta = {
        "device": device,
        "compute_type": compute_type,
        "language": detected_lang,
        "languages": languages,
        "channels": {"transcribed": done, "skipped": skipped,
                     "harmonized_drops": drops},
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    jpath, mpath = write_outputs(args, all_segments, meta)
    print(f"earshot-transcribe: wrote\n  {jpath}\n  {mpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
