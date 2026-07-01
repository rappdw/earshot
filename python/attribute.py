#!/usr/bin/env python3
"""earshot attribution - interactively name the still-anonymous remote speakers.

For each REMOTE-N speaker in a diarized meeting, plays a sample of their voice
from far_remote.wav and asks who it is. A name you give is:
  1) enrolled into the speaker library (using the embedding diarization already
     stored), so future meetings auto-identify them; and
  2) applied to this meeting's transcript (transcript.json + transcript.md).

Local + interactive. Needs ffmpeg (to cut the sample) and a player (afplay on
macOS, else ffplay). Uses the active speaker library via EARSHOT_SPEAKERS
(the earshot CLI sets this per context before calling us).

Run: earshot attribute MEETING_DIR
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import speakers as spk

REMOTE_RE = re.compile(r"^REMOTE-\d+$")


def fmt_ts(seconds):
    total = int(seconds or 0)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_transcript(meeting_dir, override):
    path = override or (os.path.join(meeting_dir, "transcript.json") if meeting_dir else None)
    if not path or not os.path.isfile(path):
        sys.exit("error: transcript.json not found; run earshot diarize first")
    with open(path, "r", encoding="utf-8") as fh:
        return path, json.load(fh)


def unidentified_labels(transcript):
    diar = transcript.get("diarization") or {}
    labels = diar.get("labels")
    if not labels:  # fall back to whatever appears on far segments
        labels = sorted({s.get("speaker") for s in transcript.get("segments", [])
                         if s.get("channel") == "far" and s.get("speaker")})
    return [l for l in labels if l and REMOTE_RE.match(l)]


def sample_segment(transcript, label, max_dur):
    """Pick a representative far-channel segment for this speaker; return
    (start, dur, text) or None."""
    cands = [s for s in transcript.get("segments", [])
             if s.get("channel") == "far" and s.get("speaker") == label
             and s.get("start") is not None and s.get("end") is not None]
    if not cands:
        return None
    best = max(cands, key=lambda s: (s["end"] - s["start"]))
    start = float(best["start"])
    dur = min(float(max_dur), float(best["end"]) - start)
    return start, max(dur, 1.0), (best.get("text") or "").strip()


def extract_clip(far, start, dur):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(start), "-i", far, "-t", str(dur), "-ac", "1", tmp.name],
        check=True,
    )
    return tmp.name


def play(path):
    """Play the clip; return the player name used, or None if none available."""
    if shutil.which("afplay"):
        subprocess.run(["afplay", path])
        return "afplay"
    if shutil.which("ffplay"):
        subprocess.run(["ffplay", "-autoexit", "-nodisp", "-loglevel", "error", path])
        return "ffplay"
    return None


def apply_names(transcript, mapping):
    """Relabel segments and rewrite the diarization metadata for named speakers."""
    for seg in transcript.get("segments", []):
        if seg.get("speaker") in mapping:
            seg["speaker"] = mapping[seg["speaker"]]
    diar = transcript.get("diarization")
    if isinstance(diar, dict):
        embs = diar.get("embeddings") or {}
        diar["embeddings"] = {mapping.get(k, k): v for k, v in embs.items()}
        if diar.get("labels"):
            diar["labels"] = sorted({mapping.get(l, l) for l in diar["labels"]})
        diar["label_map"] = {r: mapping.get(f, f) for r, f in (diar.get("label_map") or {}).items()}
        ident = diar.get("identification") or {}
        for old, new in mapping.items():
            ident[new] = {"matched": new, "score": 1.0, "source": "manual"}
            ident.pop(old, None)
        diar["identification"] = ident


def write_markdown(path, transcript):
    title = transcript.get("title") or os.path.basename(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Transcript - {title}\n\n")
        fh.write(f"_model {transcript.get('model','?')} - "
                 f"diarized {transcript.get('diarized_at','')}_\n\n")
        for seg in transcript.get("segments", []):
            text = (seg.get("text") or "").strip()
            if text:
                fh.write(f"**[{fmt_ts(seg.get('start'))}] {seg.get('speaker','?')}:** {text}\n\n")


def main():
    ap = argparse.ArgumentParser(description="Name unidentified remote speakers by ear, and learn them.")
    ap.add_argument("meeting_dir", nargs="?", help="diarized meeting folder")
    ap.add_argument("--far", help="override path to the remote-audio wav")
    ap.add_argument("--transcript", help="override path to transcript.json")
    ap.add_argument("--seconds", type=float, default=8.0, help="max sample length to play (default 8)")
    args = ap.parse_args()

    tpath, transcript = load_transcript(args.meeting_dir, args.transcript)
    far = args.far or (os.path.join(args.meeting_dir, "far_remote.wav") if args.meeting_dir else None)
    if not far or not os.path.isfile(far):
        sys.exit("error: far_remote.wav not found (pass a meeting dir or --far)")
    if not shutil.which("ffmpeg"):
        sys.exit("error: ffmpeg not found (needed to cut the audio sample)")

    todo = unidentified_labels(transcript)
    if not todo:
        print("earshot attribute: all remote speakers are already identified.", file=sys.stderr)
        return

    embeddings = (transcript.get("diarization") or {}).get("embeddings") or {}
    library = spk.load_library()

    print(f"earshot attribute: {len(todo)} unidentified speaker(s): {', '.join(todo)}",
          file=sys.stderr)
    print("For each: listen, then type a name to identify + learn them, "
          "Enter to skip, r to replay, q to quit.\n", file=sys.stderr)

    mapping = {}
    for label in todo:
        samp = sample_segment(transcript, label, args.seconds)
        if not samp:
            print(f"  {label}: no far-channel sample found, skipping.", file=sys.stderr)
            continue
        start, dur, text = samp
        clip = extract_clip(far, start, dur)
        keep_clip = False
        try:
            size = os.path.getsize(clip)
            hint = (text[:80] + "...") if len(text) > 80 else text
            print(f"--- {label}  [{fmt_ts(start)}]  ({dur:.0f}s, {size} bytes)  \"{hint}\"",
                  file=sys.stderr)
            if size < 1024:
                print("  WARNING: the extracted sample is empty/tiny - the segment may be "
                      "silent or beyond the end of far_remote.wav.", file=sys.stderr)
            while True:
                print(f"  playing sample ({dur:.0f}s)...", file=sys.stderr)
                player = play(clip)
                if player is None:
                    print("  NO AUDIO PLAYER FOUND - install afplay (macOS) or ffplay.", file=sys.stderr)
                    print(f"  clip left at {clip} - play it manually to check.", file=sys.stderr)
                    keep_clip = True
                ans = input(f"  {label} is: ").strip()
                if ans == "r":
                    continue
                if ans == "q":
                    print("  stopping.", file=sys.stderr)
                    ans = ""
                    todo = []  # break outer after applying what we have
                if not ans:
                    break
                mapping[label] = ans
                if label in embeddings:
                    sp = spk.add_sample(library, ans, embeddings[label])
                    print(f"  learned '{ans}' ({len(sp['samples'])} sample(s) in library)", file=sys.stderr)
                else:
                    print(f"  named '{ans}' for this transcript (no embedding to learn from)", file=sys.stderr)
                break
        finally:
            if not keep_clip:
                os.unlink(clip)
        if todo == []:
            break

    if not mapping:
        print("earshot attribute: nothing identified.", file=sys.stderr)
        return

    spk.save_library(library)
    apply_names(transcript, mapping)
    with open(tpath, "w", encoding="utf-8") as fh:
        json.dump(transcript, fh, ensure_ascii=False, indent=2)
    mdpath = os.path.join(os.path.dirname(tpath), "transcript.md")
    write_markdown(mdpath, transcript)

    named = ", ".join(f"{k}->{v}" for k, v in mapping.items())
    print(f"earshot attribute: {named}\nupdated {tpath} and {mdpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
