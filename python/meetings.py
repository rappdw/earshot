#!/usr/bin/env python3
"""earshot meetings tooling: list / grep / clean across your meeting roots.

  list   pipeline status per meeting (audio, transcript, diarization, notes)
  grep   search across transcripts with speaker + timestamp context
  clean  reclaim disk: FLAC-compress (or purge) meeting audio older than N days
         (dry-run by default; --apply executes)

Roots come from EARSHOT_ROOTS (colon-separated); the earshot CLI builds that
from your config. Stdlib only; `clean` compression shells out to ffmpeg.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

AUDIO_BASES = ("far_remote", "near_me")


def roots():
    out = []
    for r in os.environ.get("EARSHOT_ROOTS", "").split(":"):
        if r and os.path.isdir(r) and r not in out:
            out.append(r)
    if not out:
        sys.exit("error: no meeting roots found (EARSHOT_ROOTS empty; check your config)")
    return out


def is_meeting(d):
    return any(os.path.isfile(os.path.join(d, f)) for f in
               ("meeting.json", "transcript.json", "far_remote.wav", "far_remote.flac"))


def meetings():
    """Yield (root, dir) for every meeting folder, newest first by name."""
    seen = set()
    found = []
    for root in roots():
        for name in os.listdir(root):
            d = os.path.join(root, name)
            if os.path.isdir(d) and d not in seen and is_meeting(d):
                seen.add(d)
                found.append((root, d))
    found.sort(key=lambda rd: os.path.basename(rd[1]), reverse=True)
    return found


def fmt_ts(seconds):
    total = int(seconds or 0)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def audio_files(d):
    out = []
    for base in AUDIO_BASES + ("monitor_stereo",):
        for ext in (".wav", ".flac"):
            p = os.path.join(d, base + ext)
            if os.path.isfile(p):
                out.append(p)
    return out


def load_transcript(d):
    p = os.path.join(d, "transcript.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


# ---------------------------------------------------------------------------
def cmd_list(args):
    rows = meetings()
    if not args.all:
        rows = rows[:args.limit]
    print(f"{'meeting':44} {'ctx':9} {'audio':6} tx dz at nt  size")
    for root, d in rows:
        name = os.path.basename(d)[:44]
        ctx = os.path.basename(root.rstrip("/"))[:9]
        audio = audio_files(d)
        akind = "-"
        if any(p.endswith(".wav") and "monitor" not in p for p in audio):
            akind = "wav"
        elif any(p.endswith(".flac") for p in audio):
            akind = "flac"
        t = load_transcript(d)
        tx = "x" if t else "-"
        dz = "x" if t and t.get("diarization") else "-"
        # attributed: any far speaker that is neither YOU nor REMOTE*
        at = "-"
        if t:
            named = {s.get("speaker") for s in t.get("segments", [])
                     if s.get("channel") == "far"
                     and not str(s.get("speaker", "")).startswith("REMOTE")}
            at = "x" if named else "-"
        nt = "x" if os.path.isfile(os.path.join(d, "notes.md")) else "-"
        size = sum(os.path.getsize(p) for p in audio) / 1e6
        print(f"{name:44} {ctx:9} {akind:6} {tx}  {dz}  {at}  {nt}  {size:7.0f}MB")
    if not args.all and len(meetings()) > args.limit:
        print(f"... ({len(meetings())} total; --all to show everything)")


# ---------------------------------------------------------------------------
def cmd_grep(args):
    rx = re.compile(args.pattern, re.IGNORECASE)
    hits = 0
    cap = 0 if args.all else 50
    for root, d in meetings():
        t = load_transcript(d)
        if not t:
            continue
        name = os.path.basename(d)
        for seg in t.get("segments", []):
            text = seg.get("text") or ""
            if rx.search(text):
                print(f"{name} [{fmt_ts(seg.get('start'))}] "
                      f"{seg.get('speaker', '?')}: {text.strip()}")
                hits += 1
                if cap and hits >= cap:
                    print(f"... capped at {cap} matches (--all for everything)")
                    return
    if not hits:
        print("no matches")


# ---------------------------------------------------------------------------
def compress_wav(wav, apply_changes):
    """wav -> flac (lossless), delete the wav. Returns bytes reclaimed."""
    flac = wav[:-4] + ".flac"
    before = os.path.getsize(wav)
    if not apply_changes:
        return before // 2  # estimate
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", wav, "-c:a", "flac", flac], check=True)
    if not (os.path.isfile(flac) and os.path.getsize(flac) > 0):
        raise RuntimeError(f"compression produced no output for {wav}")
    reclaimed = before - os.path.getsize(flac)
    os.remove(wav)
    return reclaimed


def cmd_clean(args):
    import time
    if not args.purge_audio and not shutil.which("ffmpeg"):
        sys.exit("error: ffmpeg is required to compress (or use --purge-audio)")
    cutoff = time.time() - args.days * 86400
    total = 0
    actions = 0
    for root, d in meetings():
        # the stereo monitor is a derivable convenience copy: always removable
        for ext in (".wav", ".flac"):
            mon = os.path.join(d, "monitor_stereo" + ext)
            if os.path.isfile(mon):
                sz = os.path.getsize(mon)
                print(f"{'rm    ' if args.apply else 'would rm    '}{mon}  ({sz/1e6:.0f}MB)")
                if args.apply:
                    os.remove(mon)
                total += sz
                actions += 1
        for base in AUDIO_BASES:
            for ext in (".wav", ".flac"):
                p = os.path.join(d, base + ext)
                if not os.path.isfile(p) or os.path.getmtime(p) > cutoff:
                    continue
                if args.purge_audio:
                    # only purge audio that has a transcript to show for itself
                    if not os.path.isfile(os.path.join(d, "transcript.json")):
                        continue
                    sz = os.path.getsize(p)
                    print(f"{'rm    ' if args.apply else 'would rm    '}{p}  ({sz/1e6:.0f}MB)")
                    if args.apply:
                        os.remove(p)
                    total += sz
                    actions += 1
                elif ext == ".wav":
                    rec = compress_wav(p, args.apply)
                    print(f"{'flac  ' if args.apply else 'would flac  '}{p}  "
                          f"(~{rec/1e6:.0f}MB reclaimed)")
                    total += rec
                    actions += 1
    verb = "reclaimed" if args.apply else "would reclaim"
    print(f"\n{actions} action(s), {verb} ~{total/1e6:.0f}MB"
          + ("" if args.apply else "   (dry run; add --apply to execute)"))


def main():
    ap = argparse.ArgumentParser(description="List, search, and clean earshot meetings.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="pipeline status per meeting")
    p.add_argument("--all", action="store_true")
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("grep", help="search transcripts (regex, case-insensitive)")
    p.add_argument("pattern")
    p.add_argument("--all", action="store_true", help="no match cap")
    p.set_defaults(fn=cmd_grep)

    p = sub.add_parser("clean", help="compress or purge old meeting audio (dry-run by default)")
    p.add_argument("--days", type=int, default=int(os.environ.get("EARSHOT_CLEAN_DAYS", "14")),
                   help="act on audio older than this many days (default 14)")
    p.add_argument("--apply", action="store_true", help="execute (default: dry run)")
    p.add_argument("--purge-audio", action="store_true",
                   help="delete audio outright instead of compressing "
                        "(only where a transcript exists)")
    p.set_defaults(fn=cmd_clean)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
