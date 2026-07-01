#!/usr/bin/env python3
"""earshot timing - measure Whisper-vs-pyannote timestamp alignment.

Whisper far-channel segments and pyannote diarization turns describe the SAME
audio (far_remote.wav), so their alignment is measurable without listening.
If Whisper timestamps are offset or drift from the audio, diarize's speaker
assignment (segment-to-turn overlap) and transcript.md timestamps are both
skewed; this tool quantifies that so a fix can target transcription with
evidence.

Method: cross-correlation. For each third of the meeting, find the time shift
(within +/-45s) that maximizes total overlap between the Whisper segments and
the pyannote speech regions. Aligned audio yields ~0s in every third; a fixed
recording skew yields one constant nonzero shift; transcription drift yields a
shift that grows across the thirds.

Usage: earshot timing MEETING_DIR   (needs a transcribed + diarized meeting)
"""

import argparse
import bisect
import json
import os
import sys


def merge_intervals(ivs):
    ivs = sorted(ivs)
    out = []
    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap_with(merged, starts, s, e):
    """Overlap of [s,e] with merged intervals (starts = [i[0] for i in merged])."""
    total = 0.0
    i = max(bisect.bisect_right(starts, s) - 1, 0)
    while i < len(merged) and merged[i][0] < e:
        total += max(0.0, min(e, merged[i][1]) - max(s, merged[i][0]))
        i += 1
    return total


def best_shift(segs, merged, starts, lo=-45.0, hi=45.0, step=0.25):
    """Shift (subtracted from segment times) that maximizes overlap with the
    turn speech-mask, i.e. the estimated Whisper-minus-audio offset."""
    best_d, best_score = 0.0, -1.0
    d = lo
    while d <= hi + 1e-9:
        score = sum(overlap_with(merged, starts, s["start"] - d, s["end"] - d)
                    for s in segs)
        if score > best_score:
            best_score, best_d = score, d
        d += step
    return best_d


def analyze(transcript):
    turns = (transcript.get("diarization") or {}).get("turns") or []
    segs = [s for s in transcript.get("segments", [])
            if s.get("channel") == "far"
            and s.get("start") is not None and s.get("end") is not None
            and (s.get("text") or "").strip()]
    if not turns:
        sys.exit("error: no diarization turns in transcript.json (run earshot diarize first)")
    if not segs:
        sys.exit("error: no far-channel segments in transcript.json")

    segs.sort(key=lambda s: s["start"])
    merged = merge_intervals([(t["start"], t["end"]) for t in turns])
    starts = [iv[0] for iv in merged]

    span = segs[-1]["start"] - segs[0]["start"] or 1.0
    t0 = segs[0]["start"]
    third_segs = ([], [], [])
    for s in segs:
        third_segs[min(int(3 * (s["start"] - t0) / span), 2)].append(s)

    shifts = [best_shift(t, merged, starts) if t else None for t in third_segs]
    overall = best_shift(segs, merged, starts)

    # after removing the overall shift, how well do segments sit in speech?
    weak = 0
    for s in segs:
        dur = max(s["end"] - s["start"], 1e-9)
        ov = overlap_with(merged, starts, s["start"] - overall, s["end"] - overall)
        if ov / dur < 0.5:
            weak += 1

    return {
        "n_segments": len(segs),
        "n_turns": len(turns),
        "shift": overall,
        "shift_thirds": shifts,
        "residual_weak_pct": 100.0 * weak / len(segs),
    }


def verdict(r):
    lines = []
    th = [t for t in r["shift_thirds"] if t is not None]
    growth = (th[-1] - th[0]) if len(th) >= 2 else 0.0
    if abs(growth) >= 1.0:
        lines.append(f"VERDICT: DRIFT — the Whisper-vs-audio shift grows from "
                     f"{th[0]:+.2f}s to {th[-1]:+.2f}s across the meeting. Speaker "
                     "assignment degrades over time. Suspect transcription timestamp "
                     "remapping (vad_filter) or inter-channel clock drift.")
    elif abs(r["shift"]) >= 1.0:
        lines.append(f"VERDICT: constant shift of {r['shift']:+.2f}s between Whisper "
                     "and the audio (recording start skew or a fixed transcription "
                     "offset). Correctable by shifting segment times.")
    else:
        lines.append("VERDICT: aligned. Whisper and pyannote agree; speaker "
                     "assignment and transcript timestamps are trustworthy.")
    if r["residual_weak_pct"] >= 20:
        lines.append(f"NOTE: even at the best shift, {r['residual_weak_pct']:.0f}% of "
                     "segments sit mostly outside detected speech — Whisper and "
                     "pyannote disagree about where speech IS, not just when.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Measure Whisper-vs-pyannote timestamp alignment.")
    ap.add_argument("meeting_dir", nargs="?", help="meeting folder with a diarized transcript.json")
    ap.add_argument("--transcript", help="override path to transcript.json")
    args = ap.parse_args()

    tpath = args.transcript or (os.path.join(args.meeting_dir, "transcript.json")
                                if args.meeting_dir else None)
    if not tpath or not os.path.isfile(tpath):
        sys.exit("error: transcript.json not found (pass a meeting dir or --transcript)")
    with open(tpath, "r", encoding="utf-8") as fh:
        transcript = json.load(fh)

    r = analyze(transcript)
    print(f"segments (far, non-empty): {r['n_segments']}   turns: {r['n_turns']}")
    print(f"estimated shift (whisper - audio): {r['shift']:+.2f}s")
    print("shift by meeting third: "
          + "  ".join("n/a" if x is None else f"{x:+.2f}s" for x in r["shift_thirds"]))
    print(f"segments outside speech even after correction: {r['residual_weak_pct']:.0f}%")
    print()
    print(verdict(r))


if __name__ == "__main__":
    main()
