#!/usr/bin/env python3
"""earshot diarization - split the REMOTE channel into individual speakers, and
(optionally) attribute names from the speaker library.

Runs pyannote.audio on far_remote.wav (the remote mix only; near_me.wav is
already known to be YOU), relabels the REMOTE segments of an existing
transcript.json into REMOTE-1, REMOTE-2, ... by maximal time overlap, and - if
the speaker library has matches - replaces those with real names.

Per-speaker voice embeddings (pyannote return_embeddings) are saved into
transcript.json so you can enroll a speaker later with no recompute:
    earshot enroll Dan --from THIS_DIR --speaker REMOTE-1

Re-runnable: speaker labels are always recomputed from each segment's "channel"
field, so running this again just re-diarizes cleanly.

Setup gotchas (pyannote):
  1. Free Hugging Face token: https://huggingface.co/settings/tokens
  2. Accept conditions on BOTH model pages (free, just a click):
       https://huggingface.co/pyannote/speaker-diarization-3.1
       https://huggingface.co/pyannote/segmentation-3.0
  3. Provide it via --hf-token, env HF_TOKEN / HUGGINGFACE_TOKEN, or hf login.

Local-only note: audio stays on the machine; network is only the one-time
download of the pyannote model weights.
"""

import argparse
import json
import os
import sys
import datetime as dt

import speakers as spk


def pick_device(requested):
    if requested and requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def fmt_ts(seconds):
    if seconds is None:
        seconds = 0
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_token(args):
    return (args.hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or None)


def run_pyannote(far_path, args, device):
    """Return (turns, raw_embeddings) where raw_embeddings maps pyannote label
    (SPEAKER_xx) -> embedding list, skipping clusters with no valid embedding."""
    from pyannote.audio import Pipeline
    import torch

    pipeline = Pipeline.from_pretrained(args.model, use_auth_token=get_token(args))
    if pipeline is None:
        sys.exit(
            "error: pyannote returned no pipeline. This almost always means the\n"
            "       HF token is missing or you have not accepted the model conditions.\n"
            "       See the setup notes at the top of diarize.py."
        )
    pipeline.to(torch.device(device))

    kwargs = {}
    if args.num_speakers:
        kwargs["num_speakers"] = args.num_speakers
    if args.min_speakers:
        kwargs["min_speakers"] = args.min_speakers
    if args.max_speakers:
        kwargs["max_speakers"] = args.max_speakers

    raw_embeddings = {}
    try:
        out = pipeline(far_path, return_embeddings=True, **kwargs)
    except TypeError:
        out = pipeline(far_path, **kwargs)

    if isinstance(out, tuple):
        diar, emb = out
        labels = diar.labels()
        if emb is not None:
            import math
            for i, lab in enumerate(labels):
                if i >= len(emb):
                    break
                row = [float(x) for x in emb[i]]
                if any(math.isnan(x) for x in row):
                    continue
                raw_embeddings[lab] = row
    else:
        diar = out
        print("  note: this pyannote version returned no embeddings; "
              "name identification disabled (upgrade to >=3.1).", file=sys.stderr)

    turns = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        turns.append({"start": float(turn.start), "end": float(turn.end),
                      "speaker": speaker})
    turns.sort(key=lambda t: t["start"])
    return turns, raw_embeddings


def friendly_labels(turns):
    """Map pyannote SPEAKER_xx -> REMOTE-1.. by first appearance."""
    mapping = {}
    n = 0
    for t in turns:
        if t["speaker"] not in mapping:
            n += 1
            mapping[t["speaker"]] = f"REMOTE-{n}"
    return mapping


def best_raw_speaker(seg_start, seg_end, turns):
    """Pyannote label with the most overlap with [start, end]."""
    best_lab = None
    best_ov = 0.0
    for t in turns:
        ov = min(seg_end, t["end"]) - max(seg_start, t["start"])
        if ov > best_ov:
            best_ov = ov
            best_lab = t["speaker"]
    return best_lab


def relabel(transcript, turns, raw_to_final):
    for seg in transcript["segments"]:
        if seg.get("channel") != "far":
            continue  # YOU / near stays as-is
        start = seg.get("start") or 0.0
        end = seg.get("end") or start
        raw = best_raw_speaker(start, end, turns)
        seg["speaker"] = raw_to_final.get(raw, "REMOTE")


def write_markdown(path, transcript):
    title = transcript.get("title") or os.path.basename(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Transcript - {title}\n\n")
        fh.write(f"_model {transcript.get('model','?')} - "
                 f"diarized {transcript.get('diarized_at','')}_\n\n")
        for seg in transcript["segments"]:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            fh.write(f"**[{fmt_ts(seg.get('start'))}] {seg.get('speaker','?')}:** {text}\n\n")


def main():
    ap = argparse.ArgumentParser(description="Diarize the REMOTE channel, relabel a transcript, attribute names.")
    ap.add_argument("meeting_dir", nargs="?", help="meeting folder with far_remote.wav + transcript.json")
    ap.add_argument("--far", help="override path to the remote-audio wav")
    ap.add_argument("--transcript", help="override path to transcript.json")
    ap.add_argument("--model", default="pyannote/speaker-diarization-3.1")
    ap.add_argument("--hf-token", dest="hf_token", help="Hugging Face token (or set HF_TOKEN)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--num-speakers", dest="num_speakers", type=int, help="exact number of remote speakers")
    ap.add_argument("--min-speakers", dest="min_speakers", type=int)
    ap.add_argument("--max-speakers", dest="max_speakers", type=int)
    ap.add_argument("--no-identify", dest="identify", action="store_false",
                    help="skip matching against the speaker library")
    ap.add_argument("--id-threshold", dest="id_threshold", type=float,
                    default=float(os.environ.get("EARSHOT_ID_THRESHOLD", "0.5")),
                    help="cosine similarity threshold for a name match (default 0.5)")
    args = ap.parse_args()

    d = args.meeting_dir
    far = args.far or (os.path.join(d, "far_remote.wav") if d else None)
    tpath = args.transcript or (os.path.join(d, "transcript.json") if d else None)

    if not far or not os.path.isfile(far):
        sys.exit("error: far_remote.wav not found (pass a meeting dir or --far)")
    if not tpath or not os.path.isfile(tpath):
        sys.exit("error: transcript.json not found; run earshot transcribe first (or pass --transcript)")

    with open(tpath, "r", encoding="utf-8") as fh:
        transcript = json.load(fh)

    device = pick_device(args.device)
    print(f"earshot-diarize: model={args.model} device={device}", file=sys.stderr)
    if device == "mps":
        print("  note: pyannote on MPS can hit unsupported-op fallbacks; "
              "use --device cpu if it errors.", file=sys.stderr)

    turns, raw_embeddings = run_pyannote(far, args, device)

    # REMOTE-N by first appearance, then override with library names where matched.
    raw_to_final = friendly_labels(turns)
    id_report = {}
    library = spk.load_library() if args.identify else {"speakers": {}}
    have_lib = bool(library.get("speakers"))
    if args.identify and have_lib and raw_embeddings:
        for raw, emb in raw_embeddings.items():
            name, score = spk.identify(emb, library, args.id_threshold)
            id_report[raw_to_final.get(raw, raw)] = {
                "matched": name, "score": round(float(score), 3)}
            if name:
                raw_to_final[raw] = name
    elif args.identify and not have_lib:
        print("  note: speaker library empty; leaving REMOTE-N labels. "
              "Enroll with: earshot enroll NAME --from DIR --speaker REMOTE-1",
              file=sys.stderr)

    relabel(transcript, turns, raw_to_final)

    # Store embeddings keyed by FINAL label so enroll can read them back.
    final_embeddings = {}
    for raw, emb in raw_embeddings.items():
        final_embeddings[raw_to_final.get(raw, raw)] = emb

    final_labels = sorted(set(raw_to_final.values()))
    transcript["diarization"] = {
        "model": args.model,
        "device": device,
        "n_remote_speakers": len(final_labels),
        "labels": final_labels,
        "label_map": raw_to_final,
        "identification": id_report,
        "id_threshold": args.id_threshold,
        "turns": turns,
        "embeddings": final_embeddings,
    }
    transcript["diarized_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    with open(tpath, "w", encoding="utf-8") as fh:
        json.dump(transcript, fh, ensure_ascii=False, indent=2)

    mdpath = os.path.join(os.path.dirname(tpath), "transcript.md")
    write_markdown(mdpath, transcript)

    print(f"earshot-diarize: {len(final_labels)} remote speaker(s) -> "
          f"{', '.join(final_labels) if final_labels else '(none)'}", file=sys.stderr)
    if id_report:
        for lab, info in id_report.items():
            tag = info["matched"] or "(no match)"
            print(f"  {lab}: {tag}  (best cos={info['score']})", file=sys.stderr)
    print(f"earshot-diarize: updated\n  {tpath}\n  {mdpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
