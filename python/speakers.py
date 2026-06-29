#!/usr/bin/env python3
"""earshot speaker library - enroll known voices and identify them across meetings.

The library is a JSON file (default ~/.config/earshot/speakers.json) mapping each
person's name to one or more L2-normalized voice embeddings ("samples"). Identi-
fication compares a cluster's embedding to every stored sample by cosine
similarity and takes the best match above a threshold.

Embeddings come from pyannote's diarization pipeline (return_embeddings=True), so
the library and the diarizer share an embedding space. Enrolling from a meeting
you have already diarized needs no model at all - the embedding is read straight
out of transcript.json.

Subcommands:
  speakers.py list
  speakers.py enroll NAME (--from DIR --speaker LABEL | --file WAV)
  speakers.py remove NAME
  speakers.py rename OLD NEW
"""

import argparse
import json
import math
import os
import sys


def lib_path():
    env = os.environ.get("EARSHOT_SPEAKERS")
    if env:
        return env
    return os.path.join(os.path.expanduser("~/.config/earshot"), "speakers.json")


def load_library():
    p = lib_path()
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"model": None, "dim": None, "speakers": {}}


def save_library(lib):
    p = lib_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(lib, fh, indent=2)
    return p


def l2norm(v):
    v = [float(x) for x in v]
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return v
    return [x / n for x in v]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def add_sample(lib, name, emb, model=None):
    emb = l2norm(emb)
    sp = lib["speakers"].setdefault(name, {"samples": []})
    sp["samples"].append(emb)
    lib["dim"] = len(emb)
    if model and not lib.get("model"):
        lib["model"] = model
    return sp


def identify(emb, lib, threshold):
    """Return (name, score). name is None if best score < threshold."""
    emb = l2norm(emb)
    best_name = None
    best = -1.0
    for name, sp in lib.get("speakers", {}).items():
        for s in sp.get("samples", []):
            c = cosine(emb, s)
            if c > best:
                best = c
                best_name = name
    if best_name is not None and best >= threshold:
        return best_name, best
    return None, best


# --------------------------------------------------------------------------
# embedding sources
# --------------------------------------------------------------------------
def emb_from_meeting(meeting_dir, label):
    """Read a stored per-speaker embedding out of a diarized transcript.json."""
    tpath = os.path.join(meeting_dir, "transcript.json")
    if not os.path.isfile(tpath):
        sys.exit(f"error: {tpath} not found (diarize the meeting first)")
    with open(tpath, "r", encoding="utf-8") as fh:
        t = json.load(fh)
    embs = (t.get("diarization") or {}).get("embeddings") or {}
    if label not in embs:
        avail = ", ".join(sorted(embs.keys())) or "(none)"
        sys.exit(f"error: speaker '{label}' has no stored embedding in {tpath}.\n"
                 f"       available: {avail}\n"
                 f"       (embeddings need pyannote >=3.1 with return_embeddings)")
    return embs[label]


def embed_wav(path, model, hf_token, device):
    """Compute a single embedding for a clean one-speaker sample (needs pyannote)."""
    try:
        from pyannote.audio import Pipeline
        import torch
    except Exception:
        sys.exit("error: --file enrollment needs the diarization deps "
                 "(./install.sh --with-diarize)")
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    pipeline = Pipeline.from_pretrained(model, use_auth_token=token)
    if pipeline is None:
        sys.exit("error: could not load pyannote pipeline (HF token / model conditions?)")
    pipeline.to(torch.device(device))
    out = pipeline(path, num_speakers=1, return_embeddings=True)
    if not isinstance(out, tuple):
        sys.exit("error: this pyannote version does not return embeddings; upgrade to >=3.1")
    _, emb = out
    if emb is None or len(emb) == 0:
        sys.exit("error: no embedding produced from the sample")
    return [float(x) for x in emb[0]]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def cmd_list(args):
    lib = load_library()
    sp = lib.get("speakers", {})
    if not sp:
        print(f"no speakers enrolled yet ({lib_path()})")
        return
    print(f"enrolled speakers ({lib_path()}, model={lib.get('model')}):")
    for name in sorted(sp):
        print(f"  {name:20s} {len(sp[name].get('samples', []))} sample(s)")


def cmd_enroll(args):
    if not args.from_dir and not args.file:
        sys.exit("error: enroll needs --from DIR --speaker LABEL, or --file WAV")
    lib = load_library()
    if args.from_dir:
        if not args.speaker:
            sys.exit("error: --from also needs --speaker LABEL (e.g. REMOTE-1)")
        emb = emb_from_meeting(args.from_dir, args.speaker)
    else:
        emb = embed_wav(args.file, args.model, args.hf_token, args.device)
    if lib.get("dim") and len(emb) != lib["dim"]:
        sys.exit(f"error: embedding dim {len(emb)} != library dim {lib['dim']} "
                 f"(different embedding model?)")
    sp = add_sample(lib, args.name, emb, model=args.model)
    p = save_library(lib)
    print(f"enrolled '{args.name}' ({len(sp['samples'])} sample(s) total) -> {p}")


def cmd_remove(args):
    lib = load_library()
    if args.name in lib.get("speakers", {}):
        del lib["speakers"][args.name]
        save_library(lib)
        print(f"removed '{args.name}'")
    else:
        sys.exit(f"error: no speaker named '{args.name}'")


def cmd_rename(args):
    lib = load_library()
    sp = lib.get("speakers", {})
    if args.old not in sp:
        sys.exit(f"error: no speaker named '{args.old}'")
    sp[args.new] = sp.pop(args.old)
    save_library(lib)
    print(f"renamed '{args.old}' -> '{args.new}'")


def main():
    ap = argparse.ArgumentParser(description="Manage the earshot speaker library.")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list", help="list enrolled speakers")

    e = sub.add_parser("enroll", help="add a voice sample for a name")
    e.add_argument("name")
    e.add_argument("--from", dest="from_dir", help="diarized meeting dir to pull the embedding from")
    e.add_argument("--speaker", help="speaker label in that meeting (e.g. REMOTE-1)")
    e.add_argument("--file", help="clean one-speaker wav to compute an embedding from")
    e.add_argument("--model", default="pyannote/speaker-diarization-3.1")
    e.add_argument("--hf-token", dest="hf_token")
    e.add_argument("--device", default="cpu")

    r = sub.add_parser("remove", help="remove a speaker")
    r.add_argument("name")

    rn = sub.add_parser("rename", help="rename a speaker")
    rn.add_argument("old")
    rn.add_argument("new")

    args = ap.parse_args()
    cmd = args.cmd or "list"
    {"list": cmd_list, "enroll": cmd_enroll,
     "remove": cmd_remove, "rename": cmd_rename}[cmd](args)


if __name__ == "__main__":
    main()
