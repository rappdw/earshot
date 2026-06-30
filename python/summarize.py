#!/usr/bin/env python3
"""earshot summarization - turn a transcript into structured meeting notes.

Swappable backend, local by default:
  - ollama (default): fully local. Talks to a local Ollama server over HTTP.
    Nothing leaves the machine.
  - claude (opt-in): sends the transcript to Anthropic's API. This is the ONE
    step that takes text off-box, so it is off by default and prints a loud
    warning. Use it only for non-sensitive meetings.

Reads transcript.md (preferred, has speaker labels) or transcript.json from the
meeting dir and writes notes.md.

Stdlib only (urllib) so it runs without the venv. Ollama must be running
locally; the Claude backend needs ANTHROPIC_API_KEY in the environment.
"""

import argparse
import json
import os
import sys
import datetime as dt
import urllib.request
import urllib.error


SYSTEM_PROMPT = (
    "You are a meeting-notes assistant. You are given a meeting transcript with "
    "speaker labels (YOU is the person who recorded it; REMOTE-N or names are the "
    "other participants). Produce concise, accurate notes in Markdown with exactly "
    "these sections, in this order:\n"
    "## Summary  (2-4 sentences)\n"
    "## Decisions  (bullet list)\n"
    "## Action Items  (each as: - [owner] task)\n"
    "## Open Questions  (bullet list)\n"
    "## Notable Details  (bullet list)\n\n"
    "Use only information present in the transcript; do not invent facts, owners, "
    "or dates. If a section has nothing, write '_None._' under it. Attribute action "
    "items to the named speaker when the transcript makes the owner clear, otherwise "
    "use the speaker label. Output only the notes, with no preamble or sign-off."
)


def read_transcript(meeting_dir, override):
    if override:
        with open(override, "r", encoding="utf-8") as fh:
            return fh.read()
    md = os.path.join(meeting_dir, "transcript.md")
    if os.path.isfile(md):
        with open(md, "r", encoding="utf-8") as fh:
            return fh.read()
    js = os.path.join(meeting_dir, "transcript.json")
    if os.path.isfile(js):
        with open(js, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        lines = []
        for seg in data.get("segments", []):
            text = (seg.get("text") or "").strip()
            if text:
                lines.append(f"{seg.get('speaker', '?')}: {text}")
        return "\n".join(lines)
    sys.exit(f"error: no transcript.md or transcript.json in {meeting_dir} "
             f"(run earshot transcribe first)")


def http_post_json(url, payload, headers, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        sys.exit(f"error: HTTP {e.code} from {url}\n{body}")
    except urllib.error.URLError as e:
        sys.exit(f"error: could not reach {url}: {e.reason}")


def summarize_ollama(transcript, args):
    url = args.ollama_url.rstrip("/") + "/api/chat"
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    print(f"earshot-summarize: ollama model={args.model} url={url}", file=sys.stderr)
    out = http_post_json(url, payload, {"Content-Type": "application/json"}, timeout=600)
    msg = (out.get("message") or {}).get("content", "")
    if not msg:
        sys.exit(f"error: empty response from ollama (is model '{args.model}' pulled?)")
    return msg


def summarize_openai(transcript, args):
    """OpenAI-compatible /v1/chat/completions backend. Works with vLLM (reuse an
    existing server) and anything else speaking that API. Stays local if the URL
    is local."""
    base = args.openai_url.rstrip("/")
    url = base + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("EARSHOT_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
    }
    print(f"earshot-summarize: {args.backend} model={args.model} url={url}", file=sys.stderr)
    out = http_post_json(url, payload, headers, timeout=600)
    choices = out.get("choices") or []
    if not choices:
        sys.exit(f"error: empty response from {url} (is model '{args.model}' served there?)")
    text = ((choices[0].get("message") or {}).get("content") or "").strip()
    if not text:
        sys.exit(f"error: empty content from {url}")
    return text


def summarize_claude(transcript, args):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("error: claude backend needs ANTHROPIC_API_KEY in the environment")
    print("=" * 70, file=sys.stderr)
    print("earshot-summarize: WARNING - the claude backend sends this transcript",
          file=sys.stderr)
    print("  to Anthropic. Text LEAVES this machine. Use the default 'ollama'",
          file=sys.stderr)
    print("  backend to keep everything local.", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"earshot-summarize: claude model={args.model}", file=sys.stderr)
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": transcript}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    out = http_post_json("https://api.anthropic.com/v1/messages", payload, headers, timeout=300)
    if out.get("stop_reason") == "refusal":
        sys.exit("error: the model refused to produce a summary for this transcript")
    parts = [b.get("text", "") for b in out.get("content", []) if b.get("type") == "text"]
    text = "".join(parts).strip()
    if not text:
        sys.exit("error: empty response from claude")
    return text


def main():
    ap = argparse.ArgumentParser(description="Summarize a meeting transcript into notes.md.")
    ap.add_argument("meeting_dir", nargs="?", help="meeting folder with transcript.md/json")
    ap.add_argument("--transcript", help="override path to the transcript file")
    ap.add_argument("--backend", default=os.environ.get("EARSHOT_SUMMARY_BACKEND", "ollama"),
                    choices=["ollama", "vllm", "openai", "claude"],
                    help="ollama (local default) | vllm/openai (OpenAI-compatible server) | claude (opt-in)")
    ap.add_argument("--model", help="override the model for the chosen backend")
    ap.add_argument("--ollama-url", dest="ollama_url",
                    default=os.environ.get("EARSHOT_OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--openai-url", dest="openai_url",
                    default=os.environ.get("EARSHOT_OPENAI_URL", "http://localhost:8000/v1"),
                    help="OpenAI-compatible base URL (vllm/openai backends), incl. /v1")
    ap.add_argument("--max-tokens", dest="max_tokens", type=int,
                    default=int(os.environ.get("EARSHOT_SUMMARY_MAX_TOKENS", "4096")))
    args = ap.parse_args()

    if not args.meeting_dir and not args.transcript:
        sys.exit("error: pass a meeting directory or --transcript PATH")

    # resolve default model per backend if not overridden
    if not args.model:
        if args.backend == "claude":
            args.model = os.environ.get("EARSHOT_CLAUDE_MODEL", "claude-opus-4-8")
        elif args.backend in ("vllm", "openai"):
            args.model = os.environ.get("EARSHOT_OPENAI_MODEL", "")
            if not args.model:
                sys.exit("error: set the model your server serves via --model or EARSHOT_OPENAI_MODEL")
        else:
            args.model = os.environ.get("EARSHOT_OLLAMA_MODEL", "llama3.1")

    transcript = read_transcript(args.meeting_dir, args.transcript)

    if args.backend == "claude":
        notes = summarize_claude(transcript, args)
    elif args.backend in ("vllm", "openai"):
        notes = summarize_openai(transcript, args)
    else:
        notes = summarize_ollama(transcript, args)

    outdir = args.meeting_dir or os.path.dirname(os.path.abspath(args.transcript))
    npath = os.path.join(outdir, "notes.md")
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    with open(npath, "w", encoding="utf-8") as fh:
        fh.write(notes.rstrip() + "\n\n")
        fh.write(f"---\n_generated by earshot ({args.backend}/{args.model}) {stamp}_\n")
    print(f"earshot-summarize: wrote {npath}", file=sys.stderr)


if __name__ == "__main__":
    main()
