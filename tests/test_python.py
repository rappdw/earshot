#!/usr/bin/env python3
"""Unit tests for earshot's Python tools (stdlib only: python3 -m unittest)."""

import http.server
import json
import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import speakers as spk          # noqa: E402
import diarize                  # noqa: E402
import attribute                # noqa: E402
import summarize                # noqa: E402
import timing                   # noqa: E402


# ---------------------------------------------------------------------------
# speakers.py
# ---------------------------------------------------------------------------
class TestSpeakers(unittest.TestCase):
    def test_cosine_orthogonal_and_identical(self):
        self.assertAlmostEqual(spk.cosine([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(spk.cosine([1, 2], [1, 2]), 1.0)
        self.assertEqual(spk.cosine([0, 0], [1, 1]), 0.0)  # zero vector guard

    def test_l2norm(self):
        n = spk.l2norm([3, 4])
        self.assertAlmostEqual(n[0], 0.6)
        self.assertAlmostEqual(n[1], 0.8)
        self.assertEqual(spk.l2norm([0, 0]), [0, 0])

    def test_add_sample_and_identify(self):
        lib = {"model": None, "dim": None, "speakers": {}}
        spk.add_sample(lib, "Dan", [0.9, 0.1, 0.0])
        spk.add_sample(lib, "Sara", [0.0, 0.2, 0.98])
        self.assertEqual(lib["dim"], 3)
        name, score = spk.identify([0.85, 0.15, 0.05], lib, 0.5)
        self.assertEqual(name, "Dan")
        self.assertGreater(score, 0.9)
        # a voice unlike either speaker fails the threshold
        name, score = spk.identify([0.1, 0.95, 0.1], lib, 0.5)
        self.assertIsNone(name)

    def test_identify_uses_best_of_multiple_samples(self):
        lib = {"model": None, "dim": None, "speakers": {}}
        spk.add_sample(lib, "Dan", [1.0, 0.0])
        spk.add_sample(lib, "Dan", [0.0, 1.0])   # second condition/mic
        name, _ = spk.identify([0.05, 0.99], lib, 0.5)
        self.assertEqual(name, "Dan")

    def test_library_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "speakers.json")
            with mock.patch.dict(os.environ, {"EARSHOT_SPEAKERS": path}):
                lib = spk.load_library()
                spk.add_sample(lib, "Dan", [1.0, 0.0])
                spk.save_library(lib)
                lib2 = spk.load_library()
            self.assertIn("Dan", lib2["speakers"])
            self.assertEqual(len(lib2["speakers"]["Dan"]["samples"]), 1)


# ---------------------------------------------------------------------------
# diarize.py (pure logic; no torch/pyannote imports at module level)
# ---------------------------------------------------------------------------
TURNS = [
    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_01"},
    {"start": 5.2, "end": 9.0, "speaker": "SPEAKER_00"},
    {"start": 9.5, "end": 14.0, "speaker": "SPEAKER_01"},
]


class TestDiarize(unittest.TestCase):
    def test_friendly_labels_first_appearance_order(self):
        m = diarize.friendly_labels(TURNS)
        self.assertEqual(m, {"SPEAKER_01": "REMOTE-1", "SPEAKER_00": "REMOTE-2"})

    def test_best_raw_speaker_max_overlap(self):
        self.assertEqual(diarize.best_raw_speaker(0.5, 4.5, TURNS), "SPEAKER_01")
        self.assertEqual(diarize.best_raw_speaker(5.5, 8.5, TURNS), "SPEAKER_00")
        # straddles two turns; more overlap with the second
        self.assertEqual(diarize.best_raw_speaker(8.0, 13.0, TURNS), "SPEAKER_01")
        # no overlap at all
        self.assertIsNone(diarize.best_raw_speaker(50.0, 55.0, TURNS))

    def test_relabel_far_only(self):
        transcript = {"segments": [
            {"channel": "near", "speaker": "YOU", "start": 1.0, "end": 4.0, "text": "hi"},
            {"channel": "far", "speaker": "REMOTE", "start": 0.5, "end": 4.5, "text": "a"},
            {"channel": "far", "speaker": "REMOTE", "start": 5.5, "end": 8.5, "text": "b"},
            {"channel": "far", "speaker": "REMOTE", "start": 50.0, "end": 55.0, "text": "c"},
        ]}
        mapping = diarize.friendly_labels(TURNS)
        diarize.relabel(transcript, TURNS, mapping)
        got = [s["speaker"] for s in transcript["segments"]]
        # near untouched; no-overlap segment falls back to generic REMOTE
        self.assertEqual(got, ["YOU", "REMOTE-1", "REMOTE-2", "REMOTE"])


# ---------------------------------------------------------------------------
# attribute.py
# ---------------------------------------------------------------------------
def _attr_transcript():
    return {
        "segments": [
            {"channel": "near", "speaker": "YOU", "start": 0, "end": 3, "text": "hi"},
            {"channel": "far", "speaker": "REMOTE-1", "start": 5, "end": 13,
             "text": "whisper-time text"},
            {"channel": "far", "speaker": "REMOTE-2", "start": 20, "end": 24, "text": "x"},
        ],
        "diarization": {
            "labels": ["REMOTE-1", "REMOTE-2"],
            "label_map": {"SPEAKER_00": "REMOTE-1", "SPEAKER_01": "REMOTE-2"},
            "turns": [
                {"start": 100.0, "end": 130.0, "speaker": "SPEAKER_00"},
                {"start": 20.0, "end": 24.0, "speaker": "SPEAKER_01"},
            ],
            "embeddings": {"REMOTE-1": [0.9, 0.1], "REMOTE-2": [0.0, 1.0]},
            "identification": {},
        },
    }


class TestAttribute(unittest.TestCase):
    def test_unidentified_labels_only_remote_n(self):
        t = _attr_transcript()
        self.assertEqual(attribute.unidentified_labels(t), ["REMOTE-1", "REMOTE-2"])
        t["diarization"]["labels"] = ["Dan", "REMOTE-2"]
        self.assertEqual(attribute.unidentified_labels(t), ["REMOTE-2"])

    def test_generic_remote_fallback_offered(self):
        t = _attr_transcript()
        t["diarization"]["labels"] = ["REMOTE", "REMOTE-2"]
        self.assertEqual(attribute.unidentified_labels(t), ["REMOTE", "REMOTE-2"])

    def test_sample_prefers_turn_time_over_segment_time(self):
        # segment says 5s (whisper), turn says 100s (audio truth) -> use 100
        start, dur, _ = attribute.sample_segment(_attr_transcript(), "REMOTE-1", 8.0)
        self.assertEqual(start, 100.0)
        self.assertEqual(dur, 8.0)

    def test_sample_falls_back_to_segments_without_turns(self):
        t = _attr_transcript()
        t["diarization"]["turns"] = []
        start, dur, text = attribute.sample_segment(t, "REMOTE-1", 8.0)
        self.assertEqual(start, 5.0)
        self.assertEqual(text, "whisper-time text")

    def test_apply_names_relabels_and_renames_metadata(self):
        t = _attr_transcript()
        attribute.apply_names(t, {"REMOTE-1": "Dan"})
        self.assertEqual([s["speaker"] for s in t["segments"]],
                         ["YOU", "Dan", "REMOTE-2"])
        d = t["diarization"]
        self.assertIn("Dan", d["embeddings"])
        self.assertNotIn("REMOTE-1", d["embeddings"])
        self.assertEqual(d["label_map"]["SPEAKER_00"], "Dan")
        self.assertEqual(d["identification"]["Dan"]["source"], "manual")
        self.assertIn("Dan", d["labels"])


# ---------------------------------------------------------------------------
# summarize.py
# ---------------------------------------------------------------------------
class _StubHandler(http.server.BaseHTTPRequestHandler):
    payload = {}

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        type(self).last_request = json.loads(self.rfile.read(n) or b"{}")
        body = json.dumps(type(self).payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _stub_server(payload):
    handler = type("H", (_StubHandler,), {"payload": payload})
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, handler


class TestSummarize(unittest.TestCase):
    def test_read_transcript_prefers_md_falls_back_to_json(self):
        with tempfile.TemporaryDirectory() as d:
            jpath = os.path.join(d, "transcript.json")
            with open(jpath, "w") as fh:
                json.dump({"segments": [
                    {"speaker": "YOU", "text": "from json"}]}, fh)
            self.assertIn("YOU: from json", summarize.read_transcript(d, None))
            with open(os.path.join(d, "transcript.md"), "w") as fh:
                fh.write("# md wins\n")
            self.assertIn("md wins", summarize.read_transcript(d, None))

    def test_ollama_backend(self):
        srv, handler = _stub_server(
            {"message": {"content": "## Summary\nok"}})
        try:
            args = mock.Mock(backend="ollama", model="m",
                             ollama_url=f"http://127.0.0.1:{srv.server_port}",
                             max_tokens=64)
            out = summarize.summarize_ollama("transcript text", args)
            self.assertIn("## Summary", out)
            self.assertEqual(handler.last_request["model"], "m")
            self.assertFalse(handler.last_request["stream"])
        finally:
            srv.shutdown()
            srv.server_close()

    def test_openai_backend(self):
        srv, handler = _stub_server(
            {"choices": [{"message": {"content": "## Summary\nvllm"}}]})
        try:
            args = mock.Mock(backend="vllm", model="served-model",
                             openai_url=f"http://127.0.0.1:{srv.server_port}/v1",
                             max_tokens=64)
            out = summarize.summarize_openai("transcript text", args)
            self.assertIn("vllm", out)
            self.assertEqual(handler.last_request["model"], "served-model")
            self.assertEqual(handler.last_request["max_tokens"], 64)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_ollama_gets_num_predict(self):
        srv, handler = _stub_server({"message": {"content": "notes"}})
        try:
            args = mock.Mock(backend="ollama", model="m",
                             ollama_url=f"http://127.0.0.1:{srv.server_port}",
                             max_tokens=77)
            summarize.summarize_ollama("t", args)
            self.assertEqual(handler.last_request["options"]["num_predict"], 77)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_length_warning(self):
        self.assertIsNone(summarize.length_warning("short", "ollama"))
        long_text = "x" * (9000 * 4)
        self.assertIn("context window", summarize.length_warning(long_text, "ollama"))
        # vllm has more headroom before the warning fires
        self.assertIsNone(summarize.length_warning(long_text, "vllm"))

    def test_claude_backend_requires_key(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SystemExit):
                summarize.summarize_claude("t", mock.Mock(model="m", max_tokens=64))


# ---------------------------------------------------------------------------
# timing.py
# ---------------------------------------------------------------------------
def _timing_transcript(offset_fn):
    import random
    random.seed(7)
    turns, segs = [], []
    t = 0.0
    for i in range(40):
        dur = random.uniform(2.0, 9.0)
        gap = random.uniform(0.3, 4.0)
        s, e = t, t + dur
        turns.append({"start": s, "end": e, "speaker": f"S{i % 2}"})
        off = offset_fn(s)
        segs.append({"channel": "far", "speaker": "R",
                     "start": s + off, "end": e + off, "text": "x"})
        t = e + gap
    return {"segments": segs, "diarization": {"turns": turns}}


class TestTiming(unittest.TestCase):
    def test_aligned(self):
        r = timing.analyze(_timing_transcript(lambda s: 0.1))
        self.assertLess(abs(r["shift"]), 1.0)
        self.assertIn("aligned", timing.verdict(r))

    def test_constant_shift_detected_with_sign(self):
        r = timing.analyze(_timing_transcript(lambda s: 4.0))
        self.assertAlmostEqual(r["shift"], 4.0, delta=0.5)
        self.assertIn("constant shift", timing.verdict(r))
        r = timing.analyze(_timing_transcript(lambda s: -2.0))
        self.assertAlmostEqual(r["shift"], -2.0, delta=0.5)

    def test_growing_drift_detected(self):
        r = timing.analyze(_timing_transcript(lambda s: s * 0.05))
        self.assertIn("DRIFT", timing.verdict(r))

    def test_merge_and_overlap(self):
        merged = timing.merge_intervals([(0, 5), (4, 8), (10, 12)])
        self.assertEqual(merged, [[0, 8], [10, 12]])
        starts = [m[0] for m in merged]
        self.assertAlmostEqual(timing.overlap_with(merged, starts, 6, 11), 3.0)


if __name__ == "__main__":
    unittest.main()
