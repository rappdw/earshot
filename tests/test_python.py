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
import transcribe               # noqa: E402


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

    def test_library_save_is_private_atomic_and_backed_up(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "speakers.json")
            with mock.patch.dict(os.environ, {"EARSHOT_SPEAKERS": path}):
                lib = spk.load_library()
                spk.add_sample(lib, "Dan", [1.0, 0.0])
                spk.save_library(lib)
                self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
                self.assertFalse(os.path.exists(path + ".tmp"))
                spk.add_sample(lib, "Sara", [0.0, 1.0])
                spk.save_library(lib)
                # previous generation preserved
                with open(path + ".bak") as fh:
                    prev = json.load(fh)
                self.assertNotIn("Sara", prev["speakers"])
                with open(path) as fh:
                    cur = json.load(fh)
                self.assertIn("Sara", cur["speakers"])


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

    def test_dropped_manual_names(self):
        prev = {"Dan": {"matched": "Dan", "score": 1.0, "source": "manual"},
                "Sara": {"matched": "Sara", "score": 0.8},
                "Kim": {"matched": "Kim", "score": 1.0, "source": "manual"}}
        # Dan re-identified this run; Kim wasn't; Sara wasn't manual
        self.assertEqual(diarize.dropped_manual_names(prev, ["Dan", "REMOTE-1"]),
                         ["Kim"])
        self.assertEqual(diarize.dropped_manual_names(None, ["X"]), [])

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
        # rename REMOTE-1 -> Dan the way apply_names would (labels AND segments)
        attribute.apply_names(t, {"REMOTE-1": "Dan"})
        self.assertEqual(attribute.unidentified_labels(t), ["REMOTE-2"])

    def test_generic_remote_fallback_offered(self):
        t = _attr_transcript()
        t["diarization"]["labels"] = []
        for s in t["segments"]:
            if s["channel"] == "far":
                s["speaker"] = "REMOTE"
        self.assertEqual(attribute.unidentified_labels(t), ["REMOTE"])

    def test_generic_remote_offered_even_when_labels_present(self):
        # regression: diarization.labels never contains the no-overlap fallback
        # label, but segments carrying it must still be offered for naming
        t = _attr_transcript()
        t["segments"].append({"channel": "far", "speaker": "REMOTE",
                              "start": 40, "end": 44, "text": "orphan"})
        self.assertEqual(attribute.unidentified_labels(t),
                         ["REMOTE", "REMOTE-1", "REMOTE-2"])

    def test_sample_prefers_turn_time_over_segment_time(self):
        # segment says 5s (whisper), turn says 100s (audio truth) -> use 100
        start, dur, _ = attribute.sample_segment(_attr_transcript(), "REMOTE-1", 8.0)
        self.assertEqual(start, 100.0)
        self.assertEqual(dur, 8.0)

    def test_sample_hint_skips_grazing_previous_speaker(self):
        # REMOTE-1's segment tail grazes the [20,24] clip window by 0.5s and
        # sorts first; the printed hint must still be REMOTE-2's own text
        t = _attr_transcript()
        t["segments"][1]["end"] = 20.5
        _, _, text = attribute.sample_segment(t, "REMOTE-2", 8.0)
        self.assertEqual(text, "x")

    def test_sample_hint_falls_back_to_biggest_overlap(self):
        # no segment in the window carries the label: use the largest
        # overlap, not the first grazing segment
        t = _attr_transcript()
        t["segments"][2]["speaker"] = "Sara"
        t["segments"][1]["end"] = 20.5
        _, _, text = attribute.sample_segment(t, "REMOTE-2", 8.0)
        self.assertEqual(text, "x")

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
# transcribe.py (pure helpers; faster_whisper only imported inside main)
# ---------------------------------------------------------------------------
class TestTranscribe(unittest.TestCase):
    def test_resolve_inputs_flac_fallback_and_my_name(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "far_remote.flac"), "w").close()
            open(os.path.join(d, "near_me.wav"), "w").close()
            args = mock.Mock(file=None, meeting_dir=d, speaker=None, channels="both")
            with mock.patch.dict(os.environ, {"EARSHOT_MY_NAME": "Dan"}), \
                 mock.patch.object(transcribe, "channel_peak_db", return_value=-12.0):
                jobs, skipped = transcribe.resolve_inputs(args)
            chans = {c: (os.path.basename(p), s) for p, s, c in jobs}
            self.assertEqual(chans["near"], ("near_me.wav", "Dan"))
            self.assertEqual(chans["far"], ("far_remote.flac", "REMOTE"))
            self.assertEqual(skipped, {})

    def test_resolve_inputs_far_only_in_person(self):
        # in-person captures have no near channel; far alone must transcribe
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "far_remote.wav"), "w").close()
            args = mock.Mock(file=None, meeting_dir=d, speaker=None, channels="both")
            jobs, _ = transcribe.resolve_inputs(args)
            self.assertEqual(
                jobs, [(os.path.join(d, "far_remote.wav"), "REMOTE", "far")])

    def test_resolve_inputs_skips_a_silent_channel(self):
        # a misrouted call channel recorded nothing: skip it, keep the mic
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "far_remote.wav"), "w").close()
            open(os.path.join(d, "near_me.wav"), "w").close()
            args = mock.Mock(file=None, meeting_dir=d, speaker=None, channels="both")
            peaks = {"far_remote.wav": -91.0, "near_me.wav": -9.0}
            with mock.patch.object(transcribe, "channel_peak_db",
                                   side_effect=lambda p, **k: peaks[os.path.basename(p)]):
                jobs, skipped = transcribe.resolve_inputs(args)
            self.assertEqual([c for _, _, c in jobs], ["near"])
            self.assertIn("far", skipped)

    def test_resolve_inputs_keeps_a_lone_quiet_channel(self):
        # only one channel exists: transcribe it even if quiet, never nothing
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "far_remote.wav"), "w").close()
            args = mock.Mock(file=None, meeting_dir=d, speaker=None, channels="both")
            with mock.patch.object(transcribe, "channel_peak_db", return_value=-99.0):
                jobs, skipped = transcribe.resolve_inputs(args)
            self.assertEqual([c for _, _, c in jobs], ["far"])
            self.assertEqual(skipped, {})

    def test_resolve_inputs_channels_override(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "far_remote.wav"), "w").close()
            open(os.path.join(d, "near_me.wav"), "w").close()
            args = mock.Mock(file=None, meeting_dir=d, speaker=None, channels="far")
            jobs, _ = transcribe.resolve_inputs(args)
            self.assertEqual([c for _, _, c in jobs], ["far"])

    def test_harmonize_drops_speaker_bleed(self):
        segs = [
            {"channel": "far", "speaker": "REMOTE", "start": 10.0, "end": 14.0,
             "text": "so the migration lands on Friday"},
            # same words, same time, on the mic: the speakers bleeding in
            {"channel": "near", "speaker": "Dan", "start": 10.1, "end": 13.9,
             "text": "So the migration lands on Friday."},
        ]
        kept, dropped = transcribe.harmonize_channels(segs)
        self.assertEqual(dropped, 1)
        self.assertEqual([s["channel"] for s in kept], ["far"])

    def test_harmonize_keeps_genuine_simultaneous_speech(self):
        segs = [
            {"channel": "far", "speaker": "REMOTE", "start": 10.0, "end": 14.0,
             "text": "so the migration lands on Friday"},
            {"channel": "near", "speaker": "Dan", "start": 10.5, "end": 12.0,
             "text": "wait, that clashes with the audit"},
        ]
        kept, dropped = transcribe.harmonize_channels(segs)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 2)

    def test_harmonize_noop_without_far(self):
        segs = [{"channel": "near", "speaker": "Dan", "start": 1.0, "end": 2.0,
                 "text": "hello"}]
        kept, dropped = transcribe.harmonize_channels(segs)
        self.assertEqual((len(kept), dropped), (1, 0))

    def test_resolve_batch_size_auto_and_explicit(self):
        # auto: batch on CUDA, sequential on CPU
        self.assertEqual(transcribe.resolve_batch_size(None, "cuda"), 8)
        self.assertEqual(transcribe.resolve_batch_size(None, "cpu"), 0)
        # explicit wins, including forcing sequential with 0
        self.assertEqual(transcribe.resolve_batch_size(0, "cuda"), 0)
        self.assertEqual(transcribe.resolve_batch_size(16, "cpu"), 16)

    def test_transcribe_channel_routes_batched_vs_sequential(self):
        info = mock.Mock(language="en", language_probability=0.99)
        seq = mock.Mock(); seq.transcribe.return_value = ([], info)
        bat = mock.Mock(); bat.transcribe.return_value = ([], info)
        args = mock.Mock(language=None, beam_size=5)
        # batch_size > 0 with a pipeline -> batched path, no vad_filter kwarg
        transcribe.transcribe_channel(seq, bat, 8, "a.wav", "R", "far", args, None)
        bat.transcribe.assert_called_once()
        self.assertEqual(bat.transcribe.call_args.kwargs["batch_size"], 8)
        seq.transcribe.assert_not_called()
        # batch_size 0 -> sequential path with vad_filter
        transcribe.transcribe_channel(seq, bat, 0, "a.wav", "R", "far", args, None)
        seq.transcribe.assert_called_once()
        self.assertTrue(seq.transcribe.call_args.kwargs["vad_filter"])

    def test_write_outputs_schema_and_languages(self):
        with tempfile.TemporaryDirectory() as d:
            args = mock.Mock(meeting_dir=d, file=None, model="large-v3")
            meta = {"device": "cpu", "compute_type": "int8", "language": "en",
                    "languages": {"near": "en", "far": "en"},
                    "generated_at": "2026-07-02T00:00:00"}
            segs = [{"speaker": "Dan", "channel": "near", "start": 0,
                     "end": 1, "text": "hi", "words": None}]
            jpath, _ = transcribe.write_outputs(args, segs, meta)
            with open(jpath) as fh:
                js = json.load(fh)
            self.assertEqual(js["schema"], 1)
            self.assertEqual(js["languages"]["far"], "en")


# ---------------------------------------------------------------------------
# diarize alignment note
# ---------------------------------------------------------------------------
class TestAlignmentNote(unittest.TestCase):
    def test_ok_and_warning(self):
        t = _timing_transcript(lambda s: 0.1)
        self.assertIn("(ok)", diarize.alignment_note(t))
        t = _timing_transcript(lambda s: 6.0)
        self.assertIn("WARNING", diarize.alignment_note(t))
        # un-analyzable transcript -> silently None, never raises
        self.assertIsNone(diarize.alignment_note({"segments": []}))


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

    def test_sparse_junk_start_is_not_drift(self):
        # regression: an aligned meeting whose quiet first minutes contain a
        # few Whisper hallucinations (no matching turns) must not read as
        # DRIFT; the sparse third anchors to the global shift.
        t = _timing_transcript(lambda s: -0.25)
        for s in t["segments"]:
            s["start"] += 120.0
            s["end"] += 120.0
        for turn in t["diarization"]["turns"]:
            turn["start"] += 120.0
            turn["end"] += 120.0
        for js in (5.0, 40.0, 80.0):
            t["segments"].insert(0, {"channel": "far", "speaker": "R",
                                     "start": js, "end": js + 2.0, "text": "uh"})
        r = timing.analyze(t)
        self.assertIn("aligned", timing.verdict(r))
        self.assertFalse(r["thirds_confident"][0])   # sparse third flagged

    def test_merge_and_overlap(self):
        merged = timing.merge_intervals([(0, 5), (4, 8), (10, 12)])
        self.assertEqual(merged, [[0, 8], [10, 12]])
        starts = [m[0] for m in merged]
        self.assertAlmostEqual(timing.overlap_with(merged, starts, 6, 11), 3.0)


# ---------------------------------------------------------------------------
# meetings.py (list / grep / clean)
# ---------------------------------------------------------------------------
import contextlib
import io

import meetings                 # noqa: E402


def _fixture_tree(d):
    """Two meetings: one transcribed+diarized with old wav audio, one bare."""
    m1 = os.path.join(d, "2026-06-01_100000_alpha")
    m2 = os.path.join(d, "2026-07-01_100000_beta")
    os.makedirs(m1)
    os.makedirs(m2)
    for base in ("far_remote", "near_me"):
        p = os.path.join(m1, base + ".wav")
        with open(p, "wb") as fh:
            fh.write(b"\0" * 10000)
        os.utime(p, (0, 0))  # ancient
    with open(os.path.join(m1, "monitor_stereo.wav"), "wb") as fh:
        fh.write(b"\0" * 5000)
    with open(os.path.join(m1, "transcript.json"), "w") as fh:
        json.dump({"schema": 1, "diarization": {"labels": ["Dan"]},
                   "segments": [{"channel": "far", "speaker": "Dan",
                                 "start": 12, "end": 15,
                                 "text": "the Kong migration plan"}]}, fh)
    with open(os.path.join(m2, "far_remote.wav"), "wb") as fh:
        fh.write(b"\0" * 800)
    return m1, m2


class TestMeetings(unittest.TestCase):
    def _env(self, d):
        return mock.patch.dict(os.environ, {"EARSHOT_ROOTS": d})

    def test_pending_length_falls_back_to_audio(self):
        # recordings older than duration_sec in meeting.json: ask ffprobe
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "2026-07-01_090000_old")
            os.makedirs(d)
            with open(os.path.join(d, "far_remote.wav"), "wb") as fh:
                fh.write(b"\0" * 4096)
            with open(os.path.join(d, "meeting.json"), "w") as fh:
                fh.write("{}")
            with mock.patch.dict(os.environ, {"EARSHOT_ROOTS": root}), \
                 mock.patch.object(meetings, "audio_seconds", return_value=1845.0):
                rows = meetings.pending_meetings()
            self.assertEqual(rows[0][2]["duration_sec"], 1845.0)

    def test_list_shows_status(self):
        with tempfile.TemporaryDirectory() as d:
            _fixture_tree(d)
            with self._env(d):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    meetings.cmd_list(mock.Mock(all=True, limit=15))
            text = out.getvalue()
            self.assertIn("alpha", text)
            self.assertIn("beta", text)
            # alpha: transcribed+diarized+attributed; beta: nothing yet
            alpha = [l for l in text.splitlines() if "alpha" in l][0]
            self.assertRegex(alpha, r"x\s+x\s+x\s+-")

    def test_grep_finds_with_context(self):
        with tempfile.TemporaryDirectory() as d:
            _fixture_tree(d)
            with self._env(d):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    meetings.cmd_grep(mock.Mock(pattern="kong", all=False))
            self.assertIn("[00:00:12] Dan: the Kong migration plan", out.getvalue())

    def test_clean_dry_run_touches_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            m1, _ = _fixture_tree(d)
            with self._env(d):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    meetings.cmd_clean(mock.Mock(days=14, apply=False,
                                                 purge_audio=True))
            self.assertIn("would rm", out.getvalue())
            self.assertTrue(os.path.exists(os.path.join(m1, "far_remote.wav")))

    def test_clean_compress_via_stub_ffmpeg(self):
        with tempfile.TemporaryDirectory() as d, \
             tempfile.TemporaryDirectory() as stub:
            m1, _ = _fixture_tree(d)
            ff = os.path.join(stub, "ffmpeg")
            with open(ff, "w") as fh:
                fh.write("#!/bin/sh\n"
                         'out=""; for a in "$@"; do out="$a"; done\n'
                         'printf x > "$out"\n')
            os.chmod(ff, 0o755)
            env = {"EARSHOT_ROOTS": d, "PATH": stub + os.pathsep + os.environ["PATH"]}
            with mock.patch.dict(os.environ, env):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    meetings.cmd_clean(mock.Mock(days=14, apply=True,
                                                 purge_audio=False))
            # wavs compressed away, flacs exist, monitor removed
            self.assertFalse(os.path.exists(os.path.join(m1, "far_remote.wav")))
            self.assertTrue(os.path.exists(os.path.join(m1, "far_remote.flac")))
            self.assertFalse(os.path.exists(os.path.join(m1, "monitor_stereo.wav")))

    def test_clean_purge_requires_transcript(self):
        with tempfile.TemporaryDirectory() as d:
            m1, m2 = _fixture_tree(d)
            old = os.path.join(m2, "far_remote.wav")
            os.utime(old, (0, 0))  # old but NO transcript in m2
            with self._env(d):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    meetings.cmd_clean(mock.Mock(days=14, apply=True,
                                                 purge_audio=True))
            self.assertTrue(os.path.exists(old))      # protected
            self.assertFalse(os.path.exists(os.path.join(m1, "far_remote.wav")))


if __name__ == "__main__":
    unittest.main()
