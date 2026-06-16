import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from briefly.progress import ProgressReporter, read_heartbeat


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _wav(path, rate=16000):
    """0.6s tone, 0.6s silence, 0.6s tone -> VAD finds 2 mic segments."""
    n = int(0.6 * rate)
    tone = [int(8000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n)]
    samples = tone + [0] * n + tone
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<%dh" % len(samples), *samples))


class TestProgressReporter(unittest.TestCase):
    def test_heartbeat_overall_and_stage_marks(self):
        clk = _Clock()
        with tempfile.TemporaryDirectory() as td:
            stages = ["preprocess", "diarize", "transcribe", "merge"]
            r = ProgressReporter(td, "M1", stages, clock=clk, throttle_sec=0.0)
            r.stage("preprocess"); r.done("preprocess")
            r.stage("diarize"); r.done("diarize")
            r.stage("transcribe"); r.update(0.5, "33/66 utterances")
            hb = read_heartbeat(td, "M1")
            self.assertEqual(hb["stage"], "transcribe")
            self.assertEqual(hb["detail"], "33/66 utterances")
            self.assertEqual([hb["stages"][s] for s in stages],
                             ["done", "done", "running", "pending"])
            # weights: preprocess .15 + diarize .45 done + transcribe .35*0.5 = .775
            self.assertAlmostEqual(hb["overall_frac"], 0.775, places=2)

    def test_throttle_suppresses_foreground_lines(self):
        clk = _Clock()
        logs: list = []
        with tempfile.TemporaryDirectory() as td:
            r = ProgressReporter(td, "M2", ["transcribe"], clock=clk,
                                 throttle_sec=0.5, log=logs.append)
            r.stage("transcribe")          # force-write at t=0
            clk.t = 0.1; r.update(0.1, "1/10")   # within throttle -> no line
            clk.t = 0.2; r.update(0.2, "2/10")   # within throttle -> no line
            clk.t = 1.0; r.update(0.3, "3/10")   # past throttle -> one line
            self.assertEqual(len(logs), 1)
            self.assertIn("3/10", logs[0])


class TestTranscribeProgress(unittest.TestCase):
    def test_on_progress_ticks_to_total(self):
        from briefly.clients.transcribe import TranscribeConfig, transcribe_meeting
        with tempfile.TemporaryDirectory() as td:
            proc = Path(td) / "processed"; proc.mkdir()
            tx = Path(td) / "transcripts"; tx.mkdir()
            _wav(proc / "line.16k.wav"); _wav(proc / "mic.16k.wav")
            (tx / "line.diarization.json").write_text(json.dumps(
                {"segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 0.6},
                              {"speaker": "SPEAKER_00", "start": 1.2, "end": 1.8}]}))
            ticks: list = []
            transcribe_meeting(proc, tx, TranscribeConfig(concurrency=2),
                               transcribe=lambda pcm: "x",
                               on_progress=lambda d, t: ticks.append((d, t)))
            self.assertTrue(ticks)
            total = ticks[-1][1]
            self.assertGreaterEqual(total, 2)                      # >= 2 line turns
            self.assertEqual(sorted(d for d, _ in ticks), list(range(1, total + 1)))
            self.assertEqual(ticks[-1], (total, total))


class TestStatusRender(unittest.TestCase):
    def test_inferred_then_heartbeat(self):
        from briefly.orchestrator import PipelineConfig, _status_lines
        with tempfile.TemporaryDirectory() as td:
            cfg = PipelineConfig(data_root=td)
            lines = _status_lines(cfg, "M9")
            self.assertIn("inferred", lines[0])
            self.assertIn("next: preprocess", lines[2])
            ProgressReporter(td, "M9", ["preprocess"], throttle_sec=0.0).stage("preprocess")
            self.assertIn("live heartbeat", _status_lines(cfg, "M9")[0])


if __name__ == "__main__":
    unittest.main()
