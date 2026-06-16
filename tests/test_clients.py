import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from briefly.clients.diarize import (
    DiarizeConfig, diarize_file, diarize_meeting, diarize_single)
from briefly.merge import DiarDoc


def _fake_post(payload: dict):
    calls = []

    def post(url, files=None, fields=None, timeout=None):
        calls.append({"url": url, "files": files, "fields": fields, "timeout": timeout})
        return json.dumps(payload).encode()

    post.calls = calls
    return post


class TestDiarize(unittest.TestCase):
    DIAR = {"model": "pyannote/community-1", "duration_sec": 5.0, "num_speakers": 2,
            "segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
                         {"speaker": "SPEAKER_01", "start": 2.0, "end": 4.0}]}

    def test_uses_audio_field_and_merge_compat(self):
        post = _fake_post(self.DIAR)
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "line.16k.wav"
            wav.write_bytes(b"x")
            resp = diarize_file(wav, DiarizeConfig(url="http://d", max_speakers=3), post=post)
            self.assertEqual(resp["num_speakers"], 2)
            DiarDoc.from_dict(resp)
            # the homelab service expects the multipart field named "audio"
            self.assertEqual(post.calls[0]["files"][0][0], "audio")
            fields = dict((k, v) for k, v in post.calls[0]["fields"])
            self.assertEqual(fields["max_speakers"], "3")

    def test_missing_segments_raises(self):
        post = _fake_post({"model": "x"})
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "line.16k.wav"
            wav.write_bytes(b"x")
            with self.assertRaises(ValueError):
                diarize_file(wav, DiarizeConfig(url="http://d"), post=post)

    def test_diarize_meeting_writes(self):
        post = _fake_post(self.DIAR)
        with tempfile.TemporaryDirectory() as td:
            proc = Path(td) / "processed"
            proc.mkdir()
            (proc / "line.16k.wav").write_bytes(b"x")
            tx = Path(td) / "transcripts"
            out = diarize_meeting(proc, tx, DiarizeConfig(url="http://d"), post=post)
            self.assertTrue(out.exists())
            DiarDoc.from_dict(json.loads(out.read_text()))


def _write_speechy_wav(path: Path, rate: int = 16000) -> None:
    """0.6s tone, 0.6s silence, 0.6s tone -> VAD should find 2 spans (one speaker)."""
    n = int(0.6 * rate)
    tone = [int(8000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n)]
    samples = tone + [0] * n + tone
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<%dh" % len(samples), *samples))


class TestDiarizeSingle(unittest.TestCase):
    def test_single_speaker_fastpath(self):
        with tempfile.TemporaryDirectory() as td:
            proc = Path(td) / "processed"
            proc.mkdir()
            _write_speechy_wav(proc / "line.16k.wav")
            tx = Path(td) / "transcripts"
            out = diarize_single(proc, tx)
            self.assertTrue(out.exists())
            resp = json.loads(out.read_text())
            DiarDoc.from_dict(resp)  # merge-compatible schema
            self.assertEqual(resp["model"], "vad-single-speaker")
            self.assertEqual(resp["num_speakers"], 1)
            self.assertGreaterEqual(len(resp["segments"]), 1)
            self.assertTrue(all(s["speaker"] == "SPEAKER_00" for s in resp["segments"]))

    def test_missing_line_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                diarize_single(Path(td) / "processed", Path(td) / "transcripts")


if __name__ == "__main__":
    unittest.main()
