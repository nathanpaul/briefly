import json
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from briefly.clients import vad
from briefly.clients.transcribe import TranscribeConfig, transcribe_meeting
from briefly.merge import WhisperDoc

RATE = 16000


def _write_wav(path: Path, samples, rate: int = RATE) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(array("h", samples).tobytes())


class TestVad(unittest.TestCase):
    def test_segments_a_single_burst(self):
        samples = [0] * (RATE // 2) + [9000] * (RATE // 2) + [0] * (RATE // 2)  # sil/loud/sil
        a = array("h", samples)
        segs = vad.segment_speech(a, RATE)
        self.assertEqual(len(segs), 1)
        s, e = segs[0]
        self.assertLess(s, 0.6)     # starts around 0.5 s (minus pad)
        self.assertGreater(e, 0.9)  # ends around 1.0 s (plus pad)

    def test_read_and_slice(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.wav"
            _write_wav(p, [1000] * RATE)
            a, rate = vad.read_pcm16_mono(p)
            self.assertEqual(rate, RATE)
            self.assertEqual(len(vad.slice_pcm(a, rate, 0.0, 0.5)), RATE)  # 0.5 s * 2 bytes... samples


class TestTranscribeMeeting(unittest.TestCase):
    def test_line_follows_diarization_mic_follows_vad(self):
        with tempfile.TemporaryDirectory() as td:
            proc = Path(td) / "processed"
            proc.mkdir()
            _write_wav(proc / "line.16k.wav", [0] * (RATE * 3))                 # 3 s
            _write_wav(proc / "mic.16k.wav",
                       [0] * (RATE // 2) + [9000] * (RATE // 2) + [0] * (RATE // 2))
            tx = Path(td) / "transcripts"
            tx.mkdir()
            (tx / "line.diarization.json").write_text(json.dumps({
                "segments": [{"speaker": "SPEAKER_00", "start": 0.5, "end": 1.5},
                             {"speaker": "SPEAKER_01", "start": 2.0, "end": 2.8}]}))

            out = transcribe_meeting(proc, tx, TranscribeConfig(), transcribe=lambda pcm: "hello")

            line = WhisperDoc.from_dict(json.loads(out["line"].read_text()))
            self.assertEqual([(s.start, s.end) for s in line.segments], [(0.5, 1.5), (2.0, 2.8)])
            self.assertTrue(all(s.text == "hello" for s in line.segments))
            mic = WhisperDoc.from_dict(json.loads(out["mic"].read_text()))
            self.assertEqual(len(mic.segments), 1)        # one VAD utterance
            self.assertEqual(mic.segments[0].text, "hello")

    def test_empty_text_segments_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            proc = Path(td) / "processed"
            proc.mkdir()
            _write_wav(proc / "line.16k.wav", [0] * RATE)
            _write_wav(proc / "mic.16k.wav", [0] * RATE)
            tx = Path(td) / "transcripts"
            tx.mkdir()
            (tx / "line.diarization.json").write_text(json.dumps(
                {"segments": [{"speaker": "SPEAKER_00", "start": 0.1, "end": 0.5}]}))
            out = transcribe_meeting(proc, tx, TranscribeConfig(), transcribe=lambda pcm: "  ")
            self.assertEqual(json.loads(out["line"].read_text())["segments"], [])

    def test_missing_diarization_raises(self):
        with tempfile.TemporaryDirectory() as td:
            proc = Path(td) / "processed"
            proc.mkdir()
            _write_wav(proc / "line.16k.wav", [0] * RATE)
            _write_wav(proc / "mic.16k.wav", [0] * RATE)
            with self.assertRaises(FileNotFoundError):
                transcribe_meeting(proc, Path(td) / "transcripts", TranscribeConfig(),
                                   transcribe=lambda pcm: "x")


if __name__ == "__main__":
    unittest.main()
