import json
import tempfile
import unittest
from pathlib import Path

from briefly.clients.asr import AsrConfig, asr_file, to_whisper_doc, transcribe_meeting_asr

# Transcribe-only /asr response (diarization is a separate pyannote-protocol stage).
ASR_RESP = {
    "model": "whisperx-large-v2", "language": "en", "duration_sec": 6.0, "device": "cuda",
    "segments": [
        {"start": 0.0, "end": 2.0, "text": "Hello there",
         "words": [{"start": 0.0, "end": 0.5, "word": "Hello", "score": 0.9}]},
        {"start": 2.5, "end": 4.0, "text": "Hi back", "words": []},
        {"start": 4.0, "end": 4.5, "text": "   ", "words": []},   # blank -> dropped
    ]}


def _fake_post(payload):
    calls = []

    def post(url, files=None, fields=None, timeout=None):
        calls.append({"url": url, "files": files, "fields": dict(fields or [])})
        return json.dumps(payload).encode()

    post.calls = calls
    return post


class TestAsrClient(unittest.TestCase):
    def test_asr_file_fields_and_parse(self):
        post = _fake_post(ASR_RESP)
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "x.wav"; wav.write_bytes(b"x")
            resp = asr_file(wav, AsrConfig(url="http://x/asr", language="en"), post=post)
            self.assertEqual(len(resp["segments"]), 3)
            self.assertEqual(post.calls[0]["files"][0][0], "audio")
            self.assertEqual(post.calls[0]["fields"]["language"], "en")

    def test_to_whisper_doc_drops_blanks_and_ids(self):
        wd = to_whisper_doc(ASR_RESP)
        self.assertEqual([s["text"] for s in wd["segments"]], ["Hello there", "Hi back"])
        self.assertEqual([s["id"] for s in wd["segments"]], [0, 1])
        self.assertEqual(wd["language"], "en")

    def test_missing_segments_raises(self):
        post = _fake_post({"model": "x"})
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "x.wav"; wav.write_bytes(b"x")
            with self.assertRaises(ValueError):
                asr_file(wav, AsrConfig(url="http://x/asr"), post=post)


class TestAsrMeeting(unittest.TestCase):
    def test_transcribe_writes_both_channels_no_diar(self):
        post = _fake_post(ASR_RESP)
        with tempfile.TemporaryDirectory() as td:
            proc = Path(td) / "processed"; proc.mkdir()
            (proc / "line.16k.wav").write_bytes(b"x")
            (proc / "mic.16k.wav").write_bytes(b"x")
            tx = Path(td) / "tx"
            transcribe_meeting_asr(proc, tx, AsrConfig(url="http://x/asr"), post=post)
            self.assertTrue((tx / "mic.whisper.json").exists())
            self.assertTrue((tx / "line.whisper.json").exists())
            self.assertFalse((tx / "line.diarization.json").exists())   # diarization is the pyannote stage
            self.assertEqual(len(post.calls), 2)   # one per channel


if __name__ == "__main__":
    unittest.main()
