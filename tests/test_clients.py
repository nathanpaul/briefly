import json
import tempfile
import unittest
from pathlib import Path

from briefly.clients.diarize import DiarizeConfig, diarize_file, diarize_meeting
from briefly.clients.transcribe import (
    TranscribeConfig,
    normalize_openai,
    normalize_whisperx,
    transcribe_file,
    transcribe_meeting,
)
from briefly.merge import DiarDoc, WhisperDoc

OPENAI_RESP = {
    "language": "english", "duration": 5.0,
    "segments": [
        {"id": 0, "start": 0.0, "end": 2.0, "text": " Hello there",
         "avg_logprob": -0.2, "no_speech_prob": 0.01,
         "words": [{"word": "Hello", "start": 0.0, "end": 0.4, "probability": 0.98}]},
    ],
}
WHISPERX_RESP = {
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 2.0, "text": "Hello there",
         "words": [{"word": "Hello", "start": 0.0, "end": 0.4, "score": 0.9}]},
    ],
}


def _fake_post(payload: dict):
    calls = []

    def post(url, files=None, fields=None, timeout=None):
        calls.append({"url": url, "files": files, "fields": fields, "timeout": timeout})
        return json.dumps(payload).encode()

    post.calls = calls
    return post


class TestTranscribeNormalize(unittest.TestCase):
    def test_openai(self):
        d = normalize_openai(OPENAI_RESP)
        self.assertEqual(d["duration_sec"], 5.0)
        self.assertEqual(d["language"], "english")
        seg = d["segments"][0]
        self.assertEqual(seg["text"], "Hello there")          # stripped
        self.assertEqual(seg["words"][0]["prob"], 0.98)        # probability -> prob
        # merge must accept the normalized shape
        WhisperDoc.from_dict(d)

    def test_openai_top_level_words_distributed(self):
        resp = {"language": "en", "duration": 4.0,
                "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "hi"},
                             {"id": 1, "start": 2.0, "end": 4.0, "text": "bye"}],
                "words": [{"word": "hi", "start": 0.1, "end": 0.3, "probability": 0.9},
                          {"word": "bye", "start": 2.1, "end": 2.4, "probability": 0.8}]}
        d = normalize_openai(resp)
        self.assertEqual([w["word"] for w in d["segments"][0]["words"]], ["hi"])
        self.assertEqual([w["word"] for w in d["segments"][1]["words"]], ["bye"])

    def test_whisperx(self):
        d = normalize_whisperx(WHISPERX_RESP)
        self.assertEqual(d["duration_sec"], 2.0)               # from last segment
        self.assertEqual(d["segments"][0]["words"][0]["prob"], 0.9)  # score -> prob
        WhisperDoc.from_dict(d)


class TestTranscribeFile(unittest.TestCase):
    def test_request_and_normalize(self):
        post = _fake_post(OPENAI_RESP)
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "mic.16k.wav"
            wav.write_bytes(b"RIFFfake")
            cfg = TranscribeConfig(url="http://w/transcriptions", model="large-v3")
            doc = transcribe_file(wav, cfg, post=post)
        self.assertEqual(doc["segments"][0]["text"], "Hello there")
        fields = dict((k, v) for k, v in post.calls[0]["fields"])
        self.assertEqual(fields["model"], "large-v3")
        self.assertEqual(fields["response_format"], "verbose_json")
        # word granularity requested for openai
        names = [k for k, _ in post.calls[0]["fields"]]
        self.assertIn("timestamp_granularities[]", names)

    def test_transcribe_meeting_writes_both(self):
        post = _fake_post(OPENAI_RESP)
        with tempfile.TemporaryDirectory() as td:
            proc = Path(td) / "processed"
            proc.mkdir()
            (proc / "mic.16k.wav").write_bytes(b"x")
            (proc / "line.16k.wav").write_bytes(b"x")
            tx = Path(td) / "transcripts"
            out = transcribe_meeting(proc, tx, TranscribeConfig(url="http://w"), post=post)
            self.assertTrue(out["mic"].exists() and out["line"].exists())
            WhisperDoc.from_dict(json.loads(out["line"].read_text()))


class TestDiarize(unittest.TestCase):
    DIAR = {"model": "pyannote/community-1", "duration_sec": 5.0, "num_speakers": 2,
            "segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
                         {"speaker": "SPEAKER_01", "start": 2.0, "end": 4.0}]}

    def test_diarize_file_and_merge_compat(self):
        post = _fake_post(self.DIAR)
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "line.16k.wav"
            wav.write_bytes(b"x")
            resp = diarize_file(wav, DiarizeConfig(url="http://d", max_speakers=3), post=post)
        self.assertEqual(resp["num_speakers"], 2)
        DiarDoc.from_dict(resp)
        fields = dict((k, v) for k, v in post.calls[0]["fields"])
        self.assertEqual(fields["max_speakers"], "3")

    def test_diarize_missing_segments_raises(self):
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


if __name__ == "__main__":
    unittest.main()
