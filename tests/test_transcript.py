import tempfile
import unittest
from pathlib import Path

from briefly.models import Speaker, SpeakersMap, Transcript, Turn


def _transcript() -> Transcript:
    return Transcript(
        meeting_id="01J9ZC8Q9F7Y3K2N5R6T8W0X1Z",
        date="2026-06-14",
        generated_at="2026-06-14T10:05:00Z",
        partial=False,
        models={"transcription": "whisper-large-v3", "diarization": "pyannote/community-1"},
        speakers=[
            Speaker(id="me", label="Me", channel="mic", source="channel", name="Paul Nathan"),
            Speaker(id="s1", label="Speaker_1", channel="line", source="diarization", name="Jane Doe"),
            Speaker(id="s2", label="Speaker_2", channel="line", source="diarization"),
        ],
        turns=[
            Turn(0, "me", "Paul Nathan", "mic", 11.2, 13.0, "Let me walk through the phasing.",
                 confidence=0.94),
            Turn(1, "s1", "Jane Doe", "line", 73.4, 76.1, "Can we keep it read-only?",
                 confidence=0.91, diarization_confidence=0.86),
        ],
        warnings=["1 line segment had no overlapping diarization turn"],
    )


class TestTranscript(unittest.TestCase):
    def test_to_dict_order_and_nulls(self):
        d = _transcript().to_dict()
        self.assertEqual(list(d)[:5], ["schema_version", "meeting_id", "date", "generated_at", "partial"])
        # unmapped speaker keeps an explicit null name
        self.assertIsNone(d["speakers"][2]["name"])

    def test_round_trip(self):
        t = _transcript()
        self.assertEqual(Transcript.from_dict(t.to_dict()).to_dict(), t.to_dict())

    def test_to_text(self):
        txt = _transcript().to_text()
        self.assertIn("[00:00:11] Paul Nathan: Let me walk through the phasing.", txt)
        self.assertIn("[00:01:13] Jane Doe: Can we keep it read-only?", txt)

    def test_write_read(self):
        t = _transcript()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "transcript.json"
            t.write(p)
            self.assertEqual(Transcript.read(p).to_dict(), t.to_dict())


class TestSpeakersMap(unittest.TestCase):
    def test_round_trip(self):
        sm = SpeakersMap("mid", {"Me": "Paul", "Speaker_1": "Jane"},
                         [{"start": 1.0, "end": 2.0, "to": "Speaker_2"}])
        self.assertEqual(SpeakersMap.from_dict(sm.to_dict()).to_dict(), sm.to_dict())

    def test_defaults(self):
        sm = SpeakersMap.from_dict({"meeting_id": "x"})
        self.assertEqual(sm.map, {})
        self.assertEqual(sm.corrections, [])


if __name__ == "__main__":
    unittest.main()
