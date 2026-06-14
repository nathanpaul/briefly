import json
import tempfile
import unittest
from pathlib import Path

from briefly.models import CaptureInfo, ChannelInfo, MeetingManifest


def _manifest() -> MeetingManifest:
    return MeetingManifest(
        meeting_id="01J9ZC8Q9F7Y3K2N5R6T8W0X1Z",
        date="2026-06-14",
        started_at="2026-06-14T09:00:03Z",
        ended_at="2026-06-14T09:52:10Z",
        partial=False,
        attendees=["Jane Doe", "John Smith"],
        capture=CaptureInfo("dual-process", 48000, "pcm_s16le", 2, "8.1.1", "process-start-delta"),
        channels={
            "mic": ChannelInfo("mic.wav", "Cubilux CB5 MIC2", 0.0, speaker="Me",
                               duration_sec=3127.0, peak_dbfs=-1.3, mean_dbfs=-29.6, clipping=False),
            "line": ChannelInfo("line.wav", "Cubilux CB5 Line In", 0.021,
                                duration_sec=3127.0, peak_dbfs=-2.4, mean_dbfs=-21.8, clipping=False),
        },
    )


class TestManifest(unittest.TestCase):
    def test_to_dict_shape(self):
        d = _manifest().to_dict()
        self.assertEqual(list(d)[0], "schema_version")
        self.assertEqual(d["schema_version"], "1.0")
        # mic carries a fixed speaker; line is diarized later so has none.
        self.assertEqual(d["channels"]["mic"]["speaker"], "Me")
        self.assertNotIn("speaker", d["channels"]["line"])
        # None-valued optionals (e.g. device_uid) are dropped.
        self.assertNotIn("device_uid", d["channels"]["mic"])

    def test_offset_convention(self):
        d = _manifest().to_dict()
        self.assertEqual(d["channels"]["mic"]["start_offset_sec"], 0.0)
        self.assertEqual(d["channels"]["line"]["start_offset_sec"], 0.021)

    def test_round_trip(self):
        m = _manifest()
        again = MeetingManifest.from_dict(m.to_dict())
        self.assertEqual(again.to_dict(), m.to_dict())

    def test_write_read_file(self):
        m = _manifest()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "meeting.json"
            m.write(p)
            self.assertEqual(json.loads(p.read_text())["meeting_id"], m.meeting_id)
            self.assertEqual(MeetingManifest.read(p).to_dict(), m.to_dict())


if __name__ == "__main__":
    unittest.main()
