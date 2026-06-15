import tempfile
import unittest
from pathlib import Path

from briefly.models import CaptureInfo, ChannelInfo, MeetingManifest
from briefly.orchestrator import PipelineConfig
from briefly.watch import find_pending, watch_once

MID1 = "01J9ZC8Q9F7Y3K2N5R6T8W0X1A"
MID2 = "01J9ZC8Q9F7Y3K2N5R6T8W0X1B"
MID3 = "01J9ZC8Q9F7Y3K2N5R6T8W0X1C"


def _meeting(rec: Path) -> None:
    rec.mkdir(parents=True, exist_ok=True)
    MeetingManifest(
        meeting_id=rec.name, date="2026-06-14", started_at="2026-06-14T09:00:00Z",
        ended_at="2026-06-14T09:05:00Z", partial=False, attendees=[],
        capture=CaptureInfo("dual-process", 48000, "pcm_s16le", 2, "8.1.1", "process-start-delta"),
        channels={"mic": ChannelInfo("mic.wav", "MIC", 0.0, speaker="Me"),
                  "line": ChannelInfo("line.wav", "LINE", 0.0)},
    ).write(rec / "meeting.json")


class TestWatch(unittest.TestCase):
    def test_find_pending_only_finalized_and_unprocessed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = PipelineConfig(data_root=str(root))
            _meeting(root / "recordings" / MID1)               # finalized -> pending
            (root / "recordings" / MID2).mkdir(parents=True)   # no meeting.json (recording) -> skip
            _meeting(root / "recordings" / MID3)               # finalized but already merged -> skip
            (root / "transcripts" / MID3).mkdir(parents=True)
            (root / "transcripts" / MID3 / "transcript.json").write_text("{}")
            pending = find_pending(cfg, "merge", set())
            self.assertEqual(pending, [MID1])

    def test_watch_once_runs_then_skips(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = PipelineConfig(data_root=str(root))
            _meeting(root / "recordings" / MID1)
            calls = []

            def fake_run(c, mid, frm, to, force):
                calls.append((mid, frm, to, force))
                d = Path(c.data_root) / "transcripts" / mid
                d.mkdir(parents=True, exist_ok=True)
                (d / "transcript.json").write_text("{}")   # mark merge done

            self.assertEqual(watch_once(cfg, "merge", run=fake_run, log=lambda *a: None),
                             [(MID1, "ok")])
            self.assertEqual(calls, [(MID1, "preprocess", "merge", False)])
            # nothing pending on the second pass
            self.assertEqual(watch_once(cfg, "merge", run=fake_run, log=lambda *a: None), [])

    def test_watch_once_failure_is_ledgered(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = PipelineConfig(data_root=str(root))
            _meeting(root / "recordings" / MID1)
            ledger = set()

            def boom(*a):
                raise RuntimeError("pipeline blew up")

            self.assertEqual(watch_once(cfg, "merge", run=boom, ledger=ledger, log=lambda *a: None),
                             [(MID1, "error")])
            self.assertIn(MID1, ledger)
            # ledgered meetings are not retried within the same run
            self.assertEqual(watch_once(cfg, "merge", run=boom, ledger=ledger, log=lambda *a: None),
                             [])


if __name__ == "__main__":
    unittest.main()
