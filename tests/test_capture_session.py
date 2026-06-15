import json
import struct
import subprocess
import tempfile
import time
import unittest
import wave
from pathlib import Path

import briefly.audio.capture as cap
from briefly.config import CaptureConfig

FFMPEG = CaptureConfig().ffmpeg_path
MID = "01J9ZC8Q9F7Y3K2N5R6T8W0X1Z"


def _write_wav(path: Path, seconds: float = 0.3, rate: int = 48000, nch: int = 2) -> None:
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<%dh" % (n * nch), *([1500] * (n * nch))))


class TestFinalize(unittest.TestCase):
    def test_writes_manifest_and_renames(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "recordings"
            mdir = root / MID
            mdir.mkdir(parents=True)
            _write_wav(mdir / "mic.wav.part")
            _write_wav(mdir / "line.wav.part")
            cfg = CaptureConfig(recordings_dir=str(root))
            m = cap._finalize(cfg, MID, mdir, "2026-06-15T00:00:00Z", ["Jane"],
                              t_mic=100.0, t_line=100.02)
            self.assertTrue((mdir / "meeting.json").exists())
            self.assertEqual((mdir.parent / ".last-meeting-id").read_text().strip(), MID)
            self.assertFalse((mdir / "mic.wav.part").exists())   # renamed to mic.wav
            self.assertEqual(m.channels["mic"].speaker, "Me")
            self.assertEqual(m.channels["line"].start_offset_sec, 0.02)
            self.assertIsNotNone(m.channels["mic"].duration_sec)
            self.assertFalse(m.partial)

    def test_partial_when_a_channel_missing(self):
        with tempfile.TemporaryDirectory() as td:
            mdir = Path(td) / "r" / MID
            mdir.mkdir(parents=True)
            _write_wav(mdir / "mic.wav.part")   # line missing
            cfg = CaptureConfig(recordings_dir=str(Path(td) / "r"))
            m = cap._finalize(cfg, MID, mdir, "2026-06-15T00:00:00Z", [], 1.0, 1.0)
            self.assertTrue(m.partial)


class TestFindActiveSession(unittest.TestCase):
    @staticmethod
    def _cfg(root):
        return CaptureConfig(recordings_dir=str(root))

    def test_single_active(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "A"
            d.mkdir()
            (d / cap._STATE).write_text("{}")
            self.assertEqual(cap._find_active_session(self._cfg(td), None).name, "A")

    def test_finalized_is_not_active(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "A"
            d.mkdir()
            (d / cap._STATE).write_text("{}")
            (d / "meeting.json").write_text("{}")
            with self.assertRaises(cap.NoActiveSessionError):
                cap._find_active_session(self._cfg(td), None)

    def test_ambiguous_requires_id(self):
        with tempfile.TemporaryDirectory() as td:
            for nm in ("A", "B"):
                d = Path(td) / nm
                d.mkdir()
                (d / cap._STATE).write_text("{}")
            with self.assertRaises(cap.AmbiguousSessionError):
                cap._find_active_session(self._cfg(td), None)

    def test_unknown_id(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(cap.NoActiveSessionError):
                cap._find_active_session(self._cfg(td), "nope")


class TestStartStopIntegration(unittest.TestCase):
    """Exercise the detached-record + stop/finalize path with real ffmpeg lavfi sources
    (no soundcard), simulating what start() sets up."""

    def test_stop_signals_finalizes_and_clears_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "recordings"
            mdir = root / MID
            mdir.mkdir(parents=True)
            procs = []
            for ch in ("mic", "line"):
                cmd = [FFMPEG, "-hide_banner", "-y", "-f", "lavfi", "-i",
                       "sine=frequency=440:sample_rate=48000", "-ac", "2", "-c:a",
                       "pcm_s16le", "-f", "wav", str(mdir / f"{ch}.wav.part")]
                procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL, start_new_session=True))
            time.sleep(1.2)  # accumulate ~1.2 s of audio
            (mdir / cap._STATE).write_text(json.dumps({
                "meeting_id": MID, "started_at": "2026-06-15T00:00:00Z", "attendees": ["Jane"],
                "mic_device": "MIC", "line_device": "LINE",
                "mic_pid": procs[0].pid, "line_pid": procs[1].pid, "t_mic": 100.0, "t_line": 100.02,
            }))
            cfg = CaptureConfig(recordings_dir=str(root))
            manifest, _ = cap.stop(cfg, meeting_id=MID)
            self.assertTrue((mdir / "meeting.json").exists())
            self.assertFalse(manifest.partial)
            self.assertGreater(manifest.channels["mic"].duration_sec, 0.5)
            self.assertFalse((mdir / cap._STATE).exists())   # state cleared after finalize


if __name__ == "__main__":
    unittest.main()
