import tempfile
import unittest
from pathlib import Path

import briefly.audio.preprocess as pp
from briefly.audio.preprocess import PreprocessConfig, preprocess
from briefly.models import CaptureInfo, ChannelInfo, MeetingManifest

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

MID = "01J9ZC8Q9F7Y3K2N5R6T8W0X1Z"
SR = 48000


def _meeting(rec: Path) -> None:
    MeetingManifest(
        meeting_id=MID, date="2026-06-14", started_at="2026-06-14T09:00:00Z",
        ended_at="2026-06-14T09:01:00Z", partial=False, attendees=[],
        capture=CaptureInfo("dual-process", SR, "pcm_s16le", 1, "8.1.1", "process-start-delta"),
        channels={"mic": ChannelInfo("mic.wav", "Cubilux CB5 MIC2", 0.0, speaker="Me"),
                  "line": ChannelInfo("line.wav", "Cubilux CB5 Line In", 0.0)},
    ).write(rec / "meeting.json")


@unittest.skipUnless(HAVE_NUMPY, "numpy ([aec] extra) not installed")
class TestCancelEcho(unittest.TestCase):
    def test_pure_echo_is_strongly_reduced(self):
        from briefly.audio.aec import cancel_echo
        rng = np.random.default_rng(0)
        ref = (rng.standard_normal(SR) * 0.3).astype(np.float32)
        mic = 0.5 * ref                                   # pure echo, no near-end
        out, erle = cancel_echo(mic, ref)
        self.assertGreater(erle, 10.0)
        self.assertLess(float(out @ out), 0.2 * float(mic @ mic))

    def test_near_end_speech_preserved(self):
        from briefly.audio.aec import cancel_echo
        rng = np.random.default_rng(1)
        ref = (rng.standard_normal(SR) * 0.3).astype(np.float32)   # far-end (line)
        near = (rng.standard_normal(SR) * 0.3).astype(np.float32)  # your voice (proxy)
        out, _ = cancel_echo(near + 0.5 * ref, ref)
        self.assertGreater(float(np.corrcoef(out, near)[0, 1]), 0.6)  # voice survives


@unittest.skipUnless(HAVE_NUMPY, "numpy ([aec] extra) not installed")
class TestRunAecFile(unittest.TestCase):
    def test_aligns_cancels_and_writes_native_mono(self):
        from briefly.audio import aec
        rng = np.random.default_rng(2)
        ref = (rng.standard_normal(SR) * 0.3).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            aec._write_wav_mono(td / "line.wav", ref, SR)
            aec._write_wav_mono(td / "mic.wav", 0.5 * ref, SR)
            info = aec.run_aec_file(td / "mic.wav", td / "line.wav", 0.0, td / "out.wav")
            self.assertTrue(info["applied"])
            self.assertEqual(info["backend"], "wiener-numpy")
            self.assertGreater(info["reduction_db"], 10.0)
            samp, rate = aec._read_wav_mono(td / "out.wav")
            self.assertEqual(rate, SR)
            self.assertEqual(samp.ndim, 1)


@unittest.skipUnless(HAVE_NUMPY, "numpy ([aec] extra) not installed")
class TestPreprocessRealAEC(unittest.TestCase):
    def test_aec_runs_and_reduces_echo(self):
        from briefly.audio.aec import _write_wav_mono
        rng = np.random.default_rng(3)
        ref = (rng.standard_normal(SR) * 0.3).astype(np.float32)
        mic = (rng.standard_normal(SR) * 0.05).astype(np.float32) + 0.5 * ref  # echo-dominant
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rec = root / "recordings" / MID
            rec.mkdir(parents=True)
            _write_wav_mono(rec / "mic.wav", mic, SR)
            _write_wav_mono(rec / "line.wav", ref, SR)
            _meeting(rec)
            res = preprocess(MID, rec, root / "processed", PreprocessConfig(aec_enabled=True))
            self.assertTrue(res.report["aec_enabled"])
            self.assertTrue(res.report["channels"]["mic"]["aec_applied"])
            self.assertIsNotNone(res.report["estimated_echo_reduction_db"])
            self.assertGreater(res.report["estimated_echo_reduction_db"], 3.0)

    def test_fallback_when_backend_unavailable(self):
        from briefly.audio.aec import _write_wav_mono
        rng = np.random.default_rng(4)
        ref = (rng.standard_normal(SR) * 0.3).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rec = root / "recordings" / MID
            rec.mkdir(parents=True)
            _write_wav_mono(rec / "mic.wav", 0.5 * ref, SR)
            _write_wav_mono(rec / "line.wav", ref, SR)
            _meeting(rec)
            orig = pp._aec_backend_available
            pp._aec_backend_available = lambda: (False, None)  # simulate numpy absent
            try:
                res = preprocess(MID, rec, root / "processed",
                                 PreprocessConfig(aec_enabled=True))
            finally:
                pp._aec_backend_available = orig
            self.assertFalse(res.report["aec_enabled"])
            self.assertFalse(res.report["channels"]["mic"]["aec_applied"])
            self.assertTrue(any("numpy" in w.lower() or "backend" in w.lower()
                                for w in res.report["warnings"]))


if __name__ == "__main__":
    unittest.main()
