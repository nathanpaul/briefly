"""Tests for the foreground capture loop: elapsed notices + Ctrl-C = clean stop."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from briefly.audio import capture as cap
from briefly.config import CaptureConfig


class TestMonitor(unittest.TestCase):
    def test_emits_a_notice_each_interval(self):
        logs, state = [], {"n": 0}
        cap._monitor(lambda: state["n"] < 3, notify_interval=30, log=logs.append,
                     clock=lambda: state["n"] * 30, sleep=lambda _: state.__setitem__("n", state["n"] + 1))
        self.assertEqual(len(logs), 2)                    # notices at elapsed 30 and 60
        self.assertIn("00:00:30", logs[0])
        self.assertIn("00:01:00", logs[1])

    def test_no_notice_when_already_stopped(self):
        logs = []
        reason = cap._monitor(lambda: False, log=logs.append, clock=lambda: 0.0, sleep=lambda _: None)
        self.assertEqual(reason, "ended")
        self.assertEqual(logs, [])

    def test_interval_zero_is_silent(self):
        logs, state = [], {"n": 0}
        cap._monitor(lambda: state["n"] < 3, notify_interval=0, log=logs.append,
                     clock=lambda: state["n"] * 100, sleep=lambda _: state.__setitem__("n", state["n"] + 1))
        self.assertEqual(logs, [])

    def test_keyboard_interrupt_propagates(self):
        def boom(_):
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            cap._monitor(lambda: True, log=lambda _: None, clock=lambda: 0.0, sleep=boom)


class TestStartForeground(unittest.TestCase):
    def _mdir_with_state(self, td):
        mdir = Path(td) / "meeting_0001"
        mdir.mkdir()
        (mdir / cap._STATE).write_text(json.dumps({"mic_pid": 1, "line_pid": 2}))
        return mdir

    def test_ctrl_c_finalizes_via_stop(self):
        with TemporaryDirectory() as td:
            mdir = self._mdir_with_state(td)
            with mock.patch.object(cap, "start", return_value=("meeting_0001", mdir)), \
                 mock.patch.object(cap, "_monitor", side_effect=KeyboardInterrupt), \
                 mock.patch.object(cap, "stop", return_value=("MANIFEST", mdir)) as mstop:
                out = cap.start_foreground(CaptureConfig(), attendees=["A"], log=lambda *a: None)
            self.assertEqual(out, ("MANIFEST", mdir))
            mstop.assert_called_once()

    def test_recorder_exit_still_finalizes(self):
        with TemporaryDirectory() as td:
            mdir = self._mdir_with_state(td)
            warns = []
            with mock.patch.object(cap, "start", return_value=("meeting_0001", mdir)), \
                 mock.patch.object(cap, "_monitor", return_value="ended"), \
                 mock.patch.object(cap, "stop", return_value=("MANIFEST", mdir)) as mstop:
                cap.start_foreground(CaptureConfig(), log=warns.append)
            mstop.assert_called_once()
            self.assertTrue(any("exited early" in w for w in warns))


if __name__ == "__main__":
    unittest.main()
