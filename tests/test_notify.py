"""Tests for the opt-in completion notification (bell / desktop)."""
import io
import unittest

from briefly.notify import notify, resolve_mode


class TestResolveMode(unittest.TestCase):
    def test_off_by_default(self):
        self.assertEqual(resolve_mode(cli=None, env=""), "off")
        for v in ("off", "none", "0", "false", "  "):
            self.assertEqual(resolve_mode(cli=None, env=v), "off")

    def test_bell_and_desktop(self):
        for v in ("bell", "1", "on", "true", "yes"):
            self.assertEqual(resolve_mode(env=v), "bell")
        self.assertEqual(resolve_mode(env="desktop"), "desktop")
        self.assertEqual(resolve_mode(env="DeskTop"), "desktop")

    def test_cli_overrides_env(self):
        self.assertEqual(resolve_mode(cli="bell", env="off"), "bell")
        self.assertEqual(resolve_mode(cli="off", env="desktop"), "off")
        self.assertEqual(resolve_mode(cli="desktop", env=""), "desktop")


class TestNotify(unittest.TestCase):
    def test_off_is_silent(self):
        out, calls = io.StringIO(), []
        notify("t", "m", mode="off", runner=lambda *a, **k: calls.append(a), out=out)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(calls, [])

    def test_bell_rings_without_desktop(self):
        out, calls = io.StringIO(), []
        notify("t", "m", mode="bell", runner=lambda *a, **k: calls.append(a), out=out)
        self.assertIn("\a", out.getvalue())
        self.assertEqual(calls, [])

    def test_desktop_rings_and_invokes_osascript(self):
        out, calls = io.StringIO(), []
        notify("Title", 'msg "x"', mode="desktop",
               runner=lambda *a, **k: calls.append(a[0]), out=out)
        self.assertIn("\a", out.getvalue())
        self.assertTrue(calls and calls[0][0] == "osascript")

    def test_desktop_swallows_runner_errors(self):
        def boom(*a, **k):
            raise OSError("no osascript")
        # must not raise
        notify("t", "m", mode="desktop", runner=boom, out=io.StringIO())


if __name__ == "__main__":
    unittest.main()
