import json
import tempfile
import types
import unittest
from pathlib import Path

from briefly.enrich import EnrichConfig, EnrichError, build_command, enrich_meeting


def _proc(returncode=0, stdout="{}", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class TestBuildCommand(unittest.TestCase):
    def test_command_flags_and_no_bash(self):
        cfg = EnrichConfig(vault_dir="/v", claude_path="claude")
        cmd = build_command(Path("/v/20-Meetings/x.md"), cfg)
        self.assertIn("-p", cmd)
        self.assertIn("/enrich-meeting /v/20-Meetings/x.md", cmd)
        i = cmd.index("--allowedTools")
        self.assertNotIn("Bash", cmd[i + 1])          # Bash denied by design
        self.assertIn("--add-dir", cmd)
        self.assertEqual(cmd[cmd.index("--permission-mode") + 1], "acceptEdits")
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "json")


class TestEnrichMeeting(unittest.TestCase):
    def test_success_parses_json_and_uses_vault_cwd(self):
        seen = {}

        def runner(cmd, cwd, timeout):
            seen["cwd"] = cwd
            return _proc(0, json.dumps({"total_cost_usd": 0.12, "result": "ok"}))

        with tempfile.TemporaryDirectory() as td:
            notes = Path(td) / "note.md"
            notes.write_text("# note")
            res = enrich_meeting(notes, EnrichConfig(vault_dir=td), runner=runner)
        self.assertEqual(res["total_cost_usd"], 0.12)
        self.assertEqual(seen["cwd"], td)

    def test_nonzero_exit_raises(self):
        with tempfile.TemporaryDirectory() as td:
            notes = Path(td) / "note.md"
            notes.write_text("# note")
            with self.assertRaises(EnrichError):
                enrich_meeting(notes, EnrichConfig(vault_dir=td),
                               runner=lambda c, w, t: _proc(1, "", "boom"))

    def test_missing_notes_raises(self):
        with self.assertRaises(EnrichError):
            enrich_meeting("/nope/note.md", EnrichConfig(), runner=lambda c, w, t: _proc(0))


if __name__ == "__main__":
    unittest.main()
