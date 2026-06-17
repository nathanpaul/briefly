"""Tests for the prompt-driven `briefly summarize "<prompt>"` command (summarize_agent)."""
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from briefly import cli
from briefly.summarize_agent import (
    SummarizeAgentConfig,
    SummarizeAgentError,
    _event_activity_line,
    build_command,
    build_prompt,
    resolve_meeting_id,
    summarize_agent,
)

MID = "01KV674CSRY4A0SR318AH4B8CS"


class TestSummarizeAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rec = self.root / "recordings" / MID
        tx = self.root / "transcripts" / MID
        rec.mkdir(parents=True)
        tx.mkdir(parents=True)
        (rec / "meeting.json").write_text(json.dumps(
            {"meeting_id": MID, "date": "2026-06-15", "attendees": ["Paul", "Supervisor"]}))
        (tx / "transcript.txt").write_text("[00:00:00] Me: Hi.\n[00:00:01] Speaker_1: Hello.\n")
        (self.root / "recordings" / ".last-meeting-id").write_text(MID)
        self.vault = self.root / "vault"
        (self.vault / "20-Meetings").mkdir(parents=True)
        self.cfg = SummarizeAgentConfig(vault_dir=str(self.vault), data_root=str(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def test_resolve_explicit_and_last(self):
        self.assertEqual(resolve_meeting_id(self.cfg, "01XYZ"), "01XYZ")
        self.assertEqual(resolve_meeting_id(self.cfg, None), MID)  # falls back to .last-meeting-id

    def test_resolve_no_meeting_errors(self):
        cfg = SummarizeAgentConfig(vault_dir=str(self.vault), data_root=str(self.root / "empty"))
        with self.assertRaises(SummarizeAgentError):
            resolve_meeting_id(cfg, None)

    def test_missing_transcript_errors(self):
        (self.root / "transcripts" / MID / "transcript.txt").unlink()
        with self.assertRaises(SummarizeAgentError):
            summarize_agent("do X", self.cfg, meeting_id=MID, runner=lambda *a: None)

    def test_empty_prompt_errors(self):
        with self.assertRaises(SummarizeAgentError):
            summarize_agent("   ", self.cfg, meeting_id=MID, runner=lambda *a: None)

    def test_prompt_contains_instruction_and_transcript(self):
        prompt = build_prompt("EXTRACT ACTIONS", MID, "2026-06-15", ["Paul"],
                              "2026-06-15-x.md", "Me: hi\nSpeaker_1: yo")
        self.assertIn("EXTRACT ACTIONS", prompt)
        self.assertIn("Speaker_1: yo", prompt)
        self.assertIn("2026-06-15", prompt)
        self.assertIn("do NOT run shell", prompt)

    def test_default_prompt_targets_vault_root_single_page(self):
        prompt = build_prompt("SUMMARIZE", MID, "2026-06-15", [], "2026-06-15-x.md", "T", enrich=False)
        self.assertIn("vault root", prompt)
        self.assertIn("2026-06-15-x.md", prompt)
        self.assertIn("do not create or edit any other files", prompt.lower())

    def test_enrich_prompt_frames_as_cross_vault(self):
        prompt = build_prompt("BASE\n\nPUT BLOCKERS IN 30-Issues/", MID, "2026-06-15", [],
                              "2026-06-15-x.md", "T", enrich=True)
        self.assertIn("PUT BLOCKERS IN 30-Issues/", prompt)       # enrichment text present
        self.assertIn("across the vault", prompt)                 # framed as multi-note enrichment

    def test_dry_run_composes_command_no_invoke(self):
        called = []
        res = summarize_agent("do X", self.cfg, meeting_id=MID,
                              runner=lambda *a: called.append(a), dry_run=True)
        self.assertEqual(called, [])                     # claude NOT invoked
        self.assertEqual(res["meeting_id"], MID)
        self.assertEqual(res["note"], f"2026-06-15-{MID}.md")    # single page at the vault root
        cmd = res["command"]
        self.assertIn("--add-dir", cmd)
        self.assertIn(str(self.vault), cmd)
        # Bash must never be in the allowed tools (40-Personal OS guard).
        tools = cmd[cmd.index("--allowedTools") + 1]
        self.assertNotIn("Bash", tools)

    def test_runner_invoked_and_result_parsed(self):
        seen = {}

        def fake_runner(cmd, cwd, timeout):
            seen["cmd"] = cmd
            seen["cwd"] = cwd

            class P:
                returncode = 0
                stdout = json.dumps({"total_cost_usd": 0.12, "result": "ok"})
                stderr = ""
            return P()

        out = summarize_agent("link people to MOCs", self.cfg, meeting_id=MID, runner=fake_runner)
        self.assertEqual(out["total_cost_usd"], 0.12)
        self.assertEqual(out["meeting_id"], MID)
        self.assertEqual(seen["cwd"], str(self.vault))
        # the user's instruction + transcript are in the -p prompt
        prompt = seen["cmd"][seen["cmd"].index("-p") + 1]
        self.assertIn("link people to MOCs", prompt)
        self.assertIn("Speaker_1: Hello.", prompt)
        self.assertEqual(seen["cmd"][0], "claude")

    def test_build_command_uses_streaming(self):
        cmd = build_command("PROMPT", SummarizeAgentConfig(vault_dir="/v"))
        self.assertIn("stream-json", cmd)
        self.assertIn("--verbose", cmd)

    def test_event_activity_line_surfaces_tool_use(self):
        ev = {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "name": "Write", "input": {"file_path": "/vault/meetings/x.md"}}]}}
        self.assertEqual(_event_activity_line(ev, "/vault"), "  → Write meetings/x.md")

    def test_event_activity_line_ignores_non_tools(self):
        self.assertIsNone(_event_activity_line({"type": "result", "result": "done"}, "/vault"))
        self.assertIsNone(_event_activity_line(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}, "/vault"))

    def test_no_result_event_is_an_error(self):
        # claude exits 0 but streamed no result → must NOT be reported as success
        def empty_runner(cmd, cwd, timeout):
            class P:
                returncode = 0
                stdout = ""
                stderr = ""
            return P()
        with self.assertRaises(SummarizeAgentError):
            summarize_agent("x", self.cfg, meeting_id=MID, runner=empty_runner)

    def test_runner_failure_raises(self):
        def fail_runner(cmd, cwd, timeout):
            class P:
                returncode = 2
                stdout = ""
                stderr = "boom"
            return P()

        with self.assertRaises(SummarizeAgentError):
            summarize_agent("x", self.cfg, meeting_id=MID, runner=fail_runner)

    def test_cli_routes_prompt_to_agent(self):
        # `briefly summarize "<prompt>" --dry-run …` reaches the agent and returns 0.
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(["summarize", "extract action items", "--dry-run",
                           "--meeting-id", MID, "--data-root", str(self.root),
                           "--vault-dir", str(self.vault)])
        self.assertEqual(rc, 0)
        out = json.loads(buf.getvalue())
        self.assertTrue(out["dry_run"])
        self.assertEqual(out["meeting_id"], MID)


if __name__ == "__main__":
    unittest.main()
