"""Prompt-driven meeting enrichment — `briefly summarize "<instruction>" [--meeting-id <id>]`.

Unlike the structured `summarize` stage (a fixed per-person brief) and the fixed
`/enrich-meeting` skill, this runs headless Claude Code with a USER-PROVIDED instruction over
one meeting's transcript, writing/enriching the Obsidian vault however you ask for THAT meeting
— "extract action items and open tasks in 30-Tasks", "one-paragraph exec summary + link each
person to their MOC", "pull decisions into the project note", etc.

Routing: `briefly summarize "<prompt>"` (a leading non-flag arg) lands here; `briefly summarize
--meeting-id <id>` (no prompt) stays the structured `summarize` stage (see cli.py).

Safety mirrors `enrich`: the vault is added with `--add-dir`, tools are limited to
Read,Glob,Grep,Edit,Write (NO Bash — the 40-Personal OS guard must not be bypassable), and
`--permission-mode acceptEdits`. Claude Code runs with cwd = the vault root so the vault's
CLAUDE.md + templates + skills load.

PRIVACY: the transcript TEXT is sent to Claude (cloud, via Claude Code), same as the existing
summarize/enrich stages. Raw audio never leaves the device.

Exit 0 on success; non-zero (with a message on stderr) on a missing meeting/transcript or a
Claude failure. The subprocess runner is injectable so tests need no `claude` binary.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


class SummarizeAgentError(Exception):
    exit_code = 1


@dataclass
class SummarizeAgentConfig:
    vault_dir: str = "vault"
    data_root: str = "."
    transcripts_dir: str | None = None       # default: <data_root>/transcripts
    recordings_dir: str | None = None        # default: <data_root>/recordings
    meetings_subdir: str = "20-Meetings"
    claude_path: str = "claude"
    model: str | None = None
    allowed_tools: str = "Read,Glob,Grep,Edit,Write"   # NO Bash, by design (matches enrich)
    permission_mode: str = "acceptEdits"
    max_budget_usd: float | None = 1.00
    timeout_sec: float = 1800

    def tx_dir(self) -> Path:
        return Path(self.transcripts_dir or (Path(self.data_root) / "transcripts"))

    def rec_dir(self) -> Path:
        return Path(self.recordings_dir or (Path(self.data_root) / "recordings"))


def resolve_meeting_id(cfg: SummarizeAgentConfig, meeting_id: str | None) -> str:
    """Use the given meeting_id, else the last captured meeting (recordings/.last-meeting-id)."""
    if meeting_id:
        return meeting_id
    from .state import read_last_meeting

    mid = read_last_meeting(cfg.rec_dir())
    if not mid:
        raise SummarizeAgentError(
            "no --meeting-id given and no last captured meeting found "
            f"(looked in {cfg.rec_dir()}/.last-meeting-id) — capture a meeting or pass --meeting-id")
    return mid


def _read_meeting_meta(cfg: SummarizeAgentConfig, mid: str) -> tuple[str, list[str]]:
    """(date, attendees) from recordings/<id>/meeting.json — best-effort (falls back gracefully)."""
    mf = cfg.rec_dir() / mid / "meeting.json"
    if mf.exists():
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            return str(m.get("date", "0000-00-00")), list(m.get("attendees") or [])
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return "0000-00-00", []


def _read_transcript(cfg: SummarizeAgentConfig, mid: str) -> str:
    """The merged, speaker-attributed transcript.txt — required (run `briefly run` first)."""
    tp = cfg.tx_dir() / mid / "transcript.txt"
    if not tp.exists():
        raise SummarizeAgentError(
            f"transcript not found: {tp} — run `briefly run --meeting-id {mid} --to merge` first")
    text = tp.read_text(encoding="utf-8").strip()
    if not text:
        raise SummarizeAgentError(f"transcript is empty: {tp}")
    return text


def note_rel_path(cfg: SummarizeAgentConfig, mid: str, date: str) -> str:
    """Vault-relative meeting-note path: <meetings_subdir>/<date>-<id>.md (matches summarize)."""
    return f"{cfg.meetings_subdir}/{date}-{mid}.md"


def build_prompt(user_instruction: str, mid: str, date: str, attendees: list[str],
                 note_rel: str, transcript_text: str) -> str:
    """Compose the Claude Code prompt: meeting context + the user's instruction + transcript."""
    who = ", ".join(attendees) if attendees else "(unknown)"
    return (
        "You are enriching an Obsidian vault after a meeting. Your working directory IS the vault "
        "root. Use only Read/Glob/Grep/Edit/Write — do NOT run shell commands.\n\n"
        f"Meeting id: {mid}\n"
        f"Date: {date}\n"
        f"Attendees: {who}\n"
        f"Target meeting note (create if missing, else update in place): {note_rel}\n\n"
        "Follow THIS instruction for how to summarize / enrich this meeting into the vault "
        "(it is specific to this meeting):\n"
        "--- INSTRUCTION ---\n"
        f"{user_instruction}\n"
        "--- END INSTRUCTION ---\n\n"
        "Make the requested edits to the meeting note and any related vault notes the instruction "
        "calls for, following the vault's existing conventions/templates. Preserve any existing "
        "content you are not asked to change.\n\n"
        "Speaker-attributed transcript of the meeting:\n"
        "--- TRANSCRIPT ---\n"
        f"{transcript_text}\n"
        "--- END TRANSCRIPT ---\n"
    )


def build_command(prompt: str, cfg: SummarizeAgentConfig) -> list[str]:
    cmd = [
        cfg.claude_path, "-p", prompt,
        "--add-dir", str(cfg.vault_dir),
        "--allowedTools", cfg.allowed_tools,
        "--permission-mode", cfg.permission_mode,
        "--output-format", "json",
    ]
    if cfg.model:
        cmd += ["--model", cfg.model]
    if cfg.max_budget_usd is not None:
        cmd += ["--max-budget-usd", str(cfg.max_budget_usd)]
    return cmd


def _default_runner(cmd: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def summarize_agent(user_instruction: str, cfg: SummarizeAgentConfig, meeting_id: str | None = None,
                    runner=None, dry_run: bool = False) -> dict:
    """Resolve the meeting, compose the prompt, and run headless Claude Code to enrich the vault.

    Returns the parsed Claude Code JSON result (with total_cost_usd etc.), or, for dry_run, a dict
    describing the resolved command without invoking Claude. Raises SummarizeAgentError on failure.
    """
    if not (user_instruction or "").strip():
        raise SummarizeAgentError("empty prompt — pass an instruction, e.g. "
                                  "briefly summarize \"extract action items into 30-Tasks\"")
    runner = runner or _default_runner
    mid = resolve_meeting_id(cfg, meeting_id)
    date, attendees = _read_meeting_meta(cfg, mid)
    transcript_text = _read_transcript(cfg, mid)
    note_rel = note_rel_path(cfg, mid, date)
    prompt = build_prompt(user_instruction, mid, date, attendees, note_rel, transcript_text)
    cmd = build_command(prompt, cfg)

    if dry_run:
        return {"dry_run": True, "meeting_id": mid, "date": date, "note": note_rel,
                "vault_dir": str(cfg.vault_dir), "transcript_chars": len(transcript_text),
                # command with the (large) prompt elided for readability
                "command": [a if a != prompt else f"<prompt:{len(prompt)} chars>" for a in cmd]}

    if not Path(cfg.vault_dir).exists():
        raise SummarizeAgentError(f"vault not found: {cfg.vault_dir} (set --vault-dir or BRIEFLY_VAULT_DIR)")
    try:
        proc = runner(cmd, str(cfg.vault_dir), cfg.timeout_sec)
    except FileNotFoundError as e:
        raise SummarizeAgentError(f"claude CLI not found ({cfg.claude_path!r}): {e}") from e
    if proc.returncode != 0:
        raise SummarizeAgentError(
            f"claude -p failed (exit {proc.returncode}): {(proc.stderr or '')[:500]}")
    try:
        out = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        out = {"raw": proc.stdout}
    out.setdefault("meeting_id", mid)
    out.setdefault("note", note_rel)
    return out


# ----------------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="briefly summarize",
        description=(
            "Enrich a meeting into the Obsidian vault using a CUSTOM Claude instruction (agentic "
            "Claude Code). The instruction says how YOU want this particular meeting enriched. "
            "meeting-id defaults to the last captured meeting. PRIVACY: transcript text goes to "
            "Claude; raw audio never leaves the device. (Run `briefly summarize` with no prompt "
            "for the structured per-person summary stage instead.)"),
        epilog='example: briefly summarize "Pull decisions + action items into the project note, '
               'and link each attendee to their person note" --meeting-id 01K…',
    )
    p.add_argument("prompt", help="how you want Claude to summarize/enrich this meeting into the vault")
    p.add_argument("--meeting-id", default=None, help="meeting ULID (default: last captured meeting)")
    p.add_argument("--vault-dir", default=None, help="vault root (default: $BRIEFLY_VAULT_DIR or ./vault)")
    p.add_argument("--data-root", default=None, help="holds recordings/ + transcripts/ (default: $BRIEFLY_DATA_ROOT or .)")
    p.add_argument("--transcripts-dir", default=None)
    p.add_argument("--recordings-dir", default=None)
    p.add_argument("--model", default=None, help="Claude model id (optional)")
    p.add_argument("--claude-path", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="resolve the meeting + print the Claude command without invoking it")
    return p


def _config_from(args: argparse.Namespace) -> SummarizeAgentConfig:
    from .dotenv import load_dotenv

    load_dotenv()  # BRIEFLY_VAULT_DIR / BRIEFLY_DATA_ROOT (does not override real env vars)
    cfg = SummarizeAgentConfig(
        vault_dir=args.vault_dir or os.environ.get("BRIEFLY_VAULT_DIR", "vault"),
        data_root=args.data_root or os.environ.get("BRIEFLY_DATA_ROOT", "."),
        transcripts_dir=args.transcripts_dir,
        recordings_dir=args.recordings_dir,
        claude_path=args.claude_path or os.environ.get("BRIEFLY_CLAUDE_PATH", "claude"),
        model=args.model or os.environ.get("BRIEFLY_SUMMARIZE_MODEL") or None,
    )
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = _config_from(args)
    try:
        result = summarize_agent(args.prompt, cfg, meeting_id=args.meeting_id, dry_run=args.dry_run)
    except SummarizeAgentError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return 0
    cost = result.get("total_cost_usd")
    print(f"enriched {result.get('note')} for meeting {result.get('meeting_id')}"
          + (f"  (cost ${cost})" if cost is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
