"""`briefly summarize ["<instruction>"] [--enrich ["<prompt>"]] [--meeting-id <id>]` — the final
step: write a meeting into the Obsidian vault with headless Claude Code.

Default: write ONE concise summary page at the vault root. Pass a custom instruction, or omit it
to use DEFAULT_SUMMARIZE_PROMPT from .env (the built-in DEFAULT_PROMPT otherwise).
With --enrich, instead create/update notes ACROSS the vault: the ENRICHMENT_PROMPT from .env
(or the prompt passed to --enrich) is appended to the instruction to direct which files/folders
to edit. meeting_id defaults to the last captured meeting, or pass --meeting-id.

Safety: the vault is added via --add-dir, tools limited to Read,Glob,Grep,Edit,Write (NO Bash, so
the 40-Personal OS guard can't be bypassed), cwd = the vault root. permission-mode is
bypassPermissions: acceptEdits stalls on Claude Code's "suspicious path" heuristic (vaults under
iCloud / dotted paths like P.A.R.A.), so writes never land in headless mode — and with Bash
disallowed, bypassing approvals still can't run shell or escape the OS-level guard.
PRIVACY: the transcript TEXT goes to Claude (cloud, via Claude Code); raw audio never leaves the
device. The subprocess runner is injectable so tests need no `claude` binary.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Used when no instruction is passed AND DEFAULT_SUMMARIZE_PROMPT is unset in the environment.
DEFAULT_PROMPT = ("Write a concise meeting note: a 2-3 sentence summary, the key decisions, "
                  "action items with owners, and any open questions.")


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
    allowed_tools: str = "Read,Glob,Grep,Edit,Write"   # NO Bash, by design (keeps the 40-Personal guard)
    permission_mode: str = "bypassPermissions"         # acceptEdits stalls on the "suspicious path" guard in -p mode
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
    """The merged, speaker-attributed transcript.txt — required (run `briefly process` first)."""
    tp = cfg.tx_dir() / mid / "transcript.txt"
    if not tp.exists():
        raise SummarizeAgentError(
            f"transcript not found: {tp} — run `briefly process --meeting-id {mid} --to merge` first")
    text = tp.read_text(encoding="utf-8").strip()
    if not text:
        raise SummarizeAgentError(f"transcript is empty: {tp}")
    return text


def note_rel_path(cfg: SummarizeAgentConfig, mid: str, date: str) -> str:
    """Default single-summary path: <date>-<id>.md at the VAULT ROOT."""
    return f"{date}-{mid}.md"


def build_prompt(user_instruction: str, mid: str, date: str, attendees: list[str],
                 note_rel: str, transcript_text: str, enrich: bool = False) -> str:
    """Compose the Claude Code prompt: meeting context + the instruction + transcript.

    Default (enrich=False): write ONE concise summary page at the vault root (note_rel).
    enrich=True: follow the instruction to create/update notes across the vault.
    """
    who = ", ".join(attendees) if attendees else "(unknown)"
    header = (
        "You are writing meeting notes into an Obsidian vault. Your working directory IS the "
        "vault root. Use only Read/Glob/Grep/Edit/Write — do NOT run shell commands.\n\n"
        f"Meeting id: {mid}\n"
        f"Date: {date}\n"
        f"Attendees: {who}\n\n"
    )
    if enrich:
        body = (
            "Enrich the vault from this meeting. Follow THIS instruction for exactly which notes "
            "and folders to create or update (it describes where things go in this vault):\n"
            "--- INSTRUCTION ---\n"
            f"{user_instruction}\n"
            "--- END INSTRUCTION ---\n\n"
            "Make the requested edits across the vault, following its existing "
            "conventions/templates. Preserve any existing content you are not asked to change.\n\n"
        )
    else:
        body = (
            f"Write ONE concise meeting note at the vault root: {note_rel} (create if missing, "
            "else overwrite). Do not create or edit any other files. Follow this instruction:\n"
            "--- INSTRUCTION ---\n"
            f"{user_instruction}\n"
            "--- END INSTRUCTION ---\n\n"
        )
    return (
        header + body +
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
                    runner=None, dry_run: bool = False, enrich: bool = False) -> dict:
    """Resolve the meeting, compose the prompt, and run headless Claude Code on the vault.

    enrich=False (default): write one concise summary page at the vault root.
    enrich=True: follow the instruction to create/update notes across the vault.

    Returns the parsed Claude Code JSON result (with total_cost_usd etc.), or, for dry_run, a dict
    describing the resolved command without invoking Claude. Raises SummarizeAgentError on failure.
    """
    if not (user_instruction or "").strip():
        raise SummarizeAgentError("empty prompt — pass an instruction, e.g. "
                                  "briefly summarize \"3-bullet summary + action items\"")
    runner = runner or _default_runner
    mid = resolve_meeting_id(cfg, meeting_id)
    date, attendees = _read_meeting_meta(cfg, mid)
    transcript_text = _read_transcript(cfg, mid)
    note_rel = note_rel_path(cfg, mid, date)
    prompt = build_prompt(user_instruction, mid, date, attendees, note_rel, transcript_text, enrich)
    cmd = build_command(prompt, cfg)

    if dry_run:
        return {"dry_run": True, "meeting_id": mid, "date": date, "note": note_rel,
                "vault_dir": str(cfg.vault_dir), "transcript_chars": len(transcript_text),
                # command with the (large) prompt elided for readability
                "command": [a if a != prompt else f"<prompt:{len(prompt)} chars>" for a in cmd]}

    if not Path(cfg.vault_dir).exists():
        raise SummarizeAgentError(f"vault not found: {cfg.vault_dir} (set --vault-dir or VAULT_DIR)")
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

# --enrich was given with no value → fall back to ENRICHMENT_PROMPT from .env.
_ENRICH_FROM_ENV = "\0use-env-enrichment\0"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="briefly summarize",
        description=(
            "Summarize one meeting into your Obsidian vault with Claude Code. By default writes a "
            "single concise page at the vault root. Pass a custom instruction, or omit it to use "
            "DEFAULT_SUMMARIZE_PROMPT. Use --enrich to instead create/update notes across the vault "
            "per ENRICHMENT_PROMPT. meeting-id defaults to the last captured meeting. "
            "PRIVACY: transcript text goes to Claude; raw audio never leaves the device."),
        epilog='examples:\n'
               '  briefly summarize                       # one summary page at the vault root\n'
               '  briefly summarize --enrich              # enrich the vault using ENRICHMENT_PROMPT\n'
               '  briefly summarize --enrich "update the project plan"   # override the .env prompt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("prompt", nargs="?", default=None,
                   help="how to summarize this meeting; omit to use DEFAULT_SUMMARIZE_PROMPT")
    p.add_argument("--enrich", nargs="?", const=_ENRICH_FROM_ENV, default=None, metavar="PROMPT",
                   help="enrich the vault (create/update notes); appends ENRICHMENT_PROMPT from .env, "
                        "or the PROMPT you pass here, to the summarize instruction")
    p.add_argument("--meeting-id", default=None, help="meeting id (default: last captured meeting)")
    p.add_argument("--vault-dir", default=None, help="vault root (default: $VAULT_DIR or ./vault)")
    p.add_argument("--data-root", default=None, help="holds recordings/ + transcripts/ (default: $DATA_ROOT or .)")
    p.add_argument("--transcripts-dir", default=None)
    p.add_argument("--recordings-dir", default=None)
    p.add_argument("--model", default=None, help="Claude model id (optional)")
    p.add_argument("--claude-path", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="resolve the meeting + print the Claude command without invoking it")
    return p


def _config_from(args: argparse.Namespace) -> SummarizeAgentConfig:
    from .dotenv import load_dotenv

    load_dotenv()  # VAULT_DIR / DATA_ROOT (does not override real env vars)
    cfg = SummarizeAgentConfig(
        vault_dir=args.vault_dir or os.environ.get("VAULT_DIR", "vault"),
        data_root=args.data_root or os.environ.get("DATA_ROOT", "."),
        transcripts_dir=args.transcripts_dir,
        recordings_dir=args.recordings_dir,
        claude_path=args.claude_path or os.environ.get("CLAUDE_PATH", "claude"),
        model=args.model or os.environ.get("SUMMARIZE_MODEL") or None,
    )
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = _config_from(args)   # loads .env (DEFAULT_SUMMARIZE_PROMPT, ENRICHMENT_PROMPT, *)
    base = args.prompt or os.environ.get("DEFAULT_SUMMARIZE_PROMPT") or DEFAULT_PROMPT

    enrich = args.enrich is not None
    instruction = base
    if enrich:
        enrichment = (os.environ.get("ENRICHMENT_PROMPT") if args.enrich is _ENRICH_FROM_ENV
                      else args.enrich)
        if not (enrichment or "").strip():
            print("error: --enrich given but no enrichment prompt — set ENRICHMENT_PROMPT in .env "
                  'or pass one, e.g. briefly summarize --enrich "put blockers in 30-Issues/"',
                  file=sys.stderr)
            return 2
        instruction = f"{base}\n\n{enrichment}"   # enrichment prompt appended to the end

    try:
        result = summarize_agent(instruction, cfg, meeting_id=args.meeting_id,
                                 dry_run=args.dry_run, enrich=enrich)
    except SummarizeAgentError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return 0
    cost = result.get("total_cost_usd")
    summary = str(result.get("result") or "").strip()
    if result.get("is_error"):
        print(f"error: Claude reported a problem: {summary[:400] or '(no detail)'}", file=sys.stderr)
        return 1
    verb = "enriched the vault from" if enrich else f"wrote {result.get('note')} for"
    print(f"{verb} meeting {result.get('meeting_id')}"
          + (f"  (cost ${cost})" if cost is not None else ""))
    if summary:                                   # show what Claude says it did (catches "write blocked" etc.)
        print(f"  claude: {summary[:500]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
