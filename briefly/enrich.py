"""Enrich runner — invoke headless Claude Code to enrich a meeting note against the
Obsidian vault, using the version-controlled `enrich-meeting` skill.

Runs `claude -p "/enrich-meeting <note>"` with cwd = the vault root so the skill +
vault CLAUDE.md load. Bash is intentionally NOT in allowedTools (the agent must not be
able to chmod around the 40-Personal OS guard — see knowledge/decisions). The subprocess
runner is injectable so tests need no `claude` binary or network.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class EnrichError(Exception):
    exit_code = 1


@dataclass
class EnrichConfig:
    vault_dir: str = "vault"
    claude_path: str = "claude"
    skill: str = "/enrich-meeting"
    allowed_tools: str = "Read,Glob,Grep,Edit,Write"   # NO Bash, by design
    permission_mode: str = "acceptEdits"
    model: str | None = None
    max_budget_usd: float | None = 0.50
    timeout_sec: float = 1800


def build_command(notes_path: Path, cfg: EnrichConfig) -> list[str]:
    cmd = [
        cfg.claude_path, "-p", f"{cfg.skill} {notes_path}",
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


def enrich_meeting(notes_path: str | Path, cfg: EnrichConfig | None = None, runner=None) -> dict:
    """Enrich one meeting note via headless Claude Code; returns the parsed JSON result
    (with total_cost_usd etc.). Raises EnrichError on failure."""
    cfg = cfg or EnrichConfig()
    runner = runner or _default_runner
    notes = Path(notes_path)
    if not notes.exists():
        raise EnrichError(f"notes not found: {notes}")
    cmd = build_command(notes, cfg)
    try:
        proc = runner(cmd, str(cfg.vault_dir), cfg.timeout_sec)
    except FileNotFoundError as e:  # claude binary missing
        raise EnrichError(f"claude CLI not found ({cfg.claude_path!r}): {e}") from e
    if proc.returncode != 0:
        raise EnrichError(f"claude -p failed (exit {proc.returncode}): "
                          f"{(proc.stderr or '')[:500]}")
    try:
        return json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        return {"raw": proc.stdout}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="briefly enrich",
                                description="enrich a meeting note against the vault via Claude Code")
    p.add_argument("--notes", required=True, help="path to the meeting note (in the vault)")
    p.add_argument("--vault-dir", default="vault")
    p.add_argument("--claude-path", default="claude")
    p.add_argument("--model", default=None)
    args = p.parse_args(argv)
    cfg = EnrichConfig(vault_dir=args.vault_dir, claude_path=args.claude_path, model=args.model)
    try:
        result = enrich_meeting(args.notes, cfg)
    except EnrichError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    cost = result.get("total_cost_usd")
    print(f"enriched {args.notes}" + (f"  (cost ${cost})" if cost is not None else ""))
    return 0
