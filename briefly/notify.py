"""Optional completion notification — opt in with $NOTIFY or --notify (OFF by default).

Long ops (`process`, `summarize`) can ping you when they finish. Values (case-insensitive):
  off / none / 0 / ""   → nothing (default)
  bell / on / 1 / true  → terminal bell
  desktop               → terminal bell + a macOS desktop notification (osascript; best-effort)
Everything is best-effort: a failure to ring/notify never affects the command's result.
"""
from __future__ import annotations

import os
import subprocess
import sys


def resolve_mode(cli: str | None = None, env: str | None = None) -> str:
    """Resolve the notify mode. An explicit --notify value wins over $NOTIFY; default off."""
    raw = cli if cli is not None else (env if env is not None else os.environ.get("NOTIFY", ""))
    v = str(raw).strip().lower()
    if v in ("", "0", "off", "none", "false"):
        return "off"
    return "desktop" if v == "desktop" else "bell"


def notify(title: str, message: str = "", *, mode: str = "off",
           runner=subprocess.run, out=None) -> None:
    """Best-effort completion notification; a no-op unless `mode` is bell/desktop."""
    if mode == "off":
        return
    out = out if out is not None else sys.stderr
    try:
        out.write("\a")          # terminal bell
        out.flush()
    except (OSError, ValueError):
        pass
    if mode == "desktop":
        msg = message.replace('"', "'")
        ttl = title.replace('"', "'")
        try:
            runner(["osascript", "-e", f'display notification "{msg}" with title "{ttl}"'],
                   capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
