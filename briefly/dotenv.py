"""Tiny .env loader (stdlib; no python-dotenv dependency).

Parses `KEY=VALUE` lines (`#` comments, blank lines, and optional surrounding quotes or a
leading `export ` are ignored) and sets them in os.environ. By default it does NOT override
variables already present in the real environment, so an explicit `BRIEFLY_*` env var or
CLI flag still wins. `briefly process` / `briefly watch` load `.env` from the working directory
automatically (see orchestrator.load_config).
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | os.PathLike = ".env", override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from `path` into os.environ. Returns what was set. No-op if the
    file is missing."""
    p = Path(path)
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        val = val.strip()
        # Strip an inline comment: a '#' preceded by whitespace starts a comment (e.g.
        # `URL=http://h:8000/asr   # the GPU box`). Skipped for quoted values so a '#' inside a
        # quoted secret/URL survives; `pass#word` / `http://x#frag` (no leading space) are kept.
        if val[:1] not in ("'", '"'):
            for i in range(1, len(val)):
                if val[i] == "#" and val[i - 1] in " \t":
                    val = val[:i].rstrip()
                    break
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = val
            loaded[key] = val
    return loaded
