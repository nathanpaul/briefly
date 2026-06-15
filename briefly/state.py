"""Local 'last meeting' pointer.

Capture writes the finalized meeting_id to `<recordings_dir>/.last-meeting-id` so that
`briefly run` (and friends) can default to the most recently captured meeting without
re-typing the ULID.
"""
from __future__ import annotations

from pathlib import Path

_LAST = ".last-meeting-id"


def write_last_meeting(recordings_dir, meeting_id: str) -> None:
    d = Path(recordings_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / _LAST).write_text(meeting_id.strip() + "\n", encoding="utf-8")


def read_last_meeting(recordings_dir) -> str | None:
    p = Path(recordings_dir) / _LAST
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip() or None
