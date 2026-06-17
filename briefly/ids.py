"""Meeting id generation.

The default ids are short, human-typable sequential names: ``meeting_0001``, ``meeting_0002``,
… (prefix configurable via ``MEETING_ID_PREFIX``). `next_meeting_id` scans an existing
recordings/ dir and returns the next free number, so ids are easy to read and re-type later.

`new_ulid` (26 Crockford-base32 chars: 48-bit ms timestamp + 80-bit randomness) is kept for
callers that still want an opaque, time-ordered id.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

DEFAULT_MEETING_ID_PREFIX = "meeting_"

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def next_meeting_id(recordings_dir, prefix: str = DEFAULT_MEETING_ID_PREFIX, width: int = 4) -> str:
    """Return the next sequential id like ``meeting_0001`` for `recordings_dir`.

    Scans for existing ``<prefix><digits>`` directories and returns ``<prefix>`` + (max+1),
    zero-padded to `width`. Monotonic: deleting a middle id never causes reuse.
    """
    d = Path(recordings_dir)
    pat = re.compile(re.escape(prefix) + r"(\d+)$")
    highest = 0
    if d.exists():
        for child in d.iterdir():
            m = pat.match(child.name)
            if m:
                highest = max(highest, int(m.group(1)))
    return f"{prefix}{highest + 1:0{width}d}"


def _encode(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        value, rem = divmod(value, 32)
        out.append(_CROCKFORD[rem])
    return "".join(reversed(out))


def new_ulid(timestamp_ms: int | None = None, randomness: bytes | None = None) -> str:
    """Return a new 26-char ULID. Args are injectable for deterministic tests."""
    ts = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
    rand = os.urandom(10) if randomness is None else randomness
    if len(rand) != 10:
        raise ValueError("randomness must be 10 bytes (80 bits)")
    return _encode(ts, 10) + _encode(int.from_bytes(rand, "big"), 16)


def is_ulid(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 26
        and all(c in _CROCKFORD for c in value)
    )
