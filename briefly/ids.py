"""ULID generation — time-ordered, lexicographically sortable meeting ids.

A ULID is 26 Crockford-base32 chars: 48-bit millisecond timestamp + 80-bit randomness.
Generated once at capture start; the key for every downstream per-meeting directory.
"""
from __future__ import annotations

import os
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


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
