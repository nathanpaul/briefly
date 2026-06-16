"""Pipeline progress reporting.

A tiny reporter that writes a throttled JSON heartbeat to
`<data_root>/runs/<meeting_id>.progress.json` — read by `briefly status` so a running job can be
watched from another terminal — and optionally prints intra-stage lines for the foreground run.

Optional everywhere: pass `progress=None` for the original silent behavior (tests + programmatic
callers are unaffected). See docs/progress-reporting.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

# Overall-% weights; diarize + transcribe dominate the wall-clock (see knowledge/test-results/).
STAGE_WEIGHTS = {
    "preprocess": 0.15, "diarize": 0.45, "transcribe": 0.35, "merge": 0.05,
    "summarize": 0.0, "enrich": 0.0,
}


class ProgressReporter:
    """Threaded through `run_pipeline`. `stage()`/`update()`/`done()` keep the heartbeat current."""

    def __init__(self, data_root, meeting_id: str, stages, *, log=None,
                 clock=time.monotonic, throttle_sec: float = 0.5):
        self._path = Path(data_root) / "runs" / f"{meeting_id}.progress.json"
        self.meeting_id = meeting_id
        self.stages = list(stages)
        self._log = log                 # optional callable(str) for foreground intra-stage lines
        self._clock = clock
        self._throttle = throttle_sec
        self._t0 = clock()
        self._cur: str | None = None
        self._frac = 0.0
        self._detail = ""
        self._status = {s: "pending" for s in self.stages}
        self._last = float("-inf")

    def stage(self, name: str) -> None:
        if self._cur and self._status.get(self._cur) == "running":
            self._status[self._cur] = "done"
        self._cur, self._frac, self._detail = name, 0.0, ""
        if name in self._status:
            self._status[name] = "running"
        self._write(force=True)

    def update(self, frac: float, detail: str = "") -> None:
        self._frac = 0.0 if frac < 0 else 1.0 if frac > 1 else frac
        self._detail = detail
        if self._write() and self._log and detail:
            self._log(f"      {self._cur} {detail}  ({self.overall_pct()}%)")

    def done(self, name: str) -> None:
        if name in self._status:
            self._status[name] = "done"
        self._frac = 1.0
        self._write(force=True)

    def overall_frac(self) -> float:
        total = sum(STAGE_WEIGHTS.get(s, 0.0) for s in self.stages) or 1.0
        acc = 0.0
        for s in self.stages:
            w = STAGE_WEIGHTS.get(s, 0.0)
            if self._status.get(s) == "done":
                acc += w
            elif s == self._cur:
                acc += w * self._frac
        return min(1.0, acc / total)

    def overall_pct(self) -> int:
        return round(100 * self.overall_frac())

    def _write(self, force: bool = False) -> bool:
        now = self._clock()
        if not force and now - self._last < self._throttle:
            return False
        self._last = now
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meeting_id": self.meeting_id,
            "stage": self._cur,
            "stage_frac": round(self._frac, 3),
            "overall_frac": round(self.overall_frac(), 3),
            "detail": self._detail,
            "elapsed_sec": round(now - self._t0, 1),
            "stages": dict(self._status),
        }
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)
        return True


def read_heartbeat(data_root, meeting_id: str) -> dict | None:
    """Return the parsed heartbeat for a meeting, or None if absent/unreadable."""
    p = Path(data_root) / "runs" / f"{meeting_id}.progress.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
