"""Shared data models. Start with the capture manifest (meeting.json).

Output schema is docs/capture-contract.md → Output: meeting.json (schema_version 1.0).
Plain dataclasses (no third-party deps) so capture runs on a stock Python.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = "1.0"


@dataclass
class ChannelInfo:
    file: str
    device_name: str
    start_offset_sec: float
    speaker: str | None = None
    device_uid: str | None = None
    duration_sec: float | None = None
    peak_dbfs: float | None = None
    mean_dbfs: float | None = None
    clipping: bool | None = None

    def to_dict(self) -> dict:
        # Ordered to match the contract; drop None-valued optionals.
        d: dict = {"file": self.file}
        if self.speaker is not None:
            d["speaker"] = self.speaker
        d["device_name"] = self.device_name
        if self.device_uid is not None:
            d["device_uid"] = self.device_uid
        d["start_offset_sec"] = self.start_offset_sec
        for k in ("duration_sec", "peak_dbfs", "mean_dbfs", "clipping"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


@dataclass
class CaptureInfo:
    mode: str
    sample_rate: int
    format: str
    channels: int
    ffmpeg: str
    offset_method: str


@dataclass
class MeetingManifest:
    meeting_id: str
    date: str
    started_at: str
    ended_at: str | None
    partial: bool
    attendees: list[str]
    capture: CaptureInfo
    channels: dict[str, ChannelInfo]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "meeting_id": self.meeting_id,
            "date": self.date,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "partial": self.partial,
            "attendees": list(self.attendees),
            "capture": asdict(self.capture),
            "channels": {k: v.to_dict() for k, v in self.channels.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MeetingManifest":
        return cls(
            meeting_id=d["meeting_id"],
            date=d["date"],
            started_at=d["started_at"],
            ended_at=d.get("ended_at"),
            partial=d.get("partial", False),
            attendees=list(d.get("attendees", [])),
            capture=CaptureInfo(**d["capture"]),
            channels={k: ChannelInfo(**v) for k, v in d["channels"].items()},
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    def write(self, path: str | Path) -> None:
        """Atomic write: temp file → fsync → os.replace."""
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    @classmethod
    def read(cls, path: str | Path) -> "MeetingManifest":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
