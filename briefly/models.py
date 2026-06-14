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


def _hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


@dataclass
class Speaker:
    """A participant in transcript.json (merge output / summarize input)."""
    id: str
    label: str          # stable "Speaker_N" (or "Me")
    channel: str        # "mic" | "line"
    source: str         # "channel" | "diarization"
    name: str | None = None

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "name": self.name,
                "channel": self.channel, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "Speaker":
        return cls(id=d["id"], label=d["label"], channel=d["channel"],
                   source=d["source"], name=d.get("name"))


@dataclass
class Turn:
    index: int
    speaker_id: str
    speaker: str        # resolved display value (name if mapped, else label)
    channel: str
    start: float
    end: float
    text: str
    confidence: float | None = None
    diarization_confidence: float | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index, "speaker_id": self.speaker_id, "speaker": self.speaker,
            "channel": self.channel, "start": self.start, "end": self.end, "text": self.text,
            "confidence": self.confidence, "diarization_confidence": self.diarization_confidence,
            "flags": list(self.flags),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(index=d["index"], speaker_id=d["speaker_id"], speaker=d["speaker"],
                   channel=d["channel"], start=d["start"], end=d["end"], text=d["text"],
                   confidence=d.get("confidence"),
                   diarization_confidence=d.get("diarization_confidence"),
                   flags=list(d.get("flags", [])))


@dataclass
class Transcript:
    """Canonical merged transcript (transcript.json). See docs/orchestrator-merge-contract.md."""
    meeting_id: str
    date: str
    generated_at: str | None
    partial: bool
    models: dict
    speakers: list[Speaker]
    turns: list[Turn]
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "meeting_id": self.meeting_id,
            "date": self.date,
            "generated_at": self.generated_at,
            "partial": self.partial,
            "models": self.models,
            "speakers": [s.to_dict() for s in self.speakers],
            "turns": [t.to_dict() for t in self.turns],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(
            meeting_id=d["meeting_id"], date=d["date"], generated_at=d.get("generated_at"),
            partial=d.get("partial", False), models=d.get("models", {}),
            speakers=[Speaker.from_dict(s) for s in d.get("speakers", [])],
            turns=[Turn.from_dict(t) for t in d.get("turns", [])],
            warnings=list(d.get("warnings", [])),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    def to_text(self) -> str:
        """Human-readable transcript.txt companion: `[hh:mm:ss] Name: text`."""
        return "".join(f"[{_hhmmss(t.start)}] {t.speaker}: {t.text}\n" for t in self.turns)

    def write(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    @classmethod
    def read(cls, path: str | Path) -> "Transcript":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


@dataclass
class SpeakersMap:
    """Human speaker naming + corrections (speakers.json). See docs/speaker-naming-and-retrigger.md."""
    meeting_id: str
    map: dict[str, str] = field(default_factory=dict)
    corrections: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"meeting_id": self.meeting_id, "map": dict(self.map),
                "corrections": list(self.corrections)}

    @classmethod
    def from_dict(cls, d: dict) -> "SpeakersMap":
        return cls(meeting_id=d.get("meeting_id", ""), map=dict(d.get("map", {})),
                   corrections=list(d.get("corrections", [])))

    @classmethod
    def read(cls, path: str | Path) -> "SpeakersMap":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
