"""Diarize client — POST the cleaned LINE channel to the pyannote service and write
`line.diarization.json` ({model, duration_sec, num_speakers, segments:[{speaker, start,
end}]}) — exactly what `merge` consumes. Only the line (remote) channel is diarized;
the mic channel is deterministically "Me". Matches the service in
knowledge/cluster/pyannote-deployment.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from . import http


@dataclass
class DiarizeConfig:
    url: str                              # e.g. http://pyannote.briefly.svc/diarize
    timeout_sec: float = 1800
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


def diarize_file(audio_path: str | Path, cfg: DiarizeConfig, post=None) -> dict:
    post = post or http.post_multipart
    path = Path(audio_path)
    fields: list[tuple[str, str]] = []
    for name in ("num_speakers", "min_speakers", "max_speakers"):
        val = getattr(cfg, name)
        if val is not None:
            fields.append((name, str(val)))
    # The speaker-diarization service expects the multipart field named "audio"
    # (FastAPI UploadFile `audio`) — see k8s-homelab speaker-diarization/app/main.py.
    raw = post(cfg.url, files=[("audio", path.name, path.read_bytes(), "audio/wav")],
               fields=fields, timeout=cfg.timeout_sec)
    resp = json.loads(raw)
    if not isinstance(resp, dict) or "segments" not in resp:
        raise ValueError("diarization response missing 'segments'")
    return resp


def diarize_meeting(processed_dir: str | Path, transcripts_dir: str | Path,
                    cfg: DiarizeConfig, post=None) -> Path:
    src = Path(processed_dir) / "line.16k.wav"
    if not src.exists():
        raise FileNotFoundError(f"missing cleaned line audio: {src}")
    resp = diarize_file(src, cfg, post=post)
    dst = Path(transcripts_dir) / "line.diarization.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(resp, indent=2), encoding="utf-8")
    return dst


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="briefly diarize",
                                description="diarize the line channel via the pyannote service")
    p.add_argument("--meeting-id", required=True)
    p.add_argument("--processed-dir", default="processed")
    p.add_argument("--transcripts-dir", default="transcripts")
    p.add_argument("--url", required=True, help="pyannote /diarize endpoint")
    p.add_argument("--max-speakers", type=int, default=None)
    args = p.parse_args(argv)
    cfg = DiarizeConfig(url=args.url, max_speakers=args.max_speakers)
    try:
        out = diarize_meeting(Path(args.processed_dir) / args.meeting_id,
                              Path(args.transcripts_dir) / args.meeting_id, cfg)
    except (FileNotFoundError, http.HttpError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"diarization: {out}")
    return 0
