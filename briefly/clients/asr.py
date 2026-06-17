"""Transcription client for the WhisperX (GPU) and faster-whisper (CPU) HTTP services.

Both expose `POST /asr` (multipart `audio` + form language/model) -> {model, language,
duration_sec, processing_sec, device, segments:[{start, end, text, words:[...]}]}. This is
**transcription + word alignment only** — diarization is a SEPARATE stage (the pyannote-protocol
`/diarize`, served by the pyannote service OR WhisperX's own `/diarize`), exactly like the WhisperX
example keeps transcribe/align and diarize as separate steps. `merge` then assigns the line
transcribe segments to the diarization turns. So `asr_backend` only picks the transcribe engine and
the path stays interchangeable with the wyoming+pyannote path.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from . import http


@dataclass
class AsrConfig:
    url: str                              # the service's /asr endpoint
    language: str = "en"
    model: str | None = None
    timeout_sec: float = 1800


def asr_file(audio_path, cfg: AsrConfig, post=None) -> dict:
    """POST one audio file to the /asr service; return the parsed response dict."""
    post = post or http.post_multipart
    path = Path(audio_path)
    fields: list[tuple[str, str]] = [("language", cfg.language)]
    if cfg.model:
        fields.append(("model", cfg.model))
    raw = post(cfg.url, files=[("audio", path.name, path.read_bytes(), "audio/wav")],
               fields=fields, timeout=cfg.timeout_sec)
    resp = json.loads(raw)
    if not isinstance(resp, dict) or "segments" not in resp:
        raise ValueError("ASR response missing 'segments'")
    return resp


def to_whisper_doc(resp: dict) -> dict:
    """ASR response -> merge-compatible whisper.json: {language, duration_sec, segments:[{id,start,end,text}]}."""
    segs = [{"id": i, "start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3),
             "text": (s.get("text") or "").strip()}
            for i, s in enumerate(resp.get("segments", [])) if (s.get("text") or "").strip()]
    return {"language": resp.get("language"), "duration_sec": resp.get("duration_sec"), "segments": segs}


def transcribe_meeting_asr(processed_dir, transcripts_dir, cfg: AsrConfig, post=None) -> dict[str, Path]:
    """Transcribe each channel directly (real word timestamps, no diarization-guided slicing) and
    write {mic,line}.whisper.json. Diarization is the separate pyannote-protocol stage. Works for
    both the WhisperX and faster-whisper backends (same /asr contract)."""
    pdir, tdir = Path(processed_dir), Path(transcripts_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    # `line` is required; `mic` is optional (single-file/imported meetings have no mic channel).
    line_src = pdir / "line.16k.wav"
    if not line_src.exists():
        raise FileNotFoundError(f"missing cleaned audio: {line_src}")
    out: dict[str, Path] = {}
    for ch in ("mic", "line"):
        src = pdir / f"{ch}.16k.wav"
        if not src.exists():
            continue
        resp = asr_file(src, cfg, post=post)
        p = tdir / f"{ch}.whisper.json"
        p.write_text(json.dumps(to_whisper_doc(resp), indent=2), encoding="utf-8")
        out[ch] = p
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="briefly asr",
                                description="transcribe a meeting's channels via a WhisperX/faster-whisper service")
    p.add_argument("--meeting-id", required=True)
    p.add_argument("--processed-dir", default="processed")
    p.add_argument("--transcripts-dir", default="transcripts")
    p.add_argument("--url", required=True, help="the /asr endpoint")
    args = p.parse_args(argv)
    cfg = AsrConfig(url=args.url)
    try:
        out = transcribe_meeting_asr(Path(args.processed_dir) / args.meeting_id,
                                     Path(args.transcripts_dir) / args.meeting_id, cfg)
    except (FileNotFoundError, http.HttpError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for k, v in out.items():
        print(f"{k}: {v}")
    return 0
