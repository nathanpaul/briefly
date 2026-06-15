"""Transcribe client — POST a channel's 16 kHz WAV to the Whisper cluster and write a
`*.whisper.json` in the shape `merge` expects: {language, duration_sec, segments:[{id,
start, end, text, words?:[{word,start,end,prob}], avg_logprob, no_speech_prob}]}.

The cluster's response format is configurable. Default `openai` (verbose_json) is the
de-facto standard for self-hosted Whisper servers; `whisperx` is also supported. Point
`url` at your cluster; if its shape differs, add a normalizer or set `format`.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import http


@dataclass
class TranscribeConfig:
    url: str
    format: str = "openai"          # "openai" | "whisperx" | "native"
    model: str = "whisper-1"
    request_words: bool = True
    timeout_sec: float = 1800
    extra_fields: list = field(default_factory=list)  # extra (name, value) form fields


def _prob(w: dict) -> float | None:
    for k in ("prob", "probability", "score", "confidence"):
        if w.get(k) is not None:
            return float(w[k])
    return None


def normalize_openai(resp: dict) -> dict:
    segments = []
    for s in resp.get("segments", []) or []:
        seg = {
            "id": int(s.get("id", len(segments))),
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": (s.get("text") or "").strip(),
            "avg_logprob": s.get("avg_logprob"),
            "no_speech_prob": s.get("no_speech_prob"),
        }
        if s.get("words"):
            seg["words"] = [
                {"word": w.get("word", ""), "start": float(w["start"]),
                 "end": float(w["end"]), "prob": _prob(w)}
                for w in s["words"]
            ]
        segments.append(seg)
    # Some servers return words only at the top level; distribute them by time.
    top = resp.get("words") or []
    if top and not any("words" in s for s in segments):
        for s in segments:
            s["words"] = [
                {"word": w.get("word", ""), "start": float(w["start"]),
                 "end": float(w["end"]), "prob": _prob(w)}
                for w in top
                if w.get("start") is not None and s["start"] - 1e-6 <= float(w["start"]) < s["end"] + 1e-6
            ]
    return {"language": resp.get("language"),
            "duration_sec": resp.get("duration_sec") or resp.get("duration"),
            "segments": segments}


def normalize_whisperx(resp: dict) -> dict:
    segments = []
    for i, s in enumerate(resp.get("segments", []) or []):
        seg = {"id": i, "start": float(s["start"]), "end": float(s["end"]),
               "text": (s.get("text") or "").strip()}
        if s.get("words"):
            seg["words"] = [
                {"word": w.get("word", ""), "start": float(w["start"]),
                 "end": float(w["end"]), "prob": _prob(w)}
                for w in s["words"]
                if w.get("start") is not None and w.get("end") is not None
            ]
        segments.append(seg)
    dur = resp.get("duration_sec") or resp.get("duration")
    if dur is None and segments:
        dur = segments[-1]["end"]
    return {"language": resp.get("language"), "duration_sec": dur, "segments": segments}


NORMALIZERS = {"openai": normalize_openai, "whisperx": normalize_whisperx,
               "native": lambda r: r}


def transcribe_file(audio_path: str | Path, cfg: TranscribeConfig, post=None) -> dict:
    """Transcribe one WAV; returns a normalized whisper doc (merge-compatible)."""
    post = post or http.post_multipart
    path = Path(audio_path)
    fields: list[tuple[str, str]] = [("model", cfg.model), ("response_format", "verbose_json")]
    if cfg.request_words and cfg.format == "openai":
        fields += [("timestamp_granularities[]", "segment"),
                   ("timestamp_granularities[]", "word")]
    fields += list(cfg.extra_fields)
    raw = post(cfg.url, files=[("file", path.name, path.read_bytes(), "audio/wav")],
               fields=fields, timeout=cfg.timeout_sec)
    resp = json.loads(raw)
    return NORMALIZERS.get(cfg.format, normalize_openai)(resp)


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def transcribe_meeting(processed_dir: str | Path, transcripts_dir: str | Path,
                       cfg: TranscribeConfig, post=None) -> dict[str, Path]:
    """Transcribe both channels (processed/<id>/{mic,line}.16k.wav) → transcripts/<id>/
    {mic,line}.whisper.json. Returns the written paths."""
    pdir, tdir = Path(processed_dir), Path(transcripts_dir)
    out = {}
    for ch in ("mic", "line"):
        src = pdir / f"{ch}.16k.wav"
        if not src.exists():
            raise FileNotFoundError(f"missing cleaned audio: {src}")
        doc = transcribe_file(src, cfg, post=post)
        dst = tdir / f"{ch}.whisper.json"
        _write_json(dst, doc)
        out[ch] = dst
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="briefly transcribe",
                                description="transcribe both channels via the Whisper cluster")
    p.add_argument("--meeting-id", required=True)
    p.add_argument("--processed-dir", default="processed")
    p.add_argument("--transcripts-dir", default="transcripts")
    p.add_argument("--url", required=True, help="Whisper endpoint (OpenAI-compatible by default)")
    p.add_argument("--format", default="openai", choices=list(NORMALIZERS))
    p.add_argument("--model", default="whisper-1")
    args = p.parse_args(argv)
    cfg = TranscribeConfig(url=args.url, format=args.format, model=args.model)
    try:
        out = transcribe_meeting(Path(args.processed_dir) / args.meeting_id,
                                 Path(args.transcripts_dir) / args.meeting_id, cfg)
    except (FileNotFoundError, http.HttpError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for ch, path in out.items():
        print(f"{ch}: {path}")
    return 0
