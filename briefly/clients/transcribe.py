"""Transcribe stage — diarization-guided transcription against the Wyoming Whisper service.

wyoming-whisper returns text only (no timestamps), so we obtain timestamped, speaker-alignable
segments by SLICING the audio and transcribing each slice:
  * LINE channel: slice by the pyannote diarization turns -> segments whose start/end come
    straight from the turns (so merge's overlap assignment against the same turns is exact).
  * MIC channel ("Me"): VAD-segment into utterances -> timestamped segments.
Output is merge-compatible *.whisper.json. Runs AFTER diarize (needs line.diarization.json).
The per-utterance transcriber is injectable (default: the Wyoming client) so tests need no
server. See knowledge/cluster/homelab-services.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import vad


@dataclass
class TranscribeConfig:
    host: str = "localhost"
    port: int = 10300
    rate: int = 16000          # preprocess emits 16 kHz mono
    language: str = "en"
    timeout_sec: float = 300
    pad_sec: float = 0.2       # context padding around each slice (audio only; not the segment time)
    concurrency: int = 6       # parallel Wyoming requests (mic + line utterances)


def _default_transcriber(cfg: TranscribeConfig):
    from .whisper_wyoming import transcribe_pcm
    return lambda pcm: transcribe_pcm(pcm, host=cfg.host, port=cfg.port,
                                      rate=cfg.rate, language=cfg.language, timeout=cfg.timeout_sec)


def _segment(samples, rate, start, end, idx, pad, transcribe) -> dict:
    pcm = vad.slice_pcm(samples, rate, max(0.0, start - pad), end + pad)
    return {"id": idx, "start": round(start, 3), "end": round(end, 3),
            "text": (transcribe(pcm) or "").strip()}


def _write(path: Path, rate: int, n_samples: int, segs: list[dict]) -> None:
    doc = {"language": None, "duration_sec": round(n_samples / rate, 3), "segments": segs}
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def transcribe_meeting(processed_dir, transcripts_dir, cfg: TranscribeConfig,
                       transcribe=None) -> dict[str, Path]:
    """LINE: transcribe per diarization turn. MIC: transcribe per VAD utterance. All
    utterances (both channels) are transcribed CONCURRENTLY — the per-request overhead
    overlaps instead of summing. Writes {mic,line}.whisper.json. Requires
    line.diarization.json (run diarize first)."""
    pdir, tdir = Path(processed_dir), Path(transcripts_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    transcribe = transcribe or _default_transcriber(cfg)

    diar_path = tdir / "line.diarization.json"
    if not diar_path.exists():
        raise FileNotFoundError(f"missing diarization (run diarize before transcribe): {diar_path}")
    turns = json.loads(diar_path.read_text(encoding="utf-8")).get("segments", [])
    line, lrate = vad.read_pcm16_mono(pdir / "line.16k.wav")
    mic, mrate = vad.read_pcm16_mono(pdir / "mic.16k.wav")

    # One job per utterance: (channel, samples, rate, start, end, idx).
    jobs = [("line", line, lrate, float(t["start"]), float(t["end"]), i)
            for i, t in enumerate(turns)]
    jobs += [("mic", mic, mrate, a, b, i)
             for i, (a, b) in enumerate(vad.segment_speech(mic, mrate))]

    def _run(job):
        ch, samples, rate, a, b, idx = job
        return ch, _segment(samples, rate, a, b, idx, cfg.pad_sec, transcribe)

    results = []
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as ex:
            results = list(ex.map(_run, jobs))   # order-preserving

    line_segs = [s for ch, s in results if ch == "line" and s["text"]]
    mic_segs = [s for ch, s in results if ch == "mic" and s["text"]]
    _write(tdir / "line.whisper.json", lrate, len(line), line_segs)
    _write(tdir / "mic.whisper.json", mrate, len(mic), mic_segs)
    return {"mic": tdir / "mic.whisper.json", "line": tdir / "line.whisper.json"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="briefly transcribe",
                                description="diarization-guided transcription via Wyoming Whisper")
    p.add_argument("--meeting-id", required=True)
    p.add_argument("--processed-dir", default="processed")
    p.add_argument("--transcripts-dir", default="transcripts")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=10300)
    args = p.parse_args(argv)
    cfg = TranscribeConfig(host=args.host, port=args.port)
    try:
        out = transcribe_meeting(Path(args.processed_dir) / args.meeting_id,
                                 Path(args.transcripts_dir) / args.meeting_id, cfg)
    except (FileNotFoundError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for ch, path in out.items():
        print(f"{ch}: {path}")
    return 0
