"""Orchestrator — chain the file-based stages for one meeting_id:

    preprocess -> transcribe -> diarize -> merge -> [name speakers] -> summarize -> enrich

Each stage reads the previous stage's files and writes its own; a stage is SKIPPED if its
output already exists (resumable) unless --force. Default run stops after `merge` so the
human can name speakers (transcripts/<id>/speakers.json), then re-run `--from summarize
--force`. Stage runners are injectable for offline tests.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

STAGES = ["preprocess", "transcribe", "diarize", "merge", "summarize", "enrich"]


@dataclass
class PipelineConfig:
    data_root: str = "."                  # holds recordings/ processed/ transcripts/
    vault_dir: str = "vault"
    whisper_url: str = "http://localhost:8000/v1/audio/transcriptions"
    whisper_format: str = "openai"
    whisper_model: str = "whisper-1"
    diarize_url: str = "http://localhost:8080/diarize"
    summarize_model: str = "claude-opus-4-8"
    claude_path: str = "claude"
    ffmpeg_path: str = "/opt/homebrew/bin/ffmpeg"
    aec_enabled: bool = True
    timeout_sec: float = 1800

    def rec(self, mid: str) -> Path:
        return Path(self.data_root) / "recordings" / mid

    def proc(self, mid: str) -> Path:
        return Path(self.data_root) / "processed" / mid

    def tx(self, mid: str) -> Path:
        return Path(self.data_root) / "transcripts" / mid


# --- helpers to read the manifest (date / attendees) ----------------------------------

def _manifest(cfg: PipelineConfig, mid: str) -> dict:
    mf = cfg.rec(mid) / "meeting.json"
    return json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}


def notes_path(cfg: PipelineConfig, mid: str) -> Path:
    date = _manifest(cfg, mid).get("date", "0000-00-00")
    return Path(cfg.vault_dir) / "20-Meetings" / f"{date}-{mid}.md"


# --- stage runners (call the stage modules) -------------------------------------------

def _run_preprocess(cfg: PipelineConfig, mid: str) -> None:
    from .audio.preprocess import PreprocessConfig, preprocess
    preprocess(mid, cfg.rec(mid), cfg.proc(mid),
               PreprocessConfig(aec_enabled=cfg.aec_enabled, ffmpeg_path=cfg.ffmpeg_path))


def _run_transcribe(cfg: PipelineConfig, mid: str) -> None:
    from .clients.transcribe import TranscribeConfig, transcribe_meeting
    transcribe_meeting(cfg.proc(mid), cfg.tx(mid),
                       TranscribeConfig(url=cfg.whisper_url, format=cfg.whisper_format,
                                        model=cfg.whisper_model, timeout_sec=cfg.timeout_sec))


def _run_diarize(cfg: PipelineConfig, mid: str) -> None:
    from .clients.diarize import DiarizeConfig, diarize_meeting
    attendees = _manifest(cfg, mid).get("attendees") or []
    diarize_meeting(cfg.proc(mid), cfg.tx(mid),
                    DiarizeConfig(url=cfg.diarize_url, timeout_sec=cfg.timeout_sec,
                                  max_speakers=len(attendees) or None))


def _run_merge(cfg: PipelineConfig, mid: str) -> None:
    from .merge import run as merge_run
    merge_run(mid, cfg.tx(mid), cfg.rec(mid))


def _run_summarize(cfg: PipelineConfig, mid: str) -> None:
    from .summarize import SummarizeConfig, summarize
    summarize(SummarizeConfig(
        model=cfg.summarize_model, vault_dir=cfg.vault_dir,
        transcripts_dir=str(Path(cfg.data_root) / "transcripts"),
        recordings_dir=str(Path(cfg.data_root) / "recordings"),
    ), mid)


def _run_enrich(cfg: PipelineConfig, mid: str) -> None:
    from .enrich import EnrichConfig, enrich_meeting
    enrich_meeting(notes_path(cfg, mid),
                   EnrichConfig(vault_dir=cfg.vault_dir, claude_path=cfg.claude_path))


DEFAULT_RUNNERS = {
    "preprocess": _run_preprocess, "transcribe": _run_transcribe, "diarize": _run_diarize,
    "merge": _run_merge, "summarize": _run_summarize, "enrich": _run_enrich,
}


# --- done-predicates (skip a stage when its output already exists) --------------------

_ENRICH_BLOCK = re.compile(
    r"<!--\s*briefly:enrichment:start\s*-->(.*?)<!--\s*briefly:enrichment:end\s*-->", re.S)


def _enriched(cfg: PipelineConfig, mid: str) -> bool:
    """Enriched = the managed block holds real content (not just markers/placeholder)."""
    p = notes_path(cfg, mid)
    if not p.exists():
        return False
    m = _ENRICH_BLOCK.search(p.read_text(encoding="utf-8"))
    if not m:
        return False
    inner = re.sub(r"<!--.*?-->", "", m.group(1), flags=re.S)  # drop placeholder comments
    return bool(inner.strip())


DONE = {
    "preprocess": lambda c, m: (c.proc(m) / "line.16k.wav").exists(),
    "transcribe": lambda c, m: (c.tx(m) / "line.whisper.json").exists(),
    "diarize": lambda c, m: (c.tx(m) / "line.diarization.json").exists(),
    "merge": lambda c, m: (c.tx(m) / "transcript.json").exists(),
    "summarize": lambda c, m: notes_path(c, m).exists(),
    "enrich": _enriched,
}


def run_pipeline(cfg: PipelineConfig, meeting_id: str, from_stage: str = "preprocess",
                 to_stage: str = "merge", force: bool = False, runners=None,
                 log=print) -> list[tuple[str, str]]:
    """Run stages [from_stage..to_stage] for one meeting. Returns [(stage, "ok"|"skip")]."""
    runners = {**DEFAULT_RUNNERS, **(runners or {})}
    i0, i1 = STAGES.index(from_stage), STAGES.index(to_stage)
    if i0 > i1:
        raise ValueError(f"--from {from_stage} is after --to {to_stage}")
    results: list[tuple[str, str]] = []
    for stage in STAGES[i0:i1 + 1]:
        if not force and DONE[stage](cfg, meeting_id):
            log(f"skip  {stage} (already done)")
            results.append((stage, "skip"))
            continue
        log(f"run   {stage} ...")
        runners[stage](cfg, meeting_id)
        log(f"ok    {stage}")
        results.append((stage, "ok"))
    return results


# --- config loading + CLI -------------------------------------------------------------

_ENV = {
    "data_root": "BRIEFLY_DATA_ROOT", "vault_dir": "BRIEFLY_VAULT_DIR",
    "whisper_url": "BRIEFLY_WHISPER_URL", "whisper_format": "BRIEFLY_WHISPER_FORMAT",
    "whisper_model": "BRIEFLY_WHISPER_MODEL", "diarize_url": "BRIEFLY_DIARIZE_URL",
    "summarize_model": "BRIEFLY_SUMMARIZE_MODEL", "claude_path": "BRIEFLY_CLAUDE_PATH",
}


def load_config(path: str | None, overrides: dict) -> PipelineConfig:
    data: dict = {}
    if path:
        data.update(json.loads(Path(path).read_text(encoding="utf-8")))
    for field_name, env in _ENV.items():
        if env in os.environ:
            data[field_name] = os.environ[env]
    data.update({k: v for k, v in overrides.items() if v is not None})
    known = PipelineConfig().__dict__
    return PipelineConfig(**{k: v for k, v in data.items() if k in known})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="briefly run",
                                description="run the meeting pipeline for one meeting_id")
    p.add_argument("--meeting-id", required=True)
    p.add_argument("--from", dest="from_stage", default="preprocess", choices=STAGES)
    p.add_argument("--to", dest="to_stage", default="merge", choices=STAGES)
    p.add_argument("--force", action="store_true", help="re-run stages even if output exists")
    p.add_argument("--config", help="JSON config file")
    p.add_argument("--data-root")
    p.add_argument("--vault-dir")
    p.add_argument("--whisper-url")
    p.add_argument("--whisper-format")
    p.add_argument("--diarize-url")
    p.add_argument("--summarize-model")
    p.add_argument("--claude-path")
    args = p.parse_args(argv)
    cfg = load_config(args.config, {
        "data_root": args.data_root, "vault_dir": args.vault_dir,
        "whisper_url": args.whisper_url, "whisper_format": args.whisper_format,
        "diarize_url": args.diarize_url, "summarize_model": args.summarize_model,
        "claude_path": args.claude_path,
    })
    try:
        results = run_pipeline(cfg, args.meeting_id, args.from_stage, args.to_stage, args.force)
    except Exception as e:  # stage failures surface here with a clear message
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if args.to_stage == "merge" and dict(results).get("merge") == "ok":
        print(f"\nnext: name speakers in {cfg.tx(args.meeting_id) / 'speakers.json'}, then run\n"
              f"      briefly run --meeting-id {args.meeting_id} --from summarize --to enrich --force")
    return 0
