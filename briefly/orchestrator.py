"""Orchestrator — chain the file-based stages for one meeting_id:

    preprocess -> diarize -> transcribe -> merge -> [name speakers] -> summarize -> enrich

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

# diarize BEFORE transcribe: the Wyoming Whisper service is text-only, so we transcribe
# the line channel per diarization turn (knowledge/cluster/homelab-services.md).
STAGES = ["preprocess", "diarize", "transcribe", "merge", "summarize", "enrich"]


@dataclass
class PipelineConfig:
    data_root: str = "."                  # holds recordings/ processed/ transcripts/
    vault_dir: str = "vault"
    whisper_host: str = "localhost"       # wyoming-whisper TCP (port-forward or cluster DNS)
    whisper_port: int = 10300
    diarize_url: str = "http://localhost:8080/diarize"   # local default; .env / flags override
    diarize_mode: str = "pyannote"        # "single" = VAD fast-path for a one-remote-speaker 1:1
    # ASR engine for diarize+transcribe: "whisperx" (GPU box: /asr transcribe + its own /diarize),
    # "faster-whisper" (CPU service + pyannote), or "wyoming" (legacy text-only + diarization-guided
    # slicing). Diarize and transcribe stay separate steps for every backend.
    asr_backend: str = "whisperx"
    whisperx_url: str = "http://localhost:8000/asr"
    faster_whisper_url: str = "http://localhost:8001/asr"
    summarize_model: str = "claude-opus-4-8"
    summarize_backend: str = "auto"       # auto: SDK if ANTHROPIC_API_KEY else local `claude` CLI
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

def _run_preprocess(cfg: PipelineConfig, mid: str, progress=None) -> None:
    from .audio.preprocess import PreprocessConfig, preprocess
    preprocess(mid, cfg.rec(mid), cfg.proc(mid),
               PreprocessConfig(aec_enabled=cfg.aec_enabled, ffmpeg_path=cfg.ffmpeg_path))


def _run_transcribe(cfg: PipelineConfig, mid: str, progress=None) -> None:
    if cfg.asr_backend in ("whisperx", "faster-whisper"):   # POST /asr: transcribe + word align, no slicing
        from .clients.asr import AsrConfig, transcribe_meeting_asr
        url = cfg.whisperx_url if cfg.asr_backend == "whisperx" else cfg.faster_whisper_url
        transcribe_meeting_asr(cfg.proc(mid), cfg.tx(mid),
                               AsrConfig(url=url, timeout_sec=cfg.timeout_sec))
        return
    from .clients.transcribe import TranscribeConfig, transcribe_meeting
    cb = (lambda done, total: progress.update(done / total, f"{done}/{total} utterances")) \
        if progress else None
    transcribe_meeting(cfg.proc(mid), cfg.tx(mid),
                       TranscribeConfig(host=cfg.whisper_host, port=int(cfg.whisper_port),
                                        timeout_sec=cfg.timeout_sec),
                       on_progress=cb)


def _run_diarize(cfg: PipelineConfig, mid: str, progress=None) -> None:
    if cfg.diarize_mode == "single":   # 1:1 fast-path: VAD-segment line, one speaker, no pyannote
        from .clients.diarize import diarize_single
        diarize_single(cfg.proc(mid), cfg.tx(mid))
        return
    # pyannote-protocol POST /diarize — served by the pyannote service OR WhisperX's own /diarize
    # endpoint (set diarize_url accordingly). A separate step from transcribe, either way.
    from .clients.diarize import DiarizeConfig, diarize_meeting
    attendees = _manifest(cfg, mid).get("attendees") or []
    diarize_meeting(cfg.proc(mid), cfg.tx(mid),
                    DiarizeConfig(url=cfg.diarize_url, timeout_sec=cfg.timeout_sec,
                                  max_speakers=len(attendees) or None))


def _run_merge(cfg: PipelineConfig, mid: str, progress=None) -> None:
    from .merge import run as merge_run
    merge_run(mid, cfg.tx(mid), cfg.rec(mid))


def _run_summarize(cfg: PipelineConfig, mid: str, progress=None) -> None:
    from .summarize import SummarizeConfig, summarize
    summarize(SummarizeConfig(
        model=cfg.summarize_model, vault_dir=cfg.vault_dir, claude_path=cfg.claude_path,
        summarize_backend=cfg.summarize_backend,
        transcripts_dir=str(Path(cfg.data_root) / "transcripts"),
        recordings_dir=str(Path(cfg.data_root) / "recordings"),
    ), mid)


def _run_enrich(cfg: PipelineConfig, mid: str, progress=None) -> None:
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
                 log=print, progress=None) -> list[tuple[str, str]]:
    """Run stages [from_stage..to_stage] for one meeting. Returns [(stage, "ok"|"skip")].
    `progress` (a ProgressReporter) is optional; when given, a heartbeat is kept current."""
    runners = {**DEFAULT_RUNNERS, **(runners or {})}
    i0, i1 = STAGES.index(from_stage), STAGES.index(to_stage)
    if i0 > i1:
        raise ValueError(f"--from {from_stage} is after --to {to_stage}")
    results: list[tuple[str, str]] = []
    for stage in STAGES[i0:i1 + 1]:
        if not force and DONE[stage](cfg, meeting_id):
            log(f"skip  {stage} (already done)")
            if progress:
                progress.done(stage)
            results.append((stage, "skip"))
            continue
        log(f"run   {stage} ...")
        if progress:
            progress.stage(stage)
        runners[stage](cfg, meeting_id, progress)
        if progress:
            progress.done(stage)
        log(f"ok    {stage}")
        results.append((stage, "ok"))
    return results


# --- config loading + CLI -------------------------------------------------------------

_ENV = {
    "data_root": "BRIEFLY_DATA_ROOT", "vault_dir": "BRIEFLY_VAULT_DIR",
    "whisper_host": "BRIEFLY_WHISPER_HOST", "whisper_port": "BRIEFLY_WHISPER_PORT",
    "diarize_url": "BRIEFLY_DIARIZE_URL", "diarize_mode": "BRIEFLY_DIARIZE_MODE",
    "asr_backend": "BRIEFLY_ASR_BACKEND", "whisperx_url": "BRIEFLY_WHISPERX_URL",
    "faster_whisper_url": "BRIEFLY_FASTER_WHISPER_URL",
    "summarize_model": "BRIEFLY_SUMMARIZE_MODEL", "claude_path": "BRIEFLY_CLAUDE_PATH",
    "summarize_backend": "BRIEFLY_SUMMARIZE_BACKEND",
}


def load_config(path: str | None, overrides: dict) -> PipelineConfig:
    from .dotenv import load_dotenv
    load_dotenv()  # populate BRIEFLY_* from ./.env (does not override real env vars)
    data: dict = {}
    if path:
        data.update(json.loads(Path(path).read_text(encoding="utf-8")))
    for field_name, env in _ENV.items():
        if env in os.environ:
            data[field_name] = os.environ[env]
    data.update({k: v for k, v in overrides.items() if v is not None})
    if "whisper_port" in data:
        data["whisper_port"] = int(data["whisper_port"])  # env/config may pass a string
    known = PipelineConfig().__dict__
    return PipelineConfig(**{k: v for k, v in data.items() if k in known})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="briefly run",
                                description="run the meeting pipeline for one meeting_id")
    p.add_argument("--meeting-id", help="defaults to the last captured meeting "
                   "(recordings/.last-meeting-id)")
    p.add_argument("--from", dest="from_stage", default="preprocess", choices=STAGES)
    p.add_argument("--to", dest="to_stage", default="merge", choices=STAGES)
    p.add_argument("--force", action="store_true", help="re-run stages even if output exists")
    p.add_argument("--config", help="JSON config file")
    p.add_argument("--data-root")
    p.add_argument("--vault-dir")
    p.add_argument("--whisper-host")
    p.add_argument("--whisper-port", type=int)
    p.add_argument("--diarize-url")
    p.add_argument("--diarize-mode", choices=["pyannote", "single"],
                   help="'single' = VAD fast-path for a one-remote-speaker 1:1 (skips pyannote)")
    p.add_argument("--asr-backend", choices=["whisperx", "faster-whisper", "wyoming"],
                   help="diarize+transcribe engine (default whisperx)")
    p.add_argument("--whisperx-url")
    p.add_argument("--faster-whisper-url")
    p.add_argument("--summarize-model")
    p.add_argument("--summarize-backend", choices=["auto", "api", "cli"],
                   help="auto (default): Anthropic SDK if ANTHROPIC_API_KEY set, else the `claude` CLI")
    p.add_argument("--claude-path")
    args = p.parse_args(argv)
    cfg = load_config(args.config, {
        "data_root": args.data_root, "vault_dir": args.vault_dir,
        "whisper_host": args.whisper_host, "whisper_port": args.whisper_port,
        "diarize_url": args.diarize_url, "diarize_mode": args.diarize_mode,
        "asr_backend": args.asr_backend, "whisperx_url": args.whisperx_url,
        "faster_whisper_url": args.faster_whisper_url,
        "summarize_model": args.summarize_model, "claude_path": args.claude_path,
        "summarize_backend": args.summarize_backend,
    })
    from .state import read_last_meeting
    mid = args.meeting_id or read_last_meeting(Path(cfg.data_root) / "recordings")
    if not mid:
        print("error: no --meeting-id given and no last captured meeting found "
              "(run `briefly capture` first, or pass --meeting-id)", file=sys.stderr)
        return 2
    if not args.meeting_id:
        print(f"(using last captured meeting: {mid})")
    from .progress import ProgressReporter
    reporter = ProgressReporter(cfg.data_root, mid, STAGES, log=print)
    try:
        results = run_pipeline(cfg, mid, args.from_stage, args.to_stage, args.force,
                               progress=reporter)
    except Exception as e:  # stage failures surface here with a clear message
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if args.to_stage == "merge" and dict(results).get("merge") == "ok":
        print(f"\nnext: name speakers in {cfg.tx(mid) / 'speakers.json'}, then run\n"
              f"      briefly run --from summarize --to enrich --force")
    return 0


# --- status: read a running/finished job's progress -----------------------------------

_MARK = {"done": "[x]", "running": "[>]", "pending": "[ ]"}


def _status_lines(cfg: PipelineConfig, mid: str) -> list[str]:
    """Render the stage map from the live heartbeat, or infer it from artifacts on disk."""
    from .progress import read_heartbeat
    hb = read_heartbeat(cfg.data_root, mid)
    if hb:
        stages = hb.get("stages", {})
        marks = "  ".join(f"{_MARK.get(stages.get(s, 'pending'), '[ ]')} {s}" for s in STAGES)
        pct = round(100 * hb.get("overall_frac", 0.0))
        detail = f" · {hb['detail']}" if hb.get("detail") else ""
        el = hb.get("elapsed_sec")
        elapsed = f" · {int(el // 60)}m{int(el % 60):02d}s" if el is not None else ""
        return [f"meeting {mid}  (live heartbeat)", f"  {marks}",
                f"  {pct}% — {hb.get('stage') or 'idle'}{detail}{elapsed}"]
    inferred = {s: ("done" if DONE[s](cfg, mid) else "pending") for s in STAGES}
    marks = "  ".join(f"{_MARK[inferred[s]]} {s}" for s in STAGES)
    nxt = next((s for s in STAGES if inferred[s] == "pending"), None)
    return [f"meeting {mid}  (inferred from artifacts — no live heartbeat)", f"  {marks}",
            f"  next: {nxt or 'complete'}"]


def status_main(argv: list[str] | None = None) -> int:
    import time as _time
    from .progress import read_heartbeat
    from .state import read_last_meeting
    p = argparse.ArgumentParser(prog="briefly status",
                                description="show pipeline progress for a meeting")
    p.add_argument("--meeting-id")
    p.add_argument("--config")
    p.add_argument("--data-root")
    p.add_argument("--watch", action="store_true", help="repaint every 2s until done")
    args = p.parse_args(argv)
    cfg = load_config(args.config, {"data_root": args.data_root})
    mid = args.meeting_id or read_last_meeting(Path(cfg.data_root) / "recordings")
    if not mid:
        print("error: no --meeting-id and no last captured meeting", file=sys.stderr)
        return 2
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")   # clear screen
        print("\n".join(_status_lines(cfg, mid)))
        hb = read_heartbeat(cfg.data_root, mid)
        finished = (hb is not None and hb.get("overall_frac", 0) >= 0.999) \
            or (hb is None and DONE["merge"](cfg, mid))
        if not args.watch or finished:
            return 0
        _time.sleep(2)
