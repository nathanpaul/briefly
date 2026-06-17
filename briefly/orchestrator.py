"""Orchestrator — chain the file-based data stages for one meeting_id:

    preprocess -> diarize -> transcribe -> merge

Each stage reads the previous stage's files and writes its own, and is SKIPPED if its output
already exists (resumable) unless --force. Turning the transcript into a vault note is a
separate, final step — `briefly summarize`. Stage runners are injectable for offline tests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Diarize BEFORE transcribe: the legacy wyoming backend is text-only, so the line channel is
# transcribed per diarization turn. (whisperx/faster-whisper transcribe whole channels.)
STAGES = ["preprocess", "diarize", "transcribe", "merge"]


@dataclass
class PipelineConfig:
    data_root: str = "."                  # holds recordings/ processed/ transcripts/
    vault_dir: str = "vault"
    whisper_host: str = "localhost"       # wyoming-whisper TCP (legacy backend only)
    whisper_port: int = 10300
    diarize_url: str = "http://localhost:8000/diarize"   # pyannote-protocol /diarize
    diarize_mode: str = "pyannote"        # "single" = VAD fast-path for a 1-remote-speaker 1:1
    num_speakers: int | None = None       # exact total speaker count to constrain diarization (a correction)
    # diarize+transcribe engine: "whisperx" (GPU /asr + /diarize), "faster-whisper" (CPU + pyannote),
    # or "wyoming" (legacy text-only). Diarize and transcribe are separate steps for every backend.
    asr_backend: str = "whisperx"
    whisperx_url: str = "http://localhost:8000/asr"
    faster_whisper_url: str = "http://localhost:8001/asr"
    ffmpeg_path: str = "/opt/homebrew/bin/ffmpeg"
    aec_enabled: bool = True
    timeout_sec: float = 1800

    def rec(self, mid: str) -> Path:
        return Path(self.data_root) / "recordings" / mid

    def proc(self, mid: str) -> Path:
        return Path(self.data_root) / "processed" / mid

    def tx(self, mid: str) -> Path:
        return Path(self.data_root) / "transcripts" / mid


def _manifest(cfg: PipelineConfig, mid: str) -> dict:
    mf = cfg.rec(mid) / "meeting.json"
    return json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}


# --- stage runners (call the stage modules) -------------------------------------------

def _run_preprocess(cfg: PipelineConfig, mid: str, progress=None) -> None:
    from .audio.preprocess import PreprocessConfig, preprocess
    preprocess(mid, cfg.rec(mid), cfg.proc(mid),
               PreprocessConfig(aec_enabled=cfg.aec_enabled, ffmpeg_path=cfg.ffmpeg_path))


def _run_transcribe(cfg: PipelineConfig, mid: str, progress=None) -> None:
    if cfg.asr_backend in ("whisperx", "faster-whisper"):   # POST /asr: transcribe + word-align
        from .clients.asr import AsrConfig, transcribe_meeting_asr
        url = cfg.whisperx_url if cfg.asr_backend == "whisperx" else cfg.faster_whisper_url
        transcribe_meeting_asr(cfg.proc(mid), cfg.tx(mid),
                               AsrConfig(url=url, timeout_sec=cfg.timeout_sec))
        return
    from .clients.transcribe import TranscribeConfig, transcribe_meeting   # legacy wyoming
    cb = (lambda done, total: progress.update(done / total, f"{done}/{total} utterances")) \
        if progress else None
    transcribe_meeting(cfg.proc(mid), cfg.tx(mid),
                       TranscribeConfig(host=cfg.whisper_host, port=int(cfg.whisper_port),
                                        timeout_sec=cfg.timeout_sec),
                       on_progress=cb)


def _line_speaker_target(num_speakers: int | None, has_mic: bool) -> int | None:
    """Map a meeting's TOTAL speaker count to the diarized line/remote target. The mic ("Me")
    is one speaker and is never diarized, so subtract it when a mic channel exists. None ⇒ let
    pyannote decide. Clamped to ≥1."""
    if num_speakers is None:
        return None
    return max(1, num_speakers - 1) if has_mic else max(1, num_speakers)


def _run_diarize(cfg: PipelineConfig, mid: str, progress=None) -> None:
    if cfg.diarize_mode == "single":   # 1:1 fast-path: VAD-segment line, one speaker, no pyannote
        from .clients.diarize import diarize_single
        diarize_single(cfg.proc(mid), cfg.tx(mid))
        return
    from .clients.diarize import DiarizeConfig, diarize_meeting   # pyannote-protocol POST /diarize
    manifest = _manifest(cfg, mid)
    has_mic = "mic" in (manifest.get("channels") or {})
    target = _line_speaker_target(cfg.num_speakers, has_mic)
    if target is not None:             # explicit correction → force exactly this many line speakers
        dcfg = DiarizeConfig(url=cfg.diarize_url, timeout_sec=cfg.timeout_sec, num_speakers=target)
    else:                              # default: cap at the attendee count (a soft upper bound)
        attendees = manifest.get("attendees") or []
        dcfg = DiarizeConfig(url=cfg.diarize_url, timeout_sec=cfg.timeout_sec,
                             max_speakers=len(attendees) or None)
    diarize_meeting(cfg.proc(mid), cfg.tx(mid), dcfg)


def _run_merge(cfg: PipelineConfig, mid: str, progress=None) -> None:
    from .merge import run as merge_run
    merge_run(mid, cfg.tx(mid), cfg.rec(mid))


DEFAULT_RUNNERS = {
    "preprocess": _run_preprocess, "diarize": _run_diarize,
    "transcribe": _run_transcribe, "merge": _run_merge,
}

# Done-predicates: skip a stage when its output already exists.
DONE = {
    "preprocess": lambda c, m: (c.proc(m) / "line.16k.wav").exists(),
    "diarize": lambda c, m: (c.tx(m) / "line.diarization.json").exists(),
    "transcribe": lambda c, m: (c.tx(m) / "line.whisper.json").exists(),
    "merge": lambda c, m: (c.tx(m) / "transcript.json").exists(),
}


def _run_stage(work, on_tick=None, interval: float = 5.0, clock=time.monotonic):
    """Run work() on a thread, calling on_tick(elapsed_sec) every `interval`s until it
    finishes — so blocking stages (the diarize/transcribe HTTP calls) show they're alive
    instead of looking frozen. Re-raises any exception raised by work()."""
    box: dict = {}

    def runner():
        try:
            box["ok"] = work()
        except BaseException as e:  # noqa: BLE001 — ferried to the caller's thread
            box["err"] = e

    th = threading.Thread(target=runner, daemon=True)
    t0 = clock()
    th.start()
    while True:
        th.join(timeout=interval)
        if not th.is_alive():
            break
        if on_tick:
            on_tick(clock() - t0)
    if "err" in box:
        raise box["err"]
    return box.get("ok")


def _make_ticker(stage: str, log, progress):
    """A tick callback that shows elapsed time for a running stage. On a TTY it repaints one
    line in place; piped/redirected it prints a line per tick. Returns (callback, is_tty)."""
    is_tty = sys.stdout.isatty()

    def cb(elapsed: float):
        if progress:
            progress.tick()
        msg = f"…{stage} working {int(elapsed)}s"
        if is_tty:
            sys.stdout.write(f"\r      {msg}\033[K")
            sys.stdout.flush()
        else:
            log(f"      {msg}")

    return cb, is_tty


def _stage_error_hint(e: BaseException) -> str:
    """A human, actionable one-liner for a stage failure."""
    msg = f"{type(e).__name__}: {e}"
    low = str(e).lower()
    if any(s in low for s in ("connection", "refused", "timed out", "timeout",
                              "unreachable", "max retries", "failed to establish", "errno")):
        return f"{msg}  (is the service running and reachable? check the *_URL / host in your .env)"
    if any(s in low for s in ("no such file", "missing", "not found")):
        return f"{msg}  (a required input is missing — run the earlier stage first, or --force)"
    return msg


def run_pipeline(cfg: PipelineConfig, meeting_id: str, from_stage: str = "preprocess",
                 to_stage: str = "merge", force: bool = False, runners=None,
                 log=print, progress=None, tick_interval: float = 5.0,
                 clock=time.monotonic) -> list[tuple[str, str]]:
    """Run stages [from_stage..to_stage] for one meeting. Returns [(stage, "ok"|"skip")].
    `progress` (a ProgressReporter) is optional; when given, the heartbeat is kept current.
    Each stage shows an elapsed ticker while it runs; a failure logs an actionable line and
    re-raises."""
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
        cb, is_tty = _make_ticker(stage, log, progress)
        t0 = clock()
        try:
            _run_stage(lambda: runners[stage](cfg, meeting_id, progress),
                       on_tick=cb, interval=tick_interval, clock=clock)
        except Exception as e:
            if is_tty:
                sys.stdout.write("\r\033[K")
            log(f"✗ {stage} failed after {clock() - t0:.1f}s — {_stage_error_hint(e)}")
            raise
        if is_tty:
            sys.stdout.write("\r\033[K")
        if progress:
            progress.done(stage)
        log(f"ok    {stage}  ({clock() - t0:.1f}s)")
        results.append((stage, "ok"))
    return results


# --- config loading + CLI -------------------------------------------------------------

_ENV = {
    "data_root": "DATA_ROOT", "vault_dir": "VAULT_DIR",
    "whisper_host": "WHISPER_HOST", "whisper_port": "WHISPER_PORT",
    "diarize_url": "DIARIZE_URL", "diarize_mode": "DIARIZE_MODE",
    "asr_backend": "ASR_BACKEND", "whisperx_url": "TRANSCRIBE_SERVICE_URL",
    "faster_whisper_url": "FASTER_WHISPER_URL",
}


def load_config(path: str | None, overrides: dict) -> PipelineConfig:
    from .dotenv import load_dotenv
    load_dotenv()  # populate * from ./.env (real env vars / CLI flags still win)
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


def ingest_file(cfg: PipelineConfig, file_path: str | Path,
                attendees: list[str] | None = None,
                meeting_id_prefix: str = "meeting_") -> str:
    """Import a single audio file as a new line-only meeting; return its meeting_id.

    Transcodes any input (wav/mp3/m4a/flac/…) to recordings/<id>/line.wav (mono PCM) and writes
    a meeting.json with only a 'line' channel — no mic/"Me". The normal pipeline then runs
    (preprocess → diarize → transcribe → merge); diarization separates speakers in the file.
    """
    import subprocess
    from datetime import datetime, timezone

    from .audio import devices as dev
    from .audio.capture import _ffmpeg_version
    from .ids import next_meeting_id
    from .models import CaptureInfo, ChannelInfo, MeetingManifest
    from .state import write_last_meeting

    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"--from-file not found: {src}")
    rec_root = Path(cfg.data_root) / "recordings"
    mid = next_meeting_id(rec_root, meeting_id_prefix)
    mdir = rec_root / mid
    mdir.mkdir(parents=True, exist_ok=False)
    line_wav = mdir / "line.wav"
    proc = subprocess.run(
        [cfg.ffmpeg_path, "-y", "-i", str(src), "-ac", "1", "-c:a", "pcm_s16le", str(line_wav)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not line_wav.exists():
        raise RuntimeError(f"ffmpeg could not import {src}: {(proc.stderr or '')[-300:]}")
    rate, ch, dur = dev.wav_info(line_wav)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    MeetingManifest(
        meeting_id=mid, date=now[:10], started_at=now, ended_at=now, partial=False,
        attendees=list(attendees or []),
        capture=CaptureInfo(mode="import", sample_rate=rate or 0, format="pcm_s16le",
                            channels=ch or 1, ffmpeg=_ffmpeg_version(cfg.ffmpeg_path),
                            offset_method="none"),
        channels={"line": ChannelInfo(file="line.wav", device_name="import",
                                      start_offset_sec=0.0, duration_sec=dur)},
    ).write(mdir / "meeting.json")
    write_last_meeting(rec_root, mid)
    return mid


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="briefly process",
                                description="run the data pipeline (preprocess→diarize→transcribe→merge) for one meeting")
    p.add_argument("--meeting-id", help="defaults to the last captured meeting (recordings/.last-meeting-id)")
    p.add_argument("--from-file", dest="from_file", metavar="AUDIO",
                   help="import an audio file as a new meeting and process it (no soundcard capture needed)")
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
    p.add_argument("--num-speakers", type=int, default=None,
                   help="correct the speaker count: total distinct speakers in the meeting (you/the "
                        "mic counts as one). Forces diarization to exactly that many. Re-run with "
                        "--from diarize --to merge --force to apply.")
    p.add_argument("--asr-backend", choices=["whisperx", "faster-whisper", "wyoming"],
                   help="diarize+transcribe engine (default whisperx)")
    p.add_argument("--whisperx-url")
    p.add_argument("--faster-whisper-url")
    p.add_argument("--notify", nargs="?", const="bell", default=None, metavar="MODE",
                   help="ping when done: --notify (bell) or --notify desktop; default off / $NOTIFY")
    args = p.parse_args(argv)
    cfg = load_config(args.config, {
        "data_root": args.data_root, "vault_dir": args.vault_dir,
        "whisper_host": args.whisper_host, "whisper_port": args.whisper_port,
        "diarize_url": args.diarize_url, "diarize_mode": args.diarize_mode,
        "asr_backend": args.asr_backend, "whisperx_url": args.whisperx_url,
        "faster_whisper_url": args.faster_whisper_url, "num_speakers": args.num_speakers,
    })
    if args.from_file:
        if args.meeting_id:
            print("error: pass either --from-file or --meeting-id, not both", file=sys.stderr)
            return 2
        try:
            mid = ingest_file(cfg, args.from_file,
                              meeting_id_prefix=os.environ.get("MEETING_ID_PREFIX", "meeting_"))
        except (FileNotFoundError, RuntimeError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"(imported {args.from_file} as meeting: {mid})")
    else:
        from .state import read_last_meeting
        mid = args.meeting_id or read_last_meeting(Path(cfg.data_root) / "recordings")
        if not mid:
            print("error: no --meeting-id given and no last captured meeting found "
                  "(run `briefly capture`, pass --from-file <audio>, or pass --meeting-id)",
                  file=sys.stderr)
            return 2
        if not args.meeting_id:
            print(f"(using last captured meeting: {mid})")
    if cfg.num_speakers is not None:
        print(f"(diarizing for exactly {cfg.num_speakers} total speaker(s))")
    from .notify import notify, resolve_mode
    notify_mode = resolve_mode(args.notify)
    from .progress import ProgressReporter
    reporter = ProgressReporter(cfg.data_root, mid, STAGES, log=print)
    t0 = time.monotonic()
    try:
        results = run_pipeline(cfg, mid, args.from_stage, args.to_stage, args.force, progress=reporter)
    except Exception:   # run_pipeline already logged an actionable ✗ line for the failing stage
        print(f"error: process did not complete for {mid}", file=sys.stderr)
        notify("Briefly — process failed", mid, mode=notify_mode)
        return 1
    elapsed = time.monotonic() - t0
    ran = [s for s, st in results if st == "ok"]
    skipped = [s for s, st in results if st == "skip"]
    dur = f"{elapsed:.0f}s" if elapsed >= 1 else f"{elapsed:.1f}s"
    summary = f"{len(ran)} stage(s) in {dur}" if ran else "all stages already done"
    print(f"\n✓ {mid}: {summary}" + (f"  ({len(skipped)} skipped)" if skipped else ""))
    if args.to_stage == "merge" and dict(results).get("merge") in ("ok", "skip"):
        print(f"next: briefly summarize        # write this meeting into the vault")
        print(f"      (name speakers in {cfg.tx(mid) / 'speakers.json'}; wrong speaker count? re-run\n"
              f"       briefly process --from diarize --to merge --force --num-speakers N)")
    notify("Briefly — process done", f"{mid}: {summary}", mode=notify_mode)
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
