"""`briefly` CLI — top-level dispatch to the per-command entrypoints.

Workflow: `briefly capture` -> `briefly process` -> `briefly summarize`.
Each command owns its own argparse (prog="briefly <command>"); run it with --help.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .audio import capture as cap
from .config import CaptureConfig

USAGE = """briefly <command> [options]

commands:
  capture     record two soundcard channels (record --duration | start/stop)
  process     run the data pipeline (preprocess → diarize → transcribe → merge) for a meeting
  summarize   write a meeting into the vault — "<prompt>" for a custom pass, or nothing to
              use DEFAULT_SUMMARIZE_PROMPT (the usual final step)
  watch       auto-run `process` when a new meeting is captured
  status      show pipeline progress for a meeting (--watch to follow a running job)

individual stages (usually run via `process`):
  preprocess  AEC + de-clip + resample to 16 kHz mono   -> processed/<id>/
  diarize     pyannote /diarize (line channel)          -> line.diarization.json
  transcribe  whisperx /asr (or legacy wyoming)         -> *.whisper.json
  merge       whisper + diarization (+ speakers)        -> transcript.json

`briefly <command> --help` shows that command's options.
"""


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--recordings-dir")
    sp.add_argument("--mic-device")
    sp.add_argument("--line-device")
    sp.add_argument("--ffmpeg-path")


def _config_from(args: argparse.Namespace) -> CaptureConfig:
    cfg = CaptureConfig()
    for attr in ("recordings_dir", "mic_device", "line_device", "ffmpeg_path", "mode"):
        val = getattr(args, attr, None)
        if val:
            setattr(cfg, attr, val)
    return cfg


def _print_manifest(manifest, mdir) -> None:
    print(f"meeting_id: {manifest.meeting_id}")
    print(f"recordings: {mdir}")
    for role, c in manifest.channels.items():
        tail = "  *** CLIPPING ***" if c.clipping else ""
        print(f"  {role:5} {c.file}: {c.duration_sec}s  "
              f"peak {c.peak_dbfs} / mean {c.mean_dbfs} dB  "
              f"offset {c.start_offset_sec}s{tail}")
    if manifest.partial:
        print("WARNING: partial recording (a channel was truncated/missing)")
    if any(c.clipping for c in manifest.channels.values()):
        print("WARNING: clipping detected — lower mic preamp / DAC line-out to "
              "−6…−12 dB peaks and re-capture.")


def _capture_main(argv: list[str] | None) -> int:
    p = argparse.ArgumentParser(prog="briefly capture",
                                description="record a meeting's two soundcard channels")
    csub = p.add_subparsers(dest="capcmd", required=True)
    pf = csub.add_parser("preflight", help="check devices, signal, and mic permission")
    _add_common(pf)
    rec = csub.add_parser("record", help="record for a fixed duration")
    _add_common(rec)
    rec.add_argument("--duration", type=float, required=True)
    rec.add_argument("--attendees", default="")
    rec.add_argument("--mode", default=None)
    rec.add_argument("--no-preflight", action="store_true")

    st = csub.add_parser("start", help="begin an open-ended recording (finish with `stop`)")
    _add_common(st)
    st.add_argument("--attendees", default="")
    st.add_argument("--mode", default=None)
    st.add_argument("--no-preflight", action="store_true")

    sp = csub.add_parser("stop", help="finalize the recording started with `start`")
    _add_common(sp)
    sp.add_argument("--meeting-id", default=None)

    args = p.parse_args(argv)
    cfg = _config_from(args)
    if not args.recordings_dir:                      # align with `briefly process` via BRIEFLY_DATA_ROOT
        from .dotenv import load_dotenv
        load_dotenv()
        cfg.recordings_dir = str(Path(os.environ.get("BRIEFLY_DATA_ROOT", ".")) / "recordings")
    try:
        if args.capcmd == "preflight":
            for role, info in cap.preflight(cfg).items():
                state = "signal" if info["carries_signal"] else "quiet"
                print(f"{role:5} {info['name']!r}: mean {info['mean_dbfs']} / "
                      f"max {info['max_dbfs']} dB  ({state})")
            print("preflight OK")
            return 0
        if args.capcmd == "record":
            attendees = [a.strip() for a in args.attendees.split(",") if a.strip()]
            manifest, mdir = cap.record(cfg, attendees=attendees, duration=args.duration,
                                        skip_preflight=args.no_preflight)
            _print_manifest(manifest, mdir)
            return 0
        if args.capcmd == "start":
            attendees = [a.strip() for a in args.attendees.split(",") if a.strip()]
            mid, mdir = cap.start(cfg, attendees=attendees, skip_preflight=args.no_preflight)
            print(f"meeting_id: {mid}")
            print(f"recordings: {mdir}")
            print("recording in the background — run `briefly capture stop` to finish.")
            return 0
        if args.capcmd == "stop":
            manifest, mdir = cap.stop(cfg, meeting_id=args.meeting_id)
            _print_manifest(manifest, mdir)
            return 0
    except cap.CaptureError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "capture":
        return _capture_main(rest)
    if cmd == "merge":
        from .merge import main as merge_main
        return merge_main(rest)
    if cmd == "preprocess":
        from .audio.preprocess import main as preprocess_main
        return preprocess_main(rest)
    if cmd == "summarize":   # write a meeting into the vault (prompt or DEFAULT_SUMMARIZE_PROMPT)
        from .summarize_agent import main as summarize_agent_main
        return summarize_agent_main(rest)
    if cmd == "transcribe":
        from .clients.transcribe import main as transcribe_main
        return transcribe_main(rest)
    if cmd == "diarize":
        from .clients.diarize import main as diarize_main
        return diarize_main(rest)
    if cmd == "asr":
        from .clients.asr import main as asr_main
        return asr_main(rest)
    if cmd == "process":
        from .orchestrator import main as process_main
        return process_main(rest)
    if cmd == "watch":
        from .watch import main as watch_main
        return watch_main(rest)
    if cmd == "status":
        from .orchestrator import status_main
        return status_main(rest)
    print(f"unknown command: {cmd!r}\n\n{USAGE}", file=sys.stderr)
    return 2


def capture_main() -> int:
    return _capture_main(sys.argv[1:])
