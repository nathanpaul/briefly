"""`briefly` CLI — top-level dispatch to per-stage entrypoints.

  briefly capture     preflight | record --duration <sec> ...
  briefly preprocess  --meeting-id <id> ...
  briefly merge       --meeting-id <id> ...
  briefly summarize   --meeting-id <id> ...

Each stage owns its own argparse (prog="briefly <stage>"); see that stage's
--help and the matching docs/*-contract.md.
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
  run         orchestrate the whole pipeline for one meeting_id
  watch       auto-run the pipeline when a new meeting is captured
  status      show pipeline progress for a meeting (--watch to follow a running job)
  capture     record (record --duration | start/stop) two soundcard channels
  preprocess  AEC + de-clip + resample to 16 kHz mono   -> processed/<id>/
  diarize     pyannote service (line channel)           -> line.diarization.json
  transcribe  wyoming-whisper (diarization-guided)      -> *.whisper.json
  merge       whisper + diarization (+ speakers)        -> transcript.json
  summarize   transcript.json (Claude)                  -> notes.md
  enrich      enrich notes.md against the vault (Claude Code)

run `briefly <command> --help` for command options.
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
    if not args.recordings_dir:                      # align with `briefly run` via BRIEFLY_DATA_ROOT
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
    if cmd == "summarize":
        from .summarize import main as summarize_main
        return summarize_main(rest)
    if cmd == "transcribe":
        from .clients.transcribe import main as transcribe_main
        return transcribe_main(rest)
    if cmd == "diarize":
        from .clients.diarize import main as diarize_main
        return diarize_main(rest)
    if cmd == "enrich":
        from .enrich import main as enrich_main
        return enrich_main(rest)
    if cmd == "run":
        from .orchestrator import main as run_main
        return run_main(rest)
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
