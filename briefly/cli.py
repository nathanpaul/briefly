"""`briefly` CLI. Subcommand groups per pipeline stage; capture is first.

  briefly capture preflight
  briefly capture record --duration <sec> [--attendees "a,b"] [--no-preflight]
"""
from __future__ import annotations

import argparse
import sys

from .audio import capture as cap
from .config import CaptureConfig


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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="briefly", description="Briefly meeting pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    capp = sub.add_parser("capture", help="record a meeting's two soundcard channels")
    csub = capp.add_subparsers(dest="capcmd", required=True)

    pf = csub.add_parser("preflight", help="check devices, signal, and mic permission")
    _add_common(pf)

    rec = csub.add_parser("record", help="record for a fixed duration")
    _add_common(rec)
    rec.add_argument("--duration", type=float, required=True)
    rec.add_argument("--attendees", default="")
    rec.add_argument("--mode", default=None)
    rec.add_argument("--no-preflight", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = _config_from(args)
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
            manifest, mdir = cap.record(
                cfg, attendees=attendees, duration=args.duration,
                skip_preflight=args.no_preflight,
            )
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
            return 0
    except cap.CaptureError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    return 0


def capture_main() -> int:
    return main(["capture", *sys.argv[1:]])
