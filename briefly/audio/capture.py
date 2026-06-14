"""Capture stage — record the two soundcard inputs and write meeting.json.

Implements docs/capture-contract.md: resolve by name, preflight, mint ULID,
dual-process simultaneous capture at native rate (raw, no resample), finalize with
level/clip measurement + start-offset, atomic immutable output. Capture ONLY — no
resample/AEC/de-clip/transcribe (those are later stages).
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import CaptureConfig
from ..ids import new_ulid
from ..models import CaptureInfo, ChannelInfo, MeetingManifest
from . import devices as dev


class CaptureError(Exception):
    exit_code = 1


class PreflightError(CaptureError):
    exit_code = 2


class PermissionDeniedError(CaptureError):
    exit_code = 3


class AlreadyExistsError(CaptureError):
    exit_code = 4


class FinalizeError(CaptureError):
    exit_code = 5


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ffmpeg_version(ffmpeg_path: str) -> str:
    try:
        out = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True).stdout
        m = re.search(r"ffmpeg version (\S+)", out)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


def preflight(cfg: CaptureConfig) -> dict:
    """Confirm both named devices exist, carry the expected signal, and aren't virtual.

    Raises PreflightError (device missing / flat-zero / no probe) or
    PermissionDeniedError (mic present but silent → likely macOS TCC).
    """
    inputs = dev.list_audio_inputs(cfg.ffmpeg_path)
    if not inputs:
        raise PreflightError("no avfoundation audio inputs found (soundcard connected? ffmpeg ok?)")

    results: dict = {}
    for role, name in (("mic", cfg.mic_device), ("line", cfg.line_device)):
        inp = dev.find_input(name, inputs)
        if inp is None:
            found = ", ".join(repr(i.name) for i in inputs)
            raise PreflightError(f"device not found by name: {name!r}. inputs present: {found}")
        mean, mx = dev.probe_device(name, cfg.probe_sec, cfg.ffmpeg_path)
        if mean is None:
            raise PreflightError(f"could not probe {name!r} (no audio captured)")
        if dev.is_flat_zero(mean, mx):
            raise PreflightError(
                f"{name!r} probes as flat {mean}/{mx} dB = digital zero (wrong/virtual device)"
            )
        carries = mx is not None and mx > cfg.signal_floor_dbfs
        if role == "mic" and not carries:
            raise PermissionDeniedError(
                f"mic {name!r} present but silent (peak {mx} dB). grant Microphone access to your "
                "terminal in System Settings → Privacy → Microphone, or check the mic."
            )
        results[role] = {
            "name": name, "index": inp.index,
            "mean_dbfs": mean, "max_dbfs": mx, "carries_signal": carries,
        }
    return results


def _spawn(cfg: CaptureConfig, device: str, out_part: Path, duration: float | None, log):
    cmd = [cfg.ffmpeg_path, "-hide_banner", "-y", "-f", "avfoundation", "-i", f":{device}"]
    if duration is not None:
        cmd += ["-t", f"{duration}"]
    cmd += ["-ac", str(cfg.channels), "-c:a", cfg.format, "-f", "wav", str(out_part)]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log)


def _stop(p: subprocess.Popen) -> None:
    try:
        p.send_signal(signal.SIGINT)
    except ProcessLookupError:
        pass


def _finalize_file(part: Path) -> Path | None:
    if not part.exists() or part.stat().st_size <= 44:  # 44 = empty WAV header
        return None
    final = part.with_name(part.name[: -len(".part")])
    os.replace(part, final)
    return final


def _channel_info(filename, final, cfg, speaker, device_name, offset) -> ChannelInfo:
    if final is None:
        return ChannelInfo(file=filename, device_name=device_name,
                           start_offset_sec=offset, speaker=speaker)
    _, _, dur = dev.wav_info(final)
    mean, peak = dev.measure_levels(final, cfg.ffmpeg_path)
    return ChannelInfo(
        file=filename, device_name=device_name, start_offset_sec=offset, speaker=speaker,
        duration_sec=round(dur, 3) if dur else None,
        peak_dbfs=peak, mean_dbfs=mean,
        clipping=dev.is_clipping(peak, cfg.clip_warn_dbfs) if peak is not None else None,
    )


def record(cfg: CaptureConfig, attendees: list[str] | None = None,
           duration: float | None = None, skip_preflight: bool = False) -> tuple[MeetingManifest, Path]:
    """Record both channels (dual-process) and write an immutable recordings/<id>/."""
    attendees = attendees or []
    if not skip_preflight:
        preflight(cfg)
    if cfg.mode != "dual-process":
        raise CaptureError(f"capture mode {cfg.mode!r} not implemented yet (use dual-process)")

    mid = new_ulid()
    mdir = Path(cfg.recordings_dir) / mid
    if mdir.exists():
        raise AlreadyExistsError(f"recordings dir already exists: {mdir}")
    mdir.mkdir(parents=True)

    started = _utcnow()
    mic_part, line_part = mdir / "mic.wav.part", mdir / "line.wav.part"
    partial = False
    try:
        with open(mdir / "capture.log", "w") as log:
            log.write(f"# capture {mid} mic={cfg.mic_device!r} line={cfg.line_device!r}\n")
            log.flush()
            t_mic = time.monotonic()
            p_mic = _spawn(cfg, cfg.mic_device, mic_part, duration, log)
            t_line = time.monotonic()
            p_line = _spawn(cfg, cfg.line_device, line_part, duration, log)
            try:
                p_mic.wait()
                p_line.wait()
            except KeyboardInterrupt:
                partial = True
                _stop(p_mic)
                _stop(p_line)
                p_mic.wait()
                p_line.wait()
    except OSError as e:
        raise FinalizeError(f"capture IO error: {e}") from e

    ended = _utcnow()
    mic_final = _finalize_file(mic_part)
    line_final = _finalize_file(line_part)
    if mic_final is None or line_final is None:
        partial = True

    # dual-process offset: earlier channel 0.0, later one the launch delta (coarse).
    delta = round(max(0.0, t_line - t_mic), 3)
    mic_ch = _channel_info("mic.wav", mic_final, cfg, "Me", cfg.mic_device, 0.0)
    line_ch = _channel_info("line.wav", line_final, cfg, None, cfg.line_device, delta)

    rate, ch, _ = dev.wav_info(mic_final) if mic_final else (None, None, None)
    manifest = MeetingManifest(
        meeting_id=mid, date=started[:10], started_at=started, ended_at=ended,
        partial=partial, attendees=list(attendees),
        capture=CaptureInfo(
            mode=cfg.mode, sample_rate=rate or cfg.sample_rate, format=cfg.format,
            channels=ch or cfg.channels, ffmpeg=_ffmpeg_version(cfg.ffmpeg_path),
            offset_method="process-start-delta",
        ),
        channels={"mic": mic_ch, "line": line_ch},
    )
    manifest.write(mdir / "meeting.json")
    return manifest, mdir
