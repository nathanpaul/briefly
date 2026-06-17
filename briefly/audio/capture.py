"""Capture stage — record the two soundcard inputs and write meeting.json.

Resolve devices by name, preflight, mint ULID, dual-process simultaneous
capture at native rate (raw, no resample), finalize with
level/clip measurement + start-offset, atomic immutable output. Capture ONLY — no
resample/AEC/de-clip/transcribe (those are later stages).
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from ..config import CaptureConfig
from ..ids import new_ulid
from ..models import CaptureInfo, ChannelInfo, MeetingManifest
from ..state import write_last_meeting
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


class NoActiveSessionError(CaptureError):
    exit_code = 6


class AmbiguousSessionError(CaptureError):
    exit_code = 6


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


def _spawn(cfg: CaptureConfig, device: str, out_part: Path, duration: float | None, log,
           detach: bool = False):
    cmd = [cfg.ffmpeg_path, "-hide_banner", "-y", "-f", "avfoundation", "-i", f":{device}"]
    if duration is not None:
        cmd += ["-t", f"{duration}"]
    cmd += ["-ac", str(cfg.channels), "-c:a", cfg.format, "-f", "wav", str(out_part)]
    # detach=True (start_new_session) lets the recorder outlive the `start` CLI process.
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log, start_new_session=detach)


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


_STATE = "capture.state.json"


def _require_dual(cfg: CaptureConfig) -> None:
    if cfg.mode != "dual-process":
        raise CaptureError(f"capture mode {cfg.mode!r} not implemented yet (use dual-process)")


def _new_session_dir(cfg: CaptureConfig) -> tuple[str, Path]:
    mid = new_ulid()
    mdir = Path(cfg.recordings_dir) / mid
    if mdir.exists():
        raise AlreadyExistsError(f"recordings dir already exists: {mdir}")
    mdir.mkdir(parents=True)
    return mid, mdir


def _finalize(cfg: CaptureConfig, mid: str, mdir: Path, started: str,
              attendees: list[str], t_mic: float, t_line: float,
              partial: bool = False) -> MeetingManifest:
    """Rename .part -> final, measure levels/clip, compute the start-offset, and write
    meeting.json. Shared by record() (foreground) and stop() (detached start/stop)."""
    mic_final = _finalize_file(mdir / "mic.wav.part")
    line_final = _finalize_file(mdir / "line.wav.part")
    if mic_final is None or line_final is None:
        partial = True
    # dual-process offset: earlier channel 0.0, later one the launch delta (coarse).
    delta = round(max(0.0, t_line - t_mic), 3)
    mic_ch = _channel_info("mic.wav", mic_final, cfg, "Me", cfg.mic_device, 0.0)
    line_ch = _channel_info("line.wav", line_final, cfg, None, cfg.line_device, delta)
    rate, ch, _ = dev.wav_info(mic_final) if mic_final else (None, None, None)
    manifest = MeetingManifest(
        meeting_id=mid, date=started[:10], started_at=started, ended_at=_utcnow(),
        partial=partial, attendees=list(attendees),
        capture=CaptureInfo(
            mode=cfg.mode, sample_rate=rate or cfg.sample_rate, format=cfg.format,
            channels=ch or cfg.channels, ffmpeg=_ffmpeg_version(cfg.ffmpeg_path),
            offset_method="process-start-delta",
        ),
        channels={"mic": mic_ch, "line": line_ch},
    )
    manifest.write(mdir / "meeting.json")
    write_last_meeting(mdir.parent, mid)   # recordings/.last-meeting-id — default for `briefly process`
    return manifest


def record(cfg: CaptureConfig, attendees: list[str] | None = None,
           duration: float | None = None, skip_preflight: bool = False) -> tuple[MeetingManifest, Path]:
    """Record both channels for a fixed duration (or until Ctrl-C) and write an immutable
    recordings/<id>/. For meetings of unknown length use start()/stop()."""
    attendees = attendees or []
    if not skip_preflight:
        preflight(cfg)
    _require_dual(cfg)
    mid, mdir = _new_session_dir(cfg)
    started = _utcnow()
    partial = False
    try:
        with open(mdir / "capture.log", "w") as log:
            log.write(f"# capture {mid} mic={cfg.mic_device!r} line={cfg.line_device!r}\n")
            log.flush()
            t_mic = time.time()
            p_mic = _spawn(cfg, cfg.mic_device, mdir / "mic.wav.part", duration, log)
            t_line = time.time()
            p_line = _spawn(cfg, cfg.line_device, mdir / "line.wav.part", duration, log)
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
    return _finalize(cfg, mid, mdir, started, attendees, t_mic, t_line, partial), mdir


def start(cfg: CaptureConfig, attendees: list[str] | None = None,
          skip_preflight: bool = False) -> tuple[str, Path]:
    """Begin an open-ended recording and RETURN immediately — the two ffmpeg processes are
    detached and keep recording. Use stop() to finalize. For meetings of unknown length."""
    attendees = attendees or []
    if not skip_preflight:
        preflight(cfg)
    _require_dual(cfg)
    mid, mdir = _new_session_dir(cfg)
    started = _utcnow()
    log = open(mdir / "capture.log", "w")  # fd inherited by the detached ffmpeg procs
    log.write(f"# capture {mid} (start) mic={cfg.mic_device!r} line={cfg.line_device!r}\n")
    log.flush()
    t_mic = time.time()
    p_mic = _spawn(cfg, cfg.mic_device, mdir / "mic.wav.part", None, log, detach=True)
    t_line = time.time()
    p_line = _spawn(cfg, cfg.line_device, mdir / "line.wav.part", None, log, detach=True)
    state = {
        "meeting_id": mid, "started_at": started, "attendees": list(attendees),
        "mic_device": cfg.mic_device, "line_device": cfg.line_device,
        "mic_pid": p_mic.pid, "line_pid": p_line.pid, "t_mic": t_mic, "t_line": t_line,
    }
    (mdir / _STATE).write_text(json.dumps(state, indent=2), encoding="utf-8")
    return mid, mdir


def _wait_pids(pids: list[int], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    for pid in pids:
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, PermissionError):
                break
            time.sleep(0.1)


def _find_active_session(cfg: CaptureConfig, meeting_id: str | None) -> Path:
    root = Path(cfg.recordings_dir)
    if meeting_id:
        mdir = root / meeting_id
        if not (mdir / _STATE).exists():
            raise NoActiveSessionError(f"no active capture session for {meeting_id!r}")
        return mdir
    active = [p.parent for p in root.glob(f"*/{_STATE}")
              if not (p.parent / "meeting.json").exists()] if root.exists() else []
    if not active:
        raise NoActiveSessionError("no active capture session to stop")
    if len(active) > 1:
        raise AmbiguousSessionError(
            f"multiple active sessions ({', '.join(p.name for p in active)}); pass --meeting-id")
    return active[0]


def stop(cfg: CaptureConfig, meeting_id: str | None = None) -> tuple[MeetingManifest, Path]:
    """Finalize a recording started with start(): signal the detached ffmpeg processes to
    flush + exit, then measure and write meeting.json (clean stop, partial=False)."""
    mdir = _find_active_session(cfg, meeting_id)
    state = json.loads((mdir / _STATE).read_text(encoding="utf-8"))
    pids = [state["mic_pid"], state["line_pid"]]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGINT)  # ffmpeg flushes the WAV trailer on SIGINT
        except (ProcessLookupError, PermissionError):
            pass
    _wait_pids(pids)
    cfg2 = replace(cfg, mic_device=state["mic_device"], line_device=state["line_device"])
    manifest = _finalize(cfg2, state["meeting_id"], mdir, state["started_at"],
                         state["attendees"], state["t_mic"], state["t_line"], partial=False)
    try:
        (mdir / _STATE).unlink()
    except OSError:
        pass
    return manifest, mdir
