"""avfoundation device enumeration, probing, and level measurement.

Devices are resolved BY NAME — avfoundation indices reshuffle between runs and a
wrong index once resolved to a silent virtual device. Recording also targets by name (`-i :NAME`).
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

_DEV_RE = re.compile(r"\[(\d+)\]\s+(.+?)\s*$")
_MEAN_RE = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB")
_MAX_RE = re.compile(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB")


@dataclass(frozen=True)
class AudioInput:
    index: int
    name: str


def parse_audio_inputs(stderr: str) -> list[AudioInput]:
    """Parse `ffmpeg -f avfoundation -list_devices true` stderr → audio inputs only."""
    inputs: list[AudioInput] = []
    in_audio = False
    for line in stderr.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True
            continue
        if "AVFoundation video devices" in line:
            in_audio = False
            continue
        if not in_audio:
            continue
        m = _DEV_RE.search(line)
        if m:
            inputs.append(AudioInput(int(m.group(1)), m.group(2)))
    return inputs


def list_audio_inputs(ffmpeg_path: str) -> list[AudioInput]:
    proc = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )
    return parse_audio_inputs(proc.stderr)


def find_input(name: str, inputs: list[AudioInput]) -> AudioInput | None:
    for inp in inputs:
        if inp.name == name:
            return inp
    return None


def parse_levels(stderr: str) -> tuple[float | None, float | None]:
    """Return (mean_dbfs, max_dbfs) from `volumedetect` stderr."""
    mean = float(m.group(1)) if (m := _MEAN_RE.search(stderr)) else None
    mx = float(m.group(1)) if (m := _MAX_RE.search(stderr)) else None
    return mean, mx


def measure_levels(path: str | Path, ffmpeg_path: str) -> tuple[float | None, float | None]:
    proc = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    return parse_levels(proc.stderr)


def probe_device(name: str, probe_sec: int, ffmpeg_path: str) -> tuple[float | None, float | None]:
    """Record a short probe from the named device and measure its levels."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "probe.wav"
        subprocess.run(
            [ffmpeg_path, "-hide_banner", "-y", "-f", "avfoundation", "-i", f":{name}",
             "-t", str(probe_sec), "-f", "wav", str(out)],
            capture_output=True,
            text=True,
        )
        if not out.exists() or out.stat().st_size <= 44:
            return None, None
        return measure_levels(out, ffmpeg_path)


def is_flat_zero(mean: float | None, mx: float | None) -> bool:
    """A flat mean==max around the noise floor = pure digital zero = wrong/virtual device."""
    if mean is None or mx is None:
        return False
    return abs(mean - mx) < 0.05 and mean <= -90.0


def is_clipping(peak_dbfs: float | None, threshold: float) -> bool:
    return peak_dbfs is not None and peak_dbfs >= threshold


def wav_info(path: str | Path) -> tuple[int | None, int | None, float | None]:
    """Return (sample_rate, channels, duration_sec) for a WAV, or Nones if unreadable."""
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            ch = w.getnchannels()
            frames = w.getnframes()
            dur = frames / float(rate) if rate else None
            return rate, ch, dur
    except Exception:
        return None, None, None
