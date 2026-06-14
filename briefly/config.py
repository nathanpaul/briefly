"""Configuration defaults. CLI flags override; a future capture.yaml may too.

Defaults match docs/capture-contract.md and knowledge/ (device names, ffmpeg path).
Devices are addressed BY NAME — avfoundation indices are unstable
(knowledge/hardware/soundcard-cubilux-cb5.md).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaptureConfig:
    mic_device: str = "Cubilux CB5 MIC2"        # the user ("Me")
    line_device: str = "Cubilux CB5 Line In"    # remote/meeting audio (DAC line-out)
    mode: str = "dual-process"                  # "dual-process" | "aggregate"
    aggregate_device_name: str | None = None    # required when mode == "aggregate"
    sample_rate: int = 48000                    # native; capture does NOT resample
    format: str = "pcm_s16le"
    channels: int = 2                           # both CB5 inputs are 2ch
    clip_warn_dbfs: float = -0.1
    signal_floor_dbfs: float = -75.0
    probe_sec: int = 3
    ffmpeg_path: str = "/opt/homebrew/bin/ffmpeg"
    recordings_dir: str = "recordings"
