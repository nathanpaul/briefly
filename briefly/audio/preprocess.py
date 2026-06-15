"""Preprocess stage — reference-based AEC + de-clip + normalize + 16 kHz resample.

Implements docs/preprocess-contract.md: turn the raw two-channel capture into clean,
level-consistent, Whisper-ready 16 kHz mono audio. The core job is to cancel the remote
audio that leaks from open-back headphones into the studio mic, using the clean LINE
channel as the far-end reference (reference-based AEC; knowledge/audio-capture/
gain-and-leakage.md). The line channel is the clean reference and never gets AEC.

Deterministic, file-in / file-out. Reads recordings/<id>/{mic.wav,line.wav,meeting.json};
writes processed/<id>/{mic.16k.wav,line.16k.wav,preprocess.json}. No network. Shells out
to ffmpeg (de-clip / loudnorm / high-pass / resample); levels via volumedetect.

Algorithm (ordered): align line reference to mic → reference-based AEC on the mic
(default ON, pluggable, optional) → de-clip + loudness-normalize each channel → resample
to 16 kHz mono. Non-fatal recoverable conditions (unrecoverable clipping, silent line,
missing AEC backend, very short audio) → warnings[] + stderr, exit 0. Missing/invalid
required input → non-zero exit, existing output left untouched.

AEC backend is PLUGGABLE and OPTIONAL — it is lazy-imported INSIDE _run_aec so this module
imports fine when the lib is absent. Recommended backend: `webrtc-audio-processing`
(WebRTC AudioProcessing module; reference-based AEC, survives double-talk). Alternative:
`speexdsp` / `speexdsp-python`. If aec_enabled but no backend is installed, we log a clear
warning and fall back to passthrough (preprocess.json records aec_enabled=false + reason).
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..models import MeetingManifest
from . import devices as dev

SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------- #
# Exit-code exception pattern (mirrors audio/capture.py).
# --------------------------------------------------------------------------- #
class PreprocessError(Exception):
    exit_code = 1


class InputError(PreprocessError):
    """Missing/unreadable/invalid required input → leave existing output untouched."""
    exit_code = 2


class FfmpegError(PreprocessError):
    """An ffmpeg invocation failed."""
    exit_code = 5


# --------------------------------------------------------------------------- #
# Config (defaults per the contract).
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessConfig:
    aec_enabled: bool = True               # reference-based AEC on the mic (line = far-end)
    normalize_target_dbfs: float = -3.0    # peak target applied to both channels
    resample_rate: int = 16000             # Whisper-class input rate
    declip: bool = True                    # de-clip pass (ffmpeg adeclip) before normalize
    xcorr_refine: bool = True              # refine AEC delay by cross-correlation
    highpass_hz: int = 80                  # remove rumble/DC (0 disables)
    denoise: bool = False                  # off: AEC targets the dominant contaminant
    clip_threshold_dbfs: float = -0.1      # peak >= this on the RAW channel ⇒ clipping
    silence_floor_dbfs: float = -60.0      # line mean below this ⇒ "silent" ⇒ skip AEC
    ffmpeg_path: str = "/opt/homebrew/bin/ffmpeg"
    recordings_dir: str = "recordings"
    processed_dir: str = "processed"

    @classmethod
    def from_file(cls, path: str | Path) -> "PreprocessConfig":
        """Load overrides from a JSON (or simple YAML key: value) config file."""
        text = Path(path).read_text(encoding="utf-8")
        data = _load_config_text(text)
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def _load_config_text(text: str) -> dict:
    """Parse JSON; fall back to a minimal `key: value` YAML subset (no deps)."""
    text = text.strip()
    if not text:
        return {}
    if text.startswith("{"):
        return json.loads(text)
    out: dict = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        out[key.strip()] = _coerce_scalar(val.strip())
    return out


def _coerce_scalar(val: str):
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", ""):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val.strip("'\"")


# --------------------------------------------------------------------------- #
# WAV helpers (stdlib wave; PCM s16 mono int samples). Levels via ffmpeg.
# --------------------------------------------------------------------------- #
def _read_pcm_mono(path: str | Path) -> tuple[list[int], int]:
    """Read a PCM WAV as a list of mono int16 samples + its sample rate.

    Multi-channel input is downmixed (mean of channels). 8/16/32-bit PCM supported;
    anything else raises InputError. Used for analysis / AEC / xcorr, not final output
    (final output is produced by ffmpeg to preserve full fidelity until the last step).
    """
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            nch = w.getnchannels()
            width = w.getsampwidth()
            nframes = w.getnframes()
            raw = w.readframes(nframes)
    except Exception as e:  # noqa: BLE001 - any wave error is an invalid input
        raise InputError(f"cannot read WAV {path!r}: {e}") from e

    if width == 2:
        fmt = "<%dh" % (len(raw) // 2)
        flat = list(struct.unpack(fmt, raw)) if raw else []
        scale = 1
    elif width == 1:  # unsigned 8-bit → signed range
        flat = [b - 128 for b in raw]
        scale = 256  # widen to int16-ish range
    elif width == 4:
        fmt = "<%di" % (len(raw) // 4)
        wide = list(struct.unpack(fmt, raw)) if raw else []
        flat = [v >> 16 for v in wide]  # narrow to 16-bit for analysis
        scale = 1
    else:
        raise InputError(f"unsupported PCM width {width} bytes in {path!r}")

    if width == 1:
        flat = [v * scale for v in flat]

    if nch <= 1:
        return flat, rate
    # downmix to mono (mean of interleaved channels)
    mono = [
        sum(flat[i : i + nch]) // nch
        for i in range(0, len(flat) - (len(flat) % nch), nch)
    ]
    return mono, rate


def detect_clipping(path: str | Path, threshold_dbfs: float = -0.1,
                    min_run: int = 3) -> bool:
    """True if the RAW channel clips: any run of >= min_run consecutive samples at/above
    the full-scale threshold. threshold_dbfs is relative to int16 full scale (32767)."""
    samples, _ = _read_pcm_mono(path)
    if not samples:
        return False
    full_scale = 32767.0
    level = full_scale * (10.0 ** (threshold_dbfs / 20.0))
    run = 0
    for s in samples:
        if abs(s) >= level:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 0
    return False


def _peak_dbfs_from_samples(samples: list[int]) -> float | None:
    if not samples:
        return None
    peak = max(abs(s) for s in samples)
    if peak == 0:
        return -120.0
    return round(20.0 * math.log10(peak / 32767.0), 2)


def _levels(path: str | Path, cfg: PreprocessConfig) -> dict:
    """before/after level block: mean/peak dBFS (ffmpeg volumedetect) + clipping flag."""
    mean, peak = dev.measure_levels(path, cfg.ffmpeg_path)
    return {
        "mean_dbfs": mean,
        "peak_dbfs": peak,
        "clipping_detected": detect_clipping(path, cfg.clip_threshold_dbfs),
    }


# --------------------------------------------------------------------------- #
# Step 1 — alignment (coarse from manifest, optional xcorr refine).
# --------------------------------------------------------------------------- #
def coarse_delay_sec(manifest: MeetingManifest) -> float:
    """line→mic delay = line.start_offset_sec − mic.start_offset_sec (contract step 1)."""
    mic = manifest.channels["mic"].start_offset_sec
    line = manifest.channels["line"].start_offset_sec
    return float(line) - float(mic)


def refine_delay_xcorr(mic_path: str | Path, line_path: str | Path,
                       coarse_sec: float, search_ms: float = 50.0,
                       max_samples: int = 240000) -> float | None:
    """Refine the line→mic delay by cross-correlating the leaked line copy in the mic
    against the line reference, searching a small window around the coarse estimate.

    Pure-Python, bounded: both signals are read mono and capped to max_samples (decimated
    if longer) so the O(window * N) search stays cheap and deterministic. Returns the
    refined delay in seconds, or None if it cannot be estimated (too short / silent).
    """
    mic, rate_m = _read_pcm_mono(mic_path)
    line, rate_l = _read_pcm_mono(line_path)
    if not mic or not line or rate_m != rate_l or rate_m <= 0:
        return None

    rate = rate_m
    # Decimate uniformly to keep the search bounded while preserving relative timing.
    decim = max(1, (max(len(mic), len(line)) + max_samples - 1) // max_samples)
    mic_d = mic[::decim]
    line_d = line[::decim]
    eff_rate = rate / decim

    win = int(round((search_ms / 1000.0) * eff_rate))
    coarse = int(round(coarse_sec * eff_rate))
    n = min(len(mic_d), len(line_d))
    if n < 8 or win < 1:
        return None

    best_lag = coarse
    best_score = -1.0
    # Positive lag ⇒ line arrives `lag` samples later in the mic (echo delay).
    for lag in range(coarse - win, coarse + win + 1):
        num = 0.0
        cnt = 0
        for i in range(max(0, lag), n):
            j = i - lag
            if j < 0 or j >= len(line_d):
                continue
            num += mic_d[i] * line_d[j]
            cnt += 1
        if cnt == 0:
            continue
        score = num / cnt
        if score > best_score:
            best_score = score
            best_lag = lag
    if best_score <= 0:
        return None
    return best_lag / eff_rate


# --------------------------------------------------------------------------- #
# Step 2 — reference-based AEC (pluggable, optional, lazy backend import).
# --------------------------------------------------------------------------- #
def _aec_backend_available() -> tuple[bool, str | None]:
    """Lazy probe for an AEC backend WITHOUT importing it at module load time.

    Order of preference: webrtc-audio-processing, then speexdsp. Returns
    (available, module_name). Kept import-light so the module loads on stock Python.
    """
    import importlib.util  # stdlib

    if importlib.util.find_spec("numpy") is not None:
        return True, "wiener-numpy"
    for mod in ("webrtc_audio_processing", "speexdsp"):
        if importlib.util.find_spec(mod) is not None:
            return True, mod
    return False, None


def _run_aec(mic_path: Path, line_path: Path, delay_sec: float,
             cfg: PreprocessConfig, out_path: Path, warnings: list[str]) -> dict:
    """Reference-based AEC: cancel the leaked LINE audio from the MIC.

    The AEC backend is imported INSIDE this function so the module imports fine when the
    library is absent. On success, writes the echo-cancelled mic to out_path and returns
    {applied:True, reduction_db:float, backend:str}. When the backend is unavailable, logs
    a warning, copies the mic through unchanged, and returns {applied:False, ...}.
    """
    available, backend = _aec_backend_available()
    if not available:
        warnings.append(
            "aec_enabled but no AEC backend (numpy) installed (pip install 'briefly[aec]'); "
            "falling back to passthrough — mic NOT echo-cancelled."
        )
        _ffmpeg_copy(mic_path, out_path, cfg)
        return {"applied": False, "reduction_db": None, "backend": None}
    try:
        from .aec import run_aec_file
        return run_aec_file(mic_path, line_path, delay_sec, out_path)
    except Exception as e:  # noqa: BLE001 - never let AEC crash the stage
        warnings.append(f"AEC backend {backend!r} failed ({type(e).__name__}: {e}); "
                        "passthrough — mic NOT echo-cancelled.")
        _ffmpeg_copy(mic_path, out_path, cfg)
        return {"applied": False, "reduction_db": None, "backend": backend}


# --------------------------------------------------------------------------- #
# ffmpeg steps (de-clip / normalize / high-pass / resample). House style: shell
# to ffmpeg with explicit args, capture output, raise on non-zero.
# --------------------------------------------------------------------------- #
def _run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise FfmpegError("ffmpeg failed: " + " | ".join(tail))


def _ffmpeg_copy(src: Path, dst: Path, cfg: PreprocessConfig) -> None:
    """Copy/transcode mic to a working WAV at the native rate (no level change)."""
    _run_ffmpeg([cfg.ffmpeg_path, "-hide_banner", "-y", "-i", str(src),
                 "-c:a", "pcm_s16le", "-f", "wav", str(dst)])


def _normalize_and_resample(src: Path, dst: Path, cfg: PreprocessConfig,
                            declip: bool) -> None:
    """Two-pass peak normalize to normalize_target_dbfs, then de-clip/HP/resample to 16k mono.

    Pass 1: volumedetect on `src` to get max_volume. Pass 2: apply `volume=<gain>dB` so the
    new peak hits the target, alongside the cleanup+resample filterchain. Deterministic:
    identical input ⇒ identical gain ⇒ identical output.
    """
    mean, peak = dev.measure_levels(src, cfg.ffmpeg_path)
    gain_db = 0.0
    if peak is not None:
        gain_db = round(cfg.normalize_target_dbfs - peak, 3)

    filters: list[str] = []
    if declip and cfg.declip:
        filters.append("adeclip")
    if cfg.highpass_hz and cfg.highpass_hz > 0:
        filters.append(f"highpass=f={cfg.highpass_hz}")
    if cfg.denoise:
        filters.append("afftdn")
    if abs(gain_db) > 1e-3:
        filters.append(f"volume={gain_db}dB")
    filters.append(f"aresample={cfg.resample_rate}")
    filters.append("aformat=sample_fmts=s16:channel_layouts=mono")

    _run_ffmpeg([cfg.ffmpeg_path, "-hide_banner", "-y", "-i", str(src),
                 "-af", ",".join(filters),
                 "-ac", "1", "-ar", str(cfg.resample_rate),
                 "-c:a", "pcm_s16le", "-f", "wav", str(dst)])


# --------------------------------------------------------------------------- #
# Report (preprocess.json) — atomic write.
# --------------------------------------------------------------------------- #
def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ffmpeg_version(ffmpeg_path: str) -> str:
    try:
        import re
        out = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True).stdout
        m = re.search(r"ffmpeg version (\S+)", out)
        return m.group(1) if m else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Orchestration — the stage itself (testable, no argparse).
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessResult:
    report: dict
    processed_dir: Path
    warnings: list[str] = field(default_factory=list)


def preprocess(meeting_id: str, recordings_dir: str | Path,
               processed_dir: str | Path, cfg: PreprocessConfig | None = None
               ) -> PreprocessResult:
    """Run the preprocess stage for one meeting. Pure-ish: reads recordings/<id>/, writes
    processed/<id>/, no network. Returns a PreprocessResult (report + warnings).

    Raises InputError (non-zero exit) on a missing/invalid required input, leaving any
    existing processed/<id>/ untouched. Recoverable conditions are recorded in warnings[].
    """
    cfg = cfg or PreprocessConfig()
    warnings: list[str] = []

    rec_dir = Path(recordings_dir)
    # Accept either recordings/<id>/ passed directly or the parent recordings/ dir.
    if (rec_dir / "meeting.json").exists():
        in_dir = rec_dir
    else:
        in_dir = rec_dir / meeting_id
    manifest_path = in_dir / "meeting.json"
    if not manifest_path.exists():
        raise InputError(f"missing meeting.json: {manifest_path}")
    try:
        manifest = MeetingManifest.read(manifest_path)
    except Exception as e:  # noqa: BLE001
        raise InputError(f"invalid meeting.json {manifest_path}: {e}") from e

    if "mic" not in manifest.channels or "line" not in manifest.channels:
        raise InputError("meeting.json must define both 'mic' and 'line' channels")

    mic_in = in_dir / manifest.channels["mic"].file
    line_in = in_dir / manifest.channels["line"].file
    for p in (mic_in, line_in):
        if not p.exists() or p.stat().st_size <= 44:  # 44 = empty WAV header
            raise InputError(f"missing/empty required input: {p}")
        if dev.wav_info(p)[0] is None:
            raise InputError(f"unreadable WAV (bad format): {p}")

    out_dir = Path(processed_dir)
    if not (out_dir.name == meeting_id):
        out_dir = out_dir / meeting_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- before-levels (raw) -------------------------------------------------
    mic_before = _levels(mic_in, cfg)
    line_before = _levels(line_in, cfg)

    # ---- Step 1: alignment ---------------------------------------------------
    coarse = coarse_delay_sec(manifest)
    delay = coarse
    delay_source = "meeting.json"
    line_silent = (line_before["mean_dbfs"] is None
                   or line_before["mean_dbfs"] <= cfg.silence_floor_dbfs)

    if cfg.xcorr_refine and not line_silent:
        refined = refine_delay_xcorr(mic_in, line_in, coarse)
        if refined is not None:
            delay = round(refined, 4)
            delay_source = "meeting.json+xcorr"
        else:
            warnings.append("xcorr refinement unavailable (audio too short/silent); "
                            "using manifest delay.")

    # ---- Step 2: reference-based AEC on the mic (line never gets AEC) ---------
    mic_work = out_dir / "mic.aec.wav"
    aec_requested = cfg.aec_enabled
    if not aec_requested:
        aec_info = {"applied": False, "reduction_db": None, "backend": None}
        _ffmpeg_copy(mic_in, mic_work, cfg)
    elif line_silent:
        warnings.append("line channel is silent (no far-end reference); skipping AEC "
                        "(no echo to cancel).")
        aec_info = {"applied": False, "reduction_db": None, "backend": None}
        _ffmpeg_copy(mic_in, mic_work, cfg)
    else:
        aec_info = _run_aec(mic_in, line_in, delay, cfg, mic_work, warnings)

    aec_enabled_effective = bool(aec_info["applied"])

    # ---- Step 3+4: de-clip + normalize + resample each channel ---------------
    # Mic is normalized AFTER AEC (cancellation changes levels).
    mic_out = out_dir / "mic.16k.wav"
    line_out = out_dir / "line.16k.wav"
    _normalize_and_resample(mic_work, mic_out, cfg, declip=cfg.declip)
    _normalize_and_resample(line_in, line_out, cfg, declip=cfg.declip)

    try:
        mic_work.unlink()
    except OSError:
        pass

    # ---- Unrecoverable-clipping warnings (raw clipping detected) -------------
    if mic_before["clipping_detected"]:
        warnings.append(
            f"mic clipped at capture (peak {mic_before['peak_dbfs']} dBFS); de-clip is "
            "best-effort — distortion on saturated samples is unrecoverable. Lower mic "
            "preamp/DAC line-out to −6…−12 dBFS peaks and re-capture."
        )
    if line_before["clipping_detected"]:
        warnings.append(
            f"line clipped at capture (peak {line_before['peak_dbfs']} dBFS); de-clip is "
            "best-effort. Lower DAC line-out and re-capture."
        )

    # ---- after-levels --------------------------------------------------------
    mic_after = _levels(mic_out, cfg)
    line_after = _levels(line_out, cfg)

    report = {
        "schema_version": SCHEMA_VERSION,
        "meeting_id": manifest.meeting_id,
        "generated_at": _utcnow(),
        "aec_enabled": aec_enabled_effective,
        "delay_applied_sec": round(delay, 4),
        "delay_source": delay_source,
        "estimated_echo_reduction_db": aec_info["reduction_db"],
        "normalize_target": {"type": "peak", "value_dbfs": cfg.normalize_target_dbfs},
        "resample_rate": cfg.resample_rate,
        "channels": {
            "mic": {"before": mic_before, "after": mic_after,
                    "aec_applied": aec_enabled_effective},
            "line": {"before": line_before, "after": line_after,
                     "aec_applied": False},
        },
        "tool": {
            "aec": (aec_info["backend"] or "none"),
            "resample": f"ffmpeg {_ffmpeg_version(cfg.ffmpeg_path)}",
            "declip": "ffmpeg adeclip" if cfg.declip else "none",
        },
        "params": {
            "aec_enabled": aec_requested,
            "normalize_target": f"peak:{cfg.normalize_target_dbfs}dBFS",
            "resample_rate": cfg.resample_rate,
            "declip": cfg.declip,
            "highpass_hz": cfg.highpass_hz,
            "denoise": cfg.denoise,
            "xcorr_refine": cfg.xcorr_refine,
        },
        "warnings": warnings,
    }
    _write_json_atomic(out_dir / "preprocess.json", report)
    return PreprocessResult(report=report, processed_dir=out_dir, warnings=warnings)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="briefly preprocess",
        description="AEC + de-clip + normalize + 16 kHz resample (preprocess stage)",
    )
    p.add_argument("--meeting-id", required=True)
    p.add_argument("--recordings-dir", default="recordings",
                   help="recordings/ root OR recordings/<id>/ directly")
    p.add_argument("--processed-dir", default="processed",
                   help="processed/ root OR processed/<id>/ directly")
    p.add_argument("--no-aec", action="store_true",
                   help="disable reference-based AEC (closed-back/IEM monitoring)")
    p.add_argument("--config", help="JSON/YAML config file with PreprocessConfig overrides")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = PreprocessConfig.from_file(args.config) if args.config else PreprocessConfig()
    if args.no_aec:
        cfg.aec_enabled = False
    try:
        res = preprocess(args.meeting_id, args.recordings_dir, args.processed_dir, cfg)
    except PreprocessError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code

    rep = res.report
    print(f"meeting_id: {rep['meeting_id']}")
    print(f"processed:  {res.processed_dir}")
    print(f"  aec_enabled={rep['aec_enabled']}  delay={rep['delay_applied_sec']}s "
          f"({rep['delay_source']})  resample={rep['resample_rate']}Hz")
    for role in ("mic", "line"):
        ch = rep["channels"][role]
        print(f"  {role:4} peak {ch['before']['peak_dbfs']}→{ch['after']['peak_dbfs']} dBFS  "
              f"mean {ch['before']['mean_dbfs']}→{ch['after']['mean_dbfs']} dBFS  "
              f"aec_applied={ch['aec_applied']}")
    for w in res.warnings:
        print(f"  WARNING: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
