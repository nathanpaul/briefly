"""Reference-based acoustic echo cancellation (AEC).

Cancels the leaked far-end (LINE / remote) audio from the near-end (MIC / "Me") channel.
The LINE channel is the exact far-end reference, so a least-squares (Wiener) estimate of the
echo path ref→mic, subtracted from the mic, removes the measured headphone→mic leakage
(a self-keyed noise gate cannot — verified).

The leakage path (same headphones/position for a meeting) is time-invariant, so one FIR
filter estimated on a capped window and applied to the whole signal is both correct and
robust (no adaptive divergence). numpy-only; optional extra (`pip install 'briefly[aec]'`).
"""
from __future__ import annotations

import wave
from pathlib import Path


def _read_wav_mono(path: str | Path):
    """Read a 16-bit PCM WAV as float32 in [-1, 1] + rate (downmix to mono). Non-16-bit
    raises ValueError so the caller can fall back to passthrough."""
    import numpy as np

    with wave.open(str(path), "rb") as w:
        rate, nch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"AEC expects 16-bit PCM, got {width * 8}-bit")
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    return a, rate


def _write_wav_mono(path: str | Path, samples, rate: int) -> None:
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())


def cancel_echo(mic, ref, sample_rate: int = 16000, taps: int | None = None,
                est_seconds: float = 120.0, reg: float = 1e-2):
    """Estimate the FIR echo path ref→mic (least squares / Wiener) and subtract it.

    `mic` = near-end (+echo), `ref` = far-end reference (time-aligned). Returns
    (cleaned_mic_float32, erle_db). Near-end speech is uncorrelated with `ref`, so it is
    preserved; the leaked far-end is removed."""
    import numpy as np

    n = int(min(len(mic), len(ref)))
    if n == 0:
        return np.asarray(mic, dtype=np.float32), 0.0
    mic = np.asarray(mic[:n], dtype=np.float64)
    ref = np.asarray(ref[:n], dtype=np.float64)
    L = int(taps or max(256, sample_rate // 40))   # ~25 ms filter (covers path + residual delay)
    L = min(L, n)

    # ---- estimate h on a capped window (path is time-invariant) -----------------
    m = int(min(n, max(L * 4, int(est_seconds * sample_rate))))
    nfft = 1 << int(np.ceil(np.log2(2 * m)))
    Rf = np.fft.rfft(ref[:m], nfft)
    Mf = np.fft.rfft(mic[:m], nfft)
    rauto = np.fft.irfft(Rf * np.conj(Rf), nfft)[:L]            # ref autocorrelation
    pxc = np.fft.irfft(Mf * np.conj(Rf), nfft)[:L]              # mic-vs-ref cross-correlation
    Rmat = rauto[np.abs(np.subtract.outer(np.arange(L), np.arange(L)))]  # Toeplitz
    Rmat[np.diag_indices(L)] += reg * (rauto[0] + 1e-9)         # regularize
    try:
        h = np.linalg.solve(Rmat, pxc)
    except np.linalg.LinAlgError:
        h = np.zeros(L)

    # ---- apply h to the whole reference via overlap-add; subtract from mic ------
    out = mic.copy()
    B = 1 << 16
    nf = B + L - 1
    Hf = np.fft.rfft(h, nf)
    carry = np.zeros(L - 1)
    pos = 0
    while pos < n:
        seg = ref[pos:pos + B]
        bb = len(seg)
        y = np.fft.irfft(np.fft.rfft(seg, nf) * Hf, nf)[:bb + L - 1]
        if carry.size:
            y[:L - 1] += carry
        out[pos:pos + bb] -= y[:bb]
        carry = y[bb:bb + L - 1].copy()
        pos += bb

    mic_e = float(mic @ mic) + 1e-12
    res_e = float(out @ out) + 1e-12
    erle = float(max(0.0, min(60.0, 10.0 * np.log10(mic_e / res_e))))
    return out.astype(np.float32), erle


def run_aec_file(mic_path, line_path, delay_sec: float, out_path) -> dict:
    """Align LINE to MIC by `delay_sec`, cancel the echo, and write the cleaned mic as a
    16-bit mono WAV at the mic's native rate. Returns {applied, reduction_db, backend}.
    `common_t = local_t + start_offset_sec`, so line lagging mic by `delay_sec` shifts the
    reference later by that many samples (zero-padded front)."""
    import numpy as np

    mic, rate = _read_wav_mono(mic_path)
    ref, ref_rate = _read_wav_mono(line_path)
    if ref_rate != rate:  # shares a USB clock normally; resample defensively
        idx = np.arange(int(len(ref) * rate / ref_rate)) * ref_rate / rate
        ref = np.interp(idx, np.arange(len(ref)), ref).astype(np.float32)
    d = int(round(float(delay_sec) * rate))
    if d > 0:
        ref = np.concatenate([np.zeros(d, dtype=ref.dtype), ref])
    elif d < 0:
        ref = ref[-d:]
    cleaned, erle = cancel_echo(mic, ref, sample_rate=rate)
    _write_wav_mono(out_path, cleaned, rate)
    return {"applied": True, "reduction_db": round(erle, 2), "backend": "wiener-numpy"}
