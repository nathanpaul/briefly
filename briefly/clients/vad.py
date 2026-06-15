"""Energy-based voice-activity segmentation + PCM helpers (stdlib; numpy-accelerated when
available). Used to split the mic ("Me") channel into utterances for transcription, since
the Wyoming Whisper service returns no timestamps."""
from __future__ import annotations

import math
import wave
from array import array


def read_pcm16_mono(path) -> tuple[array, int]:
    """Read a 16-bit PCM WAV as a signed-short array (downmix to mono) + sample rate."""
    with wave.open(str(path), "rb") as w:
        rate, nch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = array("h")
    a.frombytes(raw)
    if nch > 1:
        a = array("h", [sum(a[i:i + nch]) // nch for i in range(0, len(a) - len(a) % nch, nch)])
    return a, rate


def slice_pcm(samples: array, rate: int, start: float, end: float) -> bytes:
    i0 = max(0, int(start * rate))
    i1 = min(len(samples), int(end * rate))
    return samples[i0:i1].tobytes()


def _frame_dbfs(samples: array, fl: int) -> list[float]:
    try:
        import numpy as np
        a = np.asarray(samples, dtype=np.float64)
        n = (len(a) // fl) * fl
        if n == 0:
            return []
        rms = np.sqrt((a[:n].reshape(-1, fl) ** 2).mean(axis=1))
        db = np.full(rms.shape, -120.0)
        nz = rms > 0
        db[nz] = 20.0 * np.log10(rms[nz] / 32768.0)
        return db.tolist()
    except ImportError:
        out = []
        for f in range(len(samples) // fl):
            seg = samples[f * fl:(f + 1) * fl]
            rms = (sum(s * s for s in seg) / len(seg)) ** 0.5 if seg else 0.0
            out.append(20.0 * math.log10(rms / 32768.0) if rms > 0 else -120.0)
        return out


def segment_speech(samples: array, rate: int, frame_ms: int = 30,
                   threshold_dbfs: float = -45.0, min_speech_sec: float = 0.3,
                   max_gap_sec: float = 0.4, pad_sec: float = 0.15) -> list[tuple[float, float]]:
    """Return [(start_sec, end_sec)] speech spans: frames above threshold, consecutive runs
    bridged across gaps <= max_gap_sec, padded, and short spans dropped."""
    fl = max(1, int(rate * frame_ms / 1000))
    speech = [d > threshold_dbfs for d in _frame_dbfs(samples, fl)]
    nframes = len(speech)
    max_gap = max(1, int(max_gap_sec * 1000 / frame_ms))
    dur = len(samples) / rate
    segs: list[tuple[float, float]] = []
    i = 0
    while i < nframes:
        if not speech[i]:
            i += 1
            continue
        j = i
        gap = 0
        k = i
        while k < nframes:
            if speech[k]:
                j, gap = k, 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            k += 1
        start = max(0.0, i * frame_ms / 1000 - pad_sec)
        end = min(dur, (j + 1) * frame_ms / 1000 + pad_sec)
        if end - start >= min_speech_sec:
            segs.append((round(start, 3), round(end, 3)))
        i = k
    return segs
