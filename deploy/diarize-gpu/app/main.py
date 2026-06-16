"""Speaker-diarization microservice (pyannote.audio; CPU by default, CUDA via DIARIZE_DEVICE=cuda).

Diarization ONLY: who-spoke-when on a single audio channel. No transcription,
no word-to-speaker assignment. An external orchestrator POSTs one audio file
(the remote side of a recorded meeting) and gets back time-stamped speaker
segments labelled SPEAKER_00, SPEAKER_01, ...

Derived from k8s-homelab/my-apps/home/speaker-diarization/app/main.py. The ONLY change
is env-driven device selection (`DIARIZE_DEVICE=cuda` runs on an NVIDIA GPU); everything
else — the `/diarize` API, the `audio` field, and the response schema — is identical, so
Briefly needs no change (just point BRIEFLY_DIARIZE_URL at this box). The same one-line
device change is safe to upstream into the homelab service (default stays CPU).

Design notes:
  * The pyannote pipeline is loaded ONCE at startup and kept warm.
  * The pipeline is NOT thread-safe -> all inference runs on a single worker
    thread, serialized by an asyncio lock. Scale out with replicas, not threads.
  * On GPU the CPU thread counts below are irrelevant (the GPU does the work).
  * A fixed seed is set; residual non-determinism is expected.
"""

import asyncio
import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# --- CPU thread tuning: must happen before torch/numpy spin up their pools ----
_THREADS = int(
    os.environ.get("CPU_LIMIT")
    or os.environ.get("OMP_NUM_THREADS")
    or os.cpu_count()
    or 1
)
os.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_THREADS))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("diarizer")

# --- Configuration (env-driven; defaults documented in the README) -----------
MODEL_ID = os.environ.get("MODEL_ID", "pyannote/speaker-diarization-community-1")
HF_TOKEN = os.environ.get("HF_TOKEN")  # only needed when not running offline
SEED = int(os.environ.get("SEED", "42"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(512 * 1024 * 1024)))
TARGET_SR = 16000
UPLOAD_CHUNK = 1024 * 1024  # 1 MiB

# --- Device selection --------------------------------------------------------
# DIARIZE_DEVICE=cuda runs on an NVIDIA GPU (e.g. an RTX 4080 Super) and is ~30-80x
# faster than CPU on long meetings. Falls back to CPU with a warning if CUDA isn't
# visible (driver/--gpus missing), so a misconfigured box still works, just slowly.
_WANT = os.environ.get("DIARIZE_DEVICE", "cpu").lower()
if _WANT == "cuda" and not torch.cuda.is_available():
    log.warning("DIARIZE_DEVICE=cuda but torch.cuda.is_available() is False -> using CPU. "
                "Check the NVIDIA driver + `--gpus all` + a CUDA build of torch.")
DEVICE = torch.device("cuda" if (_WANT == "cuda" and torch.cuda.is_available()) else "cpu")

# --- Determinism -------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# --- Module state ------------------------------------------------------------
_pipeline = None
_ready = False
_load_error: str | None = None
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diarize")
_infer_lock = asyncio.Lock()


def _load_pipeline() -> None:
    """Load and warm the pyannote pipeline. Runs once in a background thread."""
    global _pipeline, _ready, _load_error
    try:
        from pyannote.audio import Pipeline

        t0 = time.monotonic()
        log.info("loading pipeline model_id=%s device=%s offline=%s",
                 MODEL_ID, DEVICE, os.environ.get("HF_HUB_OFFLINE"))
        pipe = Pipeline.from_pretrained(MODEL_ID, token=HF_TOKEN)
        if pipe is None:
            raise RuntimeError(
                "Pipeline.from_pretrained returned None -- usually a missing HF "
                "token or un-accepted model terms for " + MODEL_ID
            )
        pipe.to(DEVICE)
        if DEVICE.type == "cuda":
            log.info("CUDA device: %s", torch.cuda.get_device_name(0))
        _pipeline = pipe
        _ready = True
        log.info("pipeline ready in %.1fs on %s", time.monotonic() - t0, DEVICE)
    except Exception as exc:  # noqa: BLE001 - surface any load failure to readiness
        _load_error = repr(exc)
        log.exception("pipeline load failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _load_pipeline)
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="speaker-diarization-gpu", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    if _ready:
        return {"status": "ready", "model": MODEL_ID, "device": str(DEVICE)}
    body = {"status": "loading", "model": MODEL_ID}
    if _load_error:
        body = {"status": "error", "model": MODEL_ID, "error": _load_error}
    return JSONResponse(status_code=503, content=body)


def _decode_to_16k_mono(src_path: str, dst_path: str) -> None:
    """Decode + resample any ffmpeg-readable audio to 16 kHz mono PCM WAV."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", src_path,
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-f", "wav", "-y", dst_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(
            status_code=422,
            detail=f"ffmpeg could not decode the audio: {proc.stderr.strip()[:500]}",
        )


def _run_diarization(wav_path: str, hints: dict) -> tuple[list[dict], float, float]:
    """Blocking inference. MUST run on the single worker thread. pyannote moves each
    window batch onto the pipeline's device internally, so the waveform stays on CPU."""
    data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    duration_sec = float(len(data)) / float(sr)
    waveform = torch.from_numpy(data).unsqueeze(0)  # (channel=1, samples)

    t0 = time.monotonic()
    result = _pipeline({"waveform": waveform, "sample_rate": sr}, **hints)
    processing_sec = time.monotonic() - t0

    annotation = getattr(result, "speaker_diarization", result)
    segments = [
        {"speaker": label, "start": round(float(seg.start), 3), "end": round(float(seg.end), 3)}
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda s: (s["start"], s["end"]))
    return segments, duration_sec, processing_sec


@app.post("/diarize")
async def diarize(
    audio: UploadFile = File(...),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    if not _ready:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    hints: dict = {}
    if num_speakers is not None:
        hints["num_speakers"] = num_speakers
    if min_speakers is not None:
        hints["min_speakers"] = min_speakers
    if max_speakers is not None:
        hints["max_speakers"] = max_speakers

    tmpdir = tempfile.mkdtemp(prefix="diarize-", dir="/tmp")
    raw_path = os.path.join(tmpdir, "input")
    wav_path = os.path.join(tmpdir, "audio16k.wav")
    try:
        total = 0
        with open(raw_path, "wb") as fh:
            while True:
                chunk = await audio.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds MAX_UPLOAD_BYTES ({MAX_UPLOAD_BYTES} bytes)",
                    )
                fh.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="empty upload")

        _decode_to_16k_mono(raw_path, wav_path)

        loop = asyncio.get_running_loop()
        async with _infer_lock:
            segments, duration_sec, processing_sec = await loop.run_in_executor(
                _executor, _run_diarization, wav_path, hints
            )

        num_spk = len({s["speaker"] for s in segments})
        rt_factor = (processing_sec / duration_sec) if duration_sec > 0 else None
        return {
            "model": MODEL_ID,
            "duration_sec": round(duration_sec, 3),
            "num_speakers": num_spk,
            "processing_sec": round(processing_sec, 3),
            "rt_factor": round(rt_factor, 4) if rt_factor is not None else None,
            "segments": segments,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
