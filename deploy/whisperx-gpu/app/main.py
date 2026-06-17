"""WhisperX GPU service — transcribe+align and diarize as TWO separate endpoints.

WhisperX (https://github.com/m-bain/whisperX) keeps transcription and diarization as
distinct stages (transcribe -> wav2vec2 align, then a separate pyannote diarization that is
assigned back onto words). This service mirrors that split and exposes the two stages
independently, so Briefly drives them as separate, independently re-runnable steps:

  POST /asr      faster-whisper ASR -> wav2vec2 forced alignment   (NO diarization)
                 -> { model, language, duration_sec, processing_sec, device,
                      segments:[{start,end,text,words:[{start,end,word,score}]}] }

  POST /diarize  pyannote diarization ONLY (WhisperX's DiarizationPipeline)
                 -> { model, duration_sec, num_speakers, processing_sec, rt_factor,
                      segments:[{speaker,start,end}] }

The two endpoints match what Briefly's clients expect:
  * /asr      == briefly/clients/asr.py        (same /asr contract as the cluster
                 faster-whisper service — point TRANSCRIBE_SERVICE_URL at …/asr)
  * /diarize  == briefly/clients/diarize.py     (byte-for-byte the SAME response shape as the
                 homelab pyannote `speaker-diarization` service — point DIARIZE_URL at
                 …/diarize; the two diarizers are interchangeable drop-ins)

Design notes:
  * The ASR model + the align model are loaded ONCE at startup in a background thread; the
    diarization pipeline is built there too (it needs HF_TOKEN for the gated model).
    `/readyz` gates on the ASR model only (503 until warm) so transcription is usable even
    without a token. Align models are loaded lazily and cached per language code.
  * /diarize without a usable pipeline (no HF_TOKEN / build failed) returns 503 with a clear
    message; /asr keeps working — the box is useful for transcription regardless.
  * pyannote (and the WhisperX inference path) is NOT thread-safe -> all inference runs on a
    single worker thread, serialized by an asyncio lock, exactly like the homelab
    speaker-diarization app. Scale out with replicas, not threads. Single uvicorn worker.
  * CUDA-first: device defaults to cuda. Falls back to CPU (slowly) with a warning if CUDA
    isn't visible, so a misconfigured box still answers.
"""

import asyncio
import logging
import os
import shutil
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

import torch  # noqa: E402

# --- torch 2.6 compatibility: force weights_only=False for trusted-checkpoint loads ----------
# whisperx / pyannote / pytorch-lightning call torch.load() on checkpoints that pickle omegaconf
# objects (e.g. ListConfig). torch 2.6 flipped torch.load's default to weights_only=True, which
# rejects those globals and crashes model loading at startup ("Weights only load failed ...
# omegaconf.listconfig.ListConfig"). Lightning passes weights_only=True *explicitly*, so we FORCE
# it back to False. These checkpoints come from trusted HF/torchaudio sources (Systran
# faster-whisper, pyannote, speechbrain, wav2vec2), so disabling the guard is safe here. Must run
# before whisperx is imported (it is imported lazily inside the load functions below).
_torch_load_orig = torch.load
def _torch_load_trusted(*args, **kwargs):  # noqa: E302
    kwargs["weights_only"] = False
    return _torch_load_orig(*args, **kwargs)
torch.load = _torch_load_trusted

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("whisperx")

# --- Configuration (env-driven; defaults documented in the README) -----------
MODEL_ID = os.environ.get("WHISPERX_MODEL", "large-v2")
COMPUTE_TYPE = os.environ.get("WHISPERX_COMPUTE_TYPE", "float16")
DEFAULT_LANGUAGE = os.environ.get("WHISPERX_LANGUAGE", "en")
BATCH_SIZE = int(os.environ.get("WHISPERX_BATCH_SIZE", "16"))
HF_TOKEN = os.environ.get("HF_TOKEN")  # required only for diarization (gated model)
# The pyannote diarization model WhisperX's DiarizationPipeline loads. Surfaced as the
# `model` field of /diarize so the response matches the homelab pyannote service shape.
DIARIZE_MODEL_ID = os.environ.get(
    "WHISPERX_DIARIZE_MODEL", "pyannote/speaker-diarization-3.1"
)
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(512 * 1024 * 1024)))
UPLOAD_CHUNK = 1024 * 1024  # 1 MiB

# --- Device selection --------------------------------------------------------
# WHISPERX_DEVICE=cuda runs on an NVIDIA GPU (e.g. an RTX 4080 Super). Falls back to CPU
# with a warning if CUDA isn't visible (driver/--gpus missing), so a misconfigured box
# still works, just slowly. WhisperX wants a plain device string ("cuda"/"cpu").
_WANT = os.environ.get("WHISPERX_DEVICE", "cuda").lower()
if _WANT == "cuda" and not torch.cuda.is_available():
    log.warning("WHISPERX_DEVICE=cuda but torch.cuda.is_available() is False -> using CPU. "
                "Check the NVIDIA driver + `--gpus all` + a CUDA build of torch.")
DEVICE = "cuda" if (_WANT == "cuda" and torch.cuda.is_available()) else "cpu"
# faster-whisper has no float16 kernels on CPU; downgrade so a CPU fallback still loads.
if DEVICE == "cpu" and COMPUTE_TYPE == "float16":
    log.warning("compute_type=float16 unsupported on CPU -> using int8")
    COMPUTE_TYPE = "int8"

# --- Module state ------------------------------------------------------------
_model = None                       # whisperx ASR model (faster-whisper backend)
_diarize_pipeline = None            # whisperx.diarize.DiarizationPipeline (built once)
_align_cache: dict[str, tuple] = {}  # language_code -> (align_model, align_metadata)
_ready = False                      # ASR model loaded -> /readyz 200; /asr usable
_load_error: str | None = None
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisperx")
_infer_lock = asyncio.Lock()


def _get_align(language_code: str):
    """Lazily load + cache the wav2vec2 alignment model for a language. Called only from
    the single inference thread, so the dict access needs no extra locking."""
    import whisperx

    cached = _align_cache.get(language_code)
    if cached is None:
        log.info("loading align model language=%s device=%s", language_code, DEVICE)
        model_a, meta = whisperx.load_align_model(language_code=language_code, device=DEVICE)
        cached = (model_a, meta)
        _align_cache[language_code] = cached
    return cached


def _load_models() -> None:
    """Load + warm the ASR model and the align model, then build the diarization pipeline.
    Runs once at startup in a background thread. /readyz gates on the ASR model only, so
    transcription is available even when HF_TOKEN is missing (diarization then 503s)."""
    global _model, _diarize_pipeline, _ready, _load_error
    try:
        import whisperx

        t0 = time.monotonic()
        log.info("loading ASR model=%s device=%s compute_type=%s", MODEL_ID, DEVICE, COMPUTE_TYPE)
        _model = whisperx.load_model(MODEL_ID, DEVICE, compute_type=COMPUTE_TYPE)
        if DEVICE == "cuda":
            log.info("CUDA device: %s", torch.cuda.get_device_name(0))

        # Pre-load the default-language align model so the first /asr call isn't cold.
        try:
            _get_align(DEFAULT_LANGUAGE)
        except Exception:  # noqa: BLE001 - non-fatal; it'll retry per request
            log.warning("could not pre-load align model for %s; will load on demand",
                        DEFAULT_LANGUAGE, exc_info=True)

        # ASR is ready now; mark readiness before the (optional) diarization build so a
        # missing/failing token never blocks transcription.
        _ready = True

        # Build the diarization pipeline once (needs the HF token for the gated model).
        # Optional: if no token, /diarize 503s clearly but /asr still works.
        if HF_TOKEN:
            try:
                from whisperx.diarize import DiarizationPipeline

                # whisperx 3.8.6's DiarizationPipeline takes `token=` (older 3.3.x used the now
                # removed `use_auth_token=`). pyannote.audio 4.x can load the gated
                # speaker-diarization-community-1 / 3.1 pipelines with this token.
                _diarize_pipeline = DiarizationPipeline(
                    model_name=DIARIZE_MODEL_ID, token=HF_TOKEN, device=DEVICE,
                )
                log.info("diarization pipeline ready model=%s", DIARIZE_MODEL_ID)
            except Exception:  # noqa: BLE001 - keep ASR available even if diarize fails to build
                log.warning("could not build diarization pipeline; /diarize will 503",
                            exc_info=True)
        else:
            log.warning("HF_TOKEN not set -> /diarize will 503 (transcription still works)")

        log.info("service ready in %.1fs on %s", time.monotonic() - t0, DEVICE)
    except Exception as exc:  # noqa: BLE001 - surface any load failure to readiness
        _load_error = repr(exc)
        log.exception("model load failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _load_models)
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="whisperx-gpu", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    # 200 once the ASR model is loaded — transcription is the baseline capability and does
    # not depend on the diarization pipeline (which may be intentionally absent).
    if _ready:
        return {"status": "ready", "model": MODEL_ID, "device": DEVICE}
    body: dict = {"status": "loading", "model": MODEL_ID}
    if _load_error:
        body = {"status": "error", "model": MODEL_ID, "error": _load_error}
    return JSONResponse(status_code=503, content=body)


def _as_float(v) -> float | None:
    """WhisperX leaves start/end/score absent on un-aligned tokens; coerce safely."""
    if v is None:
        return None
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return None


# --- /asr : transcribe + word-align ONLY (no diarization) --------------------
def _run_asr(wav_path: str, language: str, model_override: str | None) -> dict:
    """Blocking transcribe -> align. MUST run on the single worker thread (WhisperX
    inference is not thread-safe). No diarization here — that is the /diarize endpoint."""
    import whisperx

    model = _model
    used_model = MODEL_ID
    if model_override and model_override != MODEL_ID:
        # Per-request model override: load ad-hoc (not cached — the default stays warm).
        log.info("loading override model=%s", model_override)
        model = whisperx.load_model(model_override, DEVICE, compute_type=COMPUTE_TYPE)
        used_model = model_override

    audio = whisperx.load_audio(wav_path)
    duration_sec = float(len(audio)) / 16000.0  # whisperx.load_audio resamples to 16 kHz

    t0 = time.monotonic()
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=language)
    detected_language = result.get("language", language)

    # Forced alignment -> word timestamps.
    align_model, align_meta = _get_align(detected_language)
    result = whisperx.align(
        result["segments"], align_model, align_meta, audio, DEVICE,
        return_char_alignments=False,
    )
    processing_sec = time.monotonic() - t0

    segments = []
    for seg in result.get("segments", []):
        words = []
        for w in seg.get("words", []) or []:
            words.append({
                "start": _as_float(w.get("start")),
                "end": _as_float(w.get("end")),
                "word": w.get("word", ""),
                "score": _as_float(w.get("score")),
            })
        segments.append({
            "start": _as_float(seg.get("start")),
            "end": _as_float(seg.get("end")),
            "text": (seg.get("text") or "").strip(),
            "words": words,
        })

    return {
        "model": used_model,
        "language": detected_language,
        "duration_sec": round(duration_sec, 3),
        "processing_sec": round(processing_sec, 3),
        "device": DEVICE,
        "segments": segments,
    }


@app.post("/asr")
async def asr(
    audio: UploadFile = File(...),
    language: str = Form(DEFAULT_LANGUAGE),
    model: str | None = Form(None),
    # Accepted-but-ignored: diarization is now the separate /diarize endpoint. Kept in the
    # signature so older callers that still send these fields don't get a 422.
    diarize: str | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    if not _ready:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    tmpdir = tempfile.mkdtemp(prefix="whisperx-", dir="/tmp")
    raw_path = os.path.join(tmpdir, "input.wav")
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

        loop = asyncio.get_running_loop()
        async with _infer_lock:
            return await loop.run_in_executor(
                _executor, _run_asr, raw_path, language, model,
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- /diarize : diarization ONLY (drop-in for the homelab pyannote service) --
def _run_diarization(wav_path: str, hints: dict) -> tuple[list[dict], float, float]:
    """Blocking pyannote diarization via WhisperX's DiarizationPipeline. MUST run on the
    single worker thread. Returns ({speaker,start,end} segments, duration_sec, processing_sec)
    — the same shape the homelab speaker-diarization service produces."""
    import whisperx

    audio = whisperx.load_audio(wav_path)
    duration_sec = float(len(audio)) / 16000.0  # whisperx.load_audio resamples to 16 kHz

    t0 = time.monotonic()
    diarize_df = _diarize_pipeline(audio, **hints)
    processing_sec = time.monotonic() - t0

    # DiarizationPipeline returns a pandas DataFrame with columns
    # [segment, label, speaker, start, end] (one row per turn). Collapse to {speaker,start,end}.
    segments: list[dict] = []
    for row in diarize_df.itertuples(index=False):
        segments.append({
            "speaker": getattr(row, "speaker"),
            "start": round(float(getattr(row, "start")), 3),
            "end": round(float(getattr(row, "end")), 3),
        })
    segments.sort(key=lambda s: (s["start"], s["end"]))
    return segments, duration_sec, processing_sec


@app.post("/diarize")
async def diarize_endpoint(
    audio: UploadFile = File(...),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    if not _ready:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    if _diarize_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="diarization unavailable: set HF_TOKEN and accept the "
                   f"{DIARIZE_MODEL_ID} terms (see .env.example)",
        )

    hints: dict = {}
    if num_speakers is not None:
        hints["num_speakers"] = num_speakers
    if min_speakers is not None:
        hints["min_speakers"] = min_speakers
    if max_speakers is not None:
        hints["max_speakers"] = max_speakers

    tmpdir = tempfile.mkdtemp(prefix="diarize-", dir="/tmp")
    raw_path = os.path.join(tmpdir, "input.wav")
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

        loop = asyncio.get_running_loop()
        async with _infer_lock:
            segments, duration_sec, processing_sec = await loop.run_in_executor(
                _executor, _run_diarization, raw_path, hints
            )

        num_spk = len({s["speaker"] for s in segments})
        rt_factor = (processing_sec / duration_sec) if duration_sec > 0 else None
        return {
            "model": DIARIZE_MODEL_ID,
            "duration_sec": round(duration_sec, 3),
            "num_speakers": num_spk,
            "processing_sec": round(processing_sec, 3),
            "rt_factor": round(rt_factor, 4) if rt_factor is not None else None,
            "segments": segments,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
