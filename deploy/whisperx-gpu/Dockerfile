# WhisperX GPU service for a workstation with an NVIDIA GPU + Docker Desktop.
# The +cu128 torch wheels bundle their own CUDA runtime, so this slim base is enough — you only
# need the NVIDIA driver + `--gpus all` (the WSL2 / nvidia-container-toolkit integration) on the
# host. No nvidia/cuda base image required. Runs whisperX 3.8.6 (transcribe+align AND pyannote
# diarization, both on CUDA) — see requirements.txt.
#
#   docker compose up --build        # (preferred — see compose.yaml)
#   docker build -t briefly-whisperx:gpu .
#
# Pin to Debian 12 (bookworm), NOT the floating python:3.11-slim. The latter now resolves to
# Debian 13 (trixie), whose glibc hard-errors on ctranslate2's executable-stack .so
# ("cannot enable executable stack as shared object requires"), crashing faster-whisper/WhisperX
# at import. bookworm's loader accepts it. (ctranslate2 is pulled in transitively by whisperx.)
FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache \
    HF_HUB_OFFLINE=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MPLCONFIGDIR=/tmp/mpl \
    NUMBA_CACHE_DIR=/tmp/numba \
    XDG_CACHE_HOME=/tmp/cache \
    WHISPERX_DEVICE=cuda \
    WHISPERX_MODEL=large-v2 \
    WHISPERX_COMPUTE_TYPE=float16

WORKDIR /app

# Install the CUDA build of torch/torchaudio/torchvision FIRST, from the cu128 wheel index.
# These wheels bundle the CUDA runtime (no nvidia/cuda base needed); the host just needs the
# NVIDIA driver + `--gpus all`. whisperx 3.8.6 requires torch~=2.8 and torchvision~=0.23; torch
# 2.8 has no +cu124 wheels, so we use cu128 (RTX 40-series Ada supports it). torchcodec (also a
# whisperx dep) is installed transitively by requirements.txt and uses the system ffmpeg libs.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.8.0" "torchaudio==2.8.0" "torchvision==0.23.0"

COPY requirements.txt .
# whisperx 3.8.6 pulls a coherent CUDA stack: faster-whisper>=1.2 + ctranslate2>=4.5 (cuDNN 9,
# matching torch 2.8) + pyannote.audio>=4.0 (can load speaker-diarization-community-1 / 3.1) +
# torchcodec + transformers. No ctranslate2 version-forcing needed (3.3.1's <4.5 cap is gone).
RUN pip install -r requirements.txt

# ctranslate2 (faster-whisper's CUDA backend) dlopen's cuDNN + cuBLAS at transcription time but
# does NOT find the copies torch bundles under site-packages/nvidia/*/lib by default. Without
# this, /asr crashes with "Could not load libcudnn_ops*.so". Point the loader at torch's nvidia
# libs (cuDNN 9 + cuBLAS). Also set in compose.yaml so it reliably reaches the container.
# (Declared after the pip layers so it doesn't bust the cached torch download.)
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib

COPY app/ /app/

# Non-root. /cache (HF_HOME) holds the model downloads (Whisper large-v2 ~3 GB, the
# wav2vec2 align model, and the gated diarization model); mount a named volume here so they
# download once. A fresh named volume inherits this dir's ownership on first use.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /cache && chown 10001:10001 /cache
USER 10001

EXPOSE 8000
# Single worker: WhisperX / pyannote inference is per-process and not thread-safe.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
