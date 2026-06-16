# GPU speaker-diarization service for a workstation with an NVIDIA GPU + Docker Desktop.
# The cu124 torch wheel bundles its own CUDA runtime, so this slim base is enough — you
# only need the NVIDIA driver + `--gpus all` (the WSL2 / nvidia-container-toolkit
# integration) on the host. No nvidia/cuda base image required.
#
#   docker compose up --build        # (preferred — see compose.yaml)
#   docker build -t briefly-diarize:gpu .
FROM python:3.11-slim

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
    DIARIZE_DEVICE=cuda

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ /app/

# Non-root. /cache holds the (~1 GB) model download; mount a named volume here so it
# downloads once. A fresh named volume inherits this dir's ownership on first use.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /cache && chown 10001:10001 /cache
USER 10001

EXPOSE 8080
# Single worker: the pipeline is per-process and not thread-safe.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
