"""Optionally start/stop the whisperX GPU docker container around the network stages.

Enabled from `.env`: set `MANAGE_GPU_DOCKER=1`. Before diarize/transcribe run, `docker compose
up -d` the service and wait for `/readyz`; after they finish, turn it off (`stop` by default).
If the service is already up, it is left running (we only turn off what we started). Everything
is best-effort and injectable so tests need no docker.

.env keys:
  MANAGE_GPU_DOCKER=1                                  enable (off by default)
  GPU_DOCKER_COMPOSE_FILE=deploy/whisperx-gpu/compose.yaml
  GPU_DOCKER_CONTEXT=paulgaming                        docker --context (a remote GPU box); optional
  GPU_DOCKER_SERVICE=whisperx                          compose service to up/stop
  GPU_READY_URL=http://host:8000/readyz               default: derived from TRANSCRIBE_SERVICE_URL
  GPU_DOCKER_READY_TIMEOUT=240                         seconds to wait for /readyz
  GPU_DOCKER_OFF=stop                                  how to turn off: stop (default) | down
"""
from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


def _truthy(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _ready_url_from(asr_url: str | None) -> str | None:
    """Derive `<scheme>://<host>/readyz` from the transcribe URL (…/asr → …/readyz)."""
    if not asr_url:
        return None
    p = urlsplit(asr_url)
    if not p.scheme or not p.netloc:
        return None
    return urlunsplit((p.scheme, p.netloc, "/readyz", "", ""))


@dataclass
class DockerServiceConfig:
    enabled: bool = False
    compose_file: str = "deploy/whisperx-gpu/compose.yaml"
    context: str | None = None
    service: str | None = "whisperx"
    ready_url: str | None = None
    ready_timeout: float = 240.0
    poll_interval: float = 3.0
    off_command: str = "stop"          # "stop" (turn off, keep) or "down" (remove)

    @classmethod
    def from_env(cls, asr_url: str | None = None, env: dict | None = None) -> "DockerServiceConfig":
        e = env if env is not None else os.environ
        off = (e.get("GPU_DOCKER_OFF", "stop") or "stop").strip().lower()
        return cls(
            enabled=_truthy(e.get("MANAGE_GPU_DOCKER")),
            compose_file=e.get("GPU_DOCKER_COMPOSE_FILE") or "deploy/whisperx-gpu/compose.yaml",
            context=e.get("GPU_DOCKER_CONTEXT") or None,
            service=e.get("GPU_DOCKER_SERVICE", "whisperx") or None,
            ready_url=e.get("GPU_READY_URL") or _ready_url_from(asr_url),
            ready_timeout=float(e.get("GPU_DOCKER_READY_TIMEOUT", "240") or 240),
            off_command="down" if off == "down" else "stop",
        )


def _base(cfg: DockerServiceConfig) -> list[str]:
    cmd = ["docker"]
    if cfg.context:
        cmd += ["--context", cfg.context]
    return cmd + ["compose", "-f", cfg.compose_file]


def _up_cmd(cfg: DockerServiceConfig) -> list[str]:
    return _base(cfg) + ["up", "-d"] + ([cfg.service] if cfg.service else [])


def _off_cmd(cfg: DockerServiceConfig) -> list[str]:
    # `down` tears down the whole project (no service arg); `stop` can target the one service.
    if cfg.off_command == "down":
        return _base(cfg) + ["down"]
    return _base(cfg) + ["stop"] + ([cfg.service] if cfg.service else [])


def is_ready(cfg: DockerServiceConfig, *, opener=urllib.request.urlopen, timeout: float = 3.0) -> bool:
    if not cfg.ready_url:
        return False
    try:
        with opener(cfg.ready_url, timeout=timeout) as r:
            return 200 <= getattr(r, "status", 200) < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def start(cfg: DockerServiceConfig, *, runner=subprocess.run, log=print,
          clock=time.monotonic, sleep=time.sleep, ready_fn=is_ready) -> None:
    """`docker compose up -d` then wait for /readyz. Raises RuntimeError on failure."""
    where = (cfg.service or "compose") + (f" @ {cfg.context}" if cfg.context else "")
    log(f"  starting GPU service ({where}) …")
    proc = runner(_up_cmd(cfg), capture_output=True, text=True)
    if getattr(proc, "returncode", 0) != 0:
        detail = ((getattr(proc, "stderr", "") or getattr(proc, "stdout", "")) or "").strip()[:400]
        log(f"  ✗ could not start GPU service: {detail}")
        raise RuntimeError(f"GPU service start failed: {detail}")
    deadline = clock() + cfg.ready_timeout
    while True:
        if ready_fn(cfg):
            log("  GPU service ready")
            return
        if clock() >= deadline:
            raise RuntimeError(
                f"GPU service did not become ready within {cfg.ready_timeout:.0f}s "
                f"(polled {cfg.ready_url})")
        sleep(cfg.poll_interval)


def stop(cfg: DockerServiceConfig, *, runner=subprocess.run, log=print) -> None:
    """Turn the service off — best-effort; never raises (it must not mask the run's result)."""
    log(f"  stopping GPU service ({cfg.off_command}) …")
    try:
        proc = runner(_off_cmd(cfg), capture_output=True, text=True)
        if getattr(proc, "returncode", 0) != 0:
            detail = ((getattr(proc, "stderr", "") or "") or "").strip()[:200]
            log(f"  warning: could not stop GPU service: {detail}")
    except OSError as e:
        log(f"  warning: could not stop GPU service: {e}")


@contextmanager
def managed(cfg: DockerServiceConfig, *, runner=subprocess.run, log=print, ready_fn=is_ready,
            clock=time.monotonic, sleep=time.sleep):
    """Ensure the service is up for the duration, then turn it off — but ONLY if we started it.
    A no-op when disabled or when the service was already up."""
    if not cfg.enabled:
        yield
        return
    if ready_fn(cfg):
        log("  (GPU service already up — leaving it running)")
        yield
        return
    start(cfg, runner=runner, log=log, clock=clock, sleep=sleep, ready_fn=ready_fn)
    try:
        yield
    finally:
        stop(cfg, runner=runner, log=log)
