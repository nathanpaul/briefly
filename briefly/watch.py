"""Auto-trigger — watch recordings/ for newly finalized meetings and run the pipeline.

The capture stage writes `meeting.json` only when a recording is finalized, so it is the
"done" sentinel. We watch the local, non-synced `recordings/` dir for it — NOT the vault.
Single-worker (sequential) polling, stdlib-only; a stage that's already reached the target
is skipped (orchestrator idempotency), so double-runs are safe.

Runs the data pipeline to `merge`; turn meetings into vault notes afterwards with
`briefly summarize`.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from .orchestrator import DONE, STAGES, PipelineConfig, load_config, run_pipeline


@dataclass
class WatchConfig:
    interval: float = 10.0
    to_stage: str = "merge"


def find_pending(cfg: PipelineConfig, to_stage: str, ledger: set[str]) -> list[str]:
    """Finalized meetings whose pipeline hasn't reached `to_stage` and that we haven't
    already given up on this run."""
    root = Path(cfg.data_root) / "recordings"
    if not root.exists():
        return []
    pending = []
    for d in sorted(root.iterdir()):
        mid = d.name
        if not (d / "meeting.json").exists():   # capture still in progress (no sentinel yet)
            continue
        if mid in ledger or DONE[to_stage](cfg, mid):
            continue
        pending.append(mid)
    return pending


def watch_once(cfg: PipelineConfig, to_stage: str, run=run_pipeline,
               ledger: set[str] | None = None, log=print) -> list[tuple[str, str]]:
    """Process all currently-pending meetings once (sequentially). Returns [(mid, status)]."""
    ledger = ledger if ledger is not None else set()
    results: list[tuple[str, str]] = []
    for mid in find_pending(cfg, to_stage, ledger):
        log(f"[watch] new meeting {mid} -> running pipeline to {to_stage}")
        try:
            run(cfg, mid, "preprocess", to_stage, False)
            results.append((mid, "ok"))
            log(f"[watch] {mid} done")
        except Exception as e:  # noqa: BLE001 - one bad meeting must not kill the watcher
            ledger.add(mid)     # don't hammer it; retried on watcher restart
            results.append((mid, "error"))
            log(f"[watch] {mid} FAILED: {type(e).__name__}: {e}")
    return results


def watch_loop(cfg: PipelineConfig, wcfg: WatchConfig, run=run_pipeline, log=print,
               should_stop=None) -> None:
    ledger: set[str] = set()
    log(f"[watch] watching {Path(cfg.data_root) / 'recordings'} "
        f"every {wcfg.interval:g}s -> stage '{wcfg.to_stage}' (Ctrl-C to stop)")
    while True:
        watch_once(cfg, wcfg.to_stage, run=run, ledger=ledger, log=log)
        if should_stop is not None and should_stop():
            return
        time.sleep(wcfg.interval)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="briefly watch",
                                description="auto-run the pipeline on newly captured meetings")
    p.add_argument("--config")
    p.add_argument("--data-root")
    p.add_argument("--vault-dir")
    p.add_argument("--whisper-host")
    p.add_argument("--whisper-port", type=int)
    p.add_argument("--diarize-url")
    p.add_argument("--to", dest="to_stage", default="merge", choices=STAGES)
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--once", action="store_true", help="process current pending meetings and exit")
    args = p.parse_args(argv)
    cfg = load_config(args.config, {
        "data_root": args.data_root, "vault_dir": args.vault_dir,
        "whisper_host": args.whisper_host, "whisper_port": args.whisper_port,
        "diarize_url": args.diarize_url,
    })
    wcfg = WatchConfig(interval=args.interval, to_stage=args.to_stage)
    if args.once:
        watch_once(cfg, wcfg.to_stage)
        return 0
    try:
        watch_loop(cfg, wcfg)
    except KeyboardInterrupt:
        print("\n[watch] stopped")
    return 0
