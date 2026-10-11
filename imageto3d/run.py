"""Batch runner: processes every pending front image, stage by stage, each stage in a fresh process.

A fresh process per stage loads one model once for all pending assets and returns all of its RAM and
GPU memory to the OS when it exits, so long batches cannot accumulate memory.
"""
import os
import subprocess
import sys
import threading
import time
from typing import List, Optional

from . import assets as asset_store
from .config import Config

STAGES = ("views", "shape", "paint", "export")
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_stage(stage: str, keys: List[str], config_path: str, config: Config) -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [PACKAGE_ROOT, config.mv_repo, config.hunyuan_repo, env.get("PYTHONPATH", "")])
    command = [sys.executable, "-u", "-m", f"imageto3d.stage_{stage}", "--config", config_path, "--keys", *keys]
    process = subprocess.Popen(command, env=env, cwd=PACKAGE_ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # A hung stage (for example a stalled Drive read) is killed so the batch can move on.
    watchdog = threading.Timer(config.stage_timeout_minutes * 60, process.kill)
    watchdog.start()
    try:
        for line in process.stdout:
            # Hide progress-bar noise; keep stage messages and errors.
            if "it/s]" not in line and "B/s]" not in line and "s/it]" not in line:
                print(line, end="", flush=True)
    finally:
        watchdog.cancel()
    return process.wait()


def run_round(config_path: str, config: Config, only: Optional[List[str]] = None) -> int:
    """One pass over all pending assets (or only those keys). Returns how many assets had work."""
    worked = set()
    for stage in STAGES:
        if config.deadline and time.time() >= config.deadline:
            break
        queue = [asset for asset in asset_store.pending(config)
                 if (only is None or asset.key in only)
                 and not asset_store.blocked(asset, config) and asset.needs(stage, config)
                 and all(not asset.needs(previous, config) for previous in STAGES[:STAGES.index(stage)])]
        if not queue:
            continue
        # Assets that failed before go last, so a crashing image cannot hold back the rest of its group.
        queue.sort(key=lambda asset: asset_store.read_status(asset).get("failures", 0))
        # Views and shape load different models per mode, so each mode gets its own process.
        groups = ([[asset for asset in queue if asset.mode == mode] for mode in ("multiview", "single")]
                  if stage in ("views", "shape") else [queue])
        for group in filter(None, groups):
            keys = [asset.key for asset in group]
            print(f"\n=== {stage}: {', '.join(keys)}", flush=True)
            code = _run_stage(stage, keys, config_path, config)
            if code != 0:
                print(f"[{stage}] process exited with code {code}"
                      + (" (killed: out of memory or timeout)" if code < 0 else ""), flush=True)
            # A crash leaves the asset that was running marked; charge the failure to it.
            for asset in group:
                if asset_store.read_status(asset).get("running") == stage:
                    asset_store.fail(asset, stage, f"process exited with code {code}")
            worked.update(keys)
    for asset in asset_store.pending(config):
        if (only is None or asset.key in only) and asset_store.blocked(asset, config):
            print(f"SKIPPED {asset.key}: failed {config.max_failures} times "
                  f"({asset_store.read_status(asset).get('last_error')}). "
                  f"Delete {asset.work}/status.json to retry.", flush=True)
    return len(worked)


def claim_batch(config: Config, worker: Optional[int]) -> List[asset_store.Asset]:
    """Take up to claim_batch assets with work left, oldest first, assets that failed before last.

    With a worker number the assets are claimed, so other GPU workers skip them until they are released.
    """
    candidates = []
    for asset in asset_store.pending(config):
        if asset_store.blocked(asset, config):
            asset_store._warn_once(f"blocked:{asset.key}", (
                f"SKIPPED {asset.key}: failed {config.max_failures} times "
                f"({asset_store.read_status(asset).get('last_error')}). Delete {asset.work}/status.json to retry."))
        elif any(asset.needs(stage, config) for stage in STAGES):
            candidates.append(asset)
    candidates.sort(key=lambda asset: asset_store.read_status(asset).get("failures", 0))
    claimed = []
    for asset in candidates:
        if len(claimed) >= config.claim_batch:
            break
        if worker is None or asset_store.claim(asset, worker):
            claimed.append(asset)
    return claimed


def work_once(config_path: str, config: Config, worker: Optional[int]) -> int:
    """One round over the next claim_batch assets (claimed when one of several GPU workers)."""
    claimed = claim_batch(config, worker)
    if not claimed:
        return 0
    try:
        return run_round(config_path, config, [asset.key for asset in claimed])
    finally:
        if worker is not None:
            for asset in claimed:
                asset_store.release(asset)


def main(config_path: str, watch: bool = True, worker: Optional[int] = None) -> None:
    from .config import load
    config = load(config_path)
    idle_since = time.time()
    while True:
        if config.deadline and time.time() >= config.deadline:
            print("Session time limit reached; stopping so results can be saved.", flush=True)
            break
        started = time.time()
        if work_once(config_path, config, worker):
            idle_since = time.time()
            print(f"\nRound finished in {(time.time() - started) / 60:.1f} min. Outputs: {config.out_dir}", flush=True)
            continue  # New files may have arrived while this round ran.
        if not watch:
            break
        idle_minutes = (time.time() - idle_since) / 60
        if idle_minutes >= config.idle_stop_minutes:
            print(f"No new front images for {config.idle_stop_minutes} min; stopping.", flush=True)
            break
        print(f"Waiting for new front images in {config.input_dir} "
              f"(stops after {config.idle_stop_minutes - idle_minutes:.0f} more idle min)...", flush=True)
        time.sleep(config.poll_seconds)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", action="store_true")
    # Set by the Kaggle notebook: one worker per GPU (CUDA_VISIBLE_DEVICES), sharing the queue through claims.
    parser.add_argument("--worker", type=int)
    arguments = parser.parse_args()
    main(arguments.config, watch=not arguments.once, worker=arguments.worker)
