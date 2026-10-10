"""Batch runner: processes every pending front image, stage by stage, each stage in a fresh process.

A fresh process per stage loads one model once for all pending assets and returns all of its RAM and
GPU memory to the OS when it exits, so long batches cannot accumulate memory.
"""
import os
import subprocess
import sys
import threading
import time
from typing import List

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


def run_round(config_path: str, config: Config) -> int:
    """One pass over all pending assets. Returns how many assets had work."""
    worked = set()
    for stage in STAGES:
        queue = [asset for asset in asset_store.pending(config)
                 if not asset_store.blocked(asset, config) and asset.needs(stage, config)
                 and all(not asset.needs(previous, config) for previous in STAGES[:STAGES.index(stage)])]
        if not queue:
            continue
        keys = [asset.key for asset in queue]
        print(f"\n=== {stage}: {', '.join(keys)}", flush=True)
        code = _run_stage(stage, keys, config_path, config)
        if code != 0:
            print(f"[{stage}] process exited with code {code}"
                  + (" (killed: out of memory or timeout)" if code < 0 else ""), flush=True)
        # A crash leaves the asset that was running marked; charge the failure to it.
        for asset in queue:
            status = asset_store.read_status(asset)
            if status.get("running") == stage:
                asset_store.fail(asset, stage, f"process exited with code {code}")
        worked.update(keys)
    for asset in asset_store.pending(config):
        if asset_store.blocked(asset, config):
            print(f"SKIPPED {asset.key}: failed {config.max_failures} times "
                  f"({asset_store.read_status(asset).get('last_error')}). "
                  f"Delete {asset.work}/status.json to retry.", flush=True)
    return len(worked)


def main(config_path: str, watch: bool = True) -> None:
    from .config import load
    config = load(config_path)
    idle_since = time.time()
    while True:
        started = time.time()
        if run_round(config_path, config):
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
    arguments = parser.parse_args()
    main(arguments.config, watch=not arguments.once)
