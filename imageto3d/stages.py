"""Shared command-line wrapper for stage processes."""
import argparse
import time
import traceback
from typing import Callable, List

from . import assets as asset_store
from .config import Config, load


def run_stage(stage: str, load_model: Callable[[Config, List[asset_store.Asset]], object],
              process: Callable[[object, asset_store.Asset, Config], None]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    args = parser.parse_args()
    config = load(args.config)

    from . import runtime
    runtime.setup(config)
    queue: List[asset_store.Asset] = asset_store.find(config, args.keys)
    if not queue:
        return

    started = time.time()
    model = load_model(config, queue)
    runtime.release()
    runtime.report(f"{stage} model ready ({time.time() - started:.0f} s)")
    for asset in queue:
        if config.deadline and time.time() >= config.deadline:
            print(f"[{stage}] session time limit reached; leaving {asset.key} for the next session", flush=True)
            break
        asset_store.begin(asset, stage)
        started = time.time()
        try:
            process(model, asset, config)
        except Exception as error:  # One bad asset must not stop the batch.
            traceback.print_exc()
            asset_store.fail(asset, stage, f"{type(error).__name__}: {error}")
            print(f"[{stage}] {asset.key}: FAILED ({error})", flush=True)
        else:
            asset_store.succeed(asset, stage)
            print(f"[{stage}] {asset.key}: done in {time.time() - started:.0f} s", flush=True)
        runtime.release()
