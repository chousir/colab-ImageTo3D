"""Brings shapes made by the TPU shape notebook into this pipeline's work folders.

The TPU notebook (kaggle_shape_tpu_v5e.ipynb) leaves work/<key>/front.png, shape.glb and shape_import.json in
its output. Attached to the 2x T4 notebook (Add Input > the TPU notebook's output), they are copied into the
matching work folders here, so those assets skip the views and shape stages and go straight to paint.
"""
import os
import shutil
from typing import Iterable, List

from . import assets as asset_store
from .config import Config

FILES = ("front.png", "shape.glb")


def import_shapes(search_dirs: Iterable[str], config: Config) -> List[str]:
    """Copy imported shapes for assets pending in config.input_dir. Returns the keys that were copied.

    Shapes already imported (same shape_import.json) are left alone, so running this again is cheap.
    """
    pending = {asset.key: asset for asset in asset_store.pending(config)}
    own_work = os.path.realpath(config.work_dir)
    imported = []
    for search_dir in search_dirs:
        for root, _, names in os.walk(search_dir):
            if asset_store.SHAPE_IMPORT_FILE not in names or os.path.realpath(os.path.dirname(root)) == own_work:
                continue
            asset = pending.get(os.path.basename(root))
            if asset is None or not all(os.path.exists(os.path.join(root, name)) for name in FILES):
                continue
            marker = os.path.join(root, asset_store.SHAPE_IMPORT_FILE)
            if os.path.exists(asset.shape_import_path) and _same(marker, asset.shape_import_path):
                continue
            os.makedirs(asset.work, exist_ok=True)
            # A texture painted on an earlier shape would look up to date next to the copied (older) shape.glb.
            if os.path.exists(asset.textured_path):
                os.remove(asset.textured_path)
            for name in FILES:
                shutil.copy2(os.path.join(root, name), os.path.join(asset.work, name))
            shutil.copy2(marker, asset.shape_import_path)  # last: the asset counts as imported once it exists
            imported.append(asset.key)
    return imported


def _same(first: str, second: str) -> bool:
    with open(first, "rb") as a, open(second, "rb") as b:
        return a.read() == b.read()
