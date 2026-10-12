"""Kaggle without Drive access: gather input images from attached datasets or a downloaded shared folder,
and pack the results for download from the notebook's Output tab."""
import os
import shutil
import time
from typing import Iterable, List

from .assets import IMAGE_EXTENSIONS, parse_extra_view, parse_name


def collect_inputs(search_dirs: Iterable[str], input_dir: str) -> List[str]:
    """Copy correctly named images (main images and extra views) into input_dir.

    Images already there, or already finished here (renamed used-...), are skipped. Returns the copied names.
    """
    os.makedirs(input_dir, exist_ok=True)
    present = {name.lower() for name in os.listdir(input_dir)}
    copied = []
    for search_dir in search_dirs:
        for root, _, names in os.walk(search_dir):
            for name in sorted(names):
                if not name.lower().endswith(IMAGE_EXTENSIONS) or name.lower().startswith("used-"):
                    continue
                if name.lower() in present or f"used-{name}".lower() in present:
                    continue
                path = os.path.join(root, name)
                if parse_name(path) is None and parse_extra_view(path) is None:
                    continue
                shutil.copy2(path, os.path.join(input_dir, name))
                present.add(name.lower())
                copied.append(name)
    return copied


def finished_inputs(input_dir: str) -> List[str]:
    """Original names of the input images that were made into models (renamed used-... here)."""
    return sorted(name[len("used-"):] for name in os.listdir(input_dir) if name.startswith("used-"))


def zip_results(root: str, dest_dir: str) -> str:
    """Pack in/ (finished images renamed used-...) and out/ into one zip in dest_dir; returns its path."""
    import zipfile
    path = os.path.join(dest_dir, f"hunyuan3d-results-{time.strftime('%Y%m%d-%H%M')}.zip")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for folder in ("in", "out"):
            for name in sorted(os.listdir(os.path.join(root, folder))):
                bundle.write(os.path.join(root, folder, name), f"{folder}/{name}")
    return path
