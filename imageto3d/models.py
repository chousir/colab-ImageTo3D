"""Model weights for a pre-downloaded model folder (the Kaggle model dataset).

The folder mirrors Hugging Face repo ids, e.g. <root>/tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/, which is
where Hunyuan3D looks first when HY3DGEN_MODELS=<root>, and where Config.model_source finds the diffusers
models. Only the files the pipeline loads are downloaded.
"""
import json
import os
import shutil
import time
import urllib.request
from typing import Dict, List, Optional

MANIFEST = "imageto3d-models.json"
REMBG_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/{}.onnx"
REMBG_MODELS = ("isnet-general-use", "u2net")


def _hunyuan(subfolder: str) -> List[str]:
    # Hunyuan3D shape models load <subfolder>/config.yaml and model.fp16.safetensors (variant fp16).
    return [f"{subfolder}/config.yaml", f"{subfolder}/model.fp16.safetensors"]


def files(include_turbo: bool = False) -> Dict[str, List[str]]:
    """Hugging Face repo id -> allow_patterns, matching what the stages load with the default settings."""
    shape_mv = ["hunyuan3d-dit-v2-mv"] + (["hunyuan3d-dit-v2-mv-turbo"] if include_turbo else [])
    shape_single = ["hunyuan3d-dit-v2-0"] + (["hunyuan3d-dit-v2-0-turbo", "hunyuan3d-vae-v2-0-turbo"]
                                            if include_turbo else [])
    # The paint pipeline loads two diffusers pipelines; take their whole folders, as upstream does.
    paint = ["hunyuan3d-delight-v2-0", "hunyuan3d-paint-v2-0"] + (["hunyuan3d-paint-v2-0-turbo"]
                                                                   if include_turbo else [])
    sdxl = ["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*"]
    for part in ("text_encoder", "text_encoder_2"):
        sdxl += [f"{part}/config.json", f"{part}/model.fp16.safetensors"]
    for part in ("unet", "vae"):
        sdxl += [f"{part}/config.json", f"{part}/diffusion_pytorch_model.fp16.safetensors"]
    return {
        "stabilityai/stable-diffusion-xl-base-1.0": sdxl,
        "madebyollin/sdxl-vae-fp16-fix": ["config.json", "diffusion_pytorch_model.safetensors"],
        "huanngzh/mv-adapter": ["mvadapter_i2mv_sdxl.safetensors"],
        "tencent/Hunyuan3D-2mv": [path for name in shape_mv for path in _hunyuan(name)],
        "tencent/Hunyuan3D-2": ([path for name in shape_single for path in _hunyuan(name)]
                                + [f"{name}/*" for name in paint]),
    }


def folder_size_gb(path: str) -> float:
    total = 0
    for root, _, names in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, name)) for name in names)
    return total / 1024**3


def download(root: str, include_turbo: bool = False, token: Optional[str] = None) -> None:
    """Download every needed file into root (plain files, no cache symlinks) and write the manifest."""
    from huggingface_hub import snapshot_download
    os.makedirs(root, exist_ok=True)
    spec = files(include_turbo)
    for repo, patterns in spec.items():
        started = time.time()
        target = os.path.join(root, repo)
        snapshot_download(repo_id=repo, local_dir=target, allow_patterns=patterns, token=token)
        size = folder_size_gb(target)
        seconds = max(time.time() - started, 1)
        print(f"{repo}: {size:.1f} GB in {seconds / 60:.1f} min ({size * 1024 / seconds:.0f} MB/s)", flush=True)
    rembg_dir = os.path.join(root, "rembg")
    os.makedirs(rembg_dir, exist_ok=True)
    for name in REMBG_MODELS:
        if not os.path.exists(os.path.join(rembg_dir, f"{name}.onnx")):
            urllib.request.urlretrieve(REMBG_URL.format(name), os.path.join(rembg_dir, f"{name}.onnx"))
    # Downloader bookkeeping is not part of the dataset.
    for repo in spec:
        cache = os.path.join(root, repo, ".cache")
        if os.path.isdir(cache):
            shutil.rmtree(cache)
    with open(os.path.join(root, MANIFEST), "w") as file:
        json.dump({"repos": spec, "include_turbo": include_turbo,
                   "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, file, indent=2)
    print(f"Total {folder_size_gb(root):.1f} GB in {root}", flush=True)


def find_root(search_dir: str = "/kaggle/input", max_depth: int = 4) -> Optional[str]:
    """Folder holding the manifest inside an attached Kaggle dataset, wherever Kaggle mounted it."""
    base_depth = search_dir.rstrip(os.sep).count(os.sep)
    for root, dirs, names in os.walk(search_dir):
        if MANIFEST in names:
            return root
        if root.count(os.sep) - base_depth >= max_depth:
            dirs[:] = []
    return None


ARCHIVE = "imageto3d-models.zip"


def pack(root: str, archive_dir: str) -> str:
    """Move every file under root into one uncompressed zip, deleting each file once it is in the archive.

    kagglehub zips folders of more than 50 files into a temporary copy before uploading, which needs twice
    the model size on a 57 GB Kaggle disk. Packing ourselves keeps the peak at the model size plus one file.
    Kaggle extracts the archive, so the dataset keeps the folder layout.
    """
    import zipfile
    os.makedirs(archive_dir, exist_ok=True)
    archive = os.path.join(archive_dir, ARCHIVE)
    paths = sorted(os.path.join(folder, name) for folder, _, names in os.walk(root) for name in names)
    # Append mode: an interrupted run continues where it stopped (files already packed are gone from root).
    with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        packed = set(bundle.namelist())
        for path in paths:
            name = os.path.relpath(path, root)
            if name not in packed:
                bundle.write(path, name)
                bundle.fp.flush()
            os.remove(path)
    shutil.rmtree(root, ignore_errors=True)
    print(f"Packed {len(paths)} files into {archive} ({os.path.getsize(archive) / 1024**3:.1f} GB)", flush=True)
    return archive
