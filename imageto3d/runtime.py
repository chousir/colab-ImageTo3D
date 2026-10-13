"""Process setup and memory helpers for stage processes on free Colab (12.7 GB RAM, T4 15 GB VRAM)."""
import contextlib
import ctypes
import gc
import os
import sys

from .config import Config


def setup(config: Config) -> None:
    """Call before importing torch, diffusers or hy3dgen in a stage process."""
    os.environ["HF_HOME"] = config.hf_home
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["U2NET_HOME"] = config.rembg_home
    if config.model_root:
        # Hunyuan3D looks for <HY3DGEN_MODELS>/tencent/<repo>/<subfolder> before downloading.
        os.environ["HY3DGEN_MODELS"] = config.model_root
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for path in (config.hunyuan_repo, config.mv_repo):
        if path not in sys.path:
            sys.path.insert(0, path)


def background_remover(config: Config):
    """rembg cutout with the configured model; returns an RGBA image with a transparent background."""
    from rembg import new_session, remove
    session = new_session(config.rembg_model)
    return lambda image: remove(image, session=session, bgcolor=[255, 255, 255, 0])


def release() -> None:
    """Return freed Python and CUDA memory to the OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def report(stage: str) -> None:
    import torch
    allocated = torch.cuda.memory_allocated() / 1024**3
    free = torch.cuda.mem_get_info()[0] / 1024**3
    with open("/proc/meminfo") as file:
        info = dict(line.split(":", 1) for line in file)
    available_ram = int(info["MemAvailable"].split()[0]) / 1024**2
    print(f"[{stage}] GPU allocated {allocated:.1f} GB, GPU free {free:.1f} GB, RAM available {available_ram:.1f} GB",
          flush=True)


@contextlib.contextmanager
def build_on_gpu_fp16():
    """Create new modules directly on the GPU in fp16.

    Several upstream loaders build models in fp32 on the CPU next to their checkpoint, which exceeds
    free Colab RAM. Only wrap model construction that does not convert tensors to numpy.
    """
    import torch
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with torch.device("cuda"):
            yield
    finally:
        torch.set_default_dtype(default_dtype)


@contextlib.contextmanager
def mmap_torch_load():
    """Memory-map torch checkpoints so loading does not copy the whole file into RAM."""
    import torch
    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("mmap", True)
        try:
            return original(*args, **kwargs)
        except RuntimeError:
            kwargs.pop("mmap")  # Legacy checkpoint format cannot be memory-mapped.
            return original(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original
