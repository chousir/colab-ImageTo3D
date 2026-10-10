"""Stage 3: paint the cleaned mesh with Hunyuan3D-2 using the front cutout."""
from .assets import Asset
from .config import Config
from .stages import run_stage


def load_model(config: Config, queue):
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from . import runtime

    # The paint UNet loader reads its whole 3.7 GB checkpoint into RAM next to the fp32 model it builds.
    # Memory-map checkpoints instead. (Building on the GPU breaks diffusers schedulers, so only mmap here.)
    with runtime.mmap_torch_load():
        paint = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2", subfolder=config.paint_model)
    set_sizes(paint, config.texture_size, config.render_size)
    if config.low_vram:
        paint.enable_model_cpu_offload()  # Parks models in system RAM; only for GPU out-of-memory cases.
    return paint


def process(paint, asset: Asset, config: Config) -> None:
    import trimesh
    from PIL import Image

    import torch

    front = Image.open(asset.view_path("front"))
    sizes = [(config.preset(asset.category, "texture_size"), config.preset(asset.category, "render_size")),
             (config.texture_size, config.render_size)]
    for attempt, (texture_size, render_size) in enumerate(dict.fromkeys(sizes)):
        set_sizes(paint, texture_size, render_size)
        try:
            textured = paint(trimesh.load(asset.shape_path, force="mesh"), image=front)
            break
        except torch.cuda.OutOfMemoryError:
            if attempt == len(dict.fromkeys(sizes)) - 1:
                raise
            # Large-building textures may not fit on a T4; fall back to the global sizes.
            print(f"  out of GPU memory at texture {texture_size}; retrying at {config.texture_size}", flush=True)
            torch.cuda.empty_cache()
    print(f"  texture {texture_size}px, render {render_size}px", flush=True)
    textured.export(asset.textured_path)


def set_sizes(paint, texture_size: int, render_size: int) -> None:
    paint.config.texture_size = texture_size
    paint.config.render_size = render_size
    paint.render.set_default_texture_resolution(texture_size)
    paint.render.set_default_render_resolution(render_size)


if __name__ == "__main__":
    run_stage("paint", load_model, process)
