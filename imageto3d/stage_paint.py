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
    paint.config.texture_size = config.texture_size
    paint.config.render_size = config.render_size
    paint.render.set_default_texture_resolution(config.texture_size)
    paint.render.set_default_render_resolution(config.render_size)
    if config.low_vram:
        paint.enable_model_cpu_offload()  # Parks models in system RAM; only for GPU out-of-memory cases.
    return paint


def process(paint, asset: Asset, config: Config) -> None:
    import trimesh
    from PIL import Image

    mesh = trimesh.load(asset.shape_path, force="mesh")
    textured = paint(mesh, image=Image.open(asset.view_path("front")))
    textured.export(asset.textured_path)


if __name__ == "__main__":
    run_stage("paint", load_model, process)
