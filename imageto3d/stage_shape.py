"""Stage 2: build the mesh (Hunyuan3D-2mv from front/left/back, or Hunyuan3D-2 from one image), then clean it."""
from .assets import Asset
from .config import Config
from .stages import run_stage


def load_model(config: Config, queue):
    import torch
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from . import runtime

    # The runner sends one mode per process, so only one shape model is ever loaded.
    multiview = queue[0].mode == "multiview"
    repo, subfolder = (("tencent/Hunyuan3D-2mv", config.shape_model) if multiview
                       else ("tencent/Hunyuan3D-2", config.single_shape_model))
    # Upstream builds the model in fp32 on the CPU next to its 5 GB checkpoint, which exceeds free Colab RAM.
    with runtime.build_on_gpu_fp16():
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            repo, subfolder=subfolder, variant="fp16", dtype=torch.float16)
        if config.shape_flashvdm and "turbo" in subfolder:
            pipe.enable_flashvdm()  # Official turbo setting: turbo VAE plus a faster decoder.
    print(f"  shape model: {repo}/{subfolder}", flush=True)
    return {"pipe": pipe, "rembg": BackgroundRemover(), "turbo": "turbo" in subfolder}


def prepare(model, path: str):
    from PIL import Image
    image = Image.open(path)
    if image.mode == "RGBA" and image.getchannel("A").getextrema()[0] < 255:
        return image
    return model["rembg"](image.convert("RGB"))


def drop_small_components(mesh, min_ratio: float):
    """Remove disconnected specks and fragments; keeps every piece of meaningful size."""
    import numpy as np
    import trimesh
    components = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)))
    if len(components) <= 1:
        return mesh
    largest = max(len(component) for component in components)
    keep = [component for component in components if len(component) >= largest * min_ratio]
    faces = np.concatenate(keep)
    cleaned = mesh.submesh([faces], append=True)
    print(f"  removed {len(components) - len(keep)} of {len(components)} disconnected pieces", flush=True)
    return cleaned


def process(model, asset: Asset, config: Config) -> None:
    import torch
    from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

    preset = config.presets[asset.category]
    if asset.mode == "multiview":
        image = {view: prepare(model, asset.view_path(view)) for view in ("front", "left", "back")}
    else:
        image = prepare(model, asset.view_path("front"))
    steps = 5 if model["turbo"] else config.shape_steps

    def generate(octree: int):
        return model["pipe"](
            image=image,
            num_inference_steps=steps,
            guidance_scale=config.shape_guidance,
            octree_resolution=octree,
            num_chunks=8000 if octree <= 384 else 4000,
            generator=torch.Generator(device="cuda").manual_seed(config.shape_seed),
            output_type="trimesh",
        )[0]

    octree = int(preset["octree"])
    try:
        mesh = generate(octree)
    except torch.cuda.OutOfMemoryError:
        if octree <= 384:
            raise
        # Fine grids for large buildings may not fit next to the model on a T4; retry at the standard grid.
        print(f"  out of GPU memory at octree {octree}; retrying at 384", flush=True)
        torch.cuda.empty_cache()
        mesh = generate(384)
    raw_faces = len(mesh.faces)
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = drop_small_components(mesh, config.preset(asset.category, "min_component_ratio"))
    mesh = FaceReducer()(mesh, max_facenum=preset["max_faces"])
    mesh.export(asset.shape_path)
    print(f"  {raw_faces:,} -> {len(mesh.faces):,} faces", flush=True)


if __name__ == "__main__":
    run_stage("shape", load_model, process)
