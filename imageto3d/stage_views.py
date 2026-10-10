"""Stage 1: cut out the front image; for multiview categories also generate left/back/right with MV-Adapter."""
import os

from .assets import Asset
from .config import Config
from .stages import run_stage


def load_model(config: Config, queue):
    from hy3dgen.rembg import BackgroundRemover
    model = {"rembg": BackgroundRemover(), "pipe": None}
    if any(asset.mode == "multiview" for asset in queue):
        model["pipe"] = load_multiview(config)
    return model


def load_multiview(config: Config):
    import torch
    from diffusers import AutoencoderKL
    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
    from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
    from . import runtime

    # Free Colab has 12.7 GB RAM and T4 has 15 GB VRAM: load SDXL in fp16, move only the UNet to the GPU,
    # then build the adapter layers directly on the GPU (upstream creates ~0.8B fp32 parameters on the CPU).
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(
        config.view_base_model,
        vae=AutoencoderKL.from_pretrained(config.view_vae_model, torch_dtype=torch.float16),
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler, shift_mode="interpolated", shift_scale=8.0, scheduler_class=None)
    pipe.unet.to(device="cuda", dtype=torch.float16)
    runtime.release()
    with runtime.build_on_gpu_fp16():
        pipe.init_custom_adapter(num_views=len(config.view_azimuths))
    runtime.release()
    pipe.load_custom_adapter(config.view_adapter_repo, weight_name="mvadapter_i2mv_sdxl.safetensors")
    pipe.to(device="cuda", dtype=torch.float16)
    pipe.cond_encoder.to(device="cuda", dtype=torch.float16)
    pipe.enable_vae_slicing()
    return pipe


def cutout(model, path: str):
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(Image.open(path))
    if image.mode == "RGBA" and image.getchannel("A").getextrema()[0] < 255:
        return image.convert("RGBA")
    return model["rembg"](image.convert("RGB"))


def save_sheet(images, path: str, size: int = 384) -> None:
    from PIL import Image
    sheet = Image.new("RGB", (size * len(images), size), "white")
    for index, image in enumerate(images):
        tile = image.convert("RGBA")
        tile.thumbnail((size, size))
        offset = ((size - tile.width) // 2 + index * size, (size - tile.height) // 2)
        sheet.paste(tile, offset, tile)
    sheet.save(path, quality=90)


def process(model, asset: Asset, config: Config) -> None:
    os.makedirs(asset.work, exist_ok=True)
    reference = cutout(model, asset.source)
    if asset.mode == "single":
        reference.save(asset.view_path("front"))
        save_sheet([reference], asset.views_sheet)
        return
    from scripts.inference_i2mv_sdxl import run_pipeline
    prompt, negative = config.view_prompts(asset.category)
    azimuths = config.view_azimuths
    generated, _ = run_pipeline(
        model["pipe"],
        num_views=len(azimuths),
        text=prompt,
        image=reference,
        height=config.view_size,
        width=config.view_size,
        num_inference_steps=config.view_steps,
        guidance_scale=config.view_guidance,
        seed=config.view_seed,
        reference_conditioning_scale=config.view_reference_scale,
        negative_prompt=negative,
        device="cuda",
        azimuth_deg=azimuths,
    )
    if len(generated) != len(azimuths):
        raise RuntimeError(f"expected {len(azimuths)} views, received {len(generated)}")
    # MV-Adapter azimuth 90 looks at the subject's left side, matching Hunyuan3D-2mv's "left" view.
    views = {name: generated[azimuths.index(angle)] for name, angle in (("left", 90), ("back", 180), ("right", 270))}
    for name, image in views.items():
        image.convert("RGB").save(asset.view_path(name), quality=95)
    # Write the front last: its timestamp marks the view set as complete.
    reference.save(asset.view_path("front"))
    save_sheet([reference, views["left"], views["back"], views["right"]], asset.views_sheet)


if __name__ == "__main__":
    run_stage("views", load_model, process)
