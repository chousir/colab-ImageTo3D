"""Shape on a Kaggle TPU v5e-8: Hunyuan3D-2.1 from the front image, one worker process per TPU chip.

Writes the work files the GPU pipeline uses (front.png, shape.glb) plus shape_import.json, so the 2x T4
notebook imports them (see shape_import.py) and only paints and exports.

Hunyuan3D-2.1 is plain PyTorch, but three parts are rewritten here for XLA, which compiles one graph per
tensor shape and stalls on data-dependent shapes:
- the MoE layers route each token to its experts with bincount-sized slices; here every expert runs on all
  tokens and the gate weights zero the unchosen ones (same result, fixed shapes);
- the sampling loop passes timesteps as device data instead of Python numbers, so all steps share one graph;
- the volume decoder queries the grid in fixed-size chunks (the last one padded) and builds the query points
  per chunk on the CPU; the decoder runs in float32 because bfloat16 cannot resolve a 512 grid's spacing.
The image encoder (DINOv2) runs on the CPU, once per asset; marching cubes and mesh cleanup also run on the CPU.
"""
import argparse
import contextlib
import fcntl
import json
import os
import time
import traceback

from . import assets as asset_store
from .config import Config, load

MODEL_REPO = "tencent/Hunyuan3D-2.1"
MODEL_SUBFOLDER = "hunyuan3d-dit-v2-1"
BOX = 1.01  # half size of the decoded cube, as in Hunyuan3D-2.1 (box_v)


def needs_shape(asset: asset_store.Asset) -> bool:
    """No shape from this notebook yet, or the input image changed since it was made."""
    return asset_store._stale([asset.shape_import_path, asset.shape_path, asset.view_path("front")], asset.sources)


def queue(config: Config, categories) -> list:
    return [asset for asset in asset_store.pending(config)
            if asset.category in categories and needs_shape(asset) and not asset_store.blocked(asset, config)]


def patch_for_xla():
    """Call after importing hy3dshape, before building the model."""
    import torch
    from hy3dshape.models.denoisers.moe_layers import MoEBlock

    # Upstream picks CUDA attention kernels with this context manager; it has no meaning on a TPU.
    torch.backends.cuda.sdp_kernel = lambda *args, **kwargs: contextlib.nullcontext()

    def dense_moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        tokens, top_k = x.shape[0], self.moe_top_k
        gates = torch.zeros(tokens, len(self.experts), dtype=x.dtype, device=x.device)
        gates = gates.scatter_add(1, flat_expert_indices.view(tokens, top_k),
                                  flat_expert_weights.view(tokens, top_k).to(x.dtype))
        out = torch.zeros_like(x)
        for index, expert in enumerate(self.experts):
            out = out + expert(x) * gates[:, index:index + 1]
        return out

    MoEBlock.moe_infer = dense_moe_infer


def load_model(config: Config, device, lock_path: str):
    import torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from . import runtime

    patch_for_xla()
    # Workers load one at a time: building the model needs about 20 GB of RAM for a moment.
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_REPO, subfolder=MODEL_SUBFOLDER, device="cpu", dtype=torch.float32)
        pipe.model.to(dtype=torch.bfloat16).to(device).eval()
        pipe.vae.to(device).eval()  # float32, see the module docstring
        pipe.conditioner.eval()     # stays on the CPU in float32
        runtime.release()
        fcntl.flock(lock, fcntl.LOCK_UN)
    return {"pipe": pipe, "device": device, "rembg": runtime.background_remover(config)}


def sample_latents(model, image, steps: int, guidance: float, seed: int):
    """Hunyuan3D-2.1's flow-matching sampler (sigmas 0 -> 1, Euler steps) with classifier-free guidance."""
    import numpy as np
    import torch
    import torch_xla.core.xla_model as xm
    pipe, device = model["pipe"], model["device"]
    with torch.no_grad():
        inputs = pipe.prepare_image(image)
        cond = pipe.conditioner(image=inputs["image"], mask=inputs["mask"])["main"]
        uncond = pipe.conditioner.unconditional_embedding(cond.shape[0])["main"]
    cond = torch.cat([cond, uncond]).to(device, torch.bfloat16)  # conditional, unconditional
    contexts = {"main": cond}

    sigmas = np.append(np.linspace(0, 1, steps), 1.0).astype(np.float32)
    generator = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn((1, *pipe.vae.latent_shape), generator=generator).to(device, torch.bfloat16)
    for index in range(steps):
        # Built on the CPU and copied: Python numbers would become graph constants and recompile every step.
        timestep = torch.full((2,), float(sigmas[index]), dtype=torch.bfloat16).to(device)
        delta = torch.tensor(float(sigmas[index + 1] - sigmas[index]), dtype=torch.float32).to(device)
        with torch.no_grad():
            prediction = pipe.model(torch.cat([latents, latents]), timestep, contexts)
            conditional, unconditional = prediction.chunk(2)
            prediction = unconditional + guidance * (conditional - unconditional)
            latents = (latents.float() + delta * prediction.float()).to(torch.bfloat16)
        xm.mark_step()
    return latents


def decode_grid(model, latents, octree: int, chunk: int):
    """Occupancy logits on an (octree + 1)^3 grid over [-BOX, BOX]^3, as a float32 numpy array."""
    import numpy as np
    import torch
    import torch_xla.core.xla_model as xm
    pipe, device = model["pipe"], model["device"]
    side = octree + 1
    total = side ** 3
    axis = np.linspace(-BOX, BOX, side, dtype=np.float32)
    grid = np.empty(total, dtype=np.float32)
    with torch.no_grad():
        latents = pipe.vae(latents.float() / pipe.vae.scale_factor)
        xm.mark_step()  # keep the VAE transformer out of the per-chunk graph
        pending, start_of_pending = [], 0
        for start in range(0, total, chunk):
            index = np.minimum(np.arange(start, start + chunk), total - 1)  # pad the last chunk with the last point
            points = np.stack([axis[index // (side * side)], axis[(index // side) % side], axis[index % side]], -1)
            queries = torch.from_numpy(points)[None].to(device)
            pending.append(pipe.vae.geo_decoder(queries=queries, latents=latents).view(-1))
            xm.mark_step()  # runs this chunk asynchronously while the next one is prepared
            if len(pending) == 16 or start + chunk >= total:
                values = torch.cat(pending).cpu().numpy()
                count = min(len(values), total - start_of_pending)
                grid[start_of_pending:start_of_pending + count] = values[:count]
                start_of_pending += count
                pending = []
    return grid.reshape(side, side, side)


def process(model, asset: asset_store.Asset, config: Config, args) -> None:
    import torch
    from hy3dshape.pipelines import export_to_trimesh
    from hy3dshape.postprocessors import DegenerateFaceRemover, FaceReducer, FloaterRemover
    from .stage_shape import drop_small_components
    from .stage_views import cutout

    os.makedirs(asset.work, exist_ok=True)
    front = cutout(model, asset.source)
    front.save(asset.view_path("front"))
    preset = config.presets[asset.category]
    octree = int(preset["octree"])

    started = time.time()
    latents = sample_latents(model, front, config.shape_steps, config.shape_guidance, config.shape_seed)
    sampled = time.time()
    grid = decode_grid(model, latents, octree, args.chunk)
    decoded = time.time()
    outputs = model["pipe"].vae.surface_extractor(torch.from_numpy(grid)[None], mc_level=0.0, bounds=BOX,
                                                  octree_resolution=octree)
    if outputs[0] is None:
        raise RuntimeError("marching cubes found no surface (empty or full occupancy grid)")
    mesh = export_to_trimesh(outputs)[0]
    raw_faces = len(mesh.faces)
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = drop_small_components(mesh, config.preset(asset.category, "min_component_ratio"))
    mesh = FaceReducer()(mesh, max_facenum=preset["max_faces"])
    mesh.export(asset.shape_path)
    print(f"  sampling {sampled - started:.0f} s, grid {octree} {decoded - sampled:.0f} s; "
          f"{raw_faces:,} -> {len(mesh.faces):,} faces", flush=True)
    marker = {
        "shape_generator": {"model": f"{MODEL_REPO}/{MODEL_SUBFOLDER}", "code_commit": args.code_commit,
                            "seed": config.shape_seed, "steps": config.shape_steps,
                            "guidance": config.shape_guidance, "preset": preset, "device": "TPU v5e (bf16)"},
        "source_image": os.path.basename(asset.source),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(asset.shape_import_path, "w") as file:
        json.dump(marker, file, indent=2)


def worker(index: int, args) -> None:
    import torch_xla
    import torch_xla.runtime as xr

    if index >= args.workers:
        return  # fewer workers than chips (limited RAM): this chip stays idle
    config = load(args.config)
    from . import runtime
    runtime.setup(config)
    try:  # reruns reuse compiled graphs; only possible before the runtime starts
        xr.initialize_cache(os.path.join(args.xla_cache, str(index)), readonly=False)
    except Exception as error:
        print(f"[tpu{index}] no persistent compile cache ({error})", flush=True)
    device = torch_xla.device()
    tag = f"[tpu{index}]"
    if not queue(config, args.categories):
        return
    started = time.time()
    model = load_model(config, device, os.path.join(config.work_dir, ".load.lock"))
    print(f"{tag} model ready ({time.time() - started:.0f} s)", flush=True)
    while True:
        if config.deadline and time.time() >= config.deadline:
            print(f"{tag} session time limit reached; stopping", flush=True)
            break
        claimed = next((asset for asset in queue(config, args.categories) if asset_store.claim(asset, index)), None)
        if claimed is None:
            break
        asset_store.begin(claimed, "shape")
        started = time.time()
        print(f"{tag} {claimed.key}: shape (octree {config.presets[claimed.category]['octree']})", flush=True)
        try:
            process(model, claimed, config, args)
        except Exception as error:  # One bad image must not stop the batch.
            traceback.print_exc()
            asset_store.fail(claimed, "shape", f"{type(error).__name__}: {error}")
            print(f"{tag} {claimed.key}: FAILED ({error})", flush=True)
        else:
            asset_store.succeed(claimed, "shape")
            print(f"{tag} {claimed.key}: done in {time.time() - started:.0f} s", flush=True)
        finally:
            asset_store.release(claimed)
    print(f"{tag} finished", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=8192, help="grid points per volume-decoder call")
    parser.add_argument("--xla_cache", required=True)
    parser.add_argument("--code_commit", default="")
    args = parser.parse_args()
    config = load(args.config)
    asset_store.clear_claims(config)
    todo = [asset.key for asset in queue(config, args.categories)]
    print(f"{len(todo)} image(s) to shape:", ", ".join(todo) or "none", flush=True)
    if not todo:
        return
    import torch_xla.distributed.xla_multiprocessing as xmp
    from imageto3d.stage_shape_tpu import worker as importable_worker  # spawned processes import it by name
    xmp.spawn(importable_worker, args=(args,), start_method="spawn")


if __name__ == "__main__":
    main()
