"""Stage 4: scale to real-world size, stand on the ground, write the delivery files, mark the source used.

Output follows RagnarokChange docs/04-art-pipeline.md: file name `<source>-<category>-<id>.glb`,
1 unit = 1 m, +Y up, facing +Z (glTF), with a metadata JSON for the license registry.
"""
import json
import os
import shutil
import time

from .assets import Asset
from .config import LICENSE_ID, SOURCE_TAG, Config
from .stages import run_stage


def target_height(asset: Asset, config: Config):
    if asset.category == "character":
        return config.female_height_m if "female" in asset.asset_id or "male" not in asset.asset_id \
            else config.male_height_m
    return config.category_height_m.get(asset.category)


def normalize(mesh, height_m):
    """Scale to height (if given), put the lowest point on y = 0 and center x/z."""
    import numpy as np
    bounds = mesh.bounds
    if height_m:
        mesh.apply_scale(height_m / float(bounds[1][1] - bounds[0][1]))
        bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    mesh.apply_translation(np.array([-center[0], -bounds[0][1], -center[2]]))
    return mesh


def pbr_material(mesh):
    import trimesh
    material = mesh.visual.material
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=image, metallicFactor=0.0, roughnessFactor=0.9)
    return mesh


def load_model(config: Config):
    return None


def process(_, asset: Asset, config: Config) -> None:
    import trimesh

    os.makedirs(config.out_dir, exist_ok=True)
    mesh = trimesh.load(asset.textured_path, force="mesh")
    height = target_height(asset, config)
    mesh = pbr_material(normalize(mesh, height))
    glb_path = asset.output_path(config, ".glb")
    mesh.export(glb_path)
    if os.path.exists(asset.views_sheet):
        shutil.copy2(asset.views_sheet, asset.output_path(config, "-views.jpg"))

    extents = mesh.extents
    metadata = {
        "id": asset.asset_id,
        "category": asset.category,
        "file": os.path.basename(glb_path),
        "source": SOURCE_TAG,
        "license": LICENSE_ID,
        "license_note": ("Generated with Tencent Hunyuan3D-2 (Tencent Hunyuan Community License) from views made "
                         "with MV-Adapter on SDXL. Not on the RagnarokChange 04-D03 whitelist until approved."),
        "source_image": os.path.basename(asset.source),
        "head_mode": config.head_mode,
        "height_m": round(float(extents[1]), 3),
        "size_m": [round(float(value), 3) for value in extents],
        "scaled_to_height_m": height,
        "faces": int(len(mesh.faces)),
        "texture_size": config.texture_size,
        "view_generator": {"model": config.view_base_model, "adapter": config.view_adapter_repo,
                           "adapter_commit": config.view_adapter_commit, "seed": config.view_seed,
                           "azimuth_deg": config.view_azimuths},
        "shape_generator": {"model": f"tencent/Hunyuan3D-2mv/{config.shape_model}",
                            "code_commit": config.hunyuan_commit, "seed": config.shape_seed,
                            "preset": config.presets[asset.category]},
        "paint_generator": {"model": f"tencent/Hunyuan3D-2/{config.paint_model}"},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(asset.output_path(config, ".json"), "w") as file:
        json.dump(metadata, file, indent=2)
    print(f"  {os.path.basename(glb_path)}: {metadata['faces']:,} faces, "
          f"{extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f} m", flush=True)

    # Drive now holds the GLB, so the front can be marked used without risking the only copy.
    if config.mark_used and os.path.exists(asset.source):
        used = os.path.join(os.path.dirname(asset.source), "used-" + os.path.basename(asset.source))
        if os.path.exists(used):
            print(f"  WARNING: {os.path.basename(used)} already exists; source left in place", flush=True)
        else:
            os.replace(asset.source, used)


if __name__ == "__main__":
    run_stage("export", load_model, process)
