"""Pipeline settings shared by the notebook, the orchestrator and every stage process."""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# The first word of the file name picks the category. "multiview" categories get left/back/right views
# generated from the front image; "single" categories are built from the one image only.
CATEGORY_MODES = {
    "character": "multiview",
    "monster": "multiview",
    "armor": "single",
    "appearance": "single",
    "weapon": "single",
    "equipment": "single",
    "prop": "single",
    "environment": "single",
    "object": "single",
    # Large buildings need several views: supplied -left/-back/-right images, or generated ones.
    "bigobject": "multiview",
}
CATEGORIES = tuple(CATEGORY_MODES)
# Accepted last word of the file name, e.g. character-novice_female-front.png, weapon-dagger-stuff.png,
# bigobject-kronenberg_main_gate-building.png.
NAME_SUFFIXES = ("front", "stuff", "building")
# Optional extra photos/renders for multiview categories: <category>-<id>-left|back|right.<ext>.
# A supplied view replaces the generated one.
EXTRA_VIEWS = ("left", "back", "right")

# Output naming and license data follow RagnarokChange docs/04-art-pipeline.md section 6.2.
SOURCE_TAG = "hunyuan3d"
LICENSE_ID = "LicenseRef-Tencent-Hunyuan3D-Community"

BALD_PROMPT = "completely bald head, shaved head, no hair, no headwear"
BALD_NEGATIVE = "hair, hairstyle, ponytail, twin tails, braid, bangs, helmet, hat"


@dataclass
class Config:
    drive_root: str
    hf_home: str
    rembg_home: str
    hunyuan_repo: str = "/content/Hunyuan3D-2"
    mv_repo: str = "/content/MV-Adapter"

    # Multiview generation (MV-Adapter SDXL).
    view_base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    view_vae_model: str = "madebyollin/sdxl-vae-fp16-fix"
    view_adapter_repo: str = "huanngzh/mv-adapter"
    view_adapter_commit: str = "4277e0018232bac82bb2c103caf0893cedb711be"
    view_count: int = 4
    view_size: int = 768
    view_steps: int = 30
    view_guidance: float = 3.0
    view_seed: int = 1234
    view_reference_scale: float = 1.0
    view_prompt: str = ("same subject as the reference image, preserve its category, silhouette, colors and "
                        "distinctive details, orthographic multi-view turnaround, plain neutral background, no text")
    view_negative_prompt: str = "watermark, low quality, deformed, inconsistent design, extra objects, text"
    # "bald": characters (not monsters) are generated without hair; hair and head are separate assets.
    head_mode: str = "bald"

    # Shape: Hunyuan3D-2mv for multiview categories, Hunyuan3D-2 single-image model for the others.
    hunyuan_commit: str = "f8db63096c8282cb27354314d896feba5ba6ff8a"
    shape_model: str = "hunyuan3d-dit-v2-mv"
    single_shape_model: str = "hunyuan3d-dit-v2-0"
    shape_steps: int = 50
    shape_guidance: float = 5.0
    shape_flashvdm: bool = True
    shape_seed: int = 1234
    # Items keep the full octree resolution for detail and use fewer faces than characters.
    # Optional per-category keys override the global settings: texture_size, render_size, min_component_ratio.
    presets: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "character": {"max_faces": 60000, "octree": 384},
        "monster": {"max_faces": 60000, "octree": 384},
        "armor": {"max_faces": 30000, "octree": 384},
        "appearance": {"max_faces": 40000, "octree": 384},
        "weapon": {"max_faces": 20000, "octree": 384},
        "equipment": {"max_faces": 20000, "octree": 384},
        "prop": {"max_faces": 20000, "octree": 384},
        "environment": {"max_faces": 100000, "octree": 512},
        "object": {"max_faces": 30000, "octree": 384},
        # Large buildings: finest shape grid, many faces, 4K texture, and keep small decorative pieces.
        "bigobject": {"max_faces": 200000, "octree": 512, "texture_size": 4096, "render_size": 2048,
                      "min_component_ratio": 0.0002},
    })
    # Drop disconnected pieces smaller than this share of the largest piece (cloth fragments, specks).
    min_component_ratio: float = 0.002

    # Texture (Hunyuan3D-2 paint).
    paint_model: str = "hunyuan3d-paint-v2-0"
    texture_size: int = 2048
    render_size: int = 1024
    low_vram: bool = False

    # Export: real-world height in meters per category (RagnarokChange 04 section 3); None keeps the size.
    female_height_m: float = 1.65
    male_height_m: float = 1.78
    category_height_m: Dict[str, Optional[float]] = field(default_factory=dict)
    mark_used: bool = True

    # Batch runner.
    poll_seconds: int = 120
    idle_stop_minutes: int = 30
    max_failures: int = 2
    stage_timeout_minutes: int = 90

    @property
    def input_dir(self) -> str:
        return os.path.join(self.drive_root, "in")

    @property
    def out_dir(self) -> str:
        return os.path.join(self.drive_root, "out")

    @property
    def work_dir(self) -> str:
        return os.path.join(self.drive_root, "work")

    @property
    def view_azimuths(self) -> List[int]:
        return [0, 45, 90, 180, 270, 315] if self.view_count == 6 else [0, 90, 180, 270]

    def preset(self, category: str, key: str):
        """Per-category value, falling back to the global setting of the same name."""
        value = self.presets[category].get(key)
        return getattr(self, key) if value is None else type(getattr(self, key))(value)

    def view_prompts(self, category: str) -> List[str]:
        prompt, negative = self.view_prompt, self.view_negative_prompt
        if self.head_mode == "bald" and category == "character":
            prompt = f"{prompt}, {BALD_PROMPT}"
            negative = f"{negative}, {BALD_NEGATIVE}"
        return [prompt, negative]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            json.dump(asdict(self), file, indent=2)


def load(path: str) -> Config:
    with open(path) as file:
        return Config(**json.load(file))
