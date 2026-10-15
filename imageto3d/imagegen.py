"""Text to images for the 3D pipeline (Kaggle): prompts.json -> transparent <name>-<view>.png files.

The front view comes from FLUX.1-schnell; left/back/right are made from that front with MV-Adapter, so
every view shows the same object. Every image is cut out with rembg and saved as a transparent PNG.

prompts.json:
  {"defaults": {"seed": 1000, "style": "..."},
   "items": [{"name": "monster-gelly", "prompt": "a small green jelly slime", "views": ["front", "back"]}]}
Items marked "done": true are skipped; the run writes an updated prompts.json with finished items marked.

Each GPU worker runs: python -m imageto3d.imagegen --config <json> --worker <i> --workers <n> --stage front|views
Stages run in separate processes, so FLUX and MV-Adapter are never in memory together.
"""
import argparse
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Dict, List

from .config import CATEGORIES, Config

VIEWS = ("front", "left", "back", "right")
DEVICE = "cuda"  # tests run the same code on the CPU

# Default look: RagnarokChange is a realistic 3D game, and realistic materials paint well onto meshes.
DEFAULT_STYLE = "photorealistic high-detail game asset, realistic PBR materials"
# Shared framing that suits image-to-3D: one subject, fully visible, flat light, plain mid-grey backdrop
# (light objects on white get removed with the background).
FRAMING = ("single subject centered, whole subject in frame with margin, straight front view, orthographic, "
           "plain medium grey background, even studio lighting, no shadows, no text")
CATEGORY_HINTS = {
    "character": "full body, standing in A-pose with arms slightly away from the body, completely bald head "
                 "with no hair, no headwear, no weapon, no backpack",
    "monster": "full body creature, standing",
    "weapon": "single weapon standing vertically, tip pointing up",
    "armor": "single armor piece displayed on its own, no person",
    "equipment": "single item displayed on its own, no person",
    "appearance": "hairstyle only as a standalone wig, no face, no head, no mannequin",
    "prop": "single standalone object",
    "object": "single standalone object",
    "environment": "single standalone structure, no ground, no surroundings",
    "bigobject": "entire building as a front elevation, no ground, no surroundings",
}


@dataclass
class ImageGenConfig:
    root: str                      # in/ (outputs) and work/ live here
    prompts_path: str
    hf_home: str
    rembg_home: str
    model_root: str = ""           # Kaggle model dataset for SDXL and MV-Adapter (optional)
    mv_repo: str = "/content/MV-Adapter"
    hunyuan_repo: str = "/content/Hunyuan3D-2"
    flux_model: str = "black-forest-labs/FLUX.1-schnell"
    size: int = 1024
    steps: int = 4
    max_sequence_length: int = 256
    # Compute type of the 4-bit FLUX transformer. T4 has no native bfloat16; diffusers clips float16 in FLUX.
    transformer_dtype: str = "float16"
    rembg_model: str = "isnet-general-use"
    head_mode: str = "bald"
    view_seed: int = 1234
    deadline: float = 0.0
    defaults: Dict = field(default_factory=dict)

    @property
    def input_dir(self) -> str:
        return os.path.join(self.root, "in")

    @property
    def work_dir(self) -> str:
        return os.path.join(self.root, "work")

    def pipeline_config(self) -> Config:
        """3D pipeline settings, for MV-Adapter view generation and background removal."""
        return Config(drive_root=self.root, hf_home=self.hf_home, rembg_home=self.rembg_home,
                      model_root=self.model_root, mv_repo=self.mv_repo, hunyuan_repo=self.hunyuan_repo,
                      head_mode=self.head_mode, view_seed=self.view_seed, rembg_model=self.rembg_model)

    def save(self, path: str) -> None:
        with open(path, "w") as file:
            json.dump(asdict(self), file, indent=2)


def load_config(path: str) -> ImageGenConfig:
    with open(path) as file:
        return ImageGenConfig(**json.load(file))


# ---------------------------------------------------------------- prompts.json

def load_prompts(path: str) -> Dict:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data.get("items"), list):
        raise ValueError(f"{path}: needs an \"items\" list")
    seen = set()
    for index, item in enumerate(data["items"]):
        name = str(item.get("name", "")).strip().lower()
        category, _, asset_id = name.partition("-")
        if category not in CATEGORIES or not asset_id:
            raise ValueError(f"item {index}: name {item.get('name')!r} must be <category>-<id>, "
                             f"category one of {', '.join(CATEGORIES)}")
        if name in seen:
            raise ValueError(f"item {index}: duplicate name {name}")
        seen.add(name)
        if not str(item.get("prompt", "")).strip():
            raise ValueError(f"item {index} ({name}): empty prompt")
        views = [str(view).lower() for view in item.get("views", ["front"])]
        unknown = [view for view in views if view not in VIEWS]
        if unknown:
            raise ValueError(f"item {index} ({name}): unknown views {unknown}; use {', '.join(VIEWS)}")
        item["name"] = name
        # The other views are made from the front, so the front is always generated.
        item["views"] = [view for view in VIEWS if view == "front" or view in views]
    return data


def category(item: Dict) -> str:
    return item["name"].partition("-")[0]


def prompt_parts(item: Dict, defaults: Dict):
    """(subject, conditions): the user's description, and the hints, style and framing added for 3D."""
    style = item.get("style", defaults.get("style", DEFAULT_STYLE))
    conditions = ", ".join(part for part in (CATEGORY_HINTS.get(category(item), ""), style, FRAMING) if part)
    return item["prompt"].strip().rstrip("."), conditions


def full_prompt(item: Dict, defaults: Dict) -> str:
    return ", ".join(prompt_parts(item, defaults))


def clip_prompt(item: Dict, defaults: Dict) -> str:
    """CLIP reads only 77 tokens: conditions first, so the cut falls on the end of the description."""
    subject, conditions = prompt_parts(item, defaults)
    return f"{conditions}, {subject}"


def fit_prompt(subject: str, conditions: str, count_tokens, limit: int):
    """T5 prompt within `limit` tokens: the conditions are always kept, the description is shortened
    word by word from its end. Returns (prompt, dropped words)."""
    words = subject.split()
    kept = len(words)
    while kept > 0 and count_tokens(f"{' '.join(words[:kept])}, {conditions}") > limit:
        kept -= 1
    text = f"{' '.join(words[:kept])}, {conditions}" if kept else conditions
    return text, " ".join(words[kept:])


def seed(item: Dict, defaults: Dict) -> int:
    return int(item.get("seed", defaults.get("seed", 1000)))


def image_path(config: ImageGenConfig, item: Dict, view: str) -> str:
    return os.path.join(config.input_dir, f"{item['name']}-{view}.png")


def missing_views(config: ImageGenConfig, item: Dict) -> List[str]:
    return [view for view in item["views"] if not os.path.exists(image_path(config, item, view))]


def assigned(data: Dict, worker: int, workers: int) -> List[Dict]:
    """Items still to do, split between GPU workers: worker i takes every n-th one."""
    todo = [item for item in data["items"] if not item.get("done")]
    return todo[worker::workers]


def mark_done(config: ImageGenConfig, data: Dict) -> List[str]:
    """Mark items whose views all exist; returns the newly finished names."""
    finished = []
    for item in data["items"]:
        if not item.get("done") and not missing_views(config, item):
            item["done"] = True
            finished.append(item["name"])
    return finished


def save_prompts(data: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def failure_path(config: ImageGenConfig, item: Dict) -> str:
    return os.path.join(config.work_dir, f"{item['name']}.error.txt")


# ---------------------------------------------------------------- GPU stages

def _past_deadline(config: ImageGenConfig) -> bool:
    return bool(config.deadline) and time.time() >= config.deadline


def _save_cutout(remover, image, path: str) -> None:
    cutout = remover(image.convert("RGB"))
    alpha = cutout.getchannel("A")
    coverage = sum(alpha.histogram()[128:]) / (alpha.width * alpha.height)
    if coverage < 0.02:
        print(f"  WARNING: {os.path.basename(path)}: almost nothing left after background removal", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.png"
    cutout.save(tmp)
    os.replace(tmp, path)  # a finished file only ever appears complete


def load_flux(config: ImageGenConfig):
    """FLUX.1-schnell for a 16 GB T4: text encoders first (prompt embeddings), then the 4-bit transformer."""
    import torch
    from diffusers import BitsAndBytesConfig, FluxPipeline, FluxTransformer2DModel
    dtype = getattr(torch, config.transformer_dtype)
    transformer = FluxTransformer2DModel.from_pretrained(
        config.flux_model, subfolder="transformer", torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=dtype))
    return FluxPipeline.from_pretrained(config.flux_model, transformer=transformer, text_encoder=None,
                                        text_encoder_2=None, tokenizer=None, tokenizer_2=None, torch_dtype=dtype)


def encode_prompts(config: ImageGenConfig, items: List[Dict], defaults: Dict):
    """CLIP and T5 embeddings, computed in bfloat16 (T5 overflows in float16) and then unloaded.

    FLUX.1-schnell reads at most max_sequence_length T5 tokens; a long description is shortened so the
    3D conditions (background, framing) are never the part that is cut off.
    """
    import torch
    from diffusers import FluxPipeline
    from . import runtime
    text = FluxPipeline.from_pretrained(config.flux_model, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    count = lambda prompt: len(text.tokenizer_2(prompt).input_ids)
    prompts = []
    for item in items:
        t5_prompt, dropped = fit_prompt(*prompt_parts(item, defaults), count, config.max_sequence_length)
        if dropped:
            print(f"  {item['name']}: description longer than FLUX reads ({config.max_sequence_length} tokens); "
                  f"left out the end: \"{dropped}\"", flush=True)
        prompts.append((clip_prompt(item, defaults), t5_prompt))

    def encode(device: str):
        text.text_encoder.to(device)
        text.text_encoder_2.to(device)
        with torch.no_grad():
            return [tuple(part.cpu() for part in text.encode_prompt(
                prompt=clip_text, prompt_2=t5_text, device=device, max_sequence_length=config.max_sequence_length)[:2])
                for clip_text, t5_text in prompts]

    try:
        encoded = encode(DEVICE)
    except (RuntimeError, AssertionError) as error:  # bfloat16 kernels missing on this GPU: encode on the CPU instead (slower)
        print(f"  text encoders on GPU failed ({error}); encoding on the CPU", flush=True)
        text.text_encoder_2.to("cpu")
        runtime.release()
        encoded = encode("cpu")
    del text
    runtime.release()
    return encoded, [t5_text for _, t5_text in prompts]


def run_front(config: ImageGenConfig, items: List[Dict], defaults: Dict) -> None:
    import torch
    from . import runtime
    todo = [item for item in items if not os.path.exists(image_path(config, item, "front"))]
    if not todo:
        return
    pipeline_config = runtime_setup(config)
    remover = runtime.background_remover(pipeline_config)
    started = time.time()
    encoded, prompts = encode_prompts(config, todo, defaults)
    pipe = load_flux(config)
    pipe.vae.to(DEVICE, dtype=torch.float32)  # decoded separately in float32
    dtype = getattr(torch, config.transformer_dtype)
    runtime.report(f"front model ready ({time.time() - started:.0f} s)")
    for item, prompt, (embeds, pooled) in zip(todo, prompts, encoded):
        if _past_deadline(config):
            print("[front] session time limit reached; leaving the rest for the next session", flush=True)
            break
        started = time.time()
        try:
            print(f"[front] {item['name']}: {prompt}", flush=True)
            latents = pipe(prompt_embeds=embeds.to(DEVICE, dtype), pooled_prompt_embeds=pooled.to(DEVICE, dtype),
                           height=config.size, width=config.size, num_inference_steps=config.steps,
                           guidance_scale=0.0, output_type="latent",
                           generator=torch.Generator(DEVICE).manual_seed(seed(item, defaults))).images
            if not torch.isfinite(latents).all():
                raise RuntimeError(f"FLUX produced NaN in {config.transformer_dtype}; try transformer_dtype bfloat16")
            latents = pipe._unpack_latents(latents, config.size, config.size, pipe.vae_scale_factor)
            latents = latents.float() / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            with torch.no_grad():
                pixels = pipe.vae.decode(latents, return_dict=False)[0]
            image = pipe.image_processor.postprocess(pixels, output_type="pil")[0]
            os.makedirs(config.work_dir, exist_ok=True)
            image.save(os.path.join(config.work_dir, f"{item['name']}-front-raw.jpg"), quality=95)
            _save_cutout(remover, image, image_path(config, item, "front"))
            print(f"[front] {item['name']}: done in {time.time() - started:.0f} s", flush=True)
        except Exception as error:  # one bad item must not stop the batch
            traceback.print_exc()
            _record_failure(config, item, f"front: {type(error).__name__}: {error}")
        runtime.release()


def run_views(config: ImageGenConfig, items: List[Dict], defaults: Dict) -> None:
    from PIL import Image
    from . import runtime, stage_views
    todo = [item for item in items if os.path.exists(image_path(config, item, "front"))
            and [view for view in missing_views(config, item) if view != "front"]]
    if not todo:
        return
    pipeline_config = runtime_setup(config)
    remover = runtime.background_remover(pipeline_config)
    started = time.time()
    model = {"pipe": stage_views.load_multiview(pipeline_config)}
    runtime.release()
    runtime.report(f"views model ready ({time.time() - started:.0f} s)")
    for item in todo:
        if _past_deadline(config):
            print("[views] session time limit reached; leaving the rest for the next session", flush=True)
            break
        started = time.time()
        wanted = [view for view in missing_views(config, item) if view != "front"]
        try:
            reference = Image.open(image_path(config, item, "front"))
            asset = SimpleNamespace(category=category(item), generated_views=wanted)
            for view, image in stage_views.generate(model, asset, pipeline_config, reference).items():
                _save_cutout(remover, image, image_path(config, item, view))
            print(f"[views] {item['name']}: {', '.join(wanted)} done in {time.time() - started:.0f} s", flush=True)
        except Exception as error:
            traceback.print_exc()
            _record_failure(config, item, f"views: {type(error).__name__}: {error}")
        runtime.release()


def runtime_setup(config: ImageGenConfig) -> Config:
    from . import runtime
    pipeline_config = config.pipeline_config()
    runtime.setup(pipeline_config)
    return pipeline_config


def _record_failure(config: ImageGenConfig, item: Dict, message: str) -> None:
    os.makedirs(config.work_dir, exist_ok=True)
    with open(failure_path(config, item), "a") as file:
        file.write(message + "\n")
    print(f"FAILED {item['name']}: {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stage", choices=("front", "views"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = load_prompts(config.prompts_path)
    defaults = data.get("defaults", {})
    items = assigned(data, args.worker, args.workers)
    print(f"[{args.stage}] worker {args.worker}: {len(items)} item(s)", flush=True)
    (run_front if args.stage == "front" else run_views)(config, items, defaults)


if __name__ == "__main__":
    main()
