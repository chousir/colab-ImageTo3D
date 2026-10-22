"""Importing shapes made by the TPU shape notebook, without a GPU or TPU: python tests/test_shape_import.py (needs pillow)."""
import json, os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from imageto3d import config as cfgmod, assets, shape_import, stage_shape_tpu

root = tempfile.mkdtemp()
cfg = cfgmod.Config(drive_root=os.path.join(root, "t4"), hf_home="", rembg_home="")
os.makedirs(cfg.input_dir)
for name in ("character-hero-front.png", "weapon-axe-stuff.png", "monster-gelly-front.png", "monster-gelly-back.png"):
    Image.new("RGB", (8, 8)).save(os.path.join(cfg.input_dir, name))

# TPU notebook output, as attached under /kaggle/input: shapes for the hero and for an asset not pending here.
tpu_root = os.path.join(root, "input", "tpu-shapes", "hunyuan3d-shapes")
tpu_cfg = cfgmod.Config(drive_root=tpu_root, hf_home="", rembg_home="")
def tpu_shape(key, complete=True):
    work = os.path.join(tpu_cfg.work_dir, key)
    os.makedirs(work, exist_ok=True)
    Image.new("RGBA", (8, 8)).save(os.path.join(work, "front.png"))
    if complete:
        open(os.path.join(work, "shape.glb"), "wb").write(b"glb " + key.encode())
    json.dump({"shape_generator": {"model": "tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1"}, "source_image": key},
              open(os.path.join(work, assets.SHAPE_IMPORT_FILE), "w"))
tpu_shape("character-hero")
tpu_shape("prop-cup")                    # not pending in the T4 in/: ignored
tpu_shape("weapon-axe", complete=False)  # no shape.glb: ignored

pend = {a.key: a for a in assets.pending(cfg)}
hero = pend["character-hero"]
assert hero.generated_views == ["left", "back", "right"] and hero.needs("views", cfg)

# A texture painted on an earlier shape must not survive the import.
os.makedirs(hero.work); open(hero.textured_path, "w").write("old")
assert shape_import.import_shapes([os.path.join(root, "input")], cfg) == ["character-hero"]
assert not os.path.exists(hero.textured_path)
hero = {a.key: a for a in assets.pending(cfg)}["character-hero"]
assert hero.imported_shape and hero.generated_views == [] and hero.views == ("front",)
assert not hero.needs("views", cfg) and not hero.needs("shape", cfg) and hero.needs("paint", cfg)
assert open(hero.shape_path, "rb").read() == b"glb character-hero"
assert hero.shape_import()["shape_generator"]["model"].startswith("tencent/Hunyuan3D-2.1")
axe = {a.key: a for a in assets.pending(cfg)}["weapon-axe"]
assert not axe.imported_shape and axe.needs("views", cfg)

# Running again copies nothing; a painted texture is kept.
open(hero.textured_path, "w").write("new"); time.sleep(0.01)
assert shape_import.import_shapes([os.path.join(root, "input"), cfg.drive_root], cfg) == []
assert os.path.exists(hero.textured_path) and not hero.needs("paint", cfg)

# TPU side: which assets still need a shape. Done assets drop out; a newer input image brings them back.
os.makedirs(tpu_cfg.input_dir)
for name in ("character-hero-front.png", "weapon-axe-stuff.png"):
    Image.new("RGB", (8, 8)).save(os.path.join(tpu_cfg.input_dir, name))
time.sleep(0.01)
tpu_shape("character-hero")
names = lambda: [a.key for a in stage_shape_tpu.queue(tpu_cfg, ["character", "weapon"])]
assert names() == ["weapon-axe"], names()
assert [a.key for a in stage_shape_tpu.queue(tpu_cfg, ["character"])] == []
time.sleep(0.01)
Image.new("RGB", (8, 8)).save(os.path.join(tpu_cfg.input_dir, "character-hero-front.png"))
assert sorted(names()) == ["character-hero", "weapon-axe"], names()
print("shape import OK")
