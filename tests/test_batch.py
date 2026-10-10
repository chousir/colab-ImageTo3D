"""Runs the batch orchestration and export stage without a GPU: python tests/test_batch.py (needs trimesh, pillow, numpy, scipy, networkx)."""
import os, sys, shutil, tempfile, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from imageto3d import config as cfgmod, assets, run, stage_export

root = tempfile.mkdtemp()
cfg = cfgmod.Config(drive_root=root, hf_home=root + "/hf", rembg_home=root + "/rb")
os.makedirs(cfg.input_dir)
def img(name):
    Image.new("RGB", (8, 8)).save(os.path.join(cfg.input_dir, name)); time.sleep(0.01)
img("character-novice_female-front.jpg")
img("monster-gelly-front.png")
img("used-character-old-front.jpg")
img("weapon-dagger-front.jpg")          # unknown category
img("character-novice_female-back.jpg") # not a front
img("Character-Novice_Female-front.webp") # duplicate key (case-insensitive)
cfg_path = os.path.join(root, "config.json"); cfg.save(cfg_path)
loaded = cfgmod.load(cfg_path); assert loaded == cfg

pend = assets.pending(cfg)
print("pending:", [a.key for a in pend])
assert [a.key for a in pend] == ["character-novice_female", "monster-gelly"]
a = pend[0]
assert all(a.needs(s, cfg) for s in run.STAGES)
print("prompts:", cfg.view_prompts()[0][-60:])

calls = []
GLB = os.path.join(root, "sample.glb")
import numpy as np, trimesh
_box = trimesh.creation.box((0.5, 2.0, 0.3)); _uv = np.random.rand(len(_box.vertices), 2)
_box.visual = trimesh.visual.TextureVisuals(uv=_uv, image=Image.new("RGB", (64, 64), "red")); _box.export(GLB)
def fake_stage(stage, keys, config_path, config):
    calls.append((stage, tuple(keys)))
    for asset in assets.find(config, keys):
        assets.begin(asset, stage)
        if stage == "shape" and asset.asset_id == "gelly":
            return -9  # simulated OOM kill while gelly runs
        os.makedirs(asset.work, exist_ok=True)
        if stage == "views":
            for v in assets.VIEW_NAMES: Image.new("RGB", (8, 8)).save(asset.view_path(v))
        elif stage == "shape":
            shutil.copy(GLB, asset.shape_path)
        elif stage == "paint":
            shutil.copy(GLB, asset.textured_path)
        elif stage == "export":
            stage_export.process(None, asset, config)
        assets.succeed(asset, stage)
    return 0
run._run_stage = fake_stage
n = run.run_round(cfg_path, cfg)
print("round1 worked:", n, calls)
out = sorted(os.listdir(cfg.out_dir)); print("out:", out)
assert "hunyuan3d-character-novice_female.glb" in out
meta = json.load(open(os.path.join(cfg.out_dir, "hunyuan3d-character-novice_female.json")))
print("height", meta["height_m"], "size", meta["size_m"])
assert abs(meta["height_m"] - 1.65) < 0.01
print("in/:", sorted(os.listdir(cfg.input_dir)))
assert os.path.exists(os.path.join(cfg.input_dir, "used-character-novice_female-front.jpg"))
gelly = assets.pending(cfg)[0]; print("gelly status:", assets.read_status(gelly))
assert assets.read_status(gelly)["failures"] == 1
calls.clear(); run.run_round(cfg_path, cfg); print("round2:", calls)
assert assets.blocked(gelly, cfg)
calls.clear(); n = run.run_round(cfg_path, cfg); print("round3 worked:", n, calls)
assert n == 0
# exported glb sanity
import trimesh
m = trimesh.load(os.path.join(cfg.out_dir, "hunyuan3d-character-novice_female.glb"), force="mesh")
print("bounds", m.bounds.round(3).tolist(), type(m.visual.material).__name__, m.visual.material.baseColorTexture.size)
print("ALL OK")
