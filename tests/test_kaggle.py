"""Kaggle additions without a GPU: python tests/test_kaggle.py (needs pillow; the Drive sync part needs rclone,
found on PATH or through RCLONE=/path/to/rclone, and uses a local folder in place of Google Drive)."""
import os, sys, shutil, subprocess, tempfile, threading, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from imageto3d import config as cfgmod, assets, run, models, drive_sync

root = tempfile.mkdtemp()
cfg = cfgmod.Config(drive_root=os.path.join(root, "local"), hf_home=root + "/hf", rembg_home=root + "/rb")
os.makedirs(cfg.input_dir)
def img(folder, name):
    os.makedirs(folder, exist_ok=True)
    Image.new("RGB", (8, 8)).save(os.path.join(folder, name)); time.sleep(0.01)

# Model folder lookup: local copy when present, repo id otherwise; manifest found inside /kaggle/input-like trees.
model_root = os.path.join(root, "input", "datasets", "someone", "models")
os.makedirs(os.path.join(model_root, "madebyollin", "sdxl-vae-fp16-fix"))
open(os.path.join(model_root, models.MANIFEST), "w").write("{}")
cfg.model_root = model_root
assert cfg.model_source("madebyollin/sdxl-vae-fp16-fix") == os.path.join(model_root, "madebyollin", "sdxl-vae-fp16-fix")
assert cfg.model_source("huanngzh/mv-adapter") == "huanngzh/mv-adapter"
assert models.find_root(os.path.join(root, "input")) == model_root
assert models.find_root(os.path.join(root, "hf")) is None
spec = models.files()
assert "hunyuan3d-dit-v2-mv/model.fp16.safetensors" in spec["tencent/Hunyuan3D-2mv"]
assert "hunyuan3d-delight-v2-0/*" in spec["tencent/Hunyuan3D-2"] and not any("turbo" in p for p in sum(spec.values(), []))
assert any("turbo" in p for p in models.files(include_turbo=True)["tencent/Hunyuan3D-2"])
assert set(cfgmod.HIGH_QUALITY_PRESETS) == set(cfg.presets)
hq = cfgmod.Config(drive_root=root, hf_home="", rembg_home="", presets=cfgmod.HIGH_QUALITY_PRESETS)
assert hq.preset("weapon", "texture_size") == 4096 and hq.texture_size == 2048
cfg.model_root = ""

# Two workers share the queue through claims: each asset goes to exactly one worker.
for name in ("weapon-a-stuff.png", "weapon-b-stuff.png", "weapon-c-stuff.png", "armor-d-stuff.png",
             "object-e-building.png"):
    img(cfg.input_dir, name)
cfg_path = os.path.join(root, "config.json"); cfg.save(cfg_path)
GLB = os.path.join(root, "sample.glb")
import numpy as np, trimesh
box = trimesh.creation.box((0.5, 2.0, 0.3))
box.visual = trimesh.visual.TextureVisuals(uv=np.random.rand(len(box.vertices), 2), image=Image.new("RGB", (8, 8)))
box.export(GLB)
from imageto3d import stage_export
lock = threading.Lock()
done_by = {}
both_claimed = threading.Barrier(2, timeout=10)  # each worker holds its first batch before either works
started = set()
def fake_stage(stage, keys, config_path, config):
    if threading.current_thread().name not in started:
        started.add(threading.current_thread().name)
        both_claimed.wait()
    for asset in assets.find(config, keys):
        assert open(os.path.join(asset.work, assets.CLAIM_FILE)).read() in ("0", "1")
        if stage == "views":
            Image.new("RGB", (8, 8)).save(asset.view_path("front"))
        elif stage in ("shape", "paint"):
            shutil.copy(GLB, asset.shape_path if stage == "shape" else asset.textured_path)
        else:
            with lock:
                assert asset.key not in done_by, asset.key
                done_by[asset.key] = threading.current_thread().name
                stage_export.process(None, asset, config)
        time.sleep(0.05)
    return 0
run._run_stage = fake_stage
workers = [threading.Thread(target=run.main, args=(cfg_path,), kwargs={"watch": False, "worker": i}, name=str(i))
           for i in (0, 1)]
for worker in workers: worker.start()
for worker in workers: worker.join()
print("done by:", done_by)
assert sorted(done_by) == ["armor-d", "object-e", "weapon-a", "weapon-b", "weapon-c"], done_by
assert set(done_by.values()) == {"0", "1"}, "both workers should have taken assets"
assert not any(os.path.exists(os.path.join(cfg.work_dir, k, assets.CLAIM_FILE)) for k in done_by)
assert len([n for n in os.listdir(cfg.out_dir) if n.endswith(".glb")]) == 5

# Claims left behind by a killed worker are cleared at session start.
img(cfg.input_dir, "weapon-f-stuff.png")
f = [a for a in assets.pending(cfg) if a.key == "weapon-f"][0]
assert assets.claim(f, 0) and not assets.claim(f, 1)
assert run.claim_batch(cfg, 1) == []
assets.clear_claims(cfg)
assert [a.key for a in run.claim_batch(cfg, 1)] == ["weapon-f"]
assets.clear_claims(cfg)

# Deadline: no new asset starts after it.
cfg.deadline = time.time() - 1; cfg.save(cfg_path)
run.main(cfg_path, watch=True, worker=0)  # returns at once instead of waiting for images
assert not os.path.exists(os.path.join(cfg.out_dir, "hunyuan3d-weapon-f.glb"))
cfg.deadline = 0.0; cfg.save(cfg_path)

# Drive sync through rclone, with a plain folder standing in for the Drive remote.
rclone = os.environ.get("RCLONE") or shutil.which("rclone")
if not rclone:
    print("rclone not found; Drive sync part skipped")
else:
    drive = os.path.join(root, "drive")
    local = os.path.join(root, "kaggle")
    img(os.path.join(drive, "in"), "weapon-new-stuff.png")
    img(os.path.join(drive, "in"), "used-weapon-old-stuff.png")
    img(os.path.join(drive, "in"), "monster-g-front.png")
    img(os.path.join(drive, "in"), "monster-g-back.png")
    open(os.path.join(drive, "in", "notes.txt"), "w").write("x")
    os.makedirs(os.path.join(drive, "work", "monster-g")); open(os.path.join(drive, "work", "monster-g", "shape.glb"), "w").write("old")
    open(os.path.join(drive, "work", "monster-g", assets.CLAIM_FILE), "w").write("1")
    os.makedirs(os.path.join(drive, "work", "weapon-old")); open(os.path.join(drive, "work", "weapon-old", "x"), "w").write("x")
    os.makedirs(os.path.join(drive, "wheels", "py312")); open(os.path.join(drive, "wheels", "py312", "a.whl"), "w").write("w")
    sync = drive_sync.DriveSync(drive, local, rclone=rclone, filter_dir=root)
    sync.pull_inputs(); sync.pull_folder("wheels")
    assert sorted(os.listdir(os.path.join(local, "in"))) == ["monster-g-back.png", "monster-g-front.png", "weapon-new-stuff.png"]
    lcfg = cfgmod.Config(drive_root=local, hf_home="", rembg_home="")
    sync.pull_work([a.key for a in assets.pending(lcfg)])
    assert os.listdir(os.path.join(local, "work")) == ["monster-g"], os.listdir(os.path.join(local, "work"))
    assert os.listdir(os.path.join(local, "work", "monster-g")) == ["shape.glb"]
    assert os.path.exists(os.path.join(local, "wheels", "py312", "a.whl"))
    # Export here: the input is renamed locally, output and work files written.
    os.replace(os.path.join(local, "in", "weapon-new-stuff.png"), os.path.join(local, "in", "used-weapon-new-stuff.png"))
    os.makedirs(os.path.join(local, "out")); open(os.path.join(local, "out", "hunyuan3d-weapon-new.glb"), "w").write("g")
    os.makedirs(os.path.join(local, "work", "weapon-new")); open(os.path.join(local, "work", "weapon-new", assets.CLAIM_FILE), "w").write("0")
    time.sleep(1.1); open(os.path.join(local, "work", "monster-g", "shape.glb"), "w").write("new")
    # Before the push, a pull must not bring the finished original back.
    sync.pull_inputs()
    assert "weapon-new-stuff.png" not in os.listdir(os.path.join(local, "in"))
    sync.push()
    drive_in = sorted(os.listdir(os.path.join(drive, "in")))
    assert drive_in == ["monster-g-back.png", "monster-g-front.png", "notes.txt", "used-weapon-new-stuff.png",
                        "used-weapon-old-stuff.png"], drive_in
    assert os.path.exists(os.path.join(drive, "out", "hunyuan3d-weapon-new.glb"))
    assert open(os.path.join(drive, "work", "monster-g", "shape.glb")).read() == "new"
    assert not os.path.exists(os.path.join(drive, "work", "weapon-new", assets.CLAIM_FILE))
    # A newer image arriving in Drive is picked up; an older Drive copy never replaces newer local work.
    img(os.path.join(drive, "in"), "armor-late-stuff.png")
    sync.pull_inputs(); sync.pull_work(["monster-g"])
    assert "armor-late-stuff.png" in os.listdir(os.path.join(local, "in"))
    assert open(os.path.join(local, "work", "monster-g", "shape.glb")).read() == "new"
    assert sync.push_renames() == 0
print("ALL OK")

# manual mode: gather correctly named images from datasets / a downloaded folder, zip the results.
from imageto3d import kaggle_io
src_a = os.path.join(root, "dataset_a"); src_b = os.path.join(root, "drive_folder", "sub")
img(src_a, "weapon-x-stuff.png"); img(src_a, "readme.png"); img(src_a, "used-weapon-y-stuff.png")
img(src_b, "monster-z-front.png"); img(src_b, "monster-z-back.png"); img(src_b, "weapon-x-stuff.png")
manual_in = os.path.join(root, "manual", "in")
img(manual_in, "used-armor-w-stuff.png"); img(src_a, "armor-w-stuff.png")
copied = kaggle_io.collect_inputs([os.path.join(root, "drive_folder"), src_a], manual_in)
assert sorted(copied) == ["monster-z-back.png", "monster-z-front.png", "weapon-x-stuff.png"], copied
assert kaggle_io.collect_inputs([src_a, src_b], manual_in) == []
assert kaggle_io.finished_inputs(manual_in) == ["armor-w-stuff.png"]
archive = kaggle_io.zip_outputs(cfg.out_dir, root)
import zipfile
assert "hunyuan3d-weapon-a.glb" in zipfile.ZipFile(archive).namelist()
print("ALL OK (manual mode)")

# Packing for upload: files move into one zip (root emptied) and keep their relative paths.
pack_root = os.path.join(root, "pack", "models"); os.makedirs(os.path.join(pack_root, "tencent", "x"))
open(os.path.join(pack_root, "tencent", "x", "a.bin"), "wb").write(b"1" * 1000)
open(os.path.join(pack_root, models.MANIFEST), "w").write("{}")
archive = models.pack(pack_root, os.path.join(root, "pack", "upload"))
assert not os.path.exists(pack_root)
assert sorted(zipfile.ZipFile(archive).namelist()) == [models.MANIFEST, "tencent/x/a.bin"]
assert models.pack(pack_root, os.path.join(root, "pack", "upload")) == archive  # rerun keeps the archive
assert len(zipfile.ZipFile(archive).namelist()) == 2
print("ALL OK (pack)")

# Single GPU (Colab): rounds of claim_batch assets, so finished images are delivered batch by batch.
solo_root = os.path.join(root, "solo")
solo = cfgmod.Config(drive_root=solo_root, hf_home="", rembg_home="", claim_batch=2)
for name in ("weapon-p-stuff.png", "weapon-q-stuff.png", "weapon-r-stuff.png"):
    img(solo.input_dir, name)
solo_path = os.path.join(root, "solo.json"); solo.save(solo_path)
rounds = []
def solo_stage(stage, keys, config_path, config):
    rounds.append((stage, tuple(keys)))
    return fake_stage(stage, keys, config_path, config)
started.update({"MainThread"}); run._run_stage = solo_stage
for asset in assets.pending(solo): assets.claim(asset, 0)  # fake_stage expects a claim file
run.main(solo_path, watch=False)
exports = [keys for stage, keys in rounds if stage == "export"]
assert exports == [("weapon-p", "weapon-q"), ("weapon-r",)], rounds
assert sorted(os.listdir(solo.input_dir)) == ["used-weapon-p-stuff.png", "used-weapon-q-stuff.png", "used-weapon-r-stuff.png"]
print("ALL OK (rounds)")
