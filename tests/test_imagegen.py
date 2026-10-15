"""prompts.json handling for the image notebook, without a GPU: python tests/test_imagegen.py (needs pillow)."""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from imageto3d import imagegen, assets, config as cfgmod

root = tempfile.mkdtemp()
prompts = os.path.join(root, "prompts.json")
def write(data):
    json.dump(data, open(prompts, "w"))
def fails(data, text):
    write(data)
    try:
        imagegen.load_prompts(prompts)
    except ValueError as error:
        assert text in str(error), error
    else:
        raise AssertionError(f"accepted {data}")

fails({"items": [{"name": "sword-a", "prompt": "x"}]}, "must be <category>-<id>")
fails({"items": [{"name": "weapon", "prompt": "x"}]}, "must be <category>-<id>")
fails({"items": [{"name": "weapon-a", "prompt": " "}]}, "empty prompt")
fails({"items": [{"name": "weapon-a", "prompt": "x", "views": ["top"]}]}, "unknown views")
fails({"items": [{"name": "weapon-a", "prompt": "x"}, {"name": "Weapon-A", "prompt": "y"}]}, "duplicate")
fails({"things": []}, "items")

write({"defaults": {"seed": 7, "style": "watercolor"},
       "items": [{"name": "Monster-Gelly", "prompt": "a green slime.", "views": ["back"]},
                 {"name": "character-knight", "prompt": "a knight", "views": ["right", "front", "left"], "seed": 3},
                 {"name": "weapon-old", "prompt": "a dagger", "done": True},
                 {"name": "weapon-axe", "prompt": "an axe", "style": "photo"},
                 {"name": "bigobject-gate", "prompt": "a gate", "views": ["front", "back"]}]})
data = imagegen.load_prompts(prompts)
items = {item["name"]: item for item in data["items"]}
assert items["monster-gelly"]["views"] == ["front", "back"]          # front added, lower-cased name
assert items["character-knight"]["views"] == ["front", "left", "right"]  # canonical order
assert items["weapon-axe"]["views"] == ["front"]
defaults = data["defaults"]
knight = imagegen.full_prompt(items["character-knight"], defaults)
assert knight.startswith("a knight, full body") and "bald" in knight and "watercolor" in knight and "grey background" in knight
assert "photo," in imagegen.full_prompt(items["weapon-axe"], defaults) and "watercolor" not in imagegen.full_prompt(items["weapon-axe"], defaults)
assert "slime, full body creature" in imagegen.full_prompt(items["monster-gelly"], defaults)  # trailing period dropped
assert "photorealistic" in imagegen.full_prompt({"name": "prop-a", "prompt": "a cup"}, {})
assert imagegen.seed(items["character-knight"], defaults) == 3 and imagegen.seed(items["weapon-axe"], defaults) == 7

# Two GPUs split the unfinished items; done items are skipped.
todo = [i["name"] for i in data["items"] if not i.get("done")]
w0 = [i["name"] for i in imagegen.assigned(data, 0, 2)]; w1 = [i["name"] for i in imagegen.assigned(data, 1, 2)]
assert sorted(w0 + w1) == sorted(todo) and not set(w0) & set(w1) and "weapon-old" not in w0 + w1, (w0, w1)

# Outputs, done marking and names the 3D pipeline accepts.
cfg = imagegen.ImageGenConfig(root=root, prompts_path=prompts, hf_home="", rembg_home="")
assert imagegen.missing_views(cfg, items["monster-gelly"]) == ["front", "back"]
os.makedirs(cfg.input_dir)
for name in ("monster-gelly-front", "monster-gelly-back", "character-knight-front", "weapon-axe-front"):
    Image.new("RGBA", (4, 4)).save(os.path.join(cfg.input_dir, name + ".png"))
assert sorted(imagegen.mark_done(cfg, data)) == ["monster-gelly", "weapon-axe"]
assert imagegen.mark_done(cfg, data) == []
imagegen.save_prompts(data, os.path.join(root, "out.json"))
again = imagegen.load_prompts(os.path.join(root, "out.json"))
assert [i["name"] for i in imagegen.assigned(again, 0, 1)] == ["character-knight", "bigobject-gate"]
pipeline = cfgmod.Config(drive_root=root, hf_home="", rembg_home="")
# The 3D pipeline picks up every generated set (a character makes its own missing views there).
assert sorted(a.key for a in assets.pending(pipeline)) == ["character-knight", "monster-gelly", "weapon-axe"]
cfg.save(os.path.join(root, "cfg.json")); assert imagegen.load_config(os.path.join(root, "cfg.json")) == cfg
assert cfg.pipeline_config().rembg_model == "isnet-general-use"
print("ALL OK")
