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
assert knight.startswith("Straight-on front view of a knight. symmetrical full body") and "bald" in knight and "watercolor" in knight and "grey background" in knight
assert "photo," in imagegen.full_prompt(items["weapon-axe"], defaults) and "watercolor" not in imagegen.full_prompt(items["weapon-axe"], defaults)
assert "a green slime. symmetrical full body creature" in imagegen.full_prompt(items["monster-gelly"], defaults)  # trailing period dropped
assert imagegen.full_prompt(items["bigobject-gate"], defaults).startswith("Straight-on front elevation of a gate. perfectly symmetrical facade")
loose = imagegen.full_prompt({"name": "bigobject-tower", "prompt": "a leaning tower", "symmetric": False}, {})
assert "symmetrical" not in loose and "front elevation" in loose
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

# Long descriptions: FLUX reads 256 T5 tokens; view words and 3D conditions are kept, the description's end is cut.
words = lambda text: len(text.replace(",", " ").replace(".", " ").split())
lead, subject, conditions = imagegen.prompt_parts({"name": "bigobject-bridge", "prompt": "a " + "very " * 300 + "long bridge."}, {})
text, dropped = imagegen.fit_prompt(lead, subject, conditions, words, 256)
assert text.startswith("Straight-on front elevation of a very") and text.endswith(conditions) and words(text) <= 256
assert dropped.endswith("long bridge"), dropped[-30:]
assert imagegen.fit_prompt(lead, "a short bridge", conditions, words, 256) == (f"{lead} a short bridge. {conditions}", "")
assert imagegen.fit_prompt(lead, "x " * 10, conditions, words, 3)[0].endswith(conditions)  # nothing fits: conditions kept
clip = imagegen.clip_prompt({"name": "weapon-a", "prompt": "a dagger, curved blade"}, {})
assert clip.startswith("Straight-on front view of a dagger. symmetrical single weapon") and clip.endswith("a dagger, curved blade")

# Symmetry check: a straight-on symmetrical shape scores high, a lopsided (turned) one low.
from PIL import ImageDraw
def shape(points, size=200):
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0)); draw = ImageDraw.Draw(image)
    draw.polygon(points, fill=(150, 90, 60, 255)); return image
house = shape([(50, 190), (150, 190), (150, 90), (100, 30), (50, 90)])
turned = shape([(30, 190), (110, 190), (110, 80), (70, 30), (30, 80)])      # a house turned 45 degrees:
ImageDraw.Draw(turned).polygon([(110, 190), (180, 160), (180, 70), (110, 80)], fill=(70, 40, 30, 255))  # dark side wall
assert imagegen.symmetry_score(house) > 0.95 and imagegen.symmetry_score(turned) < 0.75, (
    imagegen.symmetry_score(house), imagegen.symmetry_score(turned))
assert imagegen.symmetry_score(Image.new("RGBA", (10, 10))) == 0.0

# Views share one scale, centre and baseline, whatever size and position they were generated at.
front = shape([(20, 120), (80, 120), (80, 20), (20, 20)])            # 60 wide, 100 tall, low in the frame
side = shape([(100, 60), (130, 60), (130, 160), (100, 160)], 300)   # 30 wide, 100 tall, elsewhere
placed = imagegen.normalize_views({"front": front, "left": side}, 512, 0.8)
boxes = {view: image.getchannel("A").getbbox() for view, image in placed.items()}
assert all(image.size == (512, 512) for image in placed.values())
assert boxes["front"][3] == boxes["left"][3]                         # same baseline
assert abs((boxes["front"][3] - boxes["front"][1]) - (boxes["left"][3] - boxes["left"][1])) <= 1  # same height
assert abs((boxes["front"][0] + boxes["front"][2]) / 2 - 256) <= 1 and abs((boxes["left"][0] + boxes["left"][2]) / 2 - 256) <= 1
assert abs((boxes["front"][3] - boxes["front"][1]) - 0.8 * 512) <= 2  # tallest fills 80 %
wide = imagegen.normalize_views({"front": shape([(0, 90), (199, 90), (199, 110), (0, 110)])}, 512, 0.8)["front"]
box = wide.getchannel("A").getbbox(); assert box[2] - box[0] <= 0.92 * 512 + 1   # wide subjects stay in frame
print("ALL OK (long prompts)")
