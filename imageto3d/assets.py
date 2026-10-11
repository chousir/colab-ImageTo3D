"""Finds pending front images and tracks each asset's progress through files in Drive."""
import glob
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import CATEGORIES, CATEGORY_MODES, EXTRA_VIEWS, NAME_SUFFIXES, REQUIRED_VIEWS, SOURCE_TAG, Config

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
VIEW_NAMES = ("front", "left", "back", "right")
_warned = set()


@dataclass
class Asset:
    category: str
    asset_id: str
    source: str
    work: str
    # Views the user supplied in Drive/in, e.g. {"back": ".../bigobject-gate-back.png"}.
    supplied: Dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.category}-{self.asset_id}"

    @property
    def mode(self) -> str:
        return CATEGORY_MODES[self.category]

    @property
    def generates_views(self) -> bool:
        """Only multiview categories without required views (character) fill gaps with MV-Adapter."""
        return self.mode == "multiview" and self.category not in REQUIRED_VIEWS

    @property
    def generated_views(self) -> List[str]:
        """Views MV-Adapter must generate because the user did not supply them."""
        return [view for view in EXTRA_VIEWS if self.generates_views and view not in self.supplied]

    @property
    def views(self) -> tuple:
        """Every view image this asset has in its work folder: front, supplied and generated views."""
        return tuple(view for view in VIEW_NAMES
                     if view == "front" or view in self.supplied or view in self.generated_views)

    @property
    def shape_views(self) -> List[str]:
        """Views fed to the shape model. Generated sets use front, left and back as in the official
        examples (a generated right is only for review); supplied views are always used."""
        if self.mode == "single":
            return ["front"]
        if self.generates_views:
            return ["front", "left", "back"] + (["right"] if "right" in self.supplied else [])
        return [view for view in VIEW_NAMES if view == "front" or view in self.supplied]

    @property
    def sources(self) -> List[str]:
        return [self.source] + list(self.supplied.values())

    @property
    def output_name(self) -> str:
        return f"{SOURCE_TAG}-{self.category}-{self.asset_id}"

    def view_path(self, view: str) -> str:
        return os.path.join(self.work, "front.png" if view == "front" else f"{view}.jpg")

    @property
    def views_sheet(self) -> str:
        return os.path.join(self.work, "views.jpg")

    @property
    def shape_path(self) -> str:
        return os.path.join(self.work, "shape.glb")

    @property
    def textured_path(self) -> str:
        return os.path.join(self.work, "textured.glb")

    def output_path(self, config: Config, extension: str) -> str:
        return os.path.join(config.out_dir, self.output_name + extension)

    def needs(self, stage: str, config: Config) -> bool:
        """True when the stage output is missing or older than its input."""
        if stage == "views":
            return _stale([self.view_path(view) for view in self.views], self.sources)
        if stage == "shape":
            return _stale([self.shape_path], [self.view_path(view) for view in self.views])
        if stage == "paint":
            return _stale([self.textured_path], [self.shape_path])
        if stage == "export":
            return _stale([self.output_path(config, ".glb")], [self.textured_path])
        raise ValueError(stage)


def _stale(outputs: List[str], inputs: List[str]) -> bool:
    if not all(os.path.exists(path) for path in outputs):
        return True
    newest_input = max((os.path.getmtime(path) for path in inputs if os.path.exists(path)), default=0)
    return min(os.path.getmtime(path) for path in outputs) < newest_input


def parse_name(path: str) -> Optional[Dict[str, str]]:
    """<category>-<id>-<front|stuff>.<ext>; the first word picks the category settings."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    if stem.startswith("used-"):
        return None
    category, _, rest = stem.partition("-")
    asset_id, _, suffix = rest.rpartition("-")
    if category in REQUIRED_VIEWS and suffix != "front" and suffix in NAME_SUFFIXES and asset_id:
        _warn_once(path, f"WARNING: {os.path.basename(path)} ignored; {category} main image must end in -front")
        return None
    if category in CATEGORIES and suffix in NAME_SUFFIXES and asset_id:
        return {"category": category, "asset_id": asset_id}
    return None


def _warn_once(token: str, message: str) -> None:
    if token not in _warned:
        _warned.add(token)
        print(message, flush=True)


def parse_extra_view(path: str) -> Optional[Dict[str, str]]:
    """<category>-<id>-<left|back|right>.<ext> for multiview categories."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    if stem.startswith("used-"):
        return None
    category, _, rest = stem.partition("-")
    asset_id, _, view = rest.rpartition("-")
    if CATEGORY_MODES.get(category) == "multiview" and view in EXTRA_VIEWS and asset_id:
        return {"key": f"{category}-{asset_id}", "view": view}
    return None


def pending(config: Config) -> List[Asset]:
    """Unused front images in Drive/in, oldest first so earlier uploads finish first."""
    paths = [path for path in glob.glob(os.path.join(config.input_dir, "*"))
             if path.lower().endswith(IMAGE_EXTENSIONS)]
    assets: Dict[str, Asset] = {}
    for path in sorted(paths, key=os.path.getmtime):
        parsed = parse_name(path)
        if parsed is None:
            continue
        key = f"{parsed['category']}-{parsed['asset_id']}"
        if key in assets:
            _warn_once(path, f"WARNING: duplicate front image for {key}; ignoring {os.path.basename(path)}")
            continue
        assets[key] = Asset(parsed["category"], parsed["asset_id"], path, os.path.join(config.work_dir, key))
    for path in paths:
        extra = parse_extra_view(path)
        if extra and extra["key"] in assets:
            assets[extra["key"]].supplied[extra["view"]] = path
    ready = []
    for asset in assets.values():
        missing = [view for view in REQUIRED_VIEWS.get(asset.category, ()) if view not in asset.supplied]
        if missing:
            _warn_once(f"{asset.key}:{','.join(missing)}",
                       f"WAITING {asset.key}: missing " + ", ".join(f"{asset.key}-{view}" for view in missing)
                       + " (skipped until the required views are in Drive/in)")
            continue
        ready.append(asset)
    return ready


def find(config: Config, keys: List[str]) -> List[Asset]:
    by_key = {asset.key: asset for asset in pending(config)}
    return [by_key[key] for key in keys if key in by_key]


# Failure bookkeeping: a stage writes a marker before each asset, so a crash (for example an
# out-of-memory kill) can be charged to the asset that caused it.

def _status_path(asset: Asset) -> str:
    return os.path.join(asset.work, "status.json")


def read_status(asset: Asset) -> Dict:
    try:
        with open(_status_path(asset)) as file:
            return json.load(file)
    except (OSError, ValueError):
        return {"failures": 0}


def write_status(asset: Asset, **updates) -> None:
    os.makedirs(asset.work, exist_ok=True)
    status = read_status(asset)
    status.update(updates, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(_status_path(asset), "w") as file:
        json.dump(status, file, indent=2)


def begin(asset: Asset, stage: str) -> None:
    write_status(asset, running=stage)


def succeed(asset: Asset, stage: str) -> None:
    write_status(asset, running=None, last_stage=stage, last_error=None)


def fail(asset: Asset, stage: str, error: str) -> None:
    status = read_status(asset)
    write_status(asset, running=None, failures=status.get("failures", 0) + 1,
                 last_error=f"{stage}: {error}"[-2000:])


def blocked(asset: Asset, config: Config) -> bool:
    return read_status(asset).get("failures", 0) >= config.max_failures
