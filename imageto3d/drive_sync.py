"""Keeps a local copy of the Drive folder in step with Google Drive through rclone (Kaggle has no Drive mount).

The local copy sits at the same path as the Colab mount, so the pipeline code is unchanged:
  pull: new images in Drive/in, compiled wheels, and the work folders of pending assets (to resume)
  push: renames of finished inputs to used-..., then out/, work/ and wheels/
Pulls and pushes use --update, so a newer file on either side is never replaced by an older one.
"""
import os
import subprocess
from typing import Iterable, List

from .assets import CLAIM_FILE, IMAGE_EXTENSIONS

GLOB_CHARACTERS = "\\*?[]{}"


def _escape(name: str) -> str:
    return "".join("\\" + char if char in GLOB_CHARACTERS else char for char in name)


class DriveSync:
    def __init__(self, remote: str, local: str, rclone: str = "rclone", filter_dir: str = "/tmp"):
        self.remote = remote.rstrip("/")
        self.local = local.rstrip("/")
        self.rclone = rclone
        self.filter_path = os.path.join(filter_dir, "imageto3d-rclone-filter.txt")

    def _run(self, *args: str) -> str:
        command = [self.rclone, *args, "--retries", "3", "--low-level-retries", "10"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"rclone {' '.join(args[:3])} failed: {result.stderr.strip()[-1500:]}")
        return result.stdout

    def _remote_files(self, folder: str) -> List[str]:
        try:
            return [line for line in self._run("lsf", f"{self.remote}/{folder}", "--files-only").splitlines()
                    if line]
        except RuntimeError:
            return []  # The folder does not exist in Drive yet.

    def _write_filter(self, lines: Iterable[str]) -> str:
        with open(self.filter_path, "w") as file:
            file.write("".join(line + "\n" for line in lines))
        return self.filter_path

    def pull_inputs(self) -> None:
        """Copy unused images from Drive/in. Images already finished here (renamed used-...) are skipped,
        so a rename that has not reached Drive yet cannot bring the original back."""
        local_in = os.path.join(self.local, "in")
        os.makedirs(local_in, exist_ok=True)
        finished = [name[len("used-"):] for name in os.listdir(local_in) if name.startswith("used-")]
        rules = [f"- /{_escape(name)}" for name in finished] + ["- used-*"]
        rules += [f"+ *{extension}" for extension in IMAGE_EXTENSIONS]
        rules += [f"+ *{extension.upper()}" for extension in IMAGE_EXTENSIONS] + ["- *"]
        self._run("copy", f"{self.remote}/in", local_in, "--update", "--filter-from", self._write_filter(rules))

    def pull_folder(self, folder: str) -> None:
        os.makedirs(os.path.join(self.local, folder), exist_ok=True)
        self._run("copy", f"{self.remote}/{folder}", os.path.join(self.local, folder), "--update")

    def pull_work(self, keys: Iterable[str]) -> None:
        """Copy the work folders of these assets (intermediate files from an earlier session)."""
        keys = sorted(set(keys))
        if not keys:
            return
        rules = [f"- {CLAIM_FILE}"] + [f"+ /{_escape(key)}/**" for key in keys] + ["- *"]
        os.makedirs(os.path.join(self.local, "work"), exist_ok=True)
        self._run("copy", f"{self.remote}/work", os.path.join(self.local, "work"), "--update",
                  "--filter-from", self._write_filter(rules))

    def push_renames(self) -> int:
        """Rename finished inputs in Drive/in to used-..., as the export stage did here."""
        local_in = os.path.join(self.local, "in")
        if not os.path.isdir(local_in):
            return 0
        remote_names = set(self._remote_files("in"))
        renamed = 0
        for name in sorted(os.listdir(local_in)):
            original = name[len("used-"):]
            if name.startswith("used-") and original in remote_names and name not in remote_names:
                self._run("moveto", f"{self.remote}/in/{original}", f"{self.remote}/in/{name}")
                renamed += 1
        return renamed

    def push(self) -> None:
        self.push_renames()
        for folder in ("out", "work", "wheels"):
            if os.path.isdir(os.path.join(self.local, folder)):
                self._run("copy", os.path.join(self.local, folder), f"{self.remote}/{folder}", "--update",
                          "--exclude", CLAIM_FILE)
