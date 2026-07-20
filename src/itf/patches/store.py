"""The store of patch datasets: what is on disk under `data/patch-datasets/`.

Domain logic, deliberately not in `api/app.py`. The mechanical rule of api.md §0:
**if a function of `app.py` does not mention HTTP, it is not the API's**. Listing
datasets and reading manifests mention no HTTP, so they live here -- and the CLI
gets them for free, which is the test that the boundary is real.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.npyio import NpzFile


class PatchDatasetNotFound(KeyError):
    """No such patch dataset."""


@dataclass(frozen=True)
class PatchDatasetStore:
    """`data/patch-datasets/<name>/`, one subdirectory per B."""

    root: Path

    def path(self, name: str) -> Path:
        # `name` reaches this from an URL, so it must not be able to escape the
        # store. A name is a single directory component: no separators, no `..`.
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"nombre de dataset de patches inválido: {name!r}")
        return self.root / name

    def exists(self, name: str) -> bool:
        return (self.path(name) / "manifest.json").exists()

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.parent.name for p in self.root.glob("*/manifest.json"))

    def manifest(self, name: str) -> dict:
        path = self.path(name) / "manifest.json"
        if not path.exists():
            raise PatchDatasetNotFound(name)
        return json.loads(path.read_text(encoding="utf-8"))

    def split(self, name: str) -> dict:
        path = self.path(name) / "split.json"
        if not path.exists():
            raise PatchDatasetNotFound(name)
        return json.loads(path.read_text(encoding="utf-8"))

    def split_map(self, name: str) -> dict[int, str]:
        """Source-image index -> split name. The A×B crossing of Predecir (⑥)."""
        return {
            index: split_name
            for split_name, indices in self.split(name).items()
            for index in indices
        }

    def npz_path(self, name: str) -> Path:
        """Where this B's arrays live. What `itf.patches.rows` takes."""
        path = self.path(name) / "patches.npz"
        if not path.exists():
            raise PatchDatasetNotFound(name)
        return path

    def arrays(self, name: str) -> NpzFile:
        """The `.npz`. **For consumers that want every row** -- see the warning.

        `np.load` itself is lazy, but indexing is NOT: the file is written with
        `savez_compressed`, so `data["X"][i]` inflates the entire 2,5 GB member
        to hand back 400 bytes (6,5 s, and that much RAM). Reading a *single*
        patch through here is the bug `itf.patches.rows` was written to remove --
        use `load_rows` for that. This stays for the bulk join in
        `diagnostics/service.py`, where one inflate genuinely beats millions of
        random page faults.
        """
        return np.load(self.npz_path(name))

    def delete(self, name: str) -> None:
        path = self.path(name)
        if not path.is_dir():
            raise PatchDatasetNotFound(name)
        shutil.rmtree(path)
