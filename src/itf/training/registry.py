"""E — the run registry: what is on disk under `runs/`.

Only the READING half exists so far. Fase 4 adds the loop that writes; fase 2
needs this much because deleting a B has to know which runs point at it
(contract ③), and that question is asked of E, not of B.

It lives here and not in `api/app.py` for the mechanical reason of api.md §0
(nothing here mentions HTTP), and not in `itf.patches` for a layering one: B must
not know that runs exist. The API composes the two.

This module is the seed of `exp-registry` (librerias.md). It is not extracted
yet, and must not be: **the rule is to extract on the second use, not the
first.** It has exactly one user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunStore:
    """`runs/<name>/`, one subdirectory per E."""

    root: Path

    def path(self, name: str) -> Path:
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"nombre de run inválido: {name!r}")
        return self.root / name

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.parent.name for p in self.root.glob("*/config.json"))

    def config(self, name: str) -> dict:
        path = self.path(name) / "config.json"
        if not path.exists():
            raise KeyError(name)
        return json.loads(path.read_text(encoding="utf-8"))

    def using_patch_dataset(self, patch_dataset: str) -> list[str]:
        """Which runs were trained on this B.

        The question `DELETE /patch-datasets/{name}` must ask before destroying
        anything. Getting it wrong is not loud: removing a B in use leaves every
        run that pointed at it without provenance, silently -- the old code
        followed `run.config.data` -> `manifest.config.source` and just returned
        `None` when the directory had gone, so the Predict tab lost its dataset
        without saying why.

        Reads the provenance by NAME (D2). A run whose `config.json` has no
        provenance is not a legacy case to tolerate -- D3 killed that -- but this
        query still must not explode on a half-written run: a config being
        written right now would take the whole delete down with it, and "I could
        not read one run" is not an answer to "is this in use".
        """
        used_by = []
        for name in self.names():
            try:
                config = self.config(name)
            except (json.JSONDecodeError, OSError):
                continue
            provenance = config.get("provenance") or {}
            referenced = (provenance.get("patch_dataset") or {}).get("name")
            if referenced == patch_dataset:
                used_by.append(name)
        return used_by
