"""The store of networks: `configs/networks/*.yaml`.

YAML, not JSON: **a person writes these**. And a deliberate contrast with
everything else the project writes -- these are versioned in git and are not
artifacts. They are source (formatos.md §4.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

#: Bumped only when a field has to be REJECTED or migrated, never out of habit
#: (formatos.md §3). Additive sniffing is cheap and works.
FORMAT_VERSION = 1


class NetworkNotFound(KeyError):
    """No such network."""


@dataclass(frozen=True)
class NetworkStore:
    root: Path

    def path(self, name: str) -> Path:
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"nombre de red inválido: {name!r}")
        return self.root / f"{name}.yaml"

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.yaml"))

    def get(self, name: str) -> dict:
        path = self.path(name)
        if not path.exists():
            raise NetworkNotFound(name)
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def save(self, name: str, config: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": FORMAT_VERSION, **config}
        self.path(name).write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    def delete(self, name: str) -> None:
        path = self.path(name)
        if not path.exists():
            raise NetworkNotFound(name)
        path.unlink()
