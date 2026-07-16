"""Where things live, and which roots are allowed to be read.

Paths and origins, driven by the environment. The domain modules do NOT import
this: they take explicit paths. Only the entry points (the API, the CLIs) resolve
where things are, which is what keeps `extract_dataset` callable on any directory
from a test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: The repo root, from src/itf/settings.py.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: A is an EXTERNAL dependency disguised as a local folder: it belongs to
#: image-text-sample-generator and we only ever read it.
DEFAULT_DATASETS_ROOT = REPO_ROOT.parent / "image-text-sample-generator" / "data" / "datasets"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


@dataclass(frozen=True)
class Settings:
    datasets_root: Path
    patch_datasets_root: Path
    runs_root: Path
    #: C and D live in `configs/`, and unlike everything else they are SOURCE:
    #: versioned in git, written by a person (formatos.md §4.3).
    networks_root: Path
    recipes_root: Path
    #: Every root a client-supplied path is allowed to resolve under (D4).
    allowed_roots: tuple[Path, ...]
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        datasets_root = _env_path("ITF_DATASETS_ROOT", DEFAULT_DATASETS_ROOT)
        data_root = _env_path("ITF_DATA_ROOT", REPO_ROOT / "data")
        runs_root = _env_path("ITF_RUNS_ROOT", REPO_ROOT / "runs")
        configs_root = _env_path("ITF_CONFIGS_ROOT", REPO_ROOT / "configs")

        # D4: the allowlist is DATASETS_ROOT plus whatever is declared on
        # purpose. Declaring roots keeps the "just point at that folder"
        # convenience without the API being able to read the whole disk.
        extra = os.environ.get("ITF_EXTRA_ROOTS", "")
        extra_roots = tuple(Path(p).expanduser().resolve() for p in extra.split(os.pathsep) if p.strip())

        # Closed, not `*`. The old API combined `allow_origins=["*"]` with
        # reading any image on disk by absolute path, which meant any page you
        # visited while it ran could enumerate your Pictures folder. Nobody chose
        # that: it accumulated. And the GPU is what turns it from modest to not.
        origins = os.environ.get("ITF_CORS_ORIGINS", "http://localhost:5173")
        return cls(
            datasets_root=datasets_root,
            patch_datasets_root=data_root / "patch-datasets",
            runs_root=runs_root,
            # `configs/networks/`, not `configs/models/`: "model" is the ambiguous
            # word -- it means C or E depending on who is talking -- and api.md R2
            # makes it disappear from the vocabulary. The directory was empty, so
            # the rename cost nothing (formatos.md §4.3).
            networks_root=configs_root / "networks",
            recipes_root=configs_root / "recipes",
            allowed_roots=(datasets_root, *extra_roots),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        )


class PathNotAllowed(PermissionError):
    """A path that does not hang off any allowed root (D4)."""


def resolve_within(path: str | Path, allowed_roots: tuple[Path, ...]) -> Path:
    """Resolve ``path`` and refuse it unless it lives under an allowed root.

    **The order matters and is the whole point**: resolve FIRST, check AFTER.
    Checked before resolving, `..\\..\\` walks straight out of the allowlist and
    the check reads as if it passed.
    """
    resolved = Path(path).expanduser().resolve()
    for root in allowed_roots:
        if resolved == root or root in resolved.parents:
            return resolved
    raise PathNotAllowed(str(resolved))
