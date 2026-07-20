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
    #: The per-patch tables (D1). A CACHE: recomputable exactly from run +
    #: fingerprint + split, so deleting it loses nothing and `/data/cache/` is
    #: gitignored. It has a root of its own because it is the one directory in
    #: the project that is safe to throw away at any moment.
    diagnostics_cache_root: Path
    #: The byte-offset index over each source's `labels.jsonl`
    #: (`itf.datasets.index`). A CACHE for the same reason as the one above:
    #: recomputable from the file it indexes, keyed by that file's size+mtime,
    #: and safe to delete. It has to live under OUR root and not beside
    #: `labels.jsonl` because `datasets_root` is the generator's and read-only.
    #: Defaulted so existing fixtures need not pass it.
    sources_index_cache_root: Path = REPO_ROOT / "data" / "cache" / "sources"
    #: The uncompressed, memory-mappable copy of each B's per-row arrays
    #: (`itf.patches.rows`). A CACHE, same three properties: recomputable from
    #: `patches.npz`, keyed by its size+mtime, safe to delete. It exists because
    #: `patches.npz` is deflated, so reading ONE patch inflated the whole 2,5 GB
    #: array -- and the four probes of one gallery click did it four times over.
    #: Defaulted so existing fixtures need not pass it.
    patch_rows_cache_root: Path = REPO_ROOT / "data" / "cache" / "patch-rows"
    #: H — the sweeps (`sweeps/<name>/spec.json` + optuna's SQLite). Like `runs/`,
    #: it is a RECORD: the spec is versioned (formatos.md §5), the `.db` is load
    #: and ignored. A root of its own so a test can point it at throwaway disk.
    sweeps_root: Path = REPO_ROOT / "sweeps"
    #: X — the job queue's persisted records (jobq). Under `data/`, so gitignored:
    #: a job is ephemeral, and what survives a restart is the run/sweep behind it,
    #: not the job row. Defaulted so the existing test fixtures need not pass it.
    jobs_root: Path = REPO_ROOT / "data" / "jobs"
    #: A' — where the resize writes (D19). A SECOND source root, ours and
    #: writable, because `datasets_root` belongs to image-text-sample-generator
    #: and we only ever read it (organizacion.md §1-A). Under `data/`, so
    #: gitignored like everything derived: a resized source recomputes exactly
    #: from parent + request. Defaulted so existing fixtures need not pass it.
    derived_sources_root: Path = REPO_ROOT / "data" / "sources"
    #: Every root a client-supplied path is allowed to resolve under (D4).
    allowed_roots: tuple[Path, ...] = ()
    cors_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "Settings":
        datasets_root = _env_path("ITF_DATASETS_ROOT", DEFAULT_DATASETS_ROOT)
        data_root = _env_path("ITF_DATA_ROOT", REPO_ROOT / "data")
        runs_root = _env_path("ITF_RUNS_ROOT", REPO_ROOT / "runs")
        sweeps_root = _env_path("ITF_SWEEPS_ROOT", REPO_ROOT / "sweeps")
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
            # `/data/cache/` is already gitignored, which is the point: this
            # directory is derived, and D5's criterion is that what can be
            # recomputed is not kept.
            diagnostics_cache_root=data_root / "cache" / "diagnostics",
            sources_index_cache_root=data_root / "cache" / "sources",
            patch_rows_cache_root=data_root / "cache" / "patch-rows",
            sweeps_root=sweeps_root,
            jobs_root=data_root / "jobs",
            derived_sources_root=data_root / "sources",
            # Both source roots are readable; only the derived one is ever
            # written. D4 is about what a client-supplied path may resolve to,
            # and a derived source is addressed by id exactly like an original.
            allowed_roots=(datasets_root, data_root / "sources", *extra_roots),
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
