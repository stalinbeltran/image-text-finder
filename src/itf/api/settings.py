"""API paths/settings, overridable via environment variables."""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).resolve()


# Root under which source datasets (dirs with labels.jsonl) are discovered.
DATASETS_ROOT = _env_path(
    "ITF_DATASETS_ROOT",
    "C:/Desarrollo/image-text-sample-generator/data/datasets",
)
# Where built patch datasets, training runs, and saved model configs live.
PATCH_DATASETS_DIR = _env_path("ITF_PATCH_DATASETS_DIR", "data/patch-datasets")
RUNS_DIR = _env_path("ITF_RUNS_DIR", "runs")
MODELS_DIR = _env_path("ITF_MODELS_DIR", "configs/models")


def ensure_dirs() -> None:
    for p in (PATCH_DATASETS_DIR, RUNS_DIR, MODELS_DIR):
        p.mkdir(parents=True, exist_ok=True)
