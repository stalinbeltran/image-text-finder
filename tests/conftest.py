"""Shared fixtures.

tests.md §7: tiny synthetic datasets, built in the test. A test NEVER touches the
real `data/` or `runs/` -- which is not fussiness, it is why the suite can run
anywhere and why `test_contract_03` can delete things without fear.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


def write_tiny_source(root: Path, *, num_samples: int = 6, width: int = 120, height: int = 100) -> Path:
    """A minimal source dataset (A), in the generator's SAMPLE_FORMAT.

    Only the fields formatos.md §4.5 says we consume. One axis-aligned paragraph
    per image, placed differently in each so the corner labels are not all the
    same.
    """
    (root / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(num_samples):
        img = Image.new("L", (width, height), color=255)
        x0, y0 = 10 + (i % 3) * 6, 12 + (i % 2) * 8
        x1, y1 = x0 + 62, y0 + 44
        ImageDraw.Draw(img).rectangle([x0, y0, x1, y1], fill=0)
        img.save(root / "images" / f"{i:06d}.png")
        records.append(
            {
                "index": i,
                "image": f"images/{i:06d}.png",
                "labels": {
                    "image_id": f"tiny/{i:06d}",
                    "width": width,
                    "height": height,
                    "has_overlap": False,
                    "blocks": [
                        {
                            "block_id": "b0",
                            "kind": "paragraph",
                            "text": "x",
                            "angle": 0.0,
                            "box": [x0, y0, x1 - x0, y1 - y0],
                            # clockwise from TL, as CORNER_NAMES expects
                            "quad": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                        }
                    ],
                },
            }
        )
    (root / "labels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return root


@dataclass
class Layout:
    """A throwaway disk layout: sources, patch datasets, configs and runs."""

    datasets: Path
    patch_datasets: Path
    runs: Path
    networks: Path
    recipes: Path

    def write_source(self, name: str = "tiny", **kwargs) -> str:
        write_tiny_source(self.datasets / name, **kwargs)
        return name

    def write_patch_dataset(
        self, name: str, *, patch_size: int = 40, has_border: bool = True, **manifest
    ) -> Path:
        """A B that exists as far as the store is concerned.

        Only a manifest: what `used_by`, `DELETE` and the ①/② validator care
        about is what B DECLARES, not what the pixels are -- the validator
        compares two dicts and never opens the `.npz`, which is exactly what
        makes it milliseconds. Building a real one would test the extractor
        instead of the endpoint.
        """
        path = self.patch_datasets / name
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "fingerprint": "sha256:" + "0" * 64,
            "has_border": has_border,
            "config": {"patch_size": patch_size, "stride": patch_size // 2, "seed": 1},
            "patch_shape": [patch_size, patch_size, 1],
            "corner_order": ["TL", "TR", "BR", "BL"],
            "border_order": ["top", "right", "bottom", "left"],
            "patches_per_split": {"train": 8, "val": 1, "test": 1},
            **manifest,
        }
        (path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        return path

    def write_network(self, name: str = "cnn-a", **overrides) -> dict:
        """A C in the store, as `POST /networks` would have left it."""
        from itf.models import NetworkStore

        config = {
            "input_size": 40,
            "in_channels": 1,
            "border_features": False,
            "backbone": [{"filters": 4, "kernel": 3, "stride": 1, "padding": 1, "pool": 2}],
            "head": {"hidden": [8]},
            **overrides,
        }
        NetworkStore(self.networks).save(name, config)
        return config

    def write_recipe(self, name: str = "adam-lr1e-3", **overrides):
        """A D in the store."""
        from itf.training import Recipe, RecipeStore

        recipe = Recipe(**{"epochs": 1, "batch_size": 8, **overrides})
        RecipeStore(self.recipes).save(name, recipe)
        return recipe

    def write_run(self, name: str, *, patch_dataset: str) -> Path:
        """An E whose provenance points at a B, in the shape D2 fixed."""
        path = self.runs / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "provenance": {
                        "patch_dataset": {"name": patch_dataset, "fingerprint": "sha256:" + "0" * 64},
                        "network": {"name": "cnn-a", "value": {}},
                        "recipe": {"name": "adam-lr1e-3", "value": {}},
                        "sweep": None,
                        "git_commit": "0" * 40,
                        "environment": {"python": "3.12.10", "torch": "2.13.0+cpu", "platform": "win32"},
                    },
                }
            ),
            encoding="utf-8",
        )
        return path


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    return Layout(
        datasets=tmp_path / "sources",
        patch_datasets=tmp_path / "patch-datasets",
        runs=tmp_path / "runs",
        networks=tmp_path / "configs" / "networks",
        recipes=tmp_path / "configs" / "recipes",
    )


@pytest.fixture
def itf_api(layout: Layout):
    """The API pointed at a throwaway layout.

    Built through `create_app(settings)` rather than by importing the module-level
    `app`, which is exactly why that factory exists: the module-level one reads
    the environment and would be aimed at the real `data/` and `runs/`.
    """
    from fastapi.testclient import TestClient

    from itf.api.app import create_app
    from itf.settings import Settings

    for path in (layout.datasets, layout.patch_datasets, layout.runs, layout.networks, layout.recipes):
        path.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        datasets_root=layout.datasets,
        patch_datasets_root=layout.patch_datasets,
        runs_root=layout.runs,
        networks_root=layout.networks,
        recipes_root=layout.recipes,
        allowed_roots=(layout.datasets,),
        cors_origins=("http://localhost:5173",),
    )
    with TestClient(create_app(settings)) as client:
        yield client, layout
