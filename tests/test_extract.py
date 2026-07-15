"""Tests for the patch extractor over the bundled 'reducido' dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from itf.datasets.loader import CORNER_NAMES, SourceDataset
from itf.patches.extract import PatchExtractConfig, SplitConfig, _positions, extract_dataset

SOURCE = "C:/Desarrollo/image-text-sample-generator/data/datasets/clean-paragraphs-01/reducido"

pytestmark = pytest.mark.skipif(
    not Path(SOURCE, "labels.jsonl").exists(), reason="source dataset not available"
)


def test_positions_cover_edges():
    xs = _positions(160, 40, 20)
    assert xs[0] == 0
    assert xs[-1] == 120  # flush to the right edge (120 + 40 == 160)


def test_positions_flush_when_not_divisible():
    xs = _positions(100, 40, 30)
    # 0, 30, 60 then flush 60 -> last must be 100-40 = 60 (already there)
    assert xs[-1] == 60
    xs2 = _positions(110, 40, 30)
    assert xs2[-1] == 70  # flush appended


def test_extract_shapes_and_labels(tmp_path):
    cfg = PatchExtractConfig(
        source=SOURCE,
        out=str(tmp_path / "out"),
        patch_size=40,
        stride=20,
        split=SplitConfig(0.8, 0.1, 0.1),
        seed=1,
    )
    summary = extract_dataset(cfg)

    data = np.load(tmp_path / "out" / "patches.npz")
    X, y = data["X"], data["y"]
    assert X.dtype == np.uint8
    assert X.shape[1:] == (40, 40, 1)
    assert y.shape[1:] == (4, 3)
    assert X.shape[0] == y.shape[0] == summary["num_patches"]
    # exists flags are strictly 0/1
    assert set(np.unique(y[:, :, 0]).tolist()) <= {0.0, 1.0}
    # normalized coordinates are within [0, 1]
    coords = y[:, :, 1:]
    assert coords.min() >= 0.0 and coords.max() <= 1.0


def test_label_reconstructs_true_corner(tmp_path):
    """A positive patch's (x,y) label must map back to the real quad corner."""
    cfg = PatchExtractConfig(source=SOURCE, out=str(tmp_path / "out"), patch_size=40, stride=20)
    extract_dataset(cfg)
    data = np.load(tmp_path / "out" / "patches.npz")
    X, y, sample_idx, patch_xy = data["X"], data["y"], data["sample_idx"], data["patch_xy"]

    # Ground-truth corner points per sample index.
    ds = SourceDataset(SOURCE)
    truth: dict[int, list[tuple[int, float, float]]] = {}
    for s in ds.samples():
        pts = []
        for b in s.blocks_of_kind(["paragraph"]):
            for c in range(4):
                pts.append((c, float(b.quad[c, 0]), float(b.quad[c, 1])))
        truth[s.index] = pts

    n = 40
    checked = 0
    for i in range(X.shape[0]):
        x0, y0 = int(patch_xy[i, 0]), int(patch_xy[i, 1])
        for c in range(4):
            if y[i, c, 0] == 1.0:
                gx = x0 + y[i, c, 1] * n
                gy = y0 + y[i, c, 2] * n
                # there must be a matching true corner of this type nearby
                cands = [(px, py) for (ct, px, py) in truth[int(sample_idx[i])] if ct == c]
                dists = [abs(px - gx) + abs(py - gy) for (px, py) in cands]
                assert min(dists) < 1.0, f"corner {CORNER_NAMES[c]} mismatch at patch {i}"
                checked += 1
    assert checked > 0
