"""B — the extractor. Invariants, not results (tests.md §6).

Rebuilt from the tag's `test_extract.py`, with one change worth naming: the old
suite pointed at a REAL dataset by absolute path and skipped when it was missing.
So it was green on this machine and skipped everywhere else, which is a test that
does not run. These build their own source (tests.md §7).
"""

from __future__ import annotations

import numpy as np
import pytest

from itf.geometry import CORNER_NAMES, positions
from itf.patches import PatchExtractConfig, SplitConfig, extract_dataset

from conftest import write_tiny_source


def test_positions_cover_the_edges():
    xs = positions(160, 40, 20)
    assert xs[0] == 0
    assert xs[-1] == 120  # flush right: 120 + 40 == 160


def test_positions_append_a_flush_window_when_stride_does_not_divide():
    """The strip at the far edge must not be invisible to every window.

    `size - n` is rarely a multiple of `stride`, and without the flush position
    the last few columns of every image would never be seen. It costs one extra
    overlapping window per axis.
    """
    assert positions(100, 40, 30)[-1] == 60  # 0, 30, 60 -- already flush
    assert positions(110, 40, 30)[-1] == 70  # 0, 30, 60, then 70 appended


def test_positions_refuse_a_patch_larger_than_the_image():
    with pytest.raises(ValueError, match="mayor que la dimensión"):
        positions(30, 40, 20)


def test_shapes_and_dtypes_match_the_format(tmp_path):
    source = write_tiny_source(tmp_path / "src")
    out = tmp_path / "out"
    extract_dataset(PatchExtractConfig(source=str(source), out=str(out), patch_size=40, stride=20))

    with np.load(out / "patches.npz") as data:
        X, y, border = data["X"], data["y"], data["border"]
        assert X.dtype == np.uint8 and X.shape[1:] == (40, 40, 1)
        assert y.dtype == np.float32 and y.shape[1:] == (4, 3)
        assert X.shape[0] == y.shape[0] == border.shape[0]

        assert border.shape[1:] == (4,)
        assert set(np.unique(border).tolist()) <= {0, 1}
        # Every grid has at least one window flush against each of the 4 edges.
        assert border.max(axis=0).min() == 1

        assert set(np.unique(y[:, :, 0]).tolist()) <= {0.0, 1.0}
        coords = y[:, :, 1:]
        assert coords.min() >= 0.0 and coords.max() <= 1.0


def test_label_reconstructs_the_true_corner(tmp_path):
    """The one that would catch a real bug: a positive label must map BACK.

    `(x, y)` is normalised inside the patch, so `x0 + x*n` has to land on the
    actual quad corner in the image. This is what makes the labels mean what the
    loss thinks they mean -- everything else here is shape checking.
    """
    source = write_tiny_source(tmp_path / "src")
    out = tmp_path / "out"
    extract_dataset(PatchExtractConfig(source=str(source), out=str(out), patch_size=40, stride=20))

    from itf.datasets import SourceDataset

    truth: dict[int, list[tuple[int, float, float]]] = {}
    for s in SourceDataset(source).samples():
        truth[s.index] = [
            (c, float(b.quad[c, 0]), float(b.quad[c, 1]))
            for b in s.blocks_of_kind(["paragraph"])
            for c in range(4)
        ]

    n = 40
    checked = 0
    with np.load(out / "patches.npz") as data:
        y, sample_idx, patch_xy = data["y"], data["sample_idx"], data["patch_xy"]
        for i in range(y.shape[0]):
            x0, y0 = int(patch_xy[i, 0]), int(patch_xy[i, 1])
            for c in range(4):
                if y[i, c, 0] == 1.0:
                    gx = x0 + y[i, c, 1] * n
                    gy = y0 + y[i, c, 2] * n
                    candidates = [(px, py) for (ct, px, py) in truth[int(sample_idx[i])] if ct == c]
                    dists = [abs(px - gx) + abs(py - gy) for (px, py) in candidates]
                    assert min(dists) < 1.0, f"esquina {CORNER_NAMES[c]} mal en el patch {i}"
                    checked += 1
    assert checked > 0, "el test no probó nada: no hubo ni una esquina positiva"


def test_manifest_agrees_with_the_npz(tmp_path):
    """formatos.md §4.1: if these disagree there is corruption, and it is testable."""
    source = write_tiny_source(tmp_path / "src")
    out = tmp_path / "out"
    summary = extract_dataset(
        PatchExtractConfig(source=str(source), out=str(out), patch_size=40, stride=20)
    )
    manifest = summary["manifest"]

    with np.load(out / "patches.npz") as data:
        assert manifest["num_patches"] == data["X"].shape[0]
        assert manifest["patch_shape"] == list(data["X"].shape[1:])
        assert manifest["label_shape"] == list(data["y"].shape[1:])
        assert sum(manifest["patches_per_split"].values()) == data["X"].shape[0]

    assert manifest["corner_order"] == list(CORNER_NAMES)
    assert manifest["has_border"] is True
    assert manifest["format_version"] == 1


def test_empty_val_split_is_reported_as_a_warning(tmp_path):
    """protocolo.md §1.3: a B with no val is not a measuring tool.

    `reducido-40` split 5 images into 4/0/1, so `monitor` fell back to the train
    loss and `best.pt` became the most overfitted checkpoint -- silently, and it
    was the README's own example. Building warns; TRAINING must refuse (fase 4).
    """
    source = write_tiny_source(tmp_path / "src", num_samples=2)
    out = tmp_path / "out"
    summary = extract_dataset(
        PatchExtractConfig(
            source=str(source), out=str(out), patch_size=40, stride=20, split=SplitConfig(1.0, 0.0, 0.0)
        )
    )
    codes = [w["code"] for w in summary["manifest"]["warnings"]]
    assert codes == ["empty_val_split"]
    assert summary["manifest"]["warnings"][0]["hint"]


def test_a_healthy_split_warns_about_nothing(tmp_path):
    """The control: the warning must mean something, not fire always."""
    source = write_tiny_source(tmp_path / "src", num_samples=10)
    out = tmp_path / "out"
    summary = extract_dataset(
        PatchExtractConfig(
            source=str(source), out=str(out), patch_size=40, stride=20, split=SplitConfig(0.6, 0.2, 0.2)
        )
    )
    assert summary["manifest"]["warnings"] == []
    assert summary["manifest"]["patches_per_split"]["val"] > 0
