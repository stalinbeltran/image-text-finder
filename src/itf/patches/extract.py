"""Patch-dataset builder (component 1).

Slides an ``n x n`` window over every image of a SourceDataset and, for each
patch, records whether each of the four paragraph corners (TL, TR, BR, BL)
falls inside it and, if so, its position within the patch (normalized to
``[0, 1]``). The result is a packed ``.npz`` that the CNN consumes directly.

Label array per patch has shape ``(4, 3)`` == ``[exists, x, y]`` per corner
type, in the fixed order :data:`itf.datasets.loader.CORNER_NAMES`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from itf.datasets.loader import NUM_BORDERS, NUM_CORNERS, SourceDataset

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitConfig:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def normalized(self) -> tuple[float, float, float]:
        total = self.train + self.val + self.test
        if total <= 0:
            raise ValueError("split fractions must sum to > 0")
        return (self.train / total, self.val / total, self.test / total)


@dataclass
class PatchExtractConfig:
    source: str
    out: str
    patch_size: int = 40
    stride: int = 20
    target_kinds: list[str] = field(default_factory=lambda: ["paragraph"])
    drop_overlap: bool = False
    split: SplitConfig = field(default_factory=SplitConfig)
    seed: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "PatchExtractConfig":
        d = dict(d)
        if isinstance(d.get("split"), dict):
            d["split"] = SplitConfig(**d["split"])
        return cls(**d)


def _positions(size: int, n: int, stride: int) -> list[int]:
    """Top-left offsets so the window covers ``[0, size)`` including a flush edge."""
    if n > size:
        raise ValueError(f"patch_size ({n}) larger than image dimension ({size})")
    last = size - n
    xs = list(range(0, last + 1, stride))
    if xs[-1] != last:
        xs.append(last)
    return xs


def _assign_splits(n_samples: int, split: SplitConfig, seed: int) -> np.ndarray:
    """Map each sample index -> split id (0 train, 1 val, 2 test)."""
    frac_train, frac_val, _ = split.normalized()
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    n_train = int(round(frac_train * n_samples))
    n_val = int(round(frac_val * n_samples))
    # Keep at least the requested test remainder; clamp so counts fit.
    n_train = min(n_train, n_samples)
    n_val = min(n_val, n_samples - n_train)
    ids = np.full(n_samples, 2, dtype=np.int8)  # default: test
    ids[order[:n_train]] = 0
    ids[order[n_train : n_train + n_val]] = 1
    return ids


def _corners_for_sample(sample, target_kinds) -> list[tuple[int, float, float]]:
    """Flatten paragraph quads into (corner_type, x, y) points in image pixels."""
    pts: list[tuple[int, float, float]] = []
    for block in sample.blocks_of_kind(target_kinds):
        for c in range(NUM_CORNERS):
            x, y = float(block.quad[c, 0]), float(block.quad[c, 1])
            pts.append((c, x, y))
    return pts


def extract_dataset(config: PatchExtractConfig) -> dict:
    """Build the patch dataset and write ``patches.npz`` + manifest to ``config.out``."""
    ds = SourceDataset(config.source)
    samples = ds.samples()
    if config.drop_overlap:
        samples = [s for s in samples if not s.has_overlap]
    if not samples:
        raise ValueError("no samples left after filtering")

    split_by_sample = _assign_splits(len(samples), config.split, config.seed)

    n = config.patch_size
    stride = config.stride
    half = n / 2.0

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    border_list: list[tuple[int, int, int, int]] = []
    sample_idx_list: list[int] = []
    patch_xy_list: list[tuple[int, int]] = []
    split_list: list[int] = []

    for pos, sample in enumerate(samples):
        img = sample.load_image()  # (H, W) uint8
        h, w = img.shape
        corners = _corners_for_sample(sample, config.target_kinds)
        split_id = int(split_by_sample[pos])

        for y0 in _positions(h, n, stride):
            for x0 in _positions(w, n, stride):
                patch = img[y0 : y0 + n, x0 : x0 + n]
                # Which source-image edges this patch is flush against, in
                # BORDER_NAMES order (top, right, bottom, left). Lost on crop.
                border = (
                    int(y0 == 0),
                    int(x0 + n >= w),
                    int(y0 + n >= h),
                    int(x0 == 0),
                )
                label = np.zeros((NUM_CORNERS, 3), dtype=np.float32)
                # nearest-to-center distance per corner type, to break ties
                best_d = [np.inf] * NUM_CORNERS
                for ctype, cx, cy in corners:
                    if x0 <= cx <= x0 + n and y0 <= cy <= y0 + n:
                        lx = (cx - x0) / n
                        ly = (cy - y0) / n
                        d = (cx - (x0 + half)) ** 2 + (cy - (y0 + half)) ** 2
                        if d < best_d[ctype]:
                            best_d[ctype] = d
                            label[ctype] = (1.0, min(max(lx, 0.0), 1.0), min(max(ly, 0.0), 1.0))
                X_list.append(patch)
                y_list.append(label)
                border_list.append(border)
                sample_idx_list.append(sample.index)
                patch_xy_list.append((x0, y0))
                split_list.append(split_id)

    X = np.stack(X_list).astype(np.uint8)[..., None]  # (N, n, n, 1)
    y = np.stack(y_list).astype(np.float32)            # (N, 4, 3)
    border = np.asarray(border_list, dtype=np.uint8)   # (N, 4)
    sample_idx = np.asarray(sample_idx_list, dtype=np.int32)
    patch_xy = np.asarray(patch_xy_list, dtype=np.int32)
    split_arr = np.asarray(split_list, dtype=np.int8)

    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "patches.npz",
        X=X,
        y=y,
        border=border,
        sample_idx=sample_idx,
        patch_xy=patch_xy,
        split=split_arr,
    )

    summary = _summarize(config, ds, samples, split_by_sample, X, y, split_arr)
    (out_dir / "manifest.json").write_text(json.dumps(summary["manifest"], indent=2), encoding="utf-8")
    (out_dir / "split.json").write_text(json.dumps(summary["split"], indent=2), encoding="utf-8")
    return summary


def _summarize(config, ds, samples, split_by_sample, X, y, split_arr) -> dict:
    from itf.datasets.loader import BORDER_NAMES, CORNER_NAMES

    pos_per_corner = {
        CORNER_NAMES[c]: int(y[:, c, 0].sum()) for c in range(NUM_CORNERS)
    }
    counts_by_split = {
        SPLIT_NAMES[s]: int((split_arr == s).sum()) for s in range(3)
    }
    split_samples = {
        SPLIT_NAMES[s]: [int(samples[i].index) for i in range(len(samples)) if split_by_sample[i] == s]
        for s in range(3)
    }
    cfg_dict = asdict(config)
    manifest = {
        "source_id": ds.id,
        "config": cfg_dict,
        "num_samples": len(samples),
        "num_patches": int(X.shape[0]),
        "patch_shape": list(X.shape[1:]),
        "label_shape": list(y.shape[1:]),
        "corner_order": list(CORNER_NAMES),
        "border_order": list(BORDER_NAMES),
        "patches_per_split": counts_by_split,
        "positives_per_corner": pos_per_corner,
    }
    return {"manifest": manifest, "split": split_samples,
            "num_patches": int(X.shape[0]), "positives_per_corner": pos_per_corner}
