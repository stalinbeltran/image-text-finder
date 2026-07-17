"""B — builds the patch dataset: the data the CNN actually consumes.

Slides an ``n x n`` window over every image and records, per patch, whether each
of the four paragraph corners falls inside it and where. The result is a packed
`.npz` that is **self-contained**: once built it has the pixels, and training
never needs A again.

This is where `n` is decided, and therefore where contract ① is born.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from itf.datasets.loader import SourceDataset
from itf.geometry import BORDER_NAMES, CORNER_NAMES, NUM_CORNERS, windows

#: 0 train, 1 val, 2 test -- the int->name mapping travels in the manifest,
#: because it cannot be deduced from the array (formatos.md §4.1).
SPLIT_NAMES = ("train", "val", "test")

#: Everything present from day one (D18: no old .npz to come from), so 1.
FORMAT_VERSION = 1


@dataclass
class SplitConfig:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def normalized(self) -> tuple[float, float, float]:
        total = self.train + self.val + self.test
        if total <= 0:
            raise ValueError("las fracciones del split deben sumar > 0")
        return (self.train / total, self.val / total, self.test / total)


@dataclass
class PatchExtractConfig:
    """What defines a B. Its identity, and what the fingerprint must track."""

    source: str
    out: str
    patch_size: int = 40
    stride: int = 20
    target_kinds: list[str] = field(default_factory=lambda: ["paragraph"])
    drop_overlap: bool = False
    split: SplitConfig = field(default_factory=SplitConfig)
    #: Seed of B: fixes WHICH IMAGES land in train/val/test. Not the seed of D
    #: (weight init and shuffle). In a sweep this one stays FIXED -- same split
    #: or you measure split noise instead of recipe quality (contrato ⑧).
    seed: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "PatchExtractConfig":
        d = dict(d)
        if isinstance(d.get("split"), dict):
            d["split"] = SplitConfig(**d["split"])
        return cls(**d)


def _assign_splits(n_samples: int, split: SplitConfig, seed: int) -> np.ndarray:
    """Sample index -> split id.

    **Per image, never per patch**, and it is not obvious: windows overlap by
    50%, so splitting per patch would put near-identical crops of one image in
    train and in val, and val would be contaminated. This invariant is tested.
    """
    frac_train, frac_val, _ = split.normalized()
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    n_train = min(int(round(frac_train * n_samples)), n_samples)
    n_val = min(int(round(frac_val * n_samples)), n_samples - n_train)
    ids = np.full(n_samples, 2, dtype=np.int8)  # the remainder is test
    ids[order[:n_train]] = 0
    ids[order[n_train : n_train + n_val]] = 1
    return ids


def _corners_for_sample(sample, target_kinds) -> list[tuple[int, float, float]]:
    """Flatten the quads into (corner_type, x, y) in image pixels."""
    return [
        (c, float(block.quad[c, 0]), float(block.quad[c, 1]))
        for block in sample.blocks_of_kind(target_kinds)
        for c in range(NUM_CORNERS)
    ]


def fingerprint(arrays: dict[str, np.ndarray]) -> str:
    """A hash of the CONTENT (contract ⑧).

    Without this, a B rebuilt under the same name is indistinguishable from the
    one before it, and a sweep interrupted halfway becomes incomparable **in
    silence** -- `data` is a path, and a rebuilt path still points the same way.

    Hashes the arrays, not the file: `np.savez_compressed` writes a zip, and zip
    members carry timestamps, so identical content would hash differently every
    build. Name, dtype and shape go in too -- same bytes under a different shape
    is different data.
    """
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return "sha256:" + h.hexdigest()


def extract_dataset(config: PatchExtractConfig) -> dict:
    """Build the patch dataset: `patches.npz` + `manifest.json` + `split.json`."""
    ds = SourceDataset(config.source)
    samples = ds.samples()
    if config.drop_overlap:
        samples = [s for s in samples if not s.has_overlap]
    if not samples:
        raise ValueError("no queda ninguna imagen tras filtrar")

    split_by_sample = _assign_splits(len(samples), config.split, config.seed)

    n = config.patch_size
    half = n / 2.0

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    border_list: list[tuple[int, int, int, int]] = []
    sample_idx_list: list[int] = []
    patch_xy_list: list[tuple[int, int]] = []
    split_list: list[int] = []

    for pos, sample in enumerate(samples):
        img = sample.load_image()
        h, w = img.shape
        corners = _corners_for_sample(sample, config.target_kinds)
        split_id = int(split_by_sample[pos])

        # THE window (contract ⑤). Inference calls this same function; if these
        # two ever disagree, the model meets a geometry it never trained on.
        for win in windows(w, h, n, config.stride):
            x0, y0 = win.x0, win.y0
            label = np.zeros((NUM_CORNERS, 3), dtype=np.float32)
            # A patch can contain two corners of the same type (two paragraphs
            # stacked). Only one slot exists per type, so the tie is broken by
            # distance to the patch centre: the centred one is the one this
            # window is really "about", and the other will be centred in its own.
            best_d = [np.inf] * NUM_CORNERS
            for ctype, cx, cy in corners:
                if x0 <= cx <= x0 + n and y0 <= cy <= y0 + n:
                    d = (cx - (x0 + half)) ** 2 + (cy - (y0 + half)) ** 2
                    if d < best_d[ctype]:
                        best_d[ctype] = d
                        lx = (cx - x0) / n
                        ly = (cy - y0) / n
                        label[ctype] = (1.0, min(max(lx, 0.0), 1.0), min(max(ly, 0.0), 1.0))
            X_list.append(img[y0 : y0 + n, x0 : x0 + n])
            y_list.append(label)
            border_list.append(win.border)
            sample_idx_list.append(sample.index)
            patch_xy_list.append((x0, y0))
            split_list.append(split_id)

    arrays = {
        "X": np.stack(X_list).astype(np.uint8)[..., None],  # (N, n, n, 1)
        "y": np.stack(y_list).astype(np.float32),  # (N, 4, 3)
        "border": np.asarray(border_list, dtype=np.uint8),  # (N, 4)
        "sample_idx": np.asarray(sample_idx_list, dtype=np.int32),
        "patch_xy": np.asarray(patch_xy_list, dtype=np.int32),
        "split": np.asarray(split_list, dtype=np.int8),
    }

    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "patches.npz", **arrays)

    manifest = _manifest(config, ds, samples, split_by_sample, arrays)
    split_json = {
        SPLIT_NAMES[s]: [int(samples[i].index) for i in range(len(samples)) if split_by_sample[i] == s]
        for s in range(3)
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "split.json").write_text(json.dumps(split_json, indent=2), encoding="utf-8")
    return {"manifest": manifest, "split": split_json}


def _manifest(config, ds, samples, split_by_sample, arrays: dict[str, np.ndarray]) -> dict:
    y = arrays["y"]
    split_arr = arrays["split"]
    num_patches = int(arrays["X"].shape[0])

    patches_per_split = {SPLIT_NAMES[s]: int((split_arr == s).sum()) for s in range(3)}
    positives_per_corner = {CORNER_NAMES[c]: int(y[:, c, 0].sum()) for c in range(NUM_CORNERS)}

    warnings: list[dict] = []
    # protocolo.md §1.3: a patch dataset with no val IS NOT A MEASURING TOOL.
    # `reducido-40` split 5 images into 4/0/1, so `monitor` silently fell back to
    # the train loss and `best.pt` became the most overfitted checkpoint -- and it
    # was the README's own example. Building it warns; TRAINING on it must refuse
    # (fase 4). "Falla o avisa", and the hard stop belongs where the damage is.
    if patches_per_split["val"] == 0:
        warnings.append(
            {
                "code": "empty_val_split",
                "message": (
                    f"el split de val quedó vacío: {len(samples)} imágenes con "
                    f"{config.split.train}/{config.split.val}/{config.split.test}"
                ),
                "hint": "sube el número de imágenes o la fracción de val; sin val no se puede medir",
            }
        )

    return {
        "format_version": FORMAT_VERSION,
        "fingerprint": fingerprint(arrays),
        # DECLARED, not inferred (formatos.md §2). Absent means "I don't have
        # them", which is the safe default -- it refuses more than it should,
        # never less. It is what the ② validator reads.
        "has_border": True,
        "source_id": ds.id,
        "config": asdict(config),
        "num_samples": len(samples),
        "num_patches": num_patches,
        "patch_shape": list(arrays["X"].shape[1:]),
        "label_shape": list(y.shape[1:]),
        # Semantics, not decoration: lose the order and the arrays still load and
        # mean something else.
        "corner_order": list(CORNER_NAMES),
        "border_order": list(BORDER_NAMES),
        "patches_per_split": patches_per_split,
        "positives_per_corner": positives_per_corner,
        "warnings": warnings,
    }
