"""torch Dataset over a patch `.npz`.

Loads the whole thing into RAM as float32, which is why concurrency is not free:
40x40 float32 is 6.4 KB per patch, so 100k patches ≈ 640 MB, times two for
train+val, times N for N concurrent runs. That is half of why the CPU worker
limit is 1 (organizacion.md §3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from itf.geometry import NUM_BORDERS
from itf.patches.extract import SPLIT_NAMES


class PatchDataset(Dataset):
    """Yields `(x, border, y)`: `(1, n, n)` float in [0,1], `(4,)` flags, `(4, 3)`.

    **On `border`: this class states a fact, it does not set policy.**

    A `.npz` without the `border` array gets zeros -- and zeros are harmless
    exactly as long as nobody reads them (formatos.md §2). Whether anyone reads
    them depends on the NETWORK's `border_features`, and a dataset loader does not
    know the network. So it exposes `has_border` and lets the validator decide;
    the old docstring said "no border known", which was a lie: the model cannot
    tell "unknown" from "touches no border".

    Since D18 there are no border-less `.npz` left -- every new dataset carries
    them. The fallback stays because it costs nothing and the array is optional
    by format; what does NOT exist is a path that fills the data in and lets a
    network that needs it train anyway. That is `itf.validation`'s job.
    """

    #: Row of each item in the FULL `.npz`, in this dataset's order. Identity when
    #: unsplit; the selected rows when split.
    #:
    #: Filtering by split is what makes this necessary: it renumbers everything
    #: from 0, so item 3 of the val set is not row 3 of the file, and `sample_idx`
    #: / `patch_xy` / the pixels are all indexed by the file's numbering. The
    #: per-patch table stores this as `patch_idx` (formatos.md §4.4) so a
    #: diagnostics row can be joined back to the patch it came from. Without it
    #: the table could report an error and not be able to say for which patch --
    #: which is most of what V6 and V7 are for.
    indices: np.ndarray

    def __init__(self, npz_path: str | Path, split: str | None = None):
        path = Path(npz_path)
        if path.is_dir():
            path = path / "patches.npz"
        with np.load(path) as data:
            X, y, split_arr = data["X"], data["y"], data["split"]
            self.has_border = "border" in data.files
            border = (
                data["border"]
                if self.has_border
                else np.zeros((X.shape[0], NUM_BORDERS), dtype=np.uint8)
            )
            self.indices = np.arange(X.shape[0], dtype=np.int32)
            if split is not None:
                if split not in SPLIT_NAMES:
                    raise ValueError(f"split tiene que ser uno de {SPLIT_NAMES}")
                sel = split_arr == SPLIT_NAMES.index(split)
                X, y, border = X[sel], y[sel], border[sel]
                self.indices = self.indices[sel]
            # (N, n, n, 1) uint8 -> (N, 1, n, n) float32 in [0, 1]
            self.X = torch.from_numpy(X).permute(0, 3, 1, 2).float().div_(255.0).contiguous()
            self.y = torch.from_numpy(y).float().contiguous()
            self.border = torch.from_numpy(border).float().contiguous()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[i], self.border[i], self.y[i]
