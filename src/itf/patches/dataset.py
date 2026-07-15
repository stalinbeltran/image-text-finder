"""torch Dataset over a patch ``.npz`` produced by :mod:`itf.patches.extract`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from itf.patches.extract import SPLIT_NAMES


class PatchDataset(Dataset):
    """Yields ``(x, y)`` where ``x`` is ``(1, n, n)`` float in [0,1] and ``y`` is ``(4, 3)``."""

    def __init__(self, npz_path: str | Path, split: str | None = None):
        path = Path(npz_path)
        if path.is_dir():
            path = path / "patches.npz"
        data = np.load(path)
        X, y, split_arr = data["X"], data["y"], data["split"]
        if split is not None:
            if split not in SPLIT_NAMES:
                raise ValueError(f"split must be one of {SPLIT_NAMES}")
            sel = split_arr == SPLIT_NAMES.index(split)
            X, y = X[sel], y[sel]
        # (N, n, n, 1) uint8 -> (N, 1, n, n) float32 in [0,1]
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2).float().div_(255.0).contiguous()
        self.y = torch.from_numpy(y).float().contiguous()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]
