"""Tests for the configurable CNN, corner loss, and torch dataset."""

from __future__ import annotations

import numpy as np
import torch

from itf.models import build_model
from itf.training.losses import CornerLoss

MODEL_CFG = {
    "input_size": 40,
    "in_channels": 1,
    "backbone": [
        {"filters": 8, "kernel": 3, "batchnorm": True, "activation": "relu", "pool": 2},
        {"filters": 16, "kernel": 3, "activation": "relu", "pool": 2},
    ],
    "head": {"hidden": [32], "dropout": 0.0},
}


def test_forward_shape_and_ranges():
    model = build_model(MODEL_CFG)
    out = model(torch.zeros(4, 1, 40, 40)).detach()
    assert out.shape == (4, 4, 3)
    xy = out[..., 1:3]
    assert float(xy.min()) >= 0.0 and float(xy.max()) <= 1.0  # sigmoid-bounded


def test_loss_masks_position_by_target():
    loss_fn = CornerLoss(lambda_pos=1.0)
    pred = torch.zeros(2, 4, 3)
    pred[..., 1:3] = 0.5
    target = torch.zeros(2, 4, 3)
    # no positives -> position loss must be exactly zero
    out = loss_fn(pred, target)
    assert float(out["pos_loss"]) == 0.0
    # add a positive with a coordinate mismatch -> position loss > 0
    target[0, 0, 0] = 1.0
    target[0, 0, 1:3] = torch.tensor([0.9, 0.1])
    out2 = loss_fn(pred, target)
    assert float(out2["pos_loss"]) > 0.0


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model = build_model(MODEL_CFG)
    loss_fn = CornerLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.rand(16, 1, 40, 40)
    y = torch.zeros(16, 4, 3)
    y[:, 0, 0] = 1.0
    y[:, 0, 1:3] = 0.5
    first = last = None
    for _ in range(20):
        opt.zero_grad()
        out = loss_fn(model(x), y)
        out["loss"].backward()
        opt.step()
        if first is None:
            first = float(out["loss"])
        last = float(out["loss"])
    assert last < first


def test_patch_dataset_split(tmp_path):
    from itf.patches.dataset import PatchDataset

    n, N = 40, 20
    X = (np.random.rand(N, n, n, 1) * 255).astype(np.uint8)
    y = np.zeros((N, 4, 3), dtype=np.float32)
    split = np.array([0] * 10 + [1] * 5 + [2] * 5, dtype=np.int8)
    np.savez(tmp_path / "patches.npz", X=X, y=y,
             sample_idx=np.zeros(N, np.int32), patch_xy=np.zeros((N, 2), np.int32), split=split)
    ds = PatchDataset(tmp_path, split="train")
    assert len(ds) == 10
    xb, yb = ds[0]
    assert xb.shape == (1, 40, 40)
    assert 0.0 <= float(xb.max()) <= 1.0
    assert yb.shape == (4, 3)
