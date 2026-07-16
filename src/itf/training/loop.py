"""Training loop for the configurable corner-detection CNN (component 2).

Writes a self-contained run directory:

    runs/<id>/
    ├── config.json      # frozen run config (provenance)
    ├── metrics.jsonl    # one line per epoch (train/val metrics) -- pollable live
    ├── best.pt          # checkpoint with best val loss
    └── last.pt          # checkpoint after the final epoch
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from itf.datasets.loader import CORNER_NAMES
from itf.models.builder import build_model
from itf.patches.dataset import PatchDataset
from itf.training.losses import CornerLoss

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


@dataclass
class RunConfig:
    data: str                       # patch-dataset dir or patches.npz
    out: str                        # run output dir
    model: dict = field(default_factory=dict)
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    lambda_pos: float = 1.0
    pos_weight: float | None = None
    seed: int = 1
    device: str = "cpu"
    num_workers: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def _make_optimizer(name: str, params, lr: float, weight_decay: float):
    key = name.lower()
    if key not in _OPTIMIZERS:
        raise ValueError(f"unknown optimizer '{name}', options: {sorted(_OPTIMIZERS)}")
    return _OPTIMIZERS[key](params, lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, patch_size: int) -> dict:
    model.eval()
    tot_loss = tot_cls = tot_pos = 0.0
    n_batches = 0
    tp = fp = fn = tn = 0
    pos_err_sum = 0.0
    pos_count = 0
    for x, border, y in loader:
        x, border, y = x.to(device), border.to(device), y.to(device)
        pred = model(x, border)
        out = loss_fn(pred, y)
        tot_loss += float(out["loss"]); tot_cls += float(out["cls_loss"]); tot_pos += float(out["pos_loss"])
        n_batches += 1

        prob = torch.sigmoid(pred[..., 0])
        pred_pos = prob > 0.5
        tgt_pos = y[..., 0] > 0.5
        tp += int((pred_pos & tgt_pos).sum())
        fp += int((pred_pos & ~tgt_pos).sum())
        fn += int((~pred_pos & tgt_pos).sum())
        tn += int((~pred_pos & ~tgt_pos).sum())

        mask = tgt_pos
        if mask.any():
            err = (pred[..., 1:3] - y[..., 1:3]).abs().sum(-1)  # (B,4) in patch units
            pos_err_sum += float((err[mask]).sum()) * patch_size
            pos_count += int(mask.sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "loss": tot_loss / max(n_batches, 1),
        "cls_loss": tot_cls / max(n_batches, 1),
        "pos_loss": tot_pos / max(n_batches, 1),
        "exists_acc": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pos_err_px": pos_err_sum / pos_count if pos_count else None,
    }


def train(config: RunConfig, on_epoch: Callable[[int, dict], None] | None = None) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)

    train_ds = PatchDataset(config.data, split="train")
    val_ds = PatchDataset(config.data, split="val")
    has_val = len(val_ds) > 0

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, num_workers=config.num_workers) \
        if has_val else None

    model = build_model(config.model).to(device)
    patch_size = model.config.input_size
    loss_fn = CornerLoss(lambda_pos=config.lambda_pos, pos_weight=config.pos_weight).to(device)
    optimizer = _make_optimizer(config.optimizer, model.parameters(), config.lr, config.weight_decay)

    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    best_val = float("inf")
    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        n_batches = 0
        for x, border, y in train_loader:
            x, border, y = x.to(device), border.to(device), y.to(device)
            optimizer.zero_grad()
            out = loss_fn(model(x, border), y)
            out["loss"].backward()
            optimizer.step()
            run_loss += out["loss"].detach().item(); n_batches += 1

        train_loss = run_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, loss_fn, device, patch_size) if val_loader else {}
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "seconds": round(time.time() - t0, 3),
        }
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        if on_epoch:
            on_epoch(epoch, record)

        torch.save({"model": model.state_dict(), "config": asdict(config), "epoch": epoch},
                   out_dir / "last.pt")
        monitor = val_metrics.get("loss", train_loss)
        if monitor <= best_val:
            best_val = monitor
            torch.save({"model": model.state_dict(), "config": asdict(config), "epoch": epoch},
                       out_dir / "best.pt")

    summary = {
        "run": str(out_dir),
        "epochs": config.epochs,
        "best_val_loss": best_val if has_val else None,
        "final": history[-1] if history else {},
        "corner_order": list(CORNER_NAMES),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
