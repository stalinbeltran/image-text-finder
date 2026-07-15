"""Loss for per-patch corner detection.

    L = Σ_c [ BCE(exists_c) + λ · exists_c · smoothL1((x_c, y_c)) ]

The position term is masked by the *target* existence flag, so patches without
a corner of type ``c`` contribute nothing to that corner's regression loss.
Both terms are differentiable and trained jointly.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CornerLoss(nn.Module):
    def __init__(self, lambda_pos: float = 1.0, pos_weight: float | None = None):
        super().__init__()
        self.lambda_pos = lambda_pos
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight)) if pos_weight is not None else None,
            persistent=False,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """pred/target: (B, 4, 3) == [exists, x, y] per corner type."""
        exists_logit = pred[..., 0]
        exists_tgt = target[..., 0]
        xy_pred = pred[..., 1:3]
        xy_tgt = target[..., 1:3]

        cls_loss = F.binary_cross_entropy_with_logits(
            exists_logit, exists_tgt, pos_weight=self.pos_weight
        )

        # per-element smooth L1, masked by target existence
        per_coord = F.smooth_l1_loss(xy_pred, xy_tgt, reduction="none").sum(dim=-1)  # (B, 4)
        mask = exists_tgt  # (B, 4)
        denom = mask.sum().clamp_min(1.0)
        pos_loss = (per_coord * mask).sum() / denom

        total = cls_loss + self.lambda_pos * pos_loss
        return {"loss": total, "cls_loss": cls_loss.detach(), "pos_loss": pos_loss.detach()}
