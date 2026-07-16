"""The loss.

    L = Σ_c [ BCE(exists_c) + λ · exists_c · smoothL1((x_c, y_c)) ]

The position term is masked by the TARGET existence flag, so a patch with no
corner of type `c` contributes nothing to that corner's regression. Both terms
are differentiable and train jointly.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CornerLoss(nn.Module):
    def __init__(
        self,
        lambda_pos: float = 1.0,
        pos_weight: float | None = None,
        smooth_l1_beta: float = 0.05,
    ):
        super().__init__()
        self.lambda_pos = lambda_pos
        #: **Passed on purpose, never defaulted.** torch's default is 1.0, and the
        #: coordinates are normalised to [0,1], so |error| < 1 always and the loss
        #: never leaves the quadratic branch: it is plain MSE, and Huber's
        #: robustness never activates. That was the state of things for months
        #: with nobody having decided it (organizacion.md §3).
        self.smooth_l1_beta = smooth_l1_beta
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight)) if pos_weight is not None else None,
            persistent=False,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """pred/target: `(B, 4, 3)` == `[exists, x, y]` per corner type."""
        exists_logit = pred[..., 0]
        exists_tgt = target[..., 0]

        cls_loss = F.binary_cross_entropy_with_logits(
            exists_logit, exists_tgt, pos_weight=self.pos_weight
        )

        per_coord = F.smooth_l1_loss(
            pred[..., 1:3], target[..., 1:3], reduction="none", beta=self.smooth_l1_beta
        ).sum(dim=-1)  # (B, 4)

        # Normalised by the number of REAL corners, not by the batch: otherwise
        # the position loss would shrink just because a batch happened to hold
        # fewer corners, and λ would be weighting a moving target.
        mask = exists_tgt
        denom = mask.sum().clamp_min(1.0)
        pos_loss = (per_coord * mask).sum() / denom

        return {
            "loss": cls_loss + self.lambda_pos * pos_loss,
            "cls_loss": cls_loss.detach(),
            "pos_loss": pos_loss.detach(),
        }
