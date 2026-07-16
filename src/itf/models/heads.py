"""The output head: per-patch paragraph corners.

For each of the four corner types, an existence logit plus an `(x, y)` inside the
patch. Output is `(B, 4, 3)`: channel 0 is the RAW logit (the loss applies the
sigmoid itself, for numerical stability) and channels 1:3 are sigmoid-squashed
coordinates in [0, 1].

**Not a softmax, and the difference matters downstream.** The four heads are
independent binary questions: a patch can fire two at once, or none. That is why
V9 is a "co-activation" view and not a confusion matrix -- and why TL↔TR, which
are mirror images, can both fire on the same patch.
"""

from __future__ import annotations

import torch
from torch import nn

from itf.geometry import NUM_CORNERS


class CornerHead(nn.Module):
    def __init__(self, in_features: int, hidden: list[int] | None = None, dropout: float = 0.0):
        super().__init__()
        # `hidden` is a LIST of widths, one per hidden layer, and these configs
        # are hand-written YAML -- so `hidden: 32` is a plausible slip. Without
        # this it would surface as "'int' object is not iterable" from a for-loop,
        # and the API would answer 500 instead of a 400 that says what to type.
        if hidden is not None and not isinstance(hidden, (list, tuple)):
            raise ValueError(
                f"head.hidden tiene que ser una lista de anchos, no {type(hidden).__name__} "
                f"({hidden!r}). Para una sola capa oculta: hidden: [{hidden}]"
            )
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden or []:
            layers += [nn.Linear(prev, int(h)), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = int(h)
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(prev, NUM_CORNERS * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = self.out(x).view(-1, NUM_CORNERS, 3)
        exists = x[..., 0:1]  # raw logit: BCEWithLogits wants it unsquashed
        xy = torch.sigmoid(x[..., 1:3])  # in [0, 1], like the labels
        return torch.cat([exists, xy], dim=-1)
