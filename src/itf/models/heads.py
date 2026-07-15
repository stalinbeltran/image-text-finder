"""Output head for per-patch paragraph-corner detection.

Produces, for each of the four corner types [TL, TR, BR, BL], an existence
logit plus an (x, y) position in [0, 1] (relative to the patch). Output is
shaped ``(B, 4, 3)`` where channel 0 is the raw existence logit and channels
1:3 are sigmoid-squashed coordinates.
"""

from __future__ import annotations

import torch
from torch import nn

from itf.datasets.loader import NUM_CORNERS


class CornerHead(nn.Module):
    def __init__(self, in_features: int, hidden: list[int] | None = None, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden or []:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(prev, NUM_CORNERS * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = self.out(x).view(-1, NUM_CORNERS, 3)
        exists = x[..., 0:1]                 # raw logit
        xy = torch.sigmoid(x[..., 1:3])      # in [0, 1]
        return torch.cat([exists, xy], dim=-1)
