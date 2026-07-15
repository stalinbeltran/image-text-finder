"""Build a configurable CNN from a declarative config (component 2).

The architecture is described entirely by data (a dict, usually loaded from
YAML), so different architectures can be compared without touching code:

    input_size: 40
    in_channels: 1
    backbone:
      - {filters: 32, kernel: 3, stride: 1, batchnorm: true, activation: relu, pool: 2, dropout: 0.0}
      - {filters: 64, kernel: 3, stride: 1, batchnorm: true, activation: relu, pool: 2}
    head:
      hidden: [128]
      dropout: 0.3
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from itf.models.heads import CornerHead

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
}


@dataclass
class ModelConfig:
    input_size: int = 40
    in_channels: int = 1
    backbone: list[dict] = field(default_factory=list)
    head: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            input_size=int(d.get("input_size", 40)),
            in_channels=int(d.get("in_channels", 1)),
            backbone=list(d.get("backbone", [])),
            head=dict(d.get("head", {})),
        )


def _make_activation(name: str) -> nn.Module:
    key = (name or "relu").lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"unknown activation '{name}', options: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key](inplace=True) if key in {"relu", "leaky_relu", "elu"} else _ACTIVATIONS[key]()


def _conv_block(in_ch: int, spec: dict) -> tuple[nn.Sequential, int]:
    filters = int(spec["filters"])
    kernel = int(spec.get("kernel", 3))
    stride = int(spec.get("stride", 1))
    padding = spec.get("padding", kernel // 2)  # default 'same' for stride 1
    if padding == "same":
        padding = kernel // 2
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, filters, kernel, stride=stride, padding=int(padding),
                  bias=not spec.get("batchnorm", False))
    ]
    if spec.get("batchnorm", False):
        layers.append(nn.BatchNorm2d(filters))
    layers.append(_make_activation(spec.get("activation", "relu")))
    pool = spec.get("pool")
    if pool:
        pool_size = int(pool if not isinstance(pool, bool) else 2)
        layers.append(nn.MaxPool2d(pool_size))
    dropout = float(spec.get("dropout", 0.0))
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers), filters


class ConfigurableCNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        blocks: list[nn.Module] = []
        in_ch = config.in_channels
        for spec in config.backbone:
            block, in_ch = _conv_block(in_ch, spec)
            blocks.append(block)
        self.backbone = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        flat = self._infer_flat_features(config)
        self.head = CornerHead(
            flat, hidden=config.head.get("hidden"), dropout=float(config.head.get("dropout", 0.0))
        )

    def _infer_flat_features(self, config: ModelConfig) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, config.in_channels, config.input_size, config.input_size)
            feats = self.flatten(self.backbone(dummy))
        return feats.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(x)
        return self.head(x)  # (B, 4, 3)


def build_model(config: dict | ModelConfig) -> ConfigurableCNN:
    cfg = config if isinstance(config, ModelConfig) else ModelConfig.from_dict(config)
    if not cfg.backbone:
        raise ValueError("model config needs a non-empty 'backbone'")
    return ConfigurableCNN(cfg)
