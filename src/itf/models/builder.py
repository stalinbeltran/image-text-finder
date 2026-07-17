"""C — the network: a CNN built from a declarative config.

**Config pure, zero data.** The architecture is described entirely by a dict, so
architectures can be listed, compared and versioned without touching a dataset or
training anything. That property is the whole reason C is a noun with a store and
a screen -- the old UI had the endpoints and never called them, so the
architecture only existed embedded in a training form and frozen inside runs.

What C does NOT own: weights (that is E), `lr`/`epochs` (that is D), or where the
patches come from. The three fine borders, from organizacion.md §1-D:
`dropout` and `batchnorm` are C (they change the module) however much they sound
like training; `border_features` is C (it changes the head's shape) however much
it sounds like the dataset; `lambda_pos` and `pos_weight` are D (they weight loss
terms) however much they sound like architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from itf.geometry import NUM_BORDERS
from itf.models.heads import CornerHead

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
}

_INPLACE = {"relu", "leaky_relu", "elu"}


@dataclass
class NetworkConfig:
    """What defines a C. The `input_size` here is one half of contract ①."""

    #: The patch side. Must equal B's `patch_size`, and NOTHING enforces that but
    #: the validator -- they are declared in two independent places on purpose
    #: (each domain declares its own), which is exactly why ① needs a test.
    input_size: int = 40
    in_channels: int = 1
    backbone: list[dict] = field(default_factory=list)
    head: dict = field(default_factory=dict)
    #: Feed the 4 per-patch border flags into the head alongside the flattened
    #: conv features. Contract ②: the dataset offers, the network decides.
    border_features: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "NetworkConfig":
        return cls(
            input_size=int(d.get("input_size", 40)),
            in_channels=int(d.get("in_channels", 1)),
            backbone=list(d.get("backbone", [])),
            head=dict(d.get("head", {})),
            border_features=bool(d.get("border_features", False)),
        )

    def as_dict(self) -> dict:
        return {
            "input_size": self.input_size,
            "in_channels": self.in_channels,
            "backbone": self.backbone,
            "head": self.head,
            "border_features": self.border_features,
        }


def _make_activation(name: str) -> nn.Module:
    key = (name or "relu").lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"activación desconocida '{name}'; hay: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key](inplace=True) if key in _INPLACE else _ACTIVATIONS[key]()


def _conv_block(in_ch: int, spec: dict) -> tuple[nn.Sequential, int]:
    """One backbone block. **The Conv2d is always element 0**, and that is a contract.

    `kernels()` reads `block[0].weight` to show the learned kernels (V1), and it
    keeps working with `batchnorm: true` precisely because the conv goes first.
    Reorder this and the kernel view silently shows the wrong tensor.
    """
    filters = int(spec["filters"])
    kernel = int(spec.get("kernel", 3))
    stride = int(spec.get("stride", 1))
    padding = spec.get("padding", kernel // 2)  # 'same' by default at stride 1
    if padding == "same":
        padding = kernel // 2
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_ch,
            filters,
            kernel,
            stride=stride,
            padding=int(padding),
            # BatchNorm has its own shift, so the conv bias is dead weight.
            bias=not spec.get("batchnorm", False),
        )
    ]
    if spec.get("batchnorm", False):
        layers.append(nn.BatchNorm2d(filters))
    layers.append(_make_activation(spec.get("activation", "relu")))
    if pool := spec.get("pool"):
        layers.append(nn.MaxPool2d(int(pool if not isinstance(pool, bool) else 2)))
    if (dropout := float(spec.get("dropout", 0.0))) > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers), filters


class ConfigurableCNN(nn.Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        blocks: list[nn.Module] = []
        in_ch = config.in_channels
        for spec in config.backbone:
            block, in_ch = _conv_block(in_ch, spec)
            blocks.append(block)
        self.backbone = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        self.use_border = config.border_features
        flat = self._infer_flat_features(config)
        head_in = flat + (NUM_BORDERS if self.use_border else 0)
        self.head = CornerHead(
            head_in,
            hidden=config.head.get("hidden"),
            dropout=float(config.head.get("dropout", 0.0)),
        )

    def _infer_flat_features(self, config: NetworkConfig) -> int:
        """Run a dummy patch through to find the flattened width.

        **This is where contract ① actually bites.** The dummy is `input_size`
        wide, so the Linear is sized for it. Feed 60-pixel patches to a network
        built for 40 and the flatten is a different width -- and it fails inside
        the job thread, mid-batch, with `mat1 and mat2 shapes cannot be
        multiplied`, which says nothing about the real problem.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, config.in_channels, config.input_size, config.input_size)
            return self.flatten(self.backbone(dummy)).shape[1]

    def forward(self, x: torch.Tensor, border: torch.Tensor | None = None) -> torch.Tensor:
        x = self.flatten(self.backbone(x))
        if self.use_border:
            if border is None:
                raise ValueError("la red tiene border_features pero no le has pasado el tensor border")
            x = torch.cat([x, border.to(x.dtype)], dim=1)  # (B, flat + 4)
        return self.head(x)  # (B, 4, 3)

    def feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Each block's output: `(N, filters, H, W)` per layer. The input of V2.

        No hooks: the backbone IS a Sequential of Sequentials, so re-applying the
        blocks one at a time and keeping each output is all it takes. `border`
        never enters -- `border_features` only touches the head.
        """
        maps = []
        for block in self.backbone:
            x = block(x)
            maps.append(x)
        return maps

    def kernels(self) -> list[torch.Tensor]:
        """Each conv's learned weights: `(filters, in_channels, k, k)` per layer."""
        return [block[0].weight.detach() for block in self.backbone]


def spatial_trace(config: NetworkConfig) -> list[dict]:
    """`40 → 20 → 10 → 5`, per layer, plus the channel count.

    Free (no weights needed) and it is the ONLY thing an untrained network can
    show about itself -- which is what makes the Redes screen worth having before
    anything is trained.
    """
    size = config.input_size
    trace = []
    for i, spec in enumerate(config.backbone, start=1):
        kernel = int(spec.get("kernel", 3))
        stride = int(spec.get("stride", 1))
        padding = spec.get("padding", kernel // 2)
        padding = kernel // 2 if padding == "same" else int(padding)
        conv_out = (size + 2 * padding - kernel) // stride + 1
        if conv_out < 1:
            raise ValueError(
                f"la capa {i} no cabe: entra {size}px y con kernel {kernel}, stride {stride} "
                f"y padding {padding} sale {conv_out}px"
            )
        out = conv_out
        pool = spec.get("pool")
        if pool:
            pool_size = int(pool if not isinstance(pool, bool) else 2)
            out = conv_out // pool_size
            if out < 1:
                raise ValueError(
                    f"la capa {i} no cabe: tras la conv quedan {conv_out}px y el pool de "
                    f"{pool_size} los deja en {out}"
                )
        trace.append(
            {"layer": i, "in": size, "conv": conv_out, "out": out, "channels": int(spec["filters"])}
        )
        size = out
    return trace


def flat_features(config: NetworkConfig) -> int:
    """The flattened conv width, from the trace. No weights, no torch.

    **Must agree with what `_infer_flat_features` measures with a real tensor.**
    Two ways of computing the same number is exactly the seam that contract ⑤
    warns about: the analytic one feeds the Redes screen, the empirical one sizes
    the Linear, and if they drift the screen lies about the network you are
    building. There is a test on it.
    """
    trace = spatial_trace(config)
    if not trace:
        return config.in_channels * config.input_size * config.input_size
    last = trace[-1]
    return last["channels"] * last["out"] * last["out"]


def build_model(config: dict | NetworkConfig) -> ConfigurableCNN:
    cfg = config if isinstance(config, NetworkConfig) else NetworkConfig.from_dict(config)
    if not cfg.backbone:
        raise ValueError("la config de red necesita un 'backbone' no vacío")
    return ConfigurableCNN(cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
