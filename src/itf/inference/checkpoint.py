"""Contract ④ — the checkpoint describes itself.

`load_model()` reads the network config out of the `.pt` and rebuilds the net. F
**never** needs C's YAML, nor B, nor the run's `config.json`. It is the cleanest
dependency in the project and the best property of the format: a `.pt` is
portable, so renaming a network or rebuilding a dataset cannot orphan a trained
model (organizacion.md §2-④).

Which is why the load path must not "helpfully" accept a config argument: the
moment it can be told what the network is, the checkpoint stops being the source
of truth about itself and a caller can rebuild it wrong.
"""

from __future__ import annotations

from pathlib import Path

import torch

from itf.models import ConfigurableCNN, build_model


def load_model(checkpoint: str | Path, device: str = "cpu") -> ConfigurableCNN:
    """Rebuild the network a checkpoint was saved from, and load its weights.

    In `eval()` mode: every caller in F wants inference, and a model left in
    train mode gives different answers with `dropout` or `batchnorm` in the
    config -- silently, and only for some architectures.
    """
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"no existe el checkpoint '{path}'")

    # `weights_only=False`: the config dict rides along with the tensors, which
    # is the whole point of the format.
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # A bare `state_dict` (or anything else torch.save was pointed at) is not one
    # of ours. Asked for `.get` it raises AttributeError -- a traceback, where
    # contract ④ promises a reason.
    config = (ckpt.get("config") or {}).get("model") if isinstance(ckpt, dict) else None
    if not config:
        # Ausente ≠ cero (formatos.md §2): a checkpoint with no config cannot be
        # rebuilt, and guessing an architecture would load the weights into the
        # wrong shape -- or worse, into a right-shaped wrong network.
        raise ValueError(
            f"el checkpoint '{path}' no lleva su config de red, así que no se puede "
            f"reconstruir. Un .pt tiene que describirse solo (contrato ④)."
        )

    model = build_model(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model
