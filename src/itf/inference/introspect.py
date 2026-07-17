"""V1 and V2 — looking inside a trained E: its kernels, and what they did.

Both are `matrixview` payloads (numbers, not images: the browser picks the
colour), and both **declare the colour job** because the client cannot know
whether it is holding a signed weight or a non-negative activation (api.md §3).
Getting that wrong is not cosmetic -- it is the one thing the sibling project
re-derived incorrectly (ui.md §5).

**The input is a patch, and that is contract ①**: the patch is the real input of
the CNN. A whole image belongs to F's `predict.py`, which is a different question
asked of the same weights (organizacion.md §4, last note).

No hooks anywhere. The backbone is a `Sequential` of `Sequential`s whose element
0 is always the `Conv2d` (`_conv_block` says so, and calls it a contract), so
`kernels()` reads `block[0].weight` and `feature_maps()` re-applies the blocks
one at a time. That is also why `batchnorm: true` does not break either one.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from itf.geometry import BORDER_NAMES, CORNER_NAMES, NUM_BORDERS
from itf.matrixview import MAX_MAPS_PER_LAYER, ColorJob, layer_payload
from itf.models import ConfigurableCNN

#: Activations whose output cannot be negative. Everything else produces signed
#: values, and signed values are diverging (R2/R3).
#:
#: `leaky_relu`, `gelu` and `elu` are NOT here on purpose: they all go negative
#: (elu down to -1), and painting them sequential would put the neutral wherever
#: the minimum happened to fall -- which hides the sign, exactly the failure R2
#: describes for kernels.
_NON_NEGATIVE_ACTIVATIONS = frozenset({"relu", "sigmoid"})


class NotInspectable(ValueError):
    """This model or this patch cannot produce the view, and the reason says why.

    A refusal, not a crash (R4). Every case here is the system working: a patch
    of the wrong size is contract ① arriving at F, and answering with a picture
    anyway would mean answering about a network nobody trained.
    """

    def __init__(self, code: str, message: str, hint: str = ""):
        self.code = code
        self.hint = hint
        super().__init__(message)


def _job_for_activation(activation: str) -> ColorJob:
    """R3 — **read the spec, never the data.**

    The tempting shortcut is `"diverging" if maps.min() < 0 else "sequential"`,
    and it is wrong in a way that only shows up later: a `tanh` layer whose
    activations happen to be all-positive **on this patch** would come back
    sequential, and the same layer on the next patch would come back diverging.
    The colour would then mean two different things in two screenshots of the
    same network, and nothing would say so. The activation is a property of the
    architecture, so it is answered from the architecture.
    """
    return "sequential" if (activation or "relu").lower() in _NON_NEGATIVE_ACTIVATIONS else "diverging"


def kernels(model: ConfigurableCNN, *, max_maps: int = MAX_MAPS_PER_LAYER) -> dict:
    """V1 — the learned kernels of layer 1. **Layer 1 only** (D13).

    *Decidido 2026-07-17: D13 se cierra en «nada» para las capas profundas.*

    **The rule is `in_channels == 1`, not "the first layer".** Layer 1 is just the
    only layer that satisfies it: with one input channel a filter IS a k×k matrix
    and it applies to the patch itself, so what you see is exact -- and if it
    looks like oriented edge detectors, the network learned; if it looks like
    noise, it did not, and that is information rather than a bug in the view
    (plan-ui.md fase 6). From layer 2 on, a filter is 32 or 64 matrices operating
    on channels that are not the image, and there is **no honest projection to
    one matrix**. The sibling painted `weight[k, 0]` -- one thirty-second of the
    kernel, picked arbitrarily -- and it looks like a view while telling you
    nothing. What the deep layers have to say is in their feature maps (V2), so
    that is where the reader is sent.

    **Diverging, centred on 0** (R2), and it is the whole reason V1 is worth
    drawing: a weight has sign, and what a kernel *is* is its structure of
    excitation and inhibition. Normalise min→max, as the sibling did, and zero
    lands wherever it falls and that structure goes invisible.
    """
    weights = model.kernels()
    if not weights:
        raise NotInspectable(
            "network_has_no_conv_layers",
            "esta red no tiene ninguna capa convolucional, así que no hay kernels que enseñar",
        )

    first = weights[0]  # (filters, in_channels, k, k)
    in_channels = int(first.shape[1])
    if in_channels != 1:
        # The same reason D13 gives for the deep layers, arriving one layer
        # earlier. Serving `weight[:, 0]` here would be the projection D13
        # rejected, just with a more convincing layer number on it.
        raise NotInspectable(
            "kernels_not_projectable",
            f"la capa 1 de esta red tiene {in_channels} canales de entrada, así que un filtro "
            f"no es una matriz: son {in_channels}. No hay proyección honesta a un solo mapa",
            "mira los feature maps (V2): enseñan el efecto real de cada filtro sobre un patch",
        )

    spec = dict(model.config.backbone[0])
    payload = layer_payload(
        layer=1,
        # `[:, 0]` is exact here and ONLY here: in_channels is 1, so this is the
        # whole filter and not a slice of it. The check above is what keeps that
        # sentence true.
        maps=first[:, 0].cpu().numpy(),
        # A weight is signed. Always. Independent of the activation -- that is
        # V2's question, not this one.
        job="diverging",
        max_maps=max_maps,
        kernel_size=int(first.shape[-1]),
        in_channels=in_channels,
        spec=spec,
    )
    # Said out loud rather than left as an absence: a reader who sees one layer
    # in a four-layer network deserves to know that is a decision, not a bug.
    payload["layers_in_backbone"] = len(weights)
    payload["deep_layers_note"] = (
        "solo se sirve la capa 1: con in_channels=1 un filtro es exactamente una matriz. "
        "De la capa 2 en adelante (32, 64… canales de entrada) no hay proyección honesta a "
        "una matriz — esa información está en los feature maps (V2)."
    )
    return payload


def _prepare_patch(model: ConfigurableCNN, patch: np.ndarray) -> torch.Tensor:
    """(n, n[, 1]) uint8 → (1, C, n, n) float in [0, 1], or a refusal.

    **Contract ① arriving at F.** A 60-px patch into a network built for 40 fails
    inside the head with `mat1 and mat2 shapes cannot be multiplied` -- a message
    about linear algebra, for a problem about a dataset. Same rule as `POST
    /runs`: refuse at the door, with the two numbers and the fix.
    """
    array = np.asarray(patch)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise NotInspectable(
            "invalid_patch",
            f"un patch es una matriz (n, n); me llegó {list(array.shape)}",
            "manda el patch como lista de listas de píxeles",
        )

    n = model.config.input_size
    if array.shape != (n, n):
        raise NotInspectable(
            "patch_size_mismatch",
            f"esta red espera patches de {n}x{n} y me llegó uno de "
            f"{array.shape[0]}x{array.shape[1]}",
            f"elige un patch de un dataset con patch_size {n} (contrato ①)",
        )

    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
    # /255: the same normalisation `PatchDataset` and `detect_corners` apply. A
    # view fed unnormalised pixels would show activations 255x too big and read
    # as a saturated network.
    return tensor.div_(255.0).unsqueeze(0).unsqueeze(0)


def _border_tensor(model: ConfigurableCNN, border: Sequence[int] | None) -> torch.Tensor | None:
    """The 4 border flags, or a refusal. **Never zeros** (formatos.md §2).

    Contract ② one layer down, and the same trap in miniature: 0 means "does not
    touch any edge", not "unknown". Defaulting to zeros here would answer for a
    patch that sits flush against the top of its image as if it sat in the middle
    -- quietly, with a plausible-looking prediction.
    """
    if not model.use_border:
        return None
    if border is None:
        raise NotInspectable(
            "border_required",
            "esta red usa border_features, así que necesita los 4 flags de borde del patch "
            f"({', '.join(BORDER_NAMES)}) y no me los has dado",
            "manda `border`, o pide el patch por índice y salen de su dataset",
        )
    flags = [int(v) for v in border]
    if len(flags) != NUM_BORDERS:
        raise NotInspectable(
            "border_required",
            f"border son {NUM_BORDERS} flags ({', '.join(BORDER_NAMES)}); me llegaron {len(flags)}",
        )
    return torch.tensor([flags], dtype=torch.float32)


@torch.no_grad()
def feature_maps(
    model: ConfigurableCNN,
    patch: np.ndarray,
    border: Sequence[int] | None = None,
    *,
    device: str = "cpu",
    max_maps: int = MAX_MAPS_PER_LAYER,
) -> dict:
    """V2 — every layer's activations over one patch, plus what the model said.

    The prediction rides along because the two are read together: "this filter
    lit up" and "and it decided there is a TL here" is one question. It is also
    V3's payload for an arbitrary patch, not just one from the gallery.

    `border` never enters the backbone -- `border_features` only touches the head
    -- so the maps are the same with or without it. The head is not, which is why
    it is still required when the network uses it.
    """
    x = _prepare_patch(model, patch).to(device)
    border_batch = _border_tensor(model, border)
    if border_batch is not None:
        border_batch = border_batch.to(device)

    maps = model.feature_maps(x)
    prediction = model(x, border_batch)[0]  # (4, 3)
    scores = torch.sigmoid(prediction[..., 0])

    layers = []
    for i, activation in enumerate(maps):
        spec = dict(model.config.backbone[i])
        array = activation[0].cpu().numpy()  # (filters, H, W)
        layers.append(
            layer_payload(
                layer=i + 1,
                maps=array,
                # R3: from the spec, not from `array.min()`. See the docstring
                # of `_job_for_activation` -- the shortcut is wrong per-patch.
                job=_job_for_activation(spec.get("activation", "relu")),
                max_maps=max_maps,
                height=int(array.shape[1]),
                width=int(array.shape[2]),
                spec=spec,
            )
        )

    return {
        "input_size": int(model.config.input_size),
        "layers": layers,
        "prediction": {
            "corners": [
                {
                    "corner": name,
                    "score": round(float(scores[c]), 4),
                    "x": round(float(prediction[c, 1]), 4),
                    "y": round(float(prediction[c, 2]), 4),
                }
                for c, name in enumerate(CORNER_NAMES)
            ],
            "corner_order": list(CORNER_NAMES),
        },
    }
