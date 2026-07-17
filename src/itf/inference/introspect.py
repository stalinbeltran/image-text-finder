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

from itf.geometry import BORDER_NAMES, CORNER_NAMES, NUM_BORDERS, NUM_CORNERS
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


# ── the probes (fase 8) ───────────────────────────────────────────────────────


#: The occlusion mask's side and step, in patch pixels. A 5×5 mask sliding at 2
#: over a 40-px patch is ~19² ≈ 361 forwards -- under a second in one batch (ui.md
#: §4.1). Small enough to localise, big enough to matter: a 1×1 mask over pixelated
#: text mostly lands on background and says nothing.
DEFAULT_MASK_SIZE = 5
DEFAULT_MASK_STRIDE = 2


@torch.no_grad()
def occlusion(
    model: ConfigurableCNN,
    patch: np.ndarray,
    border: Sequence[int] | None = None,
    *,
    mask_size: int = DEFAULT_MASK_SIZE,
    mask_stride: int = DEFAULT_MASK_STRIDE,
    device: str = "cpu",
) -> dict:
    """V4 — occlusion sensitivity. **Fix E and the patch, vary the mask position.**

    Slide a small mask over the patch and, at each position, ask what the four
    heads now say. Where covering a region collapses `p(exists)`, that region is
    what the head was reading. It is the rigorous version of the sibling project's
    "paint over the digit and watch it change" (ui.md §5): systematic, and -- unlike
    a pixel editor -- **in distribution**, because it only ever hides real pixels,
    never invents new ones.

    **The map is the occluded probability, and that is why it is sequential** (R3,
    ui.md §4.1). The tempting alternative is the *drop* `baseline - p`, but a drop
    is signed -- occluding an inhibitory region can raise the score -- and a signed
    quantity painted sequential puts the neutral wherever the minimum falls, which
    is the exact trap R2/R3 warn about. `p(exists | occlude here)` is a genuine
    probability, never negative, so sequential is honest: dark = covering here
    kills the score = the region that mattered. The `baseline` rides along, so the
    drop is one subtraction the reader can see rather than a colour that lies.

    **The mask is filled with the patch mean**, not black or grey: a constant equal
    to the patch's own average injects no new edge into a pixelated text crop, so
    what changes is *removal of signal*, not *addition of a dark rectangle the
    network never trained on*.
    """
    x = _prepare_patch(model, patch).to(device)  # (1, 1, n, n)
    n = model.config.input_size
    border_batch = _border_tensor(model, border)
    if border_batch is not None:
        border_batch = border_batch.to(device)

    def scores_of(batch: torch.Tensor, borders: torch.Tensor | None) -> torch.Tensor:
        return torch.sigmoid(model(batch, borders)[..., 0])  # (B, 4)

    baseline = scores_of(x, border_batch)[0]  # (4,)

    from itf.geometry import positions

    if mask_size > n:
        raise NotInspectable(
            "mask_too_big",
            f"la máscara ({mask_size}) no cabe en un patch de {n}px",
            "elige una máscara menor que el patch",
        )
    xs = positions(n, mask_size, mask_stride)
    fill = float(x.mean())

    # One occluded copy per grid cell, all forwarded in a single batch. Order is
    # row-major (y outer, x inner) so `matrix[iy][ix]` maps straight to a cell.
    occluded = x.repeat(len(xs) * len(xs), 1, 1, 1)  # (P, 1, n, n)
    for iy, y0 in enumerate(xs):
        for ix, x0 in enumerate(xs):
            occluded[iy * len(xs) + ix, 0, y0 : y0 + mask_size, x0 : x0 + mask_size] = fill

    borders_batch = (
        border_batch.repeat(occluded.shape[0], 1) if border_batch is not None else None
    )
    scores = scores_of(occluded, borders_batch)  # (P, 4)
    grid = scores.reshape(len(xs), len(xs), NUM_CORNERS).cpu().numpy()

    maps = []
    for c, name in enumerate(CORNER_NAMES):
        m = grid[:, :, c]
        maps.append(
            {
                "corner": name,
                "matrix": np.round(m, 4).tolist(),
                "min": round(float(m.min()), 4),
                "max": round(float(m.max()), 4),
                "mean": round(float(m.mean()), 4),
            }
        )

    return {
        "corner_order": list(CORNER_NAMES),
        "patch_size": n,
        "mask_size": mask_size,
        "mask_stride": mask_stride,
        "bins": len(xs),
        # Top-left offsets of each cell, so the front can place the map over the
        # patch: cell (iy, ix) covers `[offsets[iy], +mask_size)` vertically.
        "offsets": [int(v) for v in xs],
        "baseline": [round(float(baseline[c]), 4) for c in range(NUM_CORNERS)],
        "maps": maps,
        # Always sequential: `p(exists | occluded)` is a probability, never signed
        # (see the docstring). Declared, not guessed (api.md §3).
        "job": "sequential",
    }


@torch.no_grad()
def border_test(
    model: ConfigurableCNN,
    patch: np.ndarray,
    border: Sequence[int],
    *,
    device: str = "cpu",
) -> dict:
    """V10 — the border-flag test. **Fix E and the patch, flip each of the 4 flags.**

    The 4 border flags say whether the patch sits flush against the source image's
    edge (geometry.BORDER_NAMES). They are real information -- a paragraph corner
    cannot sit just outside the image, so a patch against the top has a different
    prior than one in the middle (formatos.md §2). This probe measures how much the
    network actually *uses* that: baseline with the real flags, then one forward per
    flag flipped, and the dumbbell shows each corner's score moving from before to
    after (ui.md §4.1). Five forwards.

    **Refuses when the network does not use border features**, rather than
    answering with four identical dumbbells. A network with `border_features:
    false` ignores the flags entirely, so flipping them changes nothing -- and a
    view that drew "no change" over and over would read as "the border does not
    matter to this patch", a conclusion about the data, when the truth is the
    architecture never looks at it (D13's shape: refuse, do not project a
    meaningless view).
    """
    if not model.use_border:
        raise NotInspectable(
            "border_not_used",
            "esta red no usa border_features, así que ignora los 4 flags de borde: "
            "voltearlos no cambia nada y no hay nada que medir",
            "esta sonda solo dice algo sobre redes con border_features=true",
        )

    x = _prepare_patch(model, patch).to(device)
    base_flags = _border_tensor(model, border)  # (1, 4), and validates the length
    assert base_flags is not None  # use_border is true, so it is never None
    base_flags = base_flags.to(device)

    def scores_of(borders: torch.Tensor) -> list[float]:
        s = torch.sigmoid(model(x, borders)[0, :, 0])  # (4,)
        return [round(float(v), 4) for v in s]

    baseline = scores_of(base_flags)

    flips = []
    real = [int(v) for v in border]
    for b, name in enumerate(BORDER_NAMES):
        flipped = base_flags.clone()
        flipped[0, b] = 1.0 - flipped[0, b]  # 0↔1
        flips.append(
            {
                "border": name,
                "flag_from": real[b],
                "flag_to": 1 - real[b],
                "scores": scores_of(flipped),
            }
        )

    return {
        "corner_order": list(CORNER_NAMES),
        "border_order": list(BORDER_NAMES),
        "border": real,
        "baseline": baseline,
        "flips": flips,
    }
