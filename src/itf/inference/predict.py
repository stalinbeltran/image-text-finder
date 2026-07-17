"""F — a trained E applied to a whole image: corners, then paragraphs.

Three stages, and **all three come out** (V11, api.md §3):

    1. slide the window, keep what scores above `threshold`  -> `raw`
    2. merge duplicates of the same corner across overlaps    -> `corners`
    3. pair TL->BR into boxes                                 -> `paragraphs`

Returning only the last one is what made "the paragraph came out wrong"
undiagnosable: the failure can be born in any of the three, and they fail in
different domains -- stage 1 is the model (C, D), stages 2 and 3 are knobs you
turn post-hoc without retraining (F). Without `raw` you cannot tell which one
lost the paragraph.

**Contract ⑤ is about this file, and the trap is a specific keystroke.** The
window here must be identical to the one B extracted with, or the model meets a
geometry it never trained on -- and that failure is not an exception, it is
slightly worse predictions near the image edges, forever, silently. The old code
held it together with two seams: it imported the **private** `_positions` from
`patches/extract.py`, and it **re-typed the six border-flag lines** that
`extract.py` already had. Identical, so it worked. Nothing would have noticed if
someone touched one.

So the window comes from `itf.geometry` and `win.border` is used **as given**.
Recomputing those flags here is what feels natural while writing inference, which
is why `test_contract_05` asserts this module's `windows` **is** geometry's
function and then checks the flags B actually wrote against what geometry says.

Imports `models` and `geometry`. **Never `patches`** (tests.md §4): B and F share
a window by both taking it from G, not by F reaching into B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from itf.geometry import CORNER_NAMES, windows
from itf.models import ConfigurableCNN

#: `threshold`, `stride`, the NMS radius and `min_size` are **F, not D**
#: (organizacion.md §1-D). They are chosen per call, post-hoc, over a model that
#: is already trained: sweeping them costs a forward pass, not an afternoon.
#: Keeping them out of D is worth more on CPU than it would be on GPU -- put
#: `threshold` in a training sweep and you multiply hours by something that was
#: free. **Barre D entrenando, barre F evaluando.**
DEFAULT_THRESHOLD = 0.5

#: Boxes thinner than this are not paragraphs, they are two corners that happened
#: to pair. In px.
DEFAULT_MIN_SIZE = 4.0


@dataclass(frozen=True)
class Detection:
    """One corner, in **image** pixels -- not patch-relative.

    The model speaks in fractions of a patch; everything downstream (NMS,
    pairing, the overlay) speaks in image px. Converting once, here, is what
    keeps the two from ever being confused: there is no point after this where a
    number could be either.
    """

    corner: str
    x: float
    y: float
    score: float

    def as_dict(self) -> dict:
        return {
            "corner": self.corner,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "score": round(self.score, 4),
        }


def _as_gray(image: np.ndarray) -> np.ndarray:
    """(H, W) uint8. Accepts the (H, W, 1) that B's `.npz` and PIL hand over."""
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"esperaba una imagen en escala de grises (H, W); me llegó {image.shape}")
    return image


def default_stride(patch_size: int) -> int:
    """Half the patch: every corner is seen by ~4 windows, so NMS has something
    to merge and a corner is never decided by a single crop."""
    return max(patch_size // 2, 1)


@torch.no_grad()
def detect_corners(
    model: ConfigurableCNN,
    image: np.ndarray,
    *,
    stride: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    device: str = "cpu",
) -> list[Detection]:
    """Stage 1 — the sliding window. **Raw: above threshold, before NMS.**

    Above `threshold` rather than everything: at stride `n/2` a 640×480 image is
    ~713 windows × 4 corners, and shipping ~2900 detections of which ~2800 are
    noise would not make V11 more honest, just unreadable. What `raw` is for is
    seeing what NMS *collapsed*, and that is the same set NMS is about to eat.
    """
    image = _as_gray(image)
    n = model.config.input_size
    stride = stride or default_stride(n)
    height, width = image.shape

    # THE window (contract ⑤): the same call `extract.py` makes, and `win.border`
    # used as given rather than recomputed. This is the line the test guards.
    grid = windows(width, height, n, stride)

    patches = np.stack([image[w.y0 : w.y0 + n, w.x0 : w.x0 + n] for w in grid])
    borders = np.asarray([w.border for w in grid], dtype=np.float32)

    batch = torch.from_numpy(patches).unsqueeze(1).float().div_(255.0).to(device)
    border_batch = torch.from_numpy(borders).to(device)

    pred = model(batch, border_batch)  # (P, 4, 3)
    probs = torch.sigmoid(pred[..., 0])  # (P, 4)

    # One `.cpu()` for the whole grid, not one per detection: on a 640×480 image
    # the inner loop runs ~2800 times and each `float(tensor)` is a sync.
    probs_np = probs.cpu().numpy()
    xy = pred[..., 1:3].cpu().numpy()

    detections: list[Detection] = []
    for p, c in np.argwhere(probs_np >= threshold):
        win = grid[p]
        detections.append(
            Detection(
                corner=CORNER_NAMES[c],
                # The window's offset plus the position within it. The model's
                # x/y are fractions of the patch, hence `* n` and not `+ n`.
                x=float(win.x0 + xy[p, c, 0] * n),
                y=float(win.y0 + xy[p, c, 1] * n),
                score=float(probs_np[p, c]),
            )
        )
    return detections


def merge(dets: list[Detection], radius: float) -> list[Detection]:
    """Stage 2 — NMS per corner type. Greedy, highest score first.

    Overlapping windows see the same corner ~4 times, so this is not an optional
    cleanup: without it every corner is 4 corners and the pairing in stage 3 has
    16 candidate boxes for one paragraph.

    Merged rather than suppressed: the cluster's position is its **score-weighted
    mean**, so four windows that each saw the corner slightly off average out
    instead of the loudest one winning outright. The kept `score` is the max, not
    the mean -- the cluster is one detection with the confidence of its best
    evidence, and averaging it would punish a corner for having been seen from a
    window that barely caught it.

    `radius` is Chebyshev (a square of side 2r), not Euclidean. That is what the
    old code did and it is kept deliberately: at the default `stride/2` the two
    differ by a corner case nobody has measured, and quietly "fixing" it would
    move every result while looking like a tidy-up.
    """
    merged: list[Detection] = []
    for corner in CORNER_NAMES:
        group = sorted(
            (d for d in dets if d.corner == corner), key=lambda d: d.score, reverse=True
        )
        used = [False] * len(group)
        for i, seed in enumerate(group):
            if used[i]:
                continue
            used[i] = True
            cluster = [seed]
            for j in range(i + 1, len(group)):
                if used[j]:
                    continue
                if abs(group[j].x - seed.x) <= radius and abs(group[j].y - seed.y) <= radius:
                    used[j] = True
                    cluster.append(group[j])
            weight = sum(d.score for d in cluster)
            merged.append(
                Detection(
                    corner=corner,
                    x=sum(d.x * d.score for d in cluster) / weight,
                    y=sum(d.y * d.score for d in cluster) / weight,
                    score=max(d.score for d in cluster),
                )
            )
    return merged


def reconstruct_boxes(
    dets: list[Detection], min_size: float = DEFAULT_MIN_SIZE
) -> list[dict]:
    """Stage 3 — greedy TL→BR pairing into axis-aligned boxes `[x, y, w, h]`.

    **A heuristic, and not a network** (organizacion.md §1-F). It improves on its
    own when the model improves, and it is the place to touch when the corners
    come out right and the paragraphs come out wrong -- which is a distinction
    only V11 can make, and the reason `raw` and `corners` are returned too.

    For each TL (topmost first) take the nearest unused BR below-and-right of it.
    Nearest by `dx + dy` rather than Euclidean: for axis-aligned boxes the L1
    distance is what "smallest box containing both" means, and a diagonal
    tiebreak would prefer a square over the flat wide box a paragraph actually
    is.

    TR and BL are not used. They are detected and returned, and pairing on all
    four is the obvious next move if this stage becomes the bottleneck -- but it
    is not, today, and inventing it now would be optimising a heuristic nobody
    has measured against a model still being trained.
    """
    tls = sorted((d for d in dets if d.corner == "TL"), key=lambda d: (d.y, d.x))
    brs = [d for d in dets if d.corner == "BR"]
    used = [False] * len(brs)

    boxes: list[dict] = []
    for tl in tls:
        best, best_dist = -1, float("inf")
        for j, br in enumerate(brs):
            if used[j] or br.x <= tl.x + min_size or br.y <= tl.y + min_size:
                continue
            dist = (br.x - tl.x) + (br.y - tl.y)
            if dist < best_dist:
                best, best_dist = j, dist
        if best < 0:
            continue
        used[best] = True
        br = brs[best]
        boxes.append(
            {
                "box": [
                    round(tl.x, 2),
                    round(tl.y, 2),
                    round(br.x - tl.x, 2),
                    round(br.y - tl.y, 2),
                ],
                "score": round((tl.score + br.score) / 2, 4),
            }
        )
    return boxes


def predict_image(
    model: ConfigurableCNN,
    image: np.ndarray,
    *,
    stride: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    nms_radius: float | None = None,
    min_size: float = DEFAULT_MIN_SIZE,
    device: str = "cpu",
) -> dict:
    """The three stages, all three returned (V11).

    `nms_radius` defaults to **half the stride**: two windows are `stride` apart,
    so anything closer than `stride/2` to a stronger detection of the same type
    is the same corner seen twice. Tie it to the stride rather than fix it, or
    changing the stride silently changes what counts as a duplicate.
    """
    image = _as_gray(image)
    n = model.config.input_size
    stride = stride or default_stride(n)
    radius = nms_radius if nms_radius is not None else stride / 2.0

    raw = detect_corners(model, image, stride=stride, threshold=threshold, device=device)
    corners = merge(raw, radius=radius)
    paragraphs = reconstruct_boxes(corners, min_size=min_size)

    return {
        # Pre-NMS. Without it "the paragraph came out wrong" is not diagnosable:
        # you cannot see whether the corner was never detected or NMS ate it.
        "raw": [d.as_dict() for d in raw],
        "corners": [d.as_dict() for d in corners],
        "paragraphs": paragraphs,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        # The knobs that produced this, echoed back. The sliders are live (ui.md
        # §2) and the answers arrive out of order: without knowing which stride a
        # payload came from, a slow response overwrites a newer one.
        "knobs": {
            "threshold": threshold,
            "stride": stride,
            "nms_radius": radius,
            "min_size": min_size,
        },
    }
