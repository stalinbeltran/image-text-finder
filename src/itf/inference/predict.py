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

from itf.geometry import CORNER_NAMES, positions, window_at, windows
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


def grid_coverage(width: int, height: int, n: int, stride: int) -> dict:
    """How much of the image the sliding window actually looks at.

    **A stride larger than the patch leaves gaps**, and that failure is silent in
    the worst way: the prediction comes back, it is well formed, and the corners
    that lived in the unseen strips are simply absent -- indistinguishable, on
    screen, from a model that failed to detect them. On a 10-px patch with the
    stride of 20 that the UI used to hardcode, 30 of 80 pixel columns were never
    seen by any window, and image 1 of `dirty-paragraphs-80ancho` reported zero
    TL corners for three real paragraphs.

    So the payload carries it and V11 says it out loud. Reported rather than
    refused: a coarse stride is a legitimate thing to ask for (it is F, tuned per
    call), and the honest answer is "here it is, and here is what it could not
    see" -- not an error, and not a silent hole.
    """
    seen_x, seen_y = set(), set()
    for x0 in positions(width, n, stride):
        seen_x.update(range(x0, x0 + n))
    for y0 in positions(height, n, stride):
        seen_y.update(range(y0, y0 + n))
    return {
        "stride": int(stride),
        "patch_size": int(n),
        # Gaps appear exactly when the step outruns the window.
        "has_gaps": bool(stride > n),
        "unseen_columns": int(width - len(seen_x)),
        "unseen_rows": int(height - len(seen_y)),
    }


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


@torch.no_grad()
def window_prediction(
    model: ConfigurableCNN,
    image: np.ndarray,
    x0: int,
    y0: int,
    *,
    device: str = "cpu",
) -> dict:
    """V5 — the scrubber. **Fix E and the image, vary the crop `(x0, y0)`.**

    Drag one 40×40 window over a real image and read the four heads live. Two
    things make this worth a view of its own (ui.md §4.1):

      - **The input stays in distribution.** Unlike a pixel editor, every crop is
        real pixels the extractor could have produced, so the prediction says
        something about the model rather than about an artefact the model never
        saw (ui.md §5). The border flags come from `window_at` -- the *same*
        formula B extracted with (contract ⑤) -- so a crop flush against an edge is
        flagged exactly as its trained-on twin was.

      - **It measures the stability the stride depends on.** Move the window one
        pixel and the prediction should barely move; if it jumps, that jitter is
        what sets how fine an inference `stride` can be and how big the NMS radius
        must be. So the payload carries, per corner, the largest change in score
        over the four 1-px neighbours -- a number you cannot get from a single
        prediction.
    """
    image = _as_gray(image)
    n = model.config.input_size
    height, width = image.shape
    if n > width or n > height:
        raise ValueError(
            f"la ventana ({n}px) no cabe en una imagen de {width}x{height}"
        )
    # Clamp so the crop always sits fully inside the image: the scrubber hands over
    # a top-left, and the edge of the drag must not walk the window off the image.
    x0 = max(0, min(int(x0), width - n))
    y0 = max(0, min(int(y0), height - n))

    def predict_at(cx: int, cy: int) -> np.ndarray:
        win = window_at(cx, cy, n, width, height)
        crop = image[cy : cy + n, cx : cx + n]
        batch = torch.from_numpy(np.ascontiguousarray(crop)).unsqueeze(0).unsqueeze(0)
        batch = batch.float().div_(255.0).to(device)
        borders = torch.tensor([win.border], dtype=torch.float32, device=device)
        return model(batch, borders)[0].cpu().numpy()  # (4, 3)

    pred = predict_at(x0, y0)  # (4, 3): [logit, x_frac, y_frac] per corner
    scores = 1.0 / (1.0 + np.exp(-pred[:, 0]))  # sigmoid

    # The 1-px jitter: the four axis neighbours, each clamped back inside the
    # image so a window already at the edge does not fabricate a shift.
    neighbours = [
        (min(x0 + 1, width - n), y0),
        (max(x0 - 1, 0), y0),
        (x0, min(y0 + 1, height - n)),
        (x0, max(y0 - 1, 0)),
    ]
    deltas = np.zeros(len(CORNER_NAMES))
    for cx, cy in neighbours:
        if (cx, cy) == (x0, y0):
            continue  # a clamped-away neighbour is not a real move
        other = 1.0 / (1.0 + np.exp(-predict_at(cx, cy)[:, 0]))
        deltas = np.maximum(deltas, np.abs(other - scores))

    win = window_at(x0, y0, n, width, height)
    return {
        "x0": x0,
        "y0": y0,
        "patch_size": n,
        "image_size": [int(width), int(height)],
        "border": list(win.border),
        "corner_order": list(CORNER_NAMES),
        "corners": [
            {
                "corner": CORNER_NAMES[c],
                "score": round(float(scores[c]), 4),
                # Fractions of the patch, like the head speaks.
                "x": round(float(pred[c, 1]), 4),
                "y": round(float(pred[c, 2]), 4),
                # And in image px, so the overlay can place the dot on the image.
                "image_x": round(float(x0 + pred[c, 1] * n), 2),
                "image_y": round(float(y0 + pred[c, 2] * n), 2),
            }
            for c in range(len(CORNER_NAMES))
        ],
        # How much each head trembles when the window moves 1 px (ui.md §4.1). This
        # is what a single prediction cannot show and what the stride depends on.
        "stability": {
            "per_corner": [round(float(v), 4) for v in deltas],
            "max": round(float(deltas.max()), 4),
        },
    }


def predict_image(
    model: ConfigurableCNN,
    image: np.ndarray,
    *,
    stride: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    nms_radius: float | None = None,
    min_size: float | None = None,
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
    min_size = DEFAULT_MIN_SIZE if min_size is None else min_size

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
        # The patch the model actually eats. Every knob below is measured in
        # units of it -- a stride of 20 is fine on a 40-px patch and skips half
        # the image on a 10-px one -- so a client that shows sliders cannot
        # calibrate them without this, and a hardcoded default is a number that
        # silently means something different per run.
        "patch_size": n,
        "coverage": grid_coverage(
            int(image.shape[1]), int(image.shape[0]), n, stride
        ),
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
