"""What the vistas read: the table, aggregated. V6, V7 and V8.

**The aggregate is computed here, never in the browser** (R6 of api.md). A table
is ~10⁵ rows; the PR curve is 101 points and the error map is 40×40. Shipping the
rows and reducing them client-side would be sending 10⁵ of something to compute
10² of something else -- and it would put the definition of `f1` in two languages.

The rows themselves (V6) go out **filtered and paginated**, and the filter is the
demonstration of the whole phase: `?outcome=fp&corner=TL&threshold=0.7` re-decides
what counts as a false positive **without the model running again**, because the
table stores raw scores. That is the "barrido gratis" of ui.md §4.1 -- and in CPU
the difference between free and hours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from itf.diagnostics.table import NotMeasurable, PatchTable
from itf.geometry import CORNER_NAMES
from itf.metrics import DEFAULT_THRESHOLD, prf1

#: Points of the PR curve. 101 is a threshold every 0.01 -- enough to pick one by
#: eye, small enough to ship, and it is an aggregate so its cost is the server's.
_CURVE_POINTS = 101

#: Buckets of the score histogram. The question it answers is "are positives and
#: negatives separable?", which needs shape, not resolution.
_HISTOGRAM_BINS = 20


def _nan_to_none(values) -> list:
    """NaN → None, recursively, on the way out to JSON.

    **Not cosmetic: `json.dumps` writes NaN as the bare token `NaN`, which is not
    valid JSON and which no browser can parse** -- the exact bug fase 4 found with
    `best: Infinity`, where one run whose monitor never fired would have taken
    down `GET /runs` for every other run on screen.

    Here it is not an edge case at all: `err_px` is NaN *by design* wherever there
    is no real corner, which on this data is ~80 % of the array. So every payload
    that carries an error goes through this, and `None` is also what it means --
    not measured, because there was nothing to measure (formatos.md §2).
    """
    if isinstance(values, np.ndarray):
        return [_nan_to_none(v) for v in values]
    if isinstance(values, (list, tuple)):
        return [_nan_to_none(v) for v in values]
    value = float(values)
    return None if np.isnan(value) else value


@dataclass(frozen=True)
class Diagnostics:
    """A table joined back to B's ground truth. What every view below reads.

    The join is by `patch_idx`, and it is why the table does not copy the truth:
    one place for what is true, one place for what the model said.
    """

    table: PatchTable
    #: (M, 4, 3) — `[exists, x, y]` per corner, straight from B's `.npz`.
    truth: np.ndarray
    #: (M,) — which image of A each patch came from (V15, and V6's caption).
    sample_idx: np.ndarray
    #: (M, 2) — where in that image. Provenance of the patch.
    patch_xy: np.ndarray
    patch_size: int

    @property
    def exists(self) -> np.ndarray:
        """(M, 4) bool — is there really a corner of this type in this patch?"""
        return self.truth[:, :, 0] > 0.5

    @property
    def xy_true(self) -> np.ndarray:
        """(M, 4, 2) — the real position, normalised within the patch."""
        return self.truth[:, :, 1:3]


def corner_index(corner: str | None) -> int | None:
    """`"TL"` → 0, `None`/`"all"` → None. Raises with the list if it is not one.

    The order is `CORNER_NAMES` and it is fixed project-wide: `y[:, c]` means the
    same corner in the `.npz`, in the manifest, in the meters and in the colour
    slots (R1). Accepting a name and resolving it here is what keeps the API from
    taking a bare index that would silently mean another corner if the order ever
    moved.
    """
    if corner is None or corner == "all":
        return None
    if corner not in CORNER_NAMES:
        raise NotMeasurable(
            "unknown_corner",
            f"'{corner}' no es un tipo de esquina",
            f"usa uno de {', '.join(CORNER_NAMES)}, o 'all'",
        )
    return CORNER_NAMES.index(corner)


def _select(values: np.ndarray, corner: int | None) -> np.ndarray:
    """(M, 4[, …]) → the chosen corner's column, or all of them."""
    return values if corner is None else values[:, corner : corner + 1]


def pr(diag: Diagnostics, corner: str | None = None) -> dict:
    """V8 — score histogram + PR curve. **The free sweep.**

    Every point of this curve is a threshold, and not one of them runs the model:
    the scores were computed once and stored. Entering a hyperparameter sweep
    without this is spending CPU hours looking in D for what was sitting in F
    (ui.md §6).

    The histogram and the curve go back as **two payloads and are drawn as two
    charts** (R4): precision and recall share the 0–1 scale, the bucket counts do
    not, and putting them on one plot with two y-axes would invent a correlation
    that is not in the data.
    """
    index = corner_index(corner)
    score = _select(diag.table.score, index).ravel()
    positive = _select(diag.exists, index).ravel()

    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())

    curve = []
    for threshold in np.linspace(0.0, 1.0, _CURVE_POINTS):
        predicted = score >= threshold
        tp = int((predicted & positive).sum())
        fp = int((predicted & ~positive).sum())
        fn = int((~predicted & positive).sum())
        curve.append({"threshold": round(float(threshold), 4), **prf1(tp, fp, fn)})

    # The actionable output, and the reason V8 comes before the sweep: the
    # threshold that maximises F1, chosen post-hoc, for free. Reported rather
    # than applied -- picking it is a decision, and a view that silently moved
    # the threshold would be choosing for you.
    best = max(curve, key=lambda point: point["f1"]) if n_pos else None

    edges = np.linspace(0.0, 1.0, _HISTOGRAM_BINS + 1)
    hist_pos, _ = np.histogram(score[positive], bins=edges)
    hist_neg, _ = np.histogram(score[~positive], bins=edges)

    return {
        "corner": corner or "all",
        "positives": n_pos,
        "negatives": n_neg,
        # The imbalance, in the place where it explains the shape on screen:
        # ~20 % positives means accuracy is meaningless (always saying "no
        # corner" already scores 80 %) and PR is the curve that informs.
        "positive_rate": n_pos / (n_pos + n_neg) if (n_pos + n_neg) else None,
        "curve": curve,
        "best": best,
        "default_threshold": DEFAULT_THRESHOLD,
        "histogram": {
            "edges": [round(float(e), 4) for e in edges],
            "positive": [int(v) for v in hist_pos],
            "negative": [int(v) for v in hist_neg],
        },
    }


#: Cells per side of the error map, by default.
#:
#: **ui.md §4.1 says "heatmap 40×40", and measured on real data that resolution
#: does not work** -- which is a fact about the data, not a disagreement with the
#: spec. `fase3-red`'s val split holds ~200 corners of a given type spread over
#: 1600 cells: about **0.1 samples per cell**, so the map is a mean of one sample
#: where it is anything at all, and it renders as speckle. The aggregate says the
#: edges are twice as bad as the centre (14.7 px vs 7.5 px); at 40×40 you cannot
#: see it. A view that cannot answer its own question is not a view.
#:
#: 10 per side = 4×4 px cells ≈ 8 corners each, which is enough for the
#: edge-vs-centre reading V7 exists for. It stays a parameter because the right
#: answer scales with the dataset: D6's ~2000 images would carry a finer grid,
#: and `bins = patch_size` gives exactly the map the spec describes.
DEFAULT_ERROR_MAP_BINS = 10


def error_map(diag: Diagnostics, corner: str | None = None, bins: int | None = None) -> dict:
    """V7 — where inside the patch the corner really was, and how far off we were.

    **The view that says which domain to fix**, and the most valuable in the
    catalogue for this project's actual question (ui.md §4.1): if the error piles
    up at the patch's edges -- half-visible corners -- the answer is to lower B's
    `stride`, not to add filters to C. Without this map that diagnosis is
    systematically misread as "the network is too small".

    Cells with no positive corner are **None, not 0**: nothing landed there, and a
    0 would paint "perfect localisation" over the parts of the patch the data
    never covered. `counts` travels alongside so a cell built on 2 samples is not
    read as if it were built on 200 -- and it is also the R5 number table.
    """
    index = corner_index(corner)
    n = diag.patch_size
    size = bins or DEFAULT_ERROR_MAP_BINS
    if not 1 <= size <= n:
        raise NotMeasurable(
            "invalid_bins",
            f"bins tiene que estar entre 1 y el patch_size ({n}); pediste {size}",
            f"usa bins={DEFAULT_ERROR_MAP_BINS} (celdas de {n // DEFAULT_ERROR_MAP_BINS} px), "
            f"o bins={n} para el mapa a resolución completa",
        )

    err = _select(diag.table.err_px, index)
    xy = _select(diag.xy_true, index)
    exists = _select(diag.exists, index)

    total = np.zeros((size, size), dtype=np.float64)
    counts = np.zeros((size, size), dtype=np.int64)

    rows = np.clip((xy[..., 1] * size).astype(int), 0, size - 1)
    cols = np.clip((xy[..., 0] * size).astype(int), 0, size - 1)
    # `np.add.at` rather than `total[rows, cols] += err`: fancy-index assignment
    # keeps only the LAST write when two corners land in one cell, so the map
    # would silently report one sample per cell instead of the mean of all of
    # them. Unbuffered accumulation is what makes it a mean.
    np.add.at(total, (rows[exists], cols[exists]), err[exists])
    np.add.at(counts, (rows[exists], cols[exists]), 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, total / np.maximum(counts, 1), np.nan)

    return {
        "corner": corner or "all",
        "patch_size": n,
        "bins": size,
        #: Px per cell, so the axes can be labelled in the patch's own units
        #: rather than in bin numbers, which mean nothing to a reader.
        "cell_px": n / size,
        "matrix": _nan_to_none(mean),
        "counts": [[int(v) for v in row] for row in counts],
        "samples": int(counts.sum()),
        # The client cannot know whether it is looking at a signed weight or a
        # non-negative magnitude, so the payload declares the colour work
        # (api.md §3). An error in px is a magnitude: sequential (R3).
        "job": "sequential",
    }


def _order_key(diag: Diagnostics, corner: int | None, order: str) -> np.ndarray:
    """The sort key, descending. NaN sinks: "no corner" is not "no error"."""
    if order == "patch":
        return -diag.table.patch_idx.astype(np.float64)
    if order == "score":
        return _select(diag.table.score, corner).max(axis=1)
    if order == "error":
        err = _select(diag.table.err_px, corner)
        # The worst corner of each patch. An all-NaN row is a patch with no real
        # corner at all: it has no error, so it goes last rather than first --
        # `nanmax` would warn and return NaN, and NaN sorts unpredictably.
        worst = np.where(np.isnan(err).all(axis=1), -np.inf, np.nanmax(np.nan_to_num(err, nan=-np.inf), axis=1))
        return worst
    raise NotMeasurable(
        "unknown_order", f"no sé ordenar por '{order}'", "usa order=error, score o patch"
    )


def _outcome_mask(diag: Diagnostics, corner: int | None, outcome: str, threshold: float) -> np.ndarray:
    """(M,) bool — which patches match. With `corner=all`, ANY corner matching.

    `threshold` is the point: it is applied HERE, over stored scores, so changing
    it re-decides every outcome without a single forward pass (ui.md §3).
    """
    if outcome == "all":
        return np.ones(len(diag.table), dtype=bool)

    predicted = _select(diag.table.score, corner) >= threshold
    positive = _select(diag.exists, corner)
    masks = {
        "tp": predicted & positive,
        "fp": predicted & ~positive,
        "fn": ~predicted & positive,
        "tn": ~predicted & ~positive,
    }
    if outcome not in masks:
        raise NotMeasurable(
            "unknown_outcome",
            f"no sé filtrar por '{outcome}'",
            "usa outcome=all, tp, fp, fn o tn",
        )
    return masks[outcome].any(axis=1)


def rows(
    diag: Diagnostics,
    *,
    corner: str | None = None,
    outcome: str = "all",
    order: str = "error",
    threshold: float = DEFAULT_THRESHOLD,
    offset: int = 0,
    limit: int = 24,
) -> dict:
    """V6 — the worst-first gallery, filtered and paginated.

    **No pixels here.** A row says which patch it is (`patch_idx`); the pixels
    come from B's own endpoint, which already serves exactly that. Inlining them
    would put a copy of B's data in E×B's payload and make a page of 24 thumbnails
    ~300 KB of JSON -- and it would mean two endpoints could disagree about what
    patch 37 looks like.
    """
    index = corner_index(corner)
    selected = np.nonzero(_outcome_mask(diag, index, outcome, threshold))[0]

    key = _order_key(diag, index, order)[selected]
    # Stable, so a tie between two patches with identical error does not shuffle
    # under the reader between one page and the next.
    ranked = selected[np.argsort(-key, kind="stable")]
    page = ranked[offset : offset + limit]

    return {
        "total": int(selected.size),
        "offset": offset,
        "limit": limit,
        "threshold": threshold,
        "rows": [
            {
                "patch_idx": int(diag.table.patch_idx[i]),
                "sample_idx": int(diag.sample_idx[i]),
                "patch_xy": [int(v) for v in diag.patch_xy[i]],
                "score": _nan_to_none(diag.table.score[i]),
                "xy_pred": [_nan_to_none(p) for p in diag.table.xy_pred[i]],
                "xy_true": [_nan_to_none(p) for p in diag.xy_true[i]],
                "exists": [bool(v) for v in diag.exists[i]],
                "err_px": _nan_to_none(diag.table.err_px[i]),
            }
            for i in page
        ],
        "corner_order": list(CORNER_NAMES),
    }
