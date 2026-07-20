"""What the per-corner numbers MEAN. One definition, two readers.

This module exists for the reason contract ⑤ exists, and it is worth naming
before the code: **`pos_err_px` is computed in two places now.** `evaluate()`
writes it into `metrics.jsonl` every epoch (D), and the per-patch table writes it
per patch for V7 (E×B). If those two formulas ever disagree, the error map shows
one number and the run's own curve shows another, both called "error de posición
en píxeles" -- and nothing raises. That is exactly the shape of the trap the
window duplication was (tests.md §1.2): identical copies, so it works, until
someone touches one.

So the definition lives here and both import it. `itf.geometry` is the same idea
for the window; this is the same idea for the metrics.

**Imports nothing from `itf`**, like `itf.validation`, and for a related reason:
it takes arrays, not objects. It knows nothing about runs, datasets or networks,
which is what lets D and the diagnóstico both call it without either importing
the other.

The seam is tested where a seam must be tested -- not "is `position_error_px`
correct?" (both sides call it; it cannot diverge) but **"does the table measure
the same thing the run reported?"** See `tests/test_diagnostics.py`.
"""

from __future__ import annotations

import torch

#: The default decision threshold on `p(exists)`.
#:
#: **It is F, not D** (organizacion.md §1-D): it is tuned post-hoc on a trained
#: model, and moving it costs nothing. The per-patch table stores raw scores for
#: exactly that reason -- re-thresholding filters a column instead of re-running
#: the model, which is where V8's saved CPU hours come from.
DEFAULT_THRESHOLD = 0.5


def position_error_px(pred_xy: torch.Tensor, true_xy: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Localisation error per corner, in pixels of the patch. Shape `(..., 4)`.

    L1 (Manhattan), not Euclidean, because that is what `evaluate()` has always
    measured and what every `pos_err_px` in `metrics.jsonl` already means. The
    choice is arguable; changing it silently is not -- it would move every number
    in the project's history without moving a config (protocolo.md §7).

    **This function did move `pos_err_px`, in the 9th significant digit, and that
    was measured rather than assumed** *(fase 5)*. Re-running `fase4-ui` after the
    refactor gave a bit-identical `loss`, `f1`, `precision` and `recall` -- so the
    weights are untouched -- and `pos_err_px` 12.427402796 against 12.427402806.
    The cause is float32 associativity: `evaluate()` used to sum in patch units
    and scale the total in float64, and now it scales per element and sums values
    40x larger in float32. The new order is the CORRECT one, which is the whole
    point: `evaluate()`'s number is now exactly the mean of the per-patch values
    the table stores, so V7's map and the run's own curve cannot describe two
    different quantities under one name. A ~1e-9 shift is orders of magnitude
    below what this data can resolve (protocolo.md §1: differences under ~5 % are
    not resolvable with a 20-image val), but it is recorded here rather than left
    to be rediscovered as a mystery.

    Coordinates come in normalised to [0,1] *within the patch*, so multiplying by
    `patch_size` is what puts them in pixels. That normalisation is also why
    `smooth_l1_beta` matters: with |error| < 1 always, PyTorch's default beta=1.0
    keeps the Huber in its quadratic branch forever (organizacion.md §3).
    """
    return (pred_xy - true_xy).abs().sum(-1) * patch_size


#: Below this fraction of the patch, a corner is **blind**: there is essentially
#: nothing of its paragraph inside the window to look at.
#:
#: 0.05 is a judgement and it is written here so it is made once. It was picked
#: off the measured curve, not off a hunch: the error per band rises monotonically
#: as evidence falls, and `[0, 0.05)` is where it breaks away (5.2 px vs 2.0 px on
#: a 20 px patch, `dirty-20`). Callers can pass their own cut -- the bands are a
#: reading aid, the function below is the definition.
BLIND_EVIDENCE = 0.05

#: Per corner, which way the paragraph's body extends from the corner point, as
#: `(towards_x0, towards_y0)` flags in `CORNER_NAMES` order (TL, TR, BR, BL).
#:
#: A TL sits at the paragraph's top-left, so the body runs right and down: the
#: visible part is the rectangle from the point to the patch's far corner, i.e.
#: `(1-fx)·(1-fy)`. Hence `1` means "measure towards the far edge".
#:
#: **This module imports nothing from `itf`** (see the header), so the order is
#: restated here rather than imported -- and `tests/test_metrics.py` pins it
#: against `itf.geometry.CORNER_NAMES`, which is the seam that keeps the two
#: honest. Same reasoning as contract ⑤: don't duplicate silently, test the join.
_EVIDENCE_TOWARDS = ((1, 1), (0, 1), (0, 0), (1, 0))


def corner_evidence(xy):
    """How much of the patch the corner's paragraph *can* occupy. Shape `(..., 4)`.

    **The question it answers**: when the label says "there is a TL here", is there
    anything of that paragraph inside the patch to see? A TL whose point falls near
    the patch's bottom-right has its whole paragraph *outside* the window -- the
    pixels are blank (or full of some other paragraph) and the label is asking for
    something the patch does not show.

    Purely geometric: it takes the normalised `(fx, fy)` within the patch and
    nothing else. No pixels, no model, no extra field in the `.npz` -- which is
    what makes it computable both here (over the diagnostics table) and in a
    dataloader, without either one owning it.

    It is an **upper bound**: it assumes the paragraph is at least as large as the
    patch, so a small paragraph gets credited more evidence than it has. That is
    the safe direction -- what this flags as blind *is* blind; there may be more
    blind corners it does not flag.

    Works on numpy arrays and torch tensors alike: only arithmetic and slice
    assignment, no library-specific calls. Position (`xy`) is meaningless where no
    corner exists, so the caller masks with `exists` -- this returns a number for
    every slot rather than deciding what "absent" looks like (formatos.md §2).
    """
    fx, fy = xy[..., 0], xy[..., 1]
    evidence = fx * 0
    for index, (towards_x0, towards_y0) in enumerate(_EVIDENCE_TOWARDS):
        dx = (1 - fx[..., index]) if towards_x0 else fx[..., index]
        dy = (1 - fy[..., index]) if towards_y0 else fy[..., index]
        evidence[..., index] = dx * dy
    return evidence


def prf1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall and F1 from counts, with the empty cases pinned to 0.

    Shared by `evaluate()` (one threshold, per epoch) and by the PR curve (101
    thresholds, post-hoc, off the cached scores). Same definition, so the point
    of V8's curve at 0.5 lands exactly on the run's own reported `f1` -- and that
    equality is a test, because if it did not hold one of the two would be lying.

    0 rather than NaN when nothing was predicted or nothing existed: here the
    count IS the measurement, so "no positives predicted" is a real precision of
    0, not an absence (which is the formatos.md §2 case -- see `pos_err_px`,
    where `None` is right because nothing was measured at all).
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
