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
