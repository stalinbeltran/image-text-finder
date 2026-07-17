"""E × B — the diagnóstico: what a trained run does to a split, patch by patch.

**The substrate of ui.md §3 and the reason the app becomes an instrument.** V6
(worst-first gallery), V7 (error by position) and V8 (scores + PR) are the same
pass over the val split, so they share one table instead of each re-running the
model.

Not a domain of organizacion.md §1, and that is decided rather than overlooked
(D1): the table is a pure function of things that already have identity -- run,
B's fingerprint, split -- so it can be recomputed exactly, and what can be
recomputed is not stored. **It is a cache.** No name, no CRUD, no listing, no
Evaluaciones screen. The four views come out the same.

May import `patches` (B), `training` (E), `inference` (the checkpoint), `metrics`
and `geometry` -- it is the crossing of E and B, so of course it knows both.
Nothing imports it back except `itf.api`.
"""

from itf.diagnostics.aggregate import Diagnostics, error_map, pr, rows
from itf.diagnostics.service import DEFAULT_CHECKPOINT, open_diagnostics
from itf.diagnostics.table import (
    NotMeasurable,
    PatchTable,
    TableCache,
    TableKey,
    compute_table,
)

__all__ = [
    "DEFAULT_CHECKPOINT",
    "Diagnostics",
    "NotMeasurable",
    "PatchTable",
    "TableCache",
    "TableKey",
    "compute_table",
    "error_map",
    "open_diagnostics",
    "pr",
    "rows",
]
