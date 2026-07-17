"""H — the hyperparameter sweep: a space of D explored with B and C fixed.

The public surface is deliberately split: everything here is pure (spec + store +
the ⑨ validator), so importing `itf.sweeps` does NOT pull in optuna. The engine
(`run_sweep`, `read_progress`) lives in `itf.sweeps.runner` and is imported where
a sweep is actually run -- which is what lets `POST /sweeps` answer the ⑨ 400
without the engine installed.

`itf.sweeps` is the seed of the future library, and like `matrixview` it is
written library-shaped but NOT extracted: a single project needs it, and
librerias.md §2 is explicit that the sweep is the trap you extract on the second
use, not the first.
"""

from __future__ import annotations

from itf.sweeps.spec import (
    OBJECTIVE_DIRECTION,
    OBJECTIVE_METRIC,
    STRATEGIES,
    Budget,
    SweepSpec,
    build_recipe_dict,
    check_sweep,
)
from itf.sweeps.store import SweepExists, SweepNotFound, SweepStore

__all__ = [
    "OBJECTIVE_DIRECTION",
    "OBJECTIVE_METRIC",
    "STRATEGIES",
    "Budget",
    "SweepSpec",
    "SweepStore",
    "SweepExists",
    "SweepNotFound",
    "build_recipe_dict",
    "check_sweep",
]
