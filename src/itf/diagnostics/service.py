"""Resolving a run and a split into a table you can ask questions of.

This is the composition step, and it lives here rather than in `api/app.py` for
the mechanical reason of api.md §0: **nothing in it mentions HTTP.** Following a
run's provenance to its dataset, checking that dataset is still the one that was
trained on, finding the checkpoint -- none of that is about HTTP, so none of it
is the API's. The API translates `NotMeasurable` into a status code and that is
its whole job here.

Which is also what makes the diagnóstico usable from a script: the same reason
`itf-extract` proves the boundary for B.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from itf.diagnostics.aggregate import Diagnostics
from itf.diagnostics.table import NotMeasurable, TableCache, TableKey
from itf.patches import SPLIT_NAMES
from itf.patches.store import PatchDatasetNotFound, PatchDatasetStore
from itf.training.registry import RunNotFound, RunStore

#: `best.pt`, not `last.pt`. The checkpoint the run itself selected by its own
#: monitor is the model the run IS -- fase 3 measured `best.pt` landing on epoch
#: 17 of 20, so the two are genuinely different models and diagnosing the wrong
#: one would describe a network nobody would ever use.
DEFAULT_CHECKPOINT = "best.pt"


def open_diagnostics(
    *,
    runs: RunStore,
    patch_datasets: PatchDatasetStore,
    cache: TableCache,
    run: str,
    split: str = "val",
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cpu",
) -> Diagnostics:
    """The table for `(run, split)`, from cache or computed, joined to the truth.

    Every refusal is a `NotMeasurable` carrying its `code`, message and hint (R4).
    They are refusals rather than crashes because each one is the system working:
    a run that never trained has nothing to diagnose, and saying so is the answer.
    """
    if split not in SPLIT_NAMES:
        raise NotMeasurable(
            "unknown_split", f"'{split}' no es un split", f"usa uno de {', '.join(SPLIT_NAMES)}"
        )

    try:
        config = runs.config(run)
    except RunNotFound:
        raise NotMeasurable("run_not_found", f"no existe el run '{run}'", "mira GET /runs")

    provenance = config.get("provenance")
    if not provenance:
        # **Said out loud, never worked around.** A run from before fase 4 cannot
        # say which B it was trained on, so there is no dataset to diagnose it
        # against -- and guessing one would measure the model against data it may
        # never have seen, silently. D3 killed the degrading reader on purpose;
        # this is the same rule one layer down.
        raise NotMeasurable(
            "run_without_provenance",
            f"el run '{run}' no tiene procedencia: no puede decir de qué dataset salió, "
            f"así que no hay contra qué diagnosticarlo",
            "es anterior a la fase 4. Bórralo y reentrénalo: no es comparable con nada",
        )

    reference = provenance["patch_dataset"]
    name, fingerprint = reference["name"], reference["fingerprint"]

    try:
        manifest = patch_datasets.manifest(name)
    except (PatchDatasetNotFound, ValueError):
        raise NotMeasurable(
            "patch_dataset_missing",
            f"el run '{run}' se entrenó con el dataset de patches '{name}', y ese dataset ya "
            f"no está",
            "reconstrúyelo con la misma config y semilla, o borra el run: sin su B no hay "
            "split que medir",
        )

    # **Contract ⑧, and it is exactly what the fingerprint was for.** A B rebuilt
    # under the same name is a different B: another split, other patches. Measured
    # against it, "val" is no longer the val this run's `best.pt` was selected on,
    # so the numbers would look fine and mean something else. Nothing else in the
    # system would notice -- a rebuilt path still points the same way.
    if manifest.get("fingerprint") != fingerprint:
        raise NotMeasurable(
            "patch_dataset_changed",
            f"'{name}' se reconstruyó desde que se entrenó '{run}': la huella no coincide "
            f"({fingerprint[:19]}… vs {str(manifest.get('fingerprint'))[:19]}…). Su split ya no "
            f"es el que este run usó para elegir best.pt",
            "no es comparable: reentrena el run sobre el dataset de ahora, o reconstruye el "
            "dataset con la config y la semilla originales",
        )

    checkpoint_path = runs.path(run) / checkpoint
    if not checkpoint_path.exists():
        state = runs.status(run).get("state")
        raise NotMeasurable(
            "run_has_no_checkpoint",
            f"el run '{run}' no tiene '{checkpoint}' (está '{state}'): todavía no hay ningún "
            f"modelo que diagnosticar",
            "espera a que termine una época, o elige otro run",
        )

    key = TableKey(
        run=run,
        fingerprint=fingerprint,
        split=split,
        checkpoint=checkpoint,
        # Not in D1's key, and it has to be: `best.pt` is rewritten on every
        # improving epoch, so a live run is not the immutable thing "run" implies.
        # See TableKey.
        checkpoint_mtime_ns=checkpoint_path.stat().st_mtime_ns,
    )
    table = cache.get_or_compute(
        checkpoint=checkpoint_path,
        patch_dataset=patch_datasets.path(name),
        key=key,
        device=device,
    )

    # The truth stays in B and is joined by `patch_idx`, rather than copied into
    # the cache: one place for what is true, one for what the model said.
    with patch_datasets.arrays(name) as data:
        rows = table.patch_idx
        truth = data["y"][rows]
        sample_idx = data["sample_idx"][rows]
        patch_xy = data["patch_xy"][rows]

    return Diagnostics(
        table=table,
        truth=np.asarray(truth),
        sample_idx=np.asarray(sample_idx),
        patch_xy=np.asarray(patch_xy),
        patch_size=int(manifest["config"]["patch_size"]),
    )
