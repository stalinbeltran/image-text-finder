"""H — the sweep: the ⑨ validator, and a real short sweep with pruning + resume.

The unit half tests `check_sweep` (pure, milliseconds). The integration half runs
the optuna engine over a tiny synthetic B/C/D -- the only way to prove that a
trial becomes a run with `provenance.sweep` set, that the study persists, and that
re-running is a resume, not a restart (plan-ui.md fase 7 verification).
"""

from __future__ import annotations

import json

from itf.sweeps import SweepSpec, check_sweep
from itf.sweeps.runner import read_progress, run_sweep


# --------------------------------------------------------------------------- #
# ⑨ and the rest of the spec's shape -- pure, no engine
# --------------------------------------------------------------------------- #


def test_check_sweep_rejects_loss_objective_when_lambda_pos_varies():
    problems = check_sweep(
        {
            "objective": "loss",
            "strategy": "random",
            "space": {"lambda_pos": {"type": "float", "low": 0.0, "high": 5.0}},
            "budget": {"points": 4, "epochs": 3},
        }
    )
    assert any(p["code"] == "objective_varies_with_space" for p in problems)


def test_check_sweep_allows_loss_when_lambda_pos_is_fixed():
    """The control: `loss` is only a trap while λ VARIES. Sweep lr, rank by loss."""
    problems = check_sweep(
        {
            "objective": "loss",
            "strategy": "random",
            "space": {"lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
            "budget": {"points": 4, "epochs": 3},
        }
    )
    assert not any(p["code"] == "objective_varies_with_space" for p in problems)


def test_check_sweep_flags_unknown_objective_field_and_empty_space():
    assert any(
        p["code"] == "unknown_objective"
        for p in check_sweep({"objective": "accuracy", "space": {"lr": {"type": "float", "low": 1, "high": 2}}, "budget": {"points": 1, "epochs": 1}})
    )
    assert any(
        p["code"] == "unknown_space_field"
        for p in check_sweep({"objective": "f1", "space": {"nonsense": {"type": "float", "low": 1, "high": 2}}, "budget": {"points": 1, "epochs": 1}})
    )
    assert any(
        p["code"] == "empty_space"
        for p in check_sweep({"objective": "f1", "space": {}, "budget": {"points": 1, "epochs": 1}})
    )


# --------------------------------------------------------------------------- #
# A real short sweep: pruning, provenance, and resume
# --------------------------------------------------------------------------- #


def _stores(layout):
    from itf.models import NetworkStore
    from itf.patches import PatchExtractConfig, SplitConfig, extract_dataset
    from itf.patches.store import PatchDatasetStore
    from itf.sweeps import SweepStore
    from itf.training import Recipe, RecipeStore, RunStore

    source = layout.datasets / "tiny"
    from conftest import write_tiny_source

    write_tiny_source(source, num_samples=10, width=120, height=100)

    patch_datasets = PatchDatasetStore(layout.patch_datasets)
    extract_dataset(
        PatchExtractConfig(
            source=str(source),
            out=str(patch_datasets.path("tiny-40")),
            patch_size=40,
            stride=40,
            split=SplitConfig(train=0.8, val=0.1, test=0.1),
            seed=1,
        )
    )

    networks = NetworkStore(layout.networks)
    networks.save(
        "cnn-a",
        {
            "input_size": 40,
            "in_channels": 1,
            "border_features": False,
            "backbone": [{"filters": 4, "kernel": 3, "stride": 1, "padding": 1, "pool": 2}],
            "head": {"hidden": [8]},
        },
    )

    recipes = RecipeStore(layout.recipes)
    recipes.save("base", Recipe(epochs=2, batch_size=16, monitor="val_f1"))

    return {
        "runs": RunStore(layout.runs),
        "patch_datasets": patch_datasets,
        "networks": networks,
        "recipes": recipes,
        "sweeps": SweepStore(layout.sweeps),
    }


def test_a_short_sweep_makes_runs_with_sweep_provenance_and_resumes(layout):
    """Two points, two epochs. Each becomes a run that names its parent sweep, and
    re-running the sweep does nothing -- the budget is already met (resume)."""
    stores = _stores(layout)
    spec = SweepSpec.from_dict(
        {
            "name": "s1",
            "patch_dataset": "tiny-40",
            "network": "cnn-a",
            "recipe": "base",
            "space": {"lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
            "objective": "f1",
            "strategy": "random",
            "budget": {"points": 2, "epochs": 2, "pruning": True},
            "seed": 0,
        }
    )
    stores["sweeps"].create(spec)

    progress = run_sweep(spec, **stores)
    assert progress["completed"] == 2, "el barrido debe llegar a su presupuesto"

    # Each trial is a run on disk, and it names its parent sweep (contract ③).
    run_names = [t["run"] for t in progress["trials"]]
    assert run_names == ["s1-0000", "s1-0001"]
    for run_name in run_names:
        prov = stores["runs"].config(run_name)["provenance"]
        assert prov["sweep"] == "s1", "un punto del barrido apunta a su padre"
        assert prov["network"]["name"] == "cnn-a"

    # Resume: the study is on disk, the budget is met, so a second call proposes
    # no new points -- and creates no new runs.
    before = set(stores["runs"].names())
    again = run_sweep(spec, **stores)
    assert again["completed"] == 2
    assert set(stores["runs"].names()) == before, "reanudar un barrido completo no crea runs nuevos"


def test_read_progress_on_an_unstarted_sweep_is_empty(layout):
    """A sweep reserved but never run reports an empty table, not a failure."""
    from itf.sweeps import SweepStore

    sweeps = SweepStore(layout.sweeps)
    spec = SweepSpec.from_dict(
        {
            "name": "never",
            "patch_dataset": "tiny-40",
            "network": "cnn-a",
            "space": {"lr": {"type": "float", "low": 1e-4, "high": 1e-2}},
            "objective": "f1",
            "strategy": "random",
            "budget": {"points": 2, "epochs": 2},
        }
    )
    sweeps.create(spec)
    progress = read_progress(spec, sweeps)
    assert progress["trials"] == [] and progress["completed"] == 0
    assert json.loads((sweeps.path("never") / "spec.json").read_text())["objective"] == "f1"
