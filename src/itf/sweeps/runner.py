"""H — the engine: optuna proposes points, each point becomes a run.

**optuna is inside H, not instead of it** (librerias.md, organizacion.md §3). It
owns the space, the sampler and the pruner; the organisation -- B and C fixed, the
objective declared, provenance by name -- is still ours. The load-bearing border:
**a trial is not a run.** A trial launches a run (a first-class E on disk, with
`provenance.sweep` set) and stores its name; optuna's SQLite remembers the trial,
`runs/` remembers the run.

This module is the only one that imports optuna, and it is imported lazily by the
API (after the ⑨ validation, which lives in `spec.py` and needs no engine).

**Resume is the point of the persistence** (plan-ui.md fase 7 verification): the
study lives in `optuna.db`, the runs live on disk, so a sweep that was cut by an
API restart picks up where it left off. `run_sweep` is idempotent -- it counts the
trials already done and runs only the rest -- so re-enqueuing it *is* resuming it.
"""

from __future__ import annotations

from typing import Callable

from itf.sweeps.spec import OBJECTIVE_METRIC, SweepSpec, build_recipe_dict
from itf.sweeps.store import SweepStore
from itf.training import RunStore, build_provenance
from itf.training.loop import IncompatibleError, RunSpec, frozen_config, train
from itf.training.recipe import Recipe
from itf.validation import check_run


def _make_sampler(strategy: str, space: dict, seed: int):
    import optuna

    if strategy == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if strategy == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if strategy == "grid":
        return optuna.samplers.GridSampler(_grid_search_space(space), seed=seed)
    raise ValueError(f"estrategia desconocida: {strategy}")


def _grid_search_space(space: dict) -> dict:
    """Turn the space into an explicit grid. Only discrete fields grid.

    A float range has no grid without a step, so `check_sweep` refuses `grid` with
    a float field before we get here; this enumerates the int and categorical ones.
    """
    grid: dict = {}
    for name, dist in space.items():
        if dist["type"] == "categorical":
            grid[name] = list(dist["choices"])
        elif dist["type"] == "int":
            grid[name] = list(range(int(dist["low"]), int(dist["high"]) + 1))
        else:  # pragma: no cover - guarded by check_sweep
            raise ValueError(f"grid no admite el campo continuo '{name}'")
    return grid


def _suggest(trial, space: dict) -> dict:
    """The point this trial proposes, as recipe overrides."""
    out: dict = {}
    for name, dist in space.items():
        kind = dist["type"]
        if kind == "float":
            out[name] = trial.suggest_float(name, dist["low"], dist["high"], log=bool(dist.get("log", False)))
        elif kind == "int":
            out[name] = trial.suggest_int(name, int(dist["low"]), int(dist["high"]), log=bool(dist.get("log", False)))
        elif kind == "categorical":
            out[name] = trial.suggest_categorical(name, dist["choices"])
    return out


def _make_pruner(pruning: bool):
    import optuna

    if not pruning:
        return optuna.pruners.NopPruner()
    # `n_startup_trials=1` so pruning can begin as soon as ONE point has run --
    # a short sweep (the verification is 4 points) would never prune with the
    # default 5. `n_warmup_steps=1` so the first, noisiest epoch is never the one
    # that kills a point.
    return optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1, interval_steps=1)


def _reap_running(study, runs: RunStore) -> None:
    """Fail trials left RUNNING by a crashed process, and drop their runs.

    On resume, a trial that was in flight when the API died is still RUNNING in
    optuna's storage, but its thread is gone. Left alone it would never count as
    done, so the sweep would run extra points past its budget. Its run never
    finished either, so it is deleted -- a fresh trial (a new number, a new run)
    fills the gap.
    """
    import optuna
    from optuna.trial import TrialState

    for t in study.trials:
        if t.state != TrialState.RUNNING:
            continue
        run_name = t.user_attrs.get("run")
        if run_name:
            try:
                runs.delete(run_name)
            except Exception:  # noqa: BLE001 - a missing run is fine; we are cleaning up
                pass
        try:
            study._storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
        except Exception:  # noqa: BLE001 - best effort; worst case is one extra point
            pass


def run_sweep(
    spec: SweepSpec,
    *,
    runs: RunStore,
    patch_datasets,
    networks,
    recipes,
    sweeps: SweepStore,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Run (or resume) the sweep to its budget. Returns its progress summary.

    Each point:
      1. optuna proposes a value for every field in the space;
      2. it becomes a recipe (base + point, epochs from the budget);
      3. the **same gate** `POST /runs` and `itf-train` use reserves the run;
      4. it trains, reporting the objective each epoch so optuna can PRUNE it;
      5. its run name and its (f1, pos_err_px) land on the trial for V12.
    """
    import optuna
    from optuna.trial import TrialState

    should_stop = should_stop or (lambda: False)

    study = optuna.create_study(
        study_name=spec.name,
        storage=sweeps.storage_url(spec.name),
        direction=spec.direction,
        sampler=_make_sampler(spec.strategy, spec.space, spec.seed),
        pruner=_make_pruner(spec.budget.pruning),
        load_if_exists=True,
    )
    _reap_running(study, runs)

    base = recipes.get(spec.recipe).as_dict() if spec.recipe else Recipe().as_dict()
    manifest = patch_datasets.manifest(spec.patch_dataset)
    network = networks.get(spec.network)
    network.pop("format_version", None)
    b_path = str(patch_datasets.path(spec.patch_dataset))
    metric_key = OBJECTIVE_METRIC[spec.objective]

    def _terminal(t) -> bool:
        return t.state in (TrialState.COMPLETE, TrialState.PRUNED)

    done = sum(1 for t in study.trials if _terminal(t))
    remaining = max(0, spec.budget.points - done)

    def objective(trial) -> float:
        overrides = _suggest(trial, spec.space)
        recipe = Recipe.from_dict(build_recipe_dict(base, overrides, spec.budget.epochs))
        run_name = f"{spec.name}-{trial.number:04d}"

        run_spec = RunSpec(
            data=b_path,
            out=str(runs.path(run_name)),
            network=network,
            provenance=build_provenance(
                patch_dataset={"name": spec.patch_dataset, "fingerprint": manifest["fingerprint"]},
                network={"name": spec.network, "value": network},
                # The point's recipe has no stored name; it is grouped by the
                # sweep instead (provenance.sweep). The base's name (or the
                # sweep's) satisfies R7's "it has a name".
                recipe={"name": spec.recipe or spec.name, "value": recipe.as_dict()},
                sweep=spec.name,
            ),
            recipe=recipe,
            device="cpu",
        )

        # The third door (organizacion.md §2-①): B and C are fixed and were
        # validated at sweep creation, so this should never fire -- but a sweep
        # that skipped the validator is the laxest door, and that is the one a
        # bad point comes through.
        problems = check_run(manifest, network)
        if problems:
            raise IncompatibleError(problems)
        runs.create(run_name, frozen_config(run_spec))
        trial.set_user_attr("run", run_name)

        state = {"prune": False, "best": None}

        def on_epoch(epoch: int, record: dict) -> None:
            value = (record.get("val") or {}).get(metric_key)
            if value is None:
                return
            value = float(value)
            trial.report(value, epoch)
            best = state["best"]
            improved = best is None or (
                value > best if spec.direction == "maximize" else value < best
            )
            if improved:
                state["best"] = value
            if trial.should_prune():
                state["prune"] = True

        def should_stop_run() -> bool:
            return (
                state["prune"]
                or runs.stop_requested(run_name)
                or should_stop()
                or sweeps.stop_requested(spec.name)
            )

        with runs.marking_failures(run_name):
            summary = train(run_spec, on_epoch=on_epoch, should_stop=should_stop_run)

        # Both metrics, whatever the objective, so V12 can draw the (f1,
        # pos_err_px) Pareto view regardless of what this sweep ranked by.
        final_val = (summary.get("final") or {}).get("val") or {}
        trial.set_user_attr("f1", final_val.get("f1"))
        trial.set_user_attr("pos_err_px", final_val.get("pos_err_px"))
        trial.set_user_attr("epochs_run", summary.get("epochs_run"))

        if state["prune"]:
            raise optuna.TrialPruned()
        if state["best"] is None:
            # The objective was never measurable (e.g. pos_err_px over a val with
            # no corners). It cannot be ranked, so it does not count as a point.
            raise optuna.TrialPruned()
        return state["best"]

    def _after_trial(study_, trial_) -> None:
        # Runs in the worker thread, so it is the only one touching optuna's
        # storage. It writes the JSON snapshot the API reads (SweepStore.read_
        # progress) and honours the cooperative stop. Doing both here means the
        # snapshot is fresh after every trial -- which is what makes the resume
        # decision correct to within one trial.
        sweeps.write_progress(spec.name, _snapshot(spec, study_))
        if should_stop() or sweeps.stop_requested(spec.name):
            study_.stop()

    # An initial snapshot so `GET /sweeps/{name}` shows the sweep as soon as it is
    # picked up, before the first trial finishes.
    sweeps.write_progress(spec.name, _snapshot(spec, study))

    if remaining > 0 and not (should_stop() or sweeps.stop_requested(spec.name)):
        study.optimize(objective, n_trials=remaining, callbacks=[_after_trial])

    snapshot = _snapshot(spec, study)
    sweeps.write_progress(spec.name, snapshot)
    return snapshot


def _snapshot(spec: SweepSpec, study) -> dict:
    """The trials table, ordered by optuna, as a plain dict. Worker-thread only.

    Built from the live study (which only the worker holds open) and written to
    `progress.json`; the API reads that file, never the SQLite (see SweepStore).
    """
    from optuna.trial import TrialState

    trials = []
    for t in study.trials:
        trials.append(
            {
                "number": t.number,
                "state": t.state.name.lower(),
                "run": t.user_attrs.get("run"),
                "params": t.params,
                "value": t.value if t.state == TrialState.COMPLETE else None,
                "f1": t.user_attrs.get("f1"),
                "pos_err_px": t.user_attrs.get("pos_err_px"),
                "epochs_run": t.user_attrs.get("epochs_run"),
            }
        )
    completed = sum(1 for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED))
    best = None
    if any(t.state == TrialState.COMPLETE for t in study.trials):
        bt = study.best_trial
        best = {"number": bt.number, "run": bt.user_attrs.get("run"), "value": bt.value, "params": bt.params}
    return {
        "name": spec.name,
        "objective": spec.objective,
        "direction": spec.direction,
        "budget": spec.budget.as_dict(),
        "space": spec.space,
        "patch_dataset": spec.patch_dataset,
        "network": spec.network,
        "trials": trials,
        "completed": completed,
        "best": best,
    }
