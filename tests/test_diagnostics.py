"""E × B — the per-patch table and what is read off it (fase 5).

The two tests that carry this file are the **seams**, and they are seams in the
sense tests.md §1.2 fixed the word: not "is `position_error_px` correct?" -- both
sides call the same function, it cannot diverge -- but **"does the diagnóstico
measure the same thing the run reported?"**. `evaluate()` writes `f1` and
`pos_err_px` into `metrics.jsonl` every epoch; the table recomputes both from
`best.pt` over the same split. If those two ever disagree, V7 and V8 describe a
model whose curve says something else, and nothing raises -- the numbers just
quietly stop meaning the same thing.

The rest guard the properties D1 bought by making the table a cache: it can be
thrown away, it invalidates itself, and choosing a threshold never runs the model.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from itf.diagnostics import (
    NotMeasurable,
    TableCache,
    TableKey,
    error_map,
    open_diagnostics,
    pr,
    rows,
)
from itf.metrics import DEFAULT_THRESHOLD
from itf.patches import PatchDatasetStore, PatchExtractConfig, SplitConfig, extract_dataset
from itf.training.loop import RunSpec, train
from itf.training.recipe import Recipe
from itf.training.registry import RunStore

from conftest import write_tiny_source

NETWORK = {
    "input_size": 40,
    "in_channels": 1,
    "border_features": False,
    "backbone": [{"filters": 4, "kernel": 3, "stride": 1, "pool": 2}],
    "head": {"hidden": [8]},
}


def _build(layout, *, b_name: str = "tiny-40", run_name: str = "run-01", epochs: int = 1, **extract):
    """A real B and a real E over it. **Not fakeable** here, unlike elsewhere.

    Most fixtures in this suite write a manifest and stop, because the validator
    only ever reads two dicts. This one cannot: the table is the model's output
    over real pixels, so both halves have to exist for there to be anything to
    measure.
    """
    source = write_tiny_source(layout.datasets / "tiny", num_samples=10)
    data = layout.patch_datasets / b_name
    extract_dataset(
        PatchExtractConfig(
            source=str(source),
            out=str(data),
            patch_size=40,
            stride=20,
            split=SplitConfig(0.6, 0.2, 0.2),
            **extract,
        )
    )
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))

    run = layout.runs / run_name
    summary = train(
        RunSpec(
            data=str(data),
            out=str(run),
            network=NETWORK,
            recipe=Recipe(epochs=epochs, batch_size=32),
            provenance={
                "patch_dataset": {"name": b_name, "fingerprint": manifest["fingerprint"]},
                "network": {"name": "cnn-a", "value": NETWORK},
                "recipe": {"name": "rapida", "value": {}},
                "sweep": None,
                "git_commit": "0" * 40,
                "environment": {"python": "3.12", "torch": "2.13", "platform": "win32"},
            },
        )
    )
    return manifest, summary


def _open(layout, run: str = "run-01", split: str = "val"):
    return open_diagnostics(
        runs=RunStore(layout.runs),
        patch_datasets=PatchDatasetStore(layout.patch_datasets),
        cache=TableCache(layout.cache),
        run=run,
        split=split,
    )


def _last_epoch(layout, run: str = "run-01") -> dict:
    lines = (layout.runs / run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1])


# ── the seams: the table and the run must measure the same thing ──────────────


def test_the_table_reports_the_same_pos_err_px_the_run_did(layout):
    """The seam. `evaluate()` and the table both measure `pos_err_px`.

    One epoch, so `best.pt` IS the weights that produced the epoch's val metrics
    and the two numbers are comparable at all. If they drift apart, the error map
    of V7 and the curve of V14 are two different measurements wearing one name --
    and the failure is silent, which is the whole reason `itf.metrics` exists.
    """
    _build(layout)
    diag = _open(layout)

    reported = _last_epoch(layout)["val"]["pos_err_px"]
    measured = float(np.nanmean(diag.table.err_px))

    assert reported is not None, "el dataset de prueba tiene esquinas en val"
    assert measured == pytest.approx(reported, rel=1e-4)


def test_the_pr_curve_at_the_default_threshold_lands_on_the_runs_own_f1(layout):
    """The other seam, and it is what makes V8 trustworthy.

    The curve is 101 thresholds computed off stored scores; the run computed one
    threshold in torch during the epoch. The point at 0.5 must be the same number
    the run reported -- otherwise "move the threshold and read the F1" is reading
    a different F1 than the one the project has been optimising all along.
    """
    _build(layout)
    diag = _open(layout)

    reported = _last_epoch(layout)["val"]
    point = next(p for p in pr(diag)["curve"] if p["threshold"] == DEFAULT_THRESHOLD)

    assert point["f1"] == pytest.approx(reported["f1"], rel=1e-6)
    assert point["precision"] == pytest.approx(reported["precision"], rel=1e-6)
    assert point["recall"] == pytest.approx(reported["recall"], rel=1e-6)


# ── ausente ≠ cero, and the JSON it has to survive ────────────────────────────


def test_err_px_is_nan_where_there_is_no_corner_and_never_zero(layout):
    """formatos.md §4.4: `NaN`, not 0. A 0 would say "localised it exactly".

    And it is not an edge case: ~80 % of (patch, corner) pairs have no corner, so
    filling zeros would drag every aggregate on the screen toward 0 -- the error
    map would look excellent everywhere the data never went.
    """
    _build(layout)
    diag = _open(layout)

    missing = ~diag.exists
    assert missing.any(), "el dataset de prueba tiene esquinas ausentes"
    assert np.isnan(diag.table.err_px[missing]).all(), "sin esquina real no hay error que medir"
    # The control: where a corner does exist, there IS a number.
    assert not np.isnan(diag.table.err_px[diag.exists]).any()


def test_no_payload_carries_a_nan_because_that_is_not_json(layout):
    """The fase 4 lesson, one floor down: `json.dumps` writes `NaN` verbatim.

    `Infinity` broke `GET /runs` for every run at once; `NaN` would break every
    diagnostics payload the same way -- and here it is not a corner case, it is
    the designed value of most of `err_px`. So every route that carries an error
    converts on the way out, and this asserts it end to end rather than trusting
    the helper.
    """
    _build(layout)
    diag = _open(layout)

    for payload in (pr(diag), error_map(diag), rows(diag)):
        text = json.dumps(payload)  # would not raise, which is exactly the danger
        assert "NaN" not in text and "Infinity" not in text
        json.loads(text)  # what the browser does, and what fails on a bare NaN


def test_the_error_map_leaves_empty_cells_empty(layout):
    """A cell nothing landed in is `None`, not 0 (formatos.md §2 on a heat map).

    `counts` travels with it so a cell built on 2 samples cannot be read as one
    built on 200 -- and it is the R5 number table at the same time.
    """
    _build(layout)
    payload = error_map(_open(layout), "TL")

    size = payload["bins"]
    assert len(payload["matrix"]) == size and len(payload["matrix"][0]) == size
    for row, counts in zip(payload["matrix"], payload["counts"]):
        for value, count in zip(row, counts):
            assert (value is None) == (count == 0), "celda vacía ⇒ None; celda con datos ⇒ número"
            assert value is None or math.isfinite(value)
    assert payload["samples"] > 0


def test_the_error_maps_resolution_is_a_knob_and_conserves_the_corners(layout):
    """The finding of fase 5: 40×40 is the spec, and the data cannot fill it.

    `fase3-red`'s val holds ~200 corners of a type over 1600 cells -- 0.1 per
    cell -- so the full-resolution map is a mean of one sample wherever it is
    anything, and it renders as speckle. True, and unreadable: the aggregate says
    the edges are twice as bad as the centre and at 40×40 you cannot see it.

    What must hold whatever the resolution: **every corner is counted exactly
    once**. Binning changes how finely the patch is cut, never how many corners
    there were -- a map whose total moved with its resolution would be measuring
    the grid instead of the model.
    """
    _build(layout)
    diag = _open(layout)

    coarse = error_map(diag, "TL", bins=5)
    fine = error_map(diag, "TL", bins=40)

    assert coarse["bins"] == 5 and fine["bins"] == 40
    assert coarse["cell_px"] == 8 and fine["cell_px"] == 1
    assert coarse["samples"] == fine["samples"], "el binning no crea ni pierde esquinas"
    assert sum(map(sum, coarse["counts"])) == sum(map(sum, fine["counts"]))
    # And the coarse map is denser, which is the entire point of the knob.
    filled = lambda m: sum(1 for row in m["matrix"] for v in row if v is not None)
    assert filled(coarse) / 25 > filled(fine) / 1600


def test_the_error_map_refuses_a_resolution_finer_than_the_patch(layout):
    """More cells than pixels would invent structure that cannot exist."""
    _build(layout)
    with pytest.raises(NotMeasurable) as exc:
        error_map(_open(layout), "TL", bins=64)
    assert exc.value.code == "invalid_bins"
    assert exc.value.hint


# ── the cache: what D1 bought ────────────────────────────────────────────────


def test_choosing_a_threshold_never_runs_the_model_again(layout, monkeypatch):
    """**The reason this phase comes before the sweep** (ui.md §6).

    Every threshold of V8 is free because the scores were computed once and
    stored. If this ever regressed, re-thresholding would cost a full pass over
    val per point -- and the sweep would go back to searching in D (hours of CPU)
    for what was sitting in F (free).

    Counting `load_model` calls rather than timing: it is the mechanism, and a
    timing assertion would be flaky on a busy machine.
    """
    import itf.diagnostics.table as table_module

    _build(layout)
    loads = []
    real = table_module.load_model
    monkeypatch.setattr(
        table_module, "load_model", lambda *a, **k: (loads.append(1), real(*a, **k))[1]
    )

    diag = _open(layout)
    strict = rows(diag, outcome="fp", threshold=0.9)
    loose = rows(diag, outcome="fp", threshold=0.1)
    pr(diag, "TL")
    pr(diag, "BR")

    assert len(loads) == 1, "una sola pasada: los scores están guardados"
    # And the control: the threshold must actually be DOING something, or the
    # assertion above would also pass for a filter that ignored it.
    assert strict["total"] != loose["total"], "bajar el umbral tiene que dar más falsos positivos"


def test_the_table_is_cached_and_reread_identically(layout):
    """It is a cache: a second open reads the `.npz` instead of the model.

    Identical arrays, not merely similar -- the table is a pure function of its
    key, so a re-read that differed would mean the key does not identify what it
    claims to.
    """
    _build(layout)
    first = _open(layout)
    second = _open(layout)

    assert np.array_equal(first.table.patch_idx, second.table.patch_idx)
    assert np.array_equal(first.table.score, second.table.score)
    np.testing.assert_array_equal(first.table.err_px, second.table.err_px)  # NaN-aware
    assert list(layout.cache.rglob("*.npz")), "la tabla tiene que haber aterrizado en el caché"


def test_the_cache_can_be_deleted_without_losing_anything(layout):
    """D5's criterion, made executable: what can be recomputed is not kept.

    If deleting the cache ever lost something, the table would have become an
    entity -- and D1 says it is not one, which is why there is no Evaluaciones
    screen.
    """
    import shutil

    _build(layout)
    before = _open(layout).table.score.copy()
    shutil.rmtree(layout.cache)
    after = _open(layout).table.score

    assert np.array_equal(before, after)


def test_a_new_checkpoint_invalidates_the_table(layout):
    """The key D1 did not ask for, and the reason it must be there.

    `best.pt` is rewritten on every improving epoch, so "the run" does not
    identify a model while the run is alive. Keyed on the run alone, opening
    Diagnóstico at epoch 5 and again at epoch 20 answers with epoch 5's table
    both times -- a cache lying with a straight face.
    """
    _build(layout)
    first = _open(layout)
    digest = first.table.key.digest()

    # Touch the checkpoint the way a new epoch would.
    checkpoint = layout.runs / "run-01" / "best.pt"
    stat = checkpoint.stat()
    import os

    os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))

    assert _open(layout).table.key.digest() != digest, "otro checkpoint es otra tabla"


def test_the_key_is_stable_whatever_order_it_was_built_in(layout):
    """Two identical keys must hash the same, or the cache never hits.

    It would never be *wrong* -- just permanently cold, which reads as "the
    diagnóstico is slow" and not as a bug.
    """
    a = TableKey(run="r", fingerprint="sha256:a", split="val", checkpoint="best.pt", checkpoint_mtime_ns=7)
    b = TableKey(checkpoint_mtime_ns=7, checkpoint="best.pt", split="val", fingerprint="sha256:a", run="r")
    assert a.digest() == b.digest()


# ── the refusals: each one is the system working ─────────────────────────────


def test_it_refuses_a_run_whose_dataset_was_rebuilt_underneath_it(layout):
    """**Contract ⑧, and this is what the fingerprint was always for.**

    A B rebuilt under the same name is a different B: another split, other
    patches. Measured against it, "val" is no longer the val this run's `best.pt`
    was selected on -- so the numbers come out looking perfectly normal and mean
    something else. A path still points the same way after a rebuild; only the
    huella notices.
    """
    _build(layout)
    import shutil

    # Rebuild the same name with a different stride: same path, other content.
    shutil.rmtree(layout.patch_datasets / "tiny-40")
    extract_dataset(
        PatchExtractConfig(
            source=str(layout.datasets / "tiny"),
            out=str(layout.patch_datasets / "tiny-40"),
            patch_size=40,
            stride=10,
            split=SplitConfig(0.6, 0.2, 0.2),
        )
    )

    with pytest.raises(NotMeasurable) as exc:
        _open(layout)
    assert exc.value.code == "patch_dataset_changed"
    assert exc.value.hint


def test_it_refuses_a_run_that_cannot_say_where_it_came_from(layout):
    """No degrading reader. D3 killed that, and this is the same rule one floor down.

    Without provenance there is no dataset to diagnose against, and guessing one
    would measure the model over data it may never have seen -- silently.
    """
    run = layout.runs / "fase3-ish"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"format_version": 1}), encoding="utf-8")

    with pytest.raises(NotMeasurable) as exc:
        _open(layout, run="fase3-ish")
    assert exc.value.code == "run_without_provenance"


def test_it_refuses_a_run_that_has_not_trained(layout):
    """Nothing to diagnose is an answer, not a crash."""
    _build(layout)
    (layout.runs / "run-01" / "best.pt").unlink()

    with pytest.raises(NotMeasurable) as exc:
        _open(layout)
    assert exc.value.code == "run_has_no_checkpoint"


def test_it_refuses_a_split_that_is_not_one(layout):
    _build(layout)
    with pytest.raises(NotMeasurable) as exc:
        _open(layout, split="validation")
    assert exc.value.code == "unknown_split"


# ── the gallery ──────────────────────────────────────────────────────────────


def test_worst_first_means_worst_first(layout):
    """V6's whole premise. And patches with no real corner sink, never lead.

    A patch with nothing to find has no error -- `nanmax` of an all-NaN row is not
    "infinitely bad", it is "not measured", so ordering it first would fill the
    gallery with patches that have nothing to look at.
    """
    _build(layout)
    diag = _open(layout)
    page = rows(diag, order="error", limit=200)["rows"]

    errors = [
        max((e for e in r["err_px"] if e is not None), default=None) for r in page
    ]
    measured = [e for e in errors if e is not None]
    assert measured == sorted(measured, reverse=True), "peor primero"
    # The ones with nothing to measure are at the end, not at the front.
    first_none = next((i for i, e in enumerate(errors) if e is None), len(errors))
    assert all(e is None for e in errors[first_none:])


def test_the_gallery_is_paginated_and_the_rows_carry_the_patchs_provenance(layout):
    """R6: the browser never gets the table. And a row must say WHICH patch it is.

    `patch_idx` is the join back to B's pixels; `sample_idx` and `patch_xy` are
    the patch's own provenance -- they have been in the `.npz` since fase 2 with
    nobody reading them (V15).
    """
    _build(layout)
    diag = _open(layout)

    page = rows(diag, limit=5)
    assert len(page["rows"]) == 5
    assert page["total"] == len(diag.table)
    second = rows(diag, limit=5, offset=5)["rows"]
    assert [r["patch_idx"] for r in page["rows"]] != [r["patch_idx"] for r in second]

    row = page["rows"][0]
    assert 0 <= row["patch_idx"] < page["total"] + row["patch_idx"]  # it is a real row of B
    assert row["sample_idx"] >= 0
    assert len(row["patch_xy"]) == 2
    assert len(row["score"]) == 4 and len(row["exists"]) == 4


def test_an_unknown_filter_says_so_instead_of_returning_nothing(layout):
    """Silently empty is the worst answer: it reads as "no failures here"."""
    _build(layout)
    diag = _open(layout)

    for kwargs, code in (
        ({"corner": "TOP-LEFT"}, "unknown_corner"),
        ({"outcome": "wrong"}, "unknown_outcome"),
        ({"order": "whatever"}, "unknown_order"),
    ):
        with pytest.raises(NotMeasurable) as exc:
            rows(diag, **kwargs)
        assert exc.value.code == code


# ── over HTTP: the only thing the API adds is the status code ────────────────


def test_the_three_views_answer_over_http(itf_api):
    """Synchronous, not a job (R3): one pass over val is seconds.

    The day this needs a 202 the table stopped being cheap, and V8's free
    threshold went with it -- so the status code is a design assertion here, not a
    detail.
    """
    client, layout = itf_api
    _build(layout)

    curve = client.get("/runs/run-01/diagnostics/pr?split=val&corner=TL")
    assert curve.status_code == 200
    body = curve.json()
    assert len(body["curve"]) == 101
    assert body["positives"] + body["negatives"] > 0
    assert body["best"]["threshold"] is not None

    grid = client.get("/runs/run-01/diagnostics/error-map?split=val")
    assert grid.status_code == 200
    assert grid.json()["job"] == "sequential", "un error en px es magnitud (R3 de ui.md §4.0)"

    gallery = client.get("/runs/run-01/diagnostics/patches?split=val&order=error&limit=3")
    assert gallery.status_code == 200
    assert len(gallery.json()["rows"]) == 3


def test_the_gallery_will_not_ship_the_whole_table(itf_api):
    """R6 as a cap, not as a convention: `limit` is bounded by the route.

    An unbounded `limit` would happily hand over 10⁵ rows, which is exactly what
    R6 exists to stop -- and it would do it politely, with a 200.
    """
    client, layout = itf_api
    _build(layout)
    assert client.get("/runs/run-01/diagnostics/patches?limit=100000").status_code == 422


def test_each_refusal_arrives_as_its_own_status_and_says_how_to_fix_it(itf_api):
    """R4 over HTTP. The status is the part the API adds; the reason is the domain's.

    409 for "the state says no" and 400 for "this request could never work" are
    different answers to a caller: one means fix the data, the other means fix the
    request.
    """
    client, layout = itf_api
    _build(layout)

    unknown_split = client.get("/runs/run-01/diagnostics/pr?split=validation")
    assert unknown_split.status_code == 400
    assert unknown_split.json()["detail"]["code"] == "unknown_split"

    missing = client.get("/runs/no-existe/diagnostics/pr")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "run_not_found"

    (layout.runs / "run-01" / "best.pt").unlink()
    untrained = client.get("/runs/run-01/diagnostics/pr")
    assert untrained.status_code == 409
    detail = untrained.json()["detail"]
    assert detail["code"] == "run_has_no_checkpoint"
    assert detail["hint"], "R4: un error dice por qué Y cómo se arregla"
