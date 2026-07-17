"""E over HTTP — the `/runs` resource (api.md §3, plan-ui.md fase 4).

The contracts themselves live in `test_contracts.py`; this file is the rest of
the resource: the 409 that never overwrites, the incremental polling, the
cooperative stop, and one end-to-end that trains a real (tiny) run and reads its
provenance back off disk.

tests.md §7: everything is built in the test. `data/` and `runs/` are never
touched.
"""

from __future__ import annotations

import json
import time

from conftest import write_tiny_source


def _real_patch_dataset(layout, name: str = "tiny-40", **split) -> str:
    """A B with actual pixels in it. Needed by anything that really trains."""
    from itf.patches import PatchExtractConfig, SplitConfig, extract_dataset

    source = write_tiny_source(layout.datasets / "tiny", num_samples=10)
    extract_dataset(
        PatchExtractConfig(
            source=str(source),
            out=str(layout.patch_datasets / name),
            patch_size=40,
            stride=20,
            split=SplitConfig(**{"train": 0.6, "val": 0.2, "test": 0.2, **split}),
        )
    )
    return name


def _wait_for(client, name: str, states: set[str], timeout: float = 60.0) -> dict:
    """Poll until the run reaches one of `states`. Training is a job (R3)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/runs/{name}").json()
        if body["state"]["state"] in states:
            return body
        time.sleep(0.05)
    raise AssertionError(f"el run '{name}' no llegó a {states}; se quedó en {body['state']}")


# ── never overwrite ───────────────────────────────────────────────────────────


def test_post_runs_refuses_to_overwrite_an_existing_run(itf_api):
    """409, and the run on disk must be untouched.

    The trap, measured on the old code: `mkdir(exist_ok=True)` plus truncating
    `metrics.jsonl` machaca a finished run without a word. **And a sweep that
    auto-generates names is exactly who steps on it** -- which is why this is a
    409 and not a warning: by the time a human noticed, the results would be gone.
    """
    client, layout = itf_api
    layout.write_patch_dataset("tiny-40")
    layout.write_network("cnn-a")
    layout.write_recipe("adam")
    layout.write_run("ya-existe", patch_dataset="tiny-40")
    before = (layout.runs / "ya-existe" / "config.json").read_text(encoding="utf-8")

    response = client.post(
        "/runs",
        json={
            "name": "ya-existe",
            "patch_dataset": "tiny-40",
            "network": "cnn-a",
            "recipe": "adam",
        },
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "run_exists"
    assert detail["hint"]
    assert (layout.runs / "ya-existe" / "config.json").read_text(encoding="utf-8") == before


def test_post_runs_refuses_a_dataset_with_no_val_split(itf_api):
    """protocolo.md §1.3, at the door where the damage starts.

    Building such a B warns (fase 2, `manifest.warnings`); starting a run on it
    refuses. Without val, `best.pt` gets chosen by the TRAIN loss -- i.e. the
    most overfitted checkpoint wins, silently.
    """
    client, layout = itf_api
    layout.write_patch_dataset("sin-val", patches_per_split={"train": 10, "val": 0, "test": 0})
    layout.write_network("cnn-a")
    layout.write_recipe("adam")

    response = client.post(
        "/runs",
        json={"name": "no-mide", "patch_dataset": "sin-val", "network": "cnn-a", "recipe": "adam"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "no_validation_split"
    assert not (layout.runs / "no-mide").exists()


def test_post_runs_says_which_piece_is_missing(itf_api):
    """R7: names, not values. So a name that does not exist is a 404 that says which."""
    client, layout = itf_api
    layout.write_patch_dataset("tiny-40")
    layout.write_network("cnn-a")
    layout.write_recipe("adam")
    body = {"name": "r", "patch_dataset": "tiny-40", "network": "cnn-a", "recipe": "adam"}

    assert client.post("/runs", json={**body, "network": "no-existe"}).json()["detail"]["code"] == (
        "network_not_found"
    )
    assert client.post("/runs", json={**body, "recipe": "no-existe"}).json()["detail"]["code"] == (
        "recipe_not_found"
    )
    assert client.post("/runs", json={**body, "patch_dataset": "no"}).json()["detail"]["code"] == (
        "patch_dataset_not_found"
    )


# ── polling, R5 ───────────────────────────────────────────────────────────────


def test_get_run_does_not_carry_the_metrics(itf_api):
    """R5's corollary, and it is a real cost, not a purity rule.

    The UI polls this in a loop. The old `GET /runs/{name}` returned every metric
    on every call, so watching a run got more expensive with every epoch it ran.
    """
    client, layout = itf_api
    layout.write_patch_dataset("tiny-40")
    layout.write_run("r", patch_dataset="tiny-40")

    body = client.get("/runs/r").json()

    assert "metrics" not in body
    assert body["provenance"]["patch_dataset"]["name"] == "tiny-40"


def test_a_run_without_provenance_is_reported_as_such(itf_api):
    """It is said out loud, not read around (formatos.md §4.2).

    D3 killed the degrading reader: every run born from fase 4 on has the block,
    so one without it cannot say which C or D it came from and **nothing can
    recover that** -- it is not comparable with anything. Listing it as a run
    like any other is what would be dishonest. (Fase 3's own verification left
    exactly one of these on disk, which is how this case stopped being
    hypothetical.)
    """
    client, layout = itf_api
    old = layout.runs / "de-la-fase-3"
    old.mkdir(parents=True)
    (old / "config.json").write_text(
        json.dumps({"format_version": 1, "network": {}, "recipe": {}}), encoding="utf-8"
    )
    (old / "status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")

    row = next(r for r in client.get("/runs").json()["runs"] if r["name"] == "de-la-fase-3")

    assert row["provenance"] is None
    assert "procedencia" in row["error"]
    # And the list still works: one broken run must not take the rest down.
    assert row["state"] == "done"


def test_one_broken_run_does_not_take_the_list_down(itf_api):
    """A 500 here is not "one run is broken": it is *the Runs screen shows nothing*.

    Every file of a run is written while the list is being read -- `write_text`
    is not atomic, and a delete can land between `names()` and the reads -- so a
    torn file is normal, not corruption. It must degrade to one bad row.
    """
    client, layout = itf_api
    layout.write_patch_dataset("tiny-40")
    layout.write_run("sano", patch_dataset="tiny-40")
    (layout.runs / "sano" / "status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")

    roto = layout.runs / "roto"
    roto.mkdir(parents=True)
    (roto / "status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
    # Caught mid-write: valid JSON up to here, and nothing after it.
    (roto / "summary.json").write_text('{"run": "roto", "epochs_ru', encoding="utf-8")
    (roto / "config.json").write_text('{"format_version": 1, "prov', encoding="utf-8")

    response = client.get("/runs")

    assert response.status_code == 200, "un run ilegible no puede tumbar la lista entera"
    rows = {r["name"]: r for r in response.json()["runs"]}
    assert rows["roto"]["error"]
    # And the healthy one is still there, complete.
    assert rows["sano"]["provenance"]["patch_dataset"]["name"] == "tiny-40"


def test_metrics_are_incremental(itf_api):
    """`?since=N` -> `{records, next}`. Never the whole history again."""
    client, layout = itf_api
    layout.write_patch_dataset("tiny-40")
    run = layout.write_run("r", patch_dataset="tiny-40")
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps({"epoch": i, "train_loss": 1.0 / i, "seconds": 2.0}) + "\n" for i in (1, 2, 3)),
        encoding="utf-8",
    )

    first = client.get("/runs/r/metrics").json()
    assert [r["epoch"] for r in first["records"]] == [1, 2, 3]
    assert first["next"] == 3

    # The second poll, with what the first handed back: nothing new yet.
    assert client.get("/runs/r/metrics?since=3").json() == {"records": [], "next": 3}

    # And after one more epoch lands, only that epoch travels.
    with (run / "metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"epoch": 4, "train_loss": 0.2, "seconds": 2.0}) + "\n")
    fresh = client.get("/runs/r/metrics?since=3").json()
    assert [r["epoch"] for r in fresh["records"]] == [4]
    assert fresh["next"] == 4


def test_a_half_written_metrics_line_does_not_break_the_poll(itf_api):
    """A live run is being read while it is written.

    `metrics.jsonl` is append-only and the UI polls it mid-epoch, so a torn last
    line is normal, not corruption. Parsing it would raise on a perfectly healthy
    run -- and it would do it only sometimes, which is the worst kind of bug.
    """
    client, layout = itf_api
    layout.write_patch_dataset("tiny-40")
    run = layout.write_run("r", patch_dataset="tiny-40")
    (run / "metrics.jsonl").write_text(
        json.dumps({"epoch": 1, "train_loss": 0.5, "seconds": 2.0}) + "\n" + '{"epoch": 2, "train_l',
        encoding="utf-8",
    )

    body = client.get("/runs/r/metrics").json()

    assert [r["epoch"] for r in body["records"]] == [1]
    assert body["next"] == 1


# ── renaming and deleting ─────────────────────────────────────────────────────


def test_rename_and_delete(itf_api):
    client, layout = itf_api
    layout.write_patch_dataset("tiny-40")
    layout.write_run("viejo", patch_dataset="tiny-40")
    # A run that has finished: `write_run` leaves no status.json, and a run with
    # no status reads as `error` -- which is not live, so it can be moved.
    (layout.runs / "viejo" / "status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")

    assert client.patch("/runs/viejo", json={"name": "nuevo"}).status_code == 200
    assert (layout.runs / "nuevo").exists() and not (layout.runs / "viejo").exists()

    assert client.delete("/runs/nuevo").status_code == 204
    assert not (layout.runs / "nuevo").exists()
    assert client.delete("/runs/nuevo").status_code == 404


def test_a_running_run_cannot_be_renamed_or_deleted(itf_api):
    """Moving the directory out from under the loop leaves it writing into a path
    that no longer exists."""
    client, layout = itf_api
    layout.write_patch_dataset("tiny-40")
    layout.write_run("corriendo", patch_dataset="tiny-40")
    (layout.runs / "corriendo" / "status.json").write_text(
        json.dumps({"state": "running", "epoch": 3}), encoding="utf-8"
    )

    assert client.patch("/runs/corriendo", json={"name": "otro"}).status_code == 409
    assert client.delete("/runs/corriendo").status_code == 409
    assert (layout.runs / "corriendo").exists()


# ── end to end ────────────────────────────────────────────────────────────────


def test_a_run_trained_through_the_api_carries_its_whole_provenance(itf_api):
    """The phase's verification, as a test: dato → red → receta → run.

    What it really checks is contract ③ on disk after a REAL training: the name
    of C, the name of D and the fingerprint of B, written by the API and not by
    a fixture. `write_run` fakes that file everywhere else in the suite, so
    without this nothing proves the writer and the readers agree.
    """
    client, layout = itf_api
    _real_patch_dataset(layout, "tiny-40")
    layout.write_network("cnn-a", input_size=40)
    layout.write_recipe("rapida", epochs=1, batch_size=16)
    fingerprint = json.loads(
        (layout.patch_datasets / "tiny-40" / "manifest.json").read_text(encoding="utf-8")
    )["fingerprint"]

    accepted = client.post(
        "/runs",
        json={
            "name": "e2e",
            "patch_dataset": "tiny-40",
            "network": "cnn-a",
            "recipe": "rapida",
            "device": "cpu",
        },
    )
    assert accepted.status_code == 202

    body = _wait_for(client, "e2e", {"done", "error"})
    assert body["state"]["state"] == "done", body["state"]

    prov = json.loads((layout.runs / "e2e" / "config.json").read_text(encoding="utf-8"))["provenance"]
    assert prov["patch_dataset"] == {"name": "tiny-40", "fingerprint": fingerprint}
    assert prov["network"]["name"] == "cnn-a"
    assert prov["recipe"]["name"] == "rapida"
    assert prov["recipe"]["value"]["epochs"] == 1
    assert prov["sweep"] is None
    assert prov["git_commit"] and prov["environment"]["torch"]

    # X is recorded, and OUTSIDE the recipe (contract ⑩).
    config = json.loads((layout.runs / "e2e" / "config.json").read_text(encoding="utf-8"))
    assert config["execution"] == {"device": "cpu", "num_workers": 0}
    assert "device" not in config["recipe"]

    # And the run is now visible as what it came from, by name -- the question a
    # sweep asks constantly and that was impossible before.
    row = next(r for r in client.get("/runs").json()["runs"] if r["name"] == "e2e")
    assert row["provenance"]["network"]["name"] == "cnn-a"
    assert row["seconds_per_epoch"] > 0
    assert client.get("/patch-datasets/tiny-40").json()["used_by"] == ["e2e"]


def test_polling_a_live_run_never_sees_it_as_corrupt(itf_api):
    """A healthy run must never read as broken just because it is being written.

    The bug this pins, and it was found by a flaky test rather than by reasoning:
    `Path.write_text` truncates and then writes, so a `GET` landing in that
    window read a torn `config.json` and answered **404 «el run no tiene un
    config.json legible»** — about a run that was perfectly fine. `status.json`
    had it worse: rewritten every epoch, so a poll could show a running run as
    `error`. The fix is atomic writes (temp + `os.replace`), plus not rewriting a
    `config.json` that the reservation already wrote.

    Polling hard while a real run trains is the only way to catch it: it is a
    race, so a single well-timed GET proves nothing.
    """
    client, layout = itf_api
    _real_patch_dataset(layout, "tiny-40")
    layout.write_network("cnn-a", input_size=40)
    layout.write_recipe("varias", epochs=6, batch_size=16)

    assert client.post(
        "/runs",
        json={"name": "vivo", "patch_dataset": "tiny-40", "network": "cnn-a", "recipe": "varias"},
    ).status_code == 202

    seen = set()
    deadline = time.time() + 60
    while time.time() < deadline:
        response = client.get("/runs/vivo")
        assert response.status_code == 200, f"un run sano nunca es 404: {response.json()}"
        body = response.json()
        assert body["provenance"]["network"]["name"] == "cnn-a"
        state = body["state"]["state"]
        seen.add(state)
        # The list is polled just as hard, and it must not 500 either.
        assert client.get("/runs").status_code == 200
        if state in {"done", "error", "cancelled"}:
            break

    assert "error" not in seen, f"un run sano nunca pasa por 'error': vio {seen}"
    assert "done" in seen


def test_stopping_a_run_closes_it_as_cancelled(itf_api):
    """Cooperative: it cuts at the end of an epoch, and says so.

    `cancelled`, not `done`: a run stopped at epoch 2 of 20 has real weights, so
    calling it done would let it into a comparison as if it had finished
    (protocolo.md §7).
    """
    client, layout = itf_api
    _real_patch_dataset(layout, "tiny-40")
    layout.write_network("cnn-a", input_size=40)
    layout.write_recipe("larga", epochs=50, batch_size=16)

    assert client.post(
        "/runs",
        json={"name": "parada", "patch_dataset": "tiny-40", "network": "cnn-a", "recipe": "larga"},
    ).status_code == 202

    # Ask as soon as it is actually training, so the stop lands mid-run.
    _wait_for(client, "parada", {"running", "done"})
    assert client.post("/runs/parada/stop").status_code == 202

    body = _wait_for(client, "parada", {"cancelled", "done", "error"})
    assert body["state"]["state"] == "cancelled"
    assert body["summary"]["cancelled"] is True
    # It kept what it had done: nothing was killed mid-batch.
    assert body["summary"]["epochs_run"] < 50
    assert "best.pt" in body["checkpoints"]

    # And a run that is over cannot be stopped again.
    assert client.post("/runs/parada/stop").status_code == 409
