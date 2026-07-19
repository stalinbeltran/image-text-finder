"""F over HTTP — `/kernels`, `/feature-maps`, `/predict` (plan-ui.md fase 6).

The domain logic is tested in `test_inference.py`; this file is the HTTP half,
which is the only half this layer owns (api.md §0). So what it asserts is what
the API adds and nothing else: **the status code of each refusal**, and that the
refusal arrives with a `code` the UI can switch on (R4).

The distinction the codes encode, and it is the one that is easy to get wrong:
**400 is a request that can never work** (a 60-px patch into a 40-px network --
contract ① arriving at F), **409 is a request that is fine and a state that says
no** (the run never trained). Collapse them and the UI cannot tell "fix your
request" from "fix your data", which is the whole reason R4 exists.
"""

from __future__ import annotations

import json

import numpy as np

from conftest import write_tiny_source

NETWORK = {
    "input_size": 40,
    "in_channels": 1,
    "border_features": False,
    "backbone": [{"filters": 4, "kernel": 3, "stride": 1, "pool": 2}],
    "head": {"hidden": [8]},
}


def _trained(layout, *, run: str = "run-01", b: str = "tiny-40") -> str:
    """A real B and a real E, built through the domain rather than the API.

    Training through `POST /runs` would work and would make every test here wait
    on a job. What is under test is the read side.
    """
    from itf.patches import PatchExtractConfig, SplitConfig, extract_dataset
    from itf.training.loop import RunSpec, train
    from itf.training.recipe import Recipe

    source = write_tiny_source(layout.datasets / "tiny", num_samples=8)
    data = layout.patch_datasets / b
    extract_dataset(
        PatchExtractConfig(
            source=str(source),
            out=str(data),
            patch_size=40,
            stride=20,
            split=SplitConfig(0.6, 0.2, 0.2),
        )
    )
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    train(
        RunSpec(
            data=str(data),
            out=str(layout.runs / run),
            network=NETWORK,
            recipe=Recipe(epochs=1, batch_size=32),
            provenance={
                "patch_dataset": {"name": b, "fingerprint": manifest["fingerprint"]},
                "network": {"name": "cnn-a", "value": NETWORK},
                "recipe": {"name": "rapida", "value": {}},
                "sweep": None,
                "git_commit": "0" * 40,
                "environment": {"python": "3.12", "torch": "2.13", "platform": "win32"},
            },
        )
    )
    return run


# ── V1: GET /runs/{name}/kernels ──────────────────────────────────────────────


def test_kernels_of_a_trained_run(itf_api):
    client, layout = itf_api
    _trained(layout)

    body = client.get("/runs/run-01/kernels").json()

    assert body["layer"] == 1
    assert body["job"] == "diverging"  # R2: weights are signed
    assert body["count"] == 4
    assert np.shape(body["maps"][0]["matrix"]) == (3, 3)
    # Each map carries its own stats: the normalisation is per map (ui.md §5).
    assert {"min", "max", "mean"} <= body["maps"][0].keys()


def test_kernels_of_a_run_that_never_trained_are_a_409(itf_api):
    """409, not 404: the run exists and the request is fine. The state says no.

    404 would send you looking for a run that is right there.
    """
    client, layout = itf_api
    layout.write_run("queued-01", patch_dataset="tiny-40")

    response = client.get("/runs/queued-01/kernels")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "run_has_no_checkpoint"
    assert "hint" in detail


def test_kernels_of_a_run_that_does_not_exist_are_a_404(itf_api):
    client, _ = itf_api
    assert client.get("/runs/nope/kernels").status_code == 404


# ── V2: POST /runs/{name}/feature-maps ────────────────────────────────────────


def test_feature_maps_by_patch_index(itf_api):
    """The default path: a patch OF A B, by index.

    Its border flags come out of the dataset, so they are the real ones -- which
    is exactly why this path exists rather than making the client carry them.
    """
    client, layout = itf_api
    _trained(layout)

    body = client.post(
        "/runs/run-01/feature-maps", json={"patch_dataset": "tiny-40", "index": 0}
    ).json()

    assert [layer["layer"] for layer in body["layers"]] == [1]
    assert body["layers"][0]["job"] == "sequential"  # relu: magnitude (R3)
    assert (body["layers"][0]["height"], body["layers"][0]["width"]) == (20, 20)
    assert [c["corner"] for c in body["prediction"]["corners"]] == ["TL", "TR", "BR", "BL"]


def test_feature_maps_of_inline_pixels(itf_api):
    """The other path: a patch stored nowhere (V5 crops one out of an image)."""
    client, layout = itf_api
    _trained(layout)

    body = client.post(
        "/runs/run-01/feature-maps",
        json={"patch": np.full((40, 40), 128).tolist()},
    ).json()

    assert len(body["layers"]) == 1
    assert body["input_size"] == 40


def test_feature_maps_of_the_wrong_size_are_a_400(itf_api):
    """Contract ① arriving at F, and it answers like its `POST /runs` twin.

    Inside the head this is `mat1 and mat2 shapes cannot be multiplied`: linear
    algebra, for a dataset problem. 400 because no state makes it work.
    """
    client, layout = itf_api
    _trained(layout)

    response = client.post(
        "/runs/run-01/feature-maps", json={"patch": np.zeros((60, 60)).tolist()}
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "patch_size_mismatch"
    # R4: the two numbers, so the message names the real problem.
    assert "40x40" in detail["message"] and "60x60" in detail["message"]


def test_feature_maps_without_a_patch_are_a_400(itf_api):
    client, layout = itf_api
    _trained(layout)

    response = client.post("/runs/run-01/feature-maps", json={})

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "patch_required"


def test_feature_maps_with_both_inputs_are_a_400(itf_api):
    """Both at once is a question, not a convenience to resolve.

    Picking one silently is how a view ends up describing a patch the reader did
    not choose -- and looking entirely correct while it does.
    """
    client, layout = itf_api
    _trained(layout)

    response = client.post(
        "/runs/run-01/feature-maps",
        json={"patch_dataset": "tiny-40", "index": 0, "patch": np.zeros((40, 40)).tolist()},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "patch_ambiguous"


def test_feature_maps_of_a_patch_out_of_range_are_a_404(itf_api):
    client, layout = itf_api
    _trained(layout)

    response = client.post(
        "/runs/run-01/feature-maps", json={"patch_dataset": "tiny-40", "index": 99999}
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "patch_not_found"


# ── V11: POST /runs/{name}/predict ────────────────────────────────────────────


def test_predict_returns_the_three_stages(itf_api):
    """**All three**, which is the whole point (V11).

    Without `raw`, "the paragraph came out wrong" is not diagnosable: you cannot
    tell a corner that was never detected from one NMS ate.
    """
    client, layout = itf_api
    _trained(layout)

    body = client.post(
        "/runs/run-01/predict", json={"source": "tiny", "index": 0, "threshold": 0.0}
    ).json()

    assert {"raw", "corners", "paragraphs", "image_size", "knobs"} <= body.keys()
    assert len(body["raw"]) >= len(body["corners"])
    assert body["image_size"] == [120, 100]
    # Echoed back: the sliders are live, so answers arrive out of order and a
    # slow one must not overwrite a newer one.
    assert body["knobs"]["threshold"] == 0.0
    assert body["knobs"]["stride"] == 20  # defaulted to half the patch


def test_predict_knobs_are_free_and_post_hoc(itf_api):
    """Raising `threshold` never adds detections, and never runs training.

    The knobs are F (organizacion.md §1-D): sweeping them costs a forward pass.
    That is what keeps `threshold` out of D, where it would multiply CPU hours.
    """
    client, layout = itf_api
    _trained(layout)

    def detections(threshold: float) -> int:
        body = client.post(
            "/runs/run-01/predict",
            json={"source": "tiny", "index": 0, "threshold": threshold},
        ).json()
        return len(body["raw"])

    assert detections(0.99) <= detections(0.0)


def test_predict_on_a_run_with_no_checkpoint_is_a_409(itf_api):
    client, layout = itf_api
    layout.write_run("queued-01", patch_dataset="tiny-40")

    response = client.post("/runs/queued-01/predict", json={"source": "tiny", "index": 0})

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "run_has_no_checkpoint"


def test_predict_on_a_source_that_does_not_exist_is_a_404(itf_api):
    client, layout = itf_api
    _trained(layout)

    response = client.post("/runs/run-01/predict", json={"source": "nope", "index": 0})

    assert response.status_code == 404


# ── V9: GET /runs/{name}/diagnostics/coactivation ─────────────────────────────


def test_coactivation_is_a_4x4_with_its_baseline(itf_api):
    """The matrix alone cannot tell confusion from co-occurrence.

    `matrix[TL][TR]` high means either "the TR head is confused by a TL" or
    "these patches really do contain a TR as well" -- opposite problems. So
    `truth_rate` ships with it as the control, and `counts` says how many patches
    each row rests on (the V7 lesson: a row of 3 paints like a row of 300).
    """
    client, layout = itf_api
    _trained(layout)

    body = client.get("/runs/run-01/diagnostics/coactivation?split=val").json()

    assert body["corner_order"] == ["TL", "TR", "BR", "BL"]
    assert np.shape(body["matrix"]) == (4, 4)
    assert np.shape(body["truth_rate"]) == (4, 4)
    assert len(body["counts"]) == 4
    assert body["job"] == "sequential"  # a rate is a magnitude


def test_coactivation_threshold_is_a_query_not_a_cache_key(itf_api):
    """The free sweep, again: re-deciding "fired" costs a comparison.

    If `threshold` entered the table's key, every point of a sweep would re-run
    the model -- which is exactly the cost V8 exists to avoid.
    """
    client, layout = itf_api
    _trained(layout)

    strict = client.get("/runs/run-01/diagnostics/coactivation?threshold=0.99").json()
    loose = client.get("/runs/run-01/diagnostics/coactivation?threshold=0.0").json()

    assert strict["threshold"] == 0.99
    # At threshold 0 every head fires for every patch; at 0.99 almost none can.
    for row_loose, row_strict in zip(loose["matrix"], strict["matrix"]):
        if row_loose[0] is None:
            continue  # no patch has that corner: None, not 0 (formatos.md §2)
        assert all(s <= l for s, l in zip(row_strict, row_loose))


# ── the probes (fase 8): V4, V10, V5 over HTTP ────────────────────────────────


def _border_network(layout, *, run: str, b: str) -> str:
    """A B and an E whose C uses `border_features` — what V10 needs to answer."""
    from itf.patches import PatchExtractConfig, SplitConfig, extract_dataset
    from itf.training.loop import RunSpec, train
    from itf.training.recipe import Recipe

    network = {**NETWORK, "border_features": True}
    source = write_tiny_source(layout.datasets / "tiny", num_samples=8)
    data = layout.patch_datasets / b
    extract_dataset(
        PatchExtractConfig(
            source=str(source), out=str(data), patch_size=40, stride=20,
            split=SplitConfig(0.6, 0.2, 0.2),
        )
    )
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    train(
        RunSpec(
            data=str(data), out=str(layout.runs / run), network=network,
            recipe=Recipe(epochs=1, batch_size=32),
            provenance={
                "patch_dataset": {"name": b, "fingerprint": manifest["fingerprint"]},
                "network": {"name": "cnn-b", "value": network},
                "recipe": {"name": "rapida", "value": {}},
                "sweep": None, "git_commit": "0" * 40,
                "environment": {"python": "3.12", "torch": "2.13", "platform": "win32"},
            },
        )
    )
    return run


def test_occlusion_is_four_sequential_maps(itf_api):
    """V4 — same input shape as V2 (a patch by index), a heat map per corner."""
    client, layout = itf_api
    _trained(layout)

    body = client.post(
        "/runs/run-01/occlusion", json={"patch_dataset": "tiny-40", "index": 0}
    ).json()

    assert body["job"] == "sequential"  # p(exists|occluded) is a probability
    assert len(body["maps"]) == 4
    assert len(body["baseline"]) == 4
    bins = body["bins"]
    assert np.shape(body["maps"][0]["matrix"]) == (bins, bins)


def test_occlusion_of_the_wrong_size_is_a_400(itf_api):
    """Contract ① reaching F, like the feature maps."""
    client, layout = itf_api
    _trained(layout)

    response = client.post("/runs/run-01/occlusion", json={"patch": np.zeros((60, 60)).tolist()})

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "patch_size_mismatch"


def test_deconvolution_is_diverging_maps_in_patch_space(itf_api):
    """V16 — same input shape as V2 (a patch by index), one map per filter."""
    client, layout = itf_api
    _trained(layout)

    body = client.post(
        "/runs/run-01/deconvolution", json={"patch_dataset": "tiny-40", "index": 0}
    ).json()

    n = body["input_size"]
    assert body["layers"], "una red con backbone tiene capas que retropropagar"
    for layer in body["layers"]:
        assert layer["job"] == "diverging"  # a gradient is signed (R2)
        # Patch space, not activation space: every layer overlays the same patch.
        assert np.shape(layer["maps"][0]["matrix"]) == (n, n)


def test_deconvolution_answers_a_border_network_without_flags(itf_api):
    """Over HTTP: the body carries `border`, the view does not use it.

    `_patch_from_body` is shared with V2/V4/V10, so the flags arrive either way.
    What this pins is that the endpoint does not *require* them: the backbone is
    all this probe touches, and `border_features` only reaches the head.
    """
    client, layout = itf_api
    _border_network(layout, run="run-b", b="tiny-b")

    response = client.post(
        "/runs/run-b/deconvolution", json={"patch_dataset": "tiny-b", "index": 0}
    )

    assert response.status_code == 200
    assert response.json()["layers"]


def test_deconvolution_of_the_wrong_size_is_a_400(itf_api):
    """Contract ① reaching F, like the feature maps and the occlusion."""
    client, layout = itf_api
    _trained(layout)

    response = client.post(
        "/runs/run-01/deconvolution", json={"patch": np.zeros((60, 60)).tolist()}
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "patch_size_mismatch"


def test_border_test_over_a_border_network(itf_api):
    """V10 — one flip per flag, five forwards, over a network that uses the flags."""
    client, layout = itf_api
    _border_network(layout, run="run-b", b="tiny-b")

    body = client.post(
        "/runs/run-b/border-test", json={"patch_dataset": "tiny-b", "index": 0}
    ).json()

    assert len(body["baseline"]) == 4
    assert [f["border"] for f in body["flips"]] == ["top", "right", "bottom", "left"]


def test_border_test_refuses_a_non_border_network_with_a_409(itf_api):
    """The refusal, over HTTP: 409, because the request is fine and the C says no.

    A network that ignores the flags cannot answer this probe honestly, exactly as
    `kernels` cannot project a 32-channel filter. 400 would say "fix your request",
    when the fix is "choose a network with border_features".
    """
    client, layout = itf_api
    _trained(layout)  # NETWORK has border_features=False

    response = client.post(
        "/runs/run-01/border-test", json={"patch_dataset": "tiny-40", "index": 0}
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "border_not_used"
    assert "hint" in detail


def test_window_scrubber_returns_the_four_heads_and_stability(itf_api):
    """V5 — one off-grid crop → 4 heads live + the 1-px stability the stride needs."""
    client, layout = itf_api
    _trained(layout)

    body = client.post(
        "/runs/run-01/window", json={"source": "tiny", "index": 0, "x0": 0, "y0": 0}
    ).json()

    assert [c["corner"] for c in body["corners"]] == ["TL", "TR", "BR", "BL"]
    assert body["image_size"] == [120, 100]
    # Flush top-left: the border flags come from `window_at` (contract ⑤).
    assert body["border"][0] == 1 and body["border"][3] == 1
    assert 0.0 <= body["stability"]["max"] <= 1.0
    assert body["source"] == "tiny" and body["index"] == 0


def test_window_on_a_run_with_no_checkpoint_is_a_409(itf_api):
    client, layout = itf_api
    layout.write_run("queued-01", patch_dataset="tiny-40")

    response = client.post("/runs/queued-01/window", json={"source": "tiny", "index": 0, "x0": 0, "y0": 0})

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "run_has_no_checkpoint"
