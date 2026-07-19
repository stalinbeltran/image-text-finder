"""F — the pipeline, the kernels and the feature maps (fase 6).

Contract ⑤ lives in `test_contracts.py`, where it belongs. What is here is the
rest of what fase 6 built, and the same rule picks it: **the seams and the
refusals, not the functions.**

The seam that carries this file is `test_raw_is_what_nms_ate`. `raw`, `corners`
and `paragraphs` are three stages of one pipeline, and the reason all three are
returned is that a lost paragraph is born in exactly one of them (V11). If they
ever stop being the same pipeline -- if `raw` were computed with one threshold and
`corners` with another -- V11 would still draw three tidy layers, and the layer
you would trust to tell you where the failure was would be the one lying.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from itf.inference import (
    Detection,
    ModelCache,
    NotInspectable,
    border_test,
    deconvolution,
    detect_corners,
    feature_maps,
    kernels,
    load_model,
    merge,
    occlusion,
    predict_image,
    reconstruct_boxes,
    window_prediction,
)
from itf.geometry import CORNER_NAMES, window_at, windows
from itf.models import build_model
from itf.patches import PatchExtractConfig, SplitConfig, extract_dataset
from itf.training.loop import RunSpec, train
from itf.training.recipe import Recipe

from conftest import write_tiny_source

NETWORK = {
    "input_size": 40,
    "in_channels": 1,
    "border_features": False,
    "backbone": [
        {"filters": 4, "kernel": 3, "stride": 1, "pool": 2, "activation": "relu"},
        {"filters": 6, "kernel": 3, "stride": 1, "pool": 2, "activation": "relu"},
    ],
    "head": {"hidden": [8]},
}


def _trained_run(layout, *, name: str = "run-01", network: dict | None = None):
    """A real B and a real E over it: kernels need weights, maps need a forward."""
    network = network or NETWORK
    source = write_tiny_source(layout.datasets / "tiny", num_samples=8)
    data = layout.patch_datasets / "tiny-40"
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
    run = layout.runs / name
    train(
        RunSpec(
            data=str(data),
            out=str(run),
            network=network,
            recipe=Recipe(epochs=1, batch_size=32),
            provenance={
                "patch_dataset": {"name": "tiny-40", "fingerprint": manifest["fingerprint"]},
                "network": {"name": "cnn-a", "value": network},
                "recipe": {"name": "rapida", "value": {}},
                "sweep": None,
                "git_commit": "0" * 40,
                "environment": {"python": "3.12", "torch": "2.13", "platform": "win32"},
            },
        )
    )
    return run, data


def _image(layout, index: int = 0) -> np.ndarray:
    with Image.open(layout.datasets / "tiny" / "images" / f"{index:06d}.png") as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


# ── the seam: the three stages are one pipeline ───────────────────────────────


def test_raw_is_what_nms_ate(layout):
    """`raw` ⊇ `corners`, and both come from one pass. **The point of V11.**

    Three layers over an image are a diagnosis only if they are stages of the
    same thing: `raw` is what the window found, `corners` is what survived NMS,
    `paragraphs` is what paired. If `raw` were its own recomputation, V11 would
    read as "the detector found nothing here" when NMS had eaten it, or the other
    way round -- and it would look exactly as convincing either way.
    """
    run, _ = _trained_run(layout)
    model = load_model(run / "best.pt")
    image = _image(layout)

    # `nms_radius` 25 > `stride` 20 on purpose. At the DEFAULT radius (stride/2 =
    # 10) this collapses nothing, and that is NMS being right rather than broken:
    # a barely-trained net predicts the corner at about the same place *within*
    # each patch, so neighbouring windows put it 20 px apart globally -- further
    # than 10, so they are not duplicates of one corner, they are 20 windows
    # disagreeing. Merging depends on the model localising, so a test that
    # asserted it at the default radius would be asserting a research result
    # (tests.md §6) and would fail for an honest reason.
    out = predict_image(model, image, threshold=0.0, stride=20, nms_radius=25.0)

    # threshold=0 => every window × every corner. `raw` IS the shared window,
    # which is contract ⑤ observed from F's side rather than asserted.
    grid = windows(image.shape[1], image.shape[0], 40, 20)
    assert len(out["raw"]) == len(grid) * len(CORNER_NAMES)

    assert len(out["corners"]) < len(out["raw"]), "NMS tiene que colapsar duplicados"
    # NMS never invents a corner type that was not detected.
    assert {d["corner"] for d in out["corners"]} <= {d["corner"] for d in out["raw"]}
    # Every paragraph is built from SURVIVING corners, never from raw ones.
    assert len(out["paragraphs"]) <= sum(1 for d in out["corners"] if d["corner"] == "TL")
    assert out["image_size"] == [image.shape[1], image.shape[0]]

    # The stages compose: `corners` is `merge(raw)`, not a second pass with its
    # own threshold. This is the seam -- if it broke, V11 would still draw three
    # tidy layers and the one you would trust would be the one lying.
    expected = merge(detect_corners(model, image, stride=20, threshold=0.0), radius=25.0)
    assert len(out["corners"]) == len(expected)


def test_the_knobs_come_back_with_the_answer(layout):
    """The sliders repaint live, so answers arrive out of order.

    Without knowing which stride produced a payload, a slow response overwrites a
    newer one and the view shows a picture of knobs nobody is holding.
    """
    run, _ = _trained_run(layout)
    model = load_model(run / "best.pt")
    image = np.zeros((100, 120), dtype=np.uint8)

    out = predict_image(model, image, threshold=0.7, stride=10, min_size=9.0)
    assert out["knobs"] == {
        "threshold": 0.7,
        "stride": 10,
        # Defaulted from the stride, not fixed: change the stride and what counts
        # as a duplicate has to change with it.
        "nms_radius": 5.0,
        "min_size": 9.0,
    }


def test_threshold_only_removes_detections(layout):
    """A knob of F, applied post-hoc: raising it never adds a detection.

    The invariant, not a number -- what makes `threshold` free is that it filters
    a decision the model already made (organizacion.md §1-D).
    """
    run, _ = _trained_run(layout)
    model = load_model(run / "best.pt")
    image = _image(layout, 1)

    loose = detect_corners(model, image, threshold=0.0)
    tight = detect_corners(model, image, threshold=0.9)
    assert len(tight) <= len(loose)
    assert all(d.score >= 0.9 for d in tight)


# ── NMS and reconstruction: the recovered algorithms ──────────────────────────


def test_merge_collapses_a_cluster_and_keeps_the_best_score():
    """Four windows see one corner. It is one corner, at their weighted mean."""
    cluster = [
        Detection("TL", 100.0, 100.0, 0.9),
        Detection("TL", 104.0, 100.0, 0.5),
        Detection("TL", 100.0, 103.0, 0.5),
    ]
    merged = merge(cluster, radius=10.0)
    tl = [d for d in merged if d.corner == "TL"]
    assert len(tl) == 1
    # The score is the MAX: the cluster is one detection with the confidence of
    # its best evidence. Averaging would punish a corner for also having been
    # seen from a window that barely caught it.
    assert tl[0].score == 0.9
    # The position is the score-weighted mean, so the loudest pulls hardest but
    # does not win outright.
    assert 100.0 < tl[0].x < 104.0


def test_merge_does_not_mix_corner_types():
    """A TL and a TR in the same place are two corners, not a duplicate.

    The heads are independent binaries (V9's whole point): co-located corners of
    different types are normal, and merging across types would silently delete
    one of them.
    """
    dets = [Detection("TL", 50.0, 50.0, 0.9), Detection("TR", 50.0, 50.0, 0.8)]
    merged = merge(dets, radius=20.0)
    assert sorted(d.corner for d in merged) == ["TL", "TR"]


def test_reconstruct_pairs_each_br_once():
    """Greedy TL→BR: a BR belongs to one box, or two paragraphs share a corner."""
    dets = [
        Detection("TL", 10.0, 10.0, 0.9),
        Detection("TL", 10.0, 100.0, 0.9),
        Detection("BR", 90.0, 60.0, 0.9),
    ]
    boxes = reconstruct_boxes(dets)
    assert len(boxes) == 1, "un BR no puede cerrar dos párrafos"
    assert boxes[0]["box"] == [10.0, 10.0, 80.0, 50.0]


def test_reconstruct_refuses_a_br_above_its_tl():
    """A BR up-and-left of a TL does not close a box. It closes nothing."""
    dets = [Detection("TL", 90.0, 90.0, 0.9), Detection("BR", 10.0, 10.0, 0.9)]
    assert reconstruct_boxes(dets) == []


# ── V1: the kernels ───────────────────────────────────────────────────────────


def test_kernels_are_layer_1_only_and_say_so(layout):
    """D13, closed 2026-07-17: **nothing** for the deep layers.

    Not an omission to be filled in later: with 32 input channels a filter is 32
    matrices, and `weight[:, 0]` is one of them picked arbitrarily. It looks like
    a view and tells you nothing. The payload says the layer count and why it
    stops -- an absence with no explanation reads as a bug and invites someone to
    "fix" it back into the dishonest version.
    """
    run, _ = _trained_run(layout)
    from itf.inference import load_model

    payload = kernels(load_model(run / "best.pt"))

    assert payload["layer"] == 1
    assert payload["layers_in_backbone"] == 2, "la red del test tiene dos capas"
    assert "capa 1" in payload["deep_layers_note"]
    # in_channels == 1 is what makes layer 1 exact rather than a projection.
    assert payload["in_channels"] == 1
    assert payload["count"] == 4 and payload["shown"] == 4
    assert not payload["truncated"]
    # R2: a weight has sign. A sequential ramp here puts zero wherever it falls
    # and hides what a kernel IS.
    assert payload["job"] == "diverging"
    assert len(payload["maps"]) == 4
    assert np.shape(payload["maps"][0]["matrix"]) == (3, 3)


def test_kernels_refuse_when_a_filter_is_not_a_matrix(layout):
    """The rule is `in_channels == 1`, not "the first layer".

    Layer 1 only happens to satisfy it. A 3-channel input has the same problem
    the deep layers do, one layer earlier -- and serving `weight[:, 0]` there
    would be the exact projection D13 rejected, wearing a more convincing layer
    number.
    """
    model = build_model({**NETWORK, "in_channels": 3})
    with pytest.raises(NotInspectable, match="no es una matriz") as exc:
        kernels(model)
    assert exc.value.code == "kernels_not_projectable"
    # R4: the hint sends you where the answer actually is.
    assert "V2" in exc.value.hint


# ── V2: the feature maps ──────────────────────────────────────────────────────


def test_feature_maps_are_one_layer_each_with_the_prediction(layout):
    run, data = _trained_run(layout)
    from itf.inference import load_model

    model = load_model(run / "best.pt")
    with np.load(data / "patches.npz") as arrays:
        patch = arrays["X"][0, :, :, 0]

    out = feature_maps(model, patch)

    assert [layer["layer"] for layer in out["layers"]] == [1, 2]
    # 40 -> pool 2 -> 20 -> pool 2 -> 10. The spatial trace, observed.
    assert (out["layers"][0]["height"], out["layers"][0]["width"]) == (20, 20)
    assert (out["layers"][1]["height"], out["layers"][1]["width"]) == (10, 10)
    assert out["layers"][0]["count"] == 4 and out["layers"][1]["count"] == 6
    assert [c["corner"] for c in out["prediction"]["corners"]] == ["TL", "TR", "BR", "BL"]
    assert all(0.0 <= c["score"] <= 1.0 for c in out["prediction"]["corners"])


def test_the_colour_job_follows_the_activation_not_the_data(layout):
    """R3, and the shortcut it forbids.

    `"diverging" if maps.min() < 0 else "sequential"` is the tempting version and
    it is wrong per-patch: a `tanh` layer whose activations happen to be all
    positive on THIS patch would come back sequential, and the same layer on the
    next patch would come back diverging. The colour would mean two things in two
    screenshots of one network, silently. So: read the spec.
    """
    network = {
        **NETWORK,
        "backbone": [
            {"filters": 4, "kernel": 3, "pool": 2, "activation": "relu"},
            {"filters": 4, "kernel": 3, "pool": 2, "activation": "tanh"},
        ],
    }
    run, data = _trained_run(layout, name="run-tanh", network=network)
    from itf.inference import load_model

    model = load_model(run / "best.pt")
    # A patch of a single flat value: after ReLU everything is >= 0, and after
    # tanh it may well be all-positive too. The data cannot distinguish them --
    # only the spec can, which is the whole point.
    out = feature_maps(model, np.full((40, 40), 200, dtype=np.uint8))

    assert out["layers"][0]["job"] == "sequential", "relu no da negativos: magnitud"
    assert out["layers"][1]["job"] == "diverging", "tanh tiene signo, lo enseñe este patch o no"


def test_feature_maps_refuse_a_patch_of_the_wrong_size(layout):
    """Contract ① arriving at F.

    Inside the head this is `mat1 and mat2 shapes cannot be multiplied`: a message
    about linear algebra for a problem about a dataset. Same rule as `POST /runs`
    -- refuse at the door, with both numbers and the fix.
    """
    run, _ = _trained_run(layout)
    from itf.inference import load_model

    model = load_model(run / "best.pt")
    with pytest.raises(NotInspectable, match="40x40") as exc:
        feature_maps(model, np.zeros((60, 60), dtype=np.uint8))
    assert exc.value.code == "patch_size_mismatch"


def test_feature_maps_refuse_to_invent_border_flags(layout):
    """Contract ② one layer down. **Zeros are a claim, not a default.**

    0 means "touches no edge". Defaulting to it would answer for a patch flush
    against the top of its image as if it sat in the middle -- quietly, with a
    plausible-looking prediction (formatos.md §2).
    """
    network = {**NETWORK, "border_features": True}
    run, data = _trained_run(layout, name="run-border", network=network)
    from itf.inference import load_model

    model = load_model(run / "best.pt")
    with np.load(data / "patches.npz") as arrays:
        patch = arrays["X"][0, :, :, 0]

    with pytest.raises(NotInspectable) as exc:
        feature_maps(model, patch)
    assert exc.value.code == "border_required"

    # The control: with the flags, it answers. Without it, "refuses" would also
    # be satisfied by a view that was simply broken.
    out = feature_maps(model, patch, [1, 0, 0, 1])
    assert len(out["layers"]) == 2


# ── the model cache ───────────────────────────────────────────────────────────


# ── the probes (fase 8) ───────────────────────────────────────────────────────


def test_occlusion_baseline_is_the_unoccluded_prediction(layout):
    """V4 — the seam: `baseline` is the real forward, and the map is more of it.

    An occlusion map is only a diagnosis if the baseline it is read against is the
    prediction the rest of the app shows. If `baseline` were computed some other
    way, "covering here dropped the score by 0.3" would be measured against a
    number no other view reports, and the drop would be quietly meaningless.
    """
    run, data = _trained_run(layout)
    model = load_model(run / "best.pt")
    with np.load(data / "patches.npz") as arrays:
        patch = arrays["X"][0, :, :, 0]

    occ = occlusion(model, patch)
    fm = feature_maps(model, patch)

    # Same four scores as V2/V3 report for this patch: one prediction, many views.
    baseline_from_maps = [c["score"] for c in fm["prediction"]["corners"]]
    assert occ["baseline"] == baseline_from_maps

    # Shape and the honest colour job: p(exists|occluded) is a probability, never
    # signed, so sequential (R3) -- not the drop, which would be signed and would
    # put the neutral wherever the min fell.
    assert occ["job"] == "sequential"
    assert len(occ["maps"]) == len(CORNER_NAMES)
    bins = occ["bins"]
    for m in occ["maps"]:
        assert np.shape(m["matrix"]) == (bins, bins)
        flat = np.asarray(m["matrix"]).ravel()
        assert ((flat >= 0.0) & (flat <= 1.0)).all(), "es una probabilidad"


def test_occlusion_refuses_to_invent_border_flags(layout):
    """The same contract ② the feature maps obey, reused through the shared helper.

    Zeros would mean "touches no edge" -- a claim, not a default (formatos.md §2).
    """
    network = {**NETWORK, "border_features": True}
    run, data = _trained_run(layout, name="run-occ-border", network=network)
    model = load_model(run / "best.pt")
    with np.load(data / "patches.npz") as arrays:
        patch = arrays["X"][0, :, :, 0]

    with pytest.raises(NotInspectable) as exc:
        occlusion(model, patch)
    assert exc.value.code == "border_required"


def test_border_test_refuses_a_network_that_ignores_the_flags(layout):
    """V10 — **the refusal is the point**, and it is tests.md §1.1 exactly.

    A network with `border_features: false` never reads the 4 flags, so flipping
    them changes nothing. Drawing "no change" four times would read as "the border
    does not matter to this patch" -- a conclusion about the data, when the truth
    is the architecture never looks at it. So it refuses, like `kernels` refuses a
    projection it cannot make honestly.
    """
    run, data = _trained_run(layout)  # NETWORK has border_features=False
    model = load_model(run / "best.pt")
    with np.load(data / "patches.npz") as arrays:
        patch = arrays["X"][0, :, :, 0]
        border = arrays["border"][0].astype(int).tolist()

    with pytest.raises(NotInspectable, match="border_features") as exc:
        border_test(model, patch, border)
    assert exc.value.code == "border_not_used"


def test_border_test_flips_exactly_one_flag_at_a_time(layout):
    """Baseline is the real flags; each flip toggles one and forwards again.

    The invariant, not a number: `baseline` must be what the model says with the
    patch's own flags (so the dumbbell's "before" is honest), and `flips[b]` must
    differ from the real flags in exactly position `b`.
    """
    network = {**NETWORK, "border_features": True}
    run, data = _trained_run(layout, name="run-bt", network=network)
    model = load_model(run / "best.pt")
    with np.load(data / "patches.npz") as arrays:
        patch = arrays["X"][0, :, :, 0]
        border = arrays["border"][0].astype(int).tolist()

    bt = border_test(model, patch, border)

    # `baseline` == the prediction with the REAL flags (same forward V2/V3 make).
    fm = feature_maps(model, patch, border)
    assert bt["baseline"] == [c["score"] for c in fm["prediction"]["corners"]]

    assert bt["border"] == border
    assert len(bt["flips"]) == 4
    for b, flip in enumerate(bt["flips"]):
        assert flip["flag_from"] == border[b]
        assert flip["flag_to"] == 1 - border[b]
        # The flip's scores are the forward with exactly that flag toggled.
        flipped = list(border)
        flipped[b] = 1 - flipped[b]
        expected = feature_maps(model, patch, flipped)
        assert flip["scores"] == [c["score"] for c in expected["prediction"]["corners"]]


# ── V16, la deconvolución ─────────────────────────────────────────────────────


def test_deconvolution_peak_is_the_max_of_the_feature_map(layout):
    """V16 — **the seam**: the peak it differentiates is the one V2 draws.

    Same family as `test_occlusion_baseline_is_the_unoccluded_prediction`, and the
    same reason. V16's map answers "what lit filter k up", and that sentence only
    means something if the *up* is the activation the rest of the app shows for
    filter k. Compute the peak some other way -- a mean, a different sample of the
    batch, an off-by-one in the `h[idx, idx]` diagonal -- and the map is still a
    plausible-looking attribution, of a number no other view reports.

    The diagonal is the part that would fail silently: `h[i, j]` for i != j is a
    real activation of the wrong filter, so the view would keep working and keep
    being wrong. Checking the position too (not just the value) is what pins it.
    """
    run, data = _trained_run(layout)
    model = load_model(run / "best.pt")
    with np.load(data / "patches.npz") as arrays:
        patch = arrays["X"][0, :, :, 0]

    dec = deconvolution(model, patch)
    fm = feature_maps(model, patch)

    assert len(dec["layers"]) == len(fm["layers"])
    for maps_layer, grad_layer in zip(fm["layers"], dec["layers"]):
        assert grad_layer["count"] == maps_layer["count"]
        for shown, peak in zip(maps_layer["maps"], grad_layer["peaks"]):
            activation = np.asarray(shown["matrix"])
            assert peak["activation"] == pytest.approx(activation.max(), abs=1e-3)
            # ...and at that position, which is what catches a wrong diagonal.
            assert activation[peak["y"], peak["x"]] == pytest.approx(
                peak["activation"], abs=1e-3
            )


def test_deconvolution_is_signed_and_lives_in_patch_space(layout):
    """Two invariants that decide whether the picture means anything.

    **Diverging**: a gradient has sign -- pixels that push the filter down are half
    the information -- so painting it sequential would put the neutral wherever the
    minimum fell and erase that half (R2, the kernel bug the sibling shipped).

    **Patch space**: the map is `n×n` for every layer, whatever the pools did to
    the activation. That is what lets a layer-3 map be laid over the patch and
    compared against a layer-1 map without rescaling either. A map that came back
    at the activation's size would still render -- as a smaller square, silently
    misaligned with the patch under it.
    """
    run, data = _trained_run(layout)
    model = load_model(run / "best.pt")
    with np.load(data / "patches.npz") as arrays:
        patch = arrays["X"][0, :, :, 0]

    dec = deconvolution(model, patch)
    n = dec["input_size"]

    signed = False
    for layer in dec["layers"]:
        assert layer["job"] == "diverging"
        assert len(layer["peaks"]) == layer["count"]
        assert 0 <= layer["silent"] <= layer["count"]
        for m in layer["maps"]:
            assert np.shape(m["matrix"]) == (n, n)
            signed |= m["min"] < 0

    # The control: `diverging` would be a wrong declaration over data that is
    # never negative, so at least some map must actually go both ways.
    assert signed, "un gradiente sin valores negativos no justificaría la paleta divergente"


def test_deconvolution_does_not_demand_border_flags(layout):
    """The deliberate asymmetry with V4, pinned so nobody "fixes" it.

    `occlusion` refuses without the flags because it runs the head, and the head is
    the only thing `border_features` touches. The deconvolution never leaves the
    backbone, so the flags cannot change a single number in it -- and requiring
    them would be demanding a datum the view does not read, which is how a probe
    ends up refusing patches it could have answered.
    """
    network = {**NETWORK, "border_features": True}
    run, data = _trained_run(layout, name="run-deconv-border", network=network)
    model = load_model(run / "best.pt")
    with np.load(data / "patches.npz") as arrays:
        patch = arrays["X"][0, :, :, 0]

    # The control first: this same network/patch DOES refuse the occlusion probe,
    # so "it answered" below is about the backbone, not about a lax network.
    with pytest.raises(NotInspectable):
        occlusion(model, patch)

    dec = deconvolution(model, patch)
    assert len(dec["layers"]) == len(model.config.backbone)


def test_the_scrubber_window_is_the_window_B_extracts(layout):
    """V5 — contract ⑤ from F's side: an off-grid crop is flagged like its twin.

    The scrubber puts the window anywhere, but its border flags come from
    `window_at`, the *same* formula B grids over in `windows`. So a scrubber crop
    that happens to land on a grid position must get identical flags to the patch B
    extracted there -- otherwise the model would meet, live, a geometry it never
    trained on (the exact silent failure ⑤ exists to stop, tests.md §1.2).
    """
    run, _ = _trained_run(layout)
    model = load_model(run / "best.pt")
    image = _image(layout)
    h, w = image.shape

    # Every on-grid window's flags == `window_at` at the same coords. This is the
    # seam: `windows` and the scrubber both go through `window_at`.
    for win in windows(w, h, 40, 20):
        assert window_at(win.x0, win.y0, 40, w, h).border == win.border

    # And the scrubber reports that same window (border used as given).
    out = window_prediction(model, image, 0, 0)
    assert tuple(out["border"]) == window_at(0, 0, 40, w, h).border
    assert out["border"][0] == 1 and out["border"][3] == 1  # flush top-left


def test_the_scrubber_clamps_the_window_inside_the_image(layout):
    """A drag off the edge does not walk the window off the image.

    The response is authoritative: it reports the clamped `(x0, y0)`, so the box
    always sits where the model really looked, and `image_x` = `x0 + x*n` is
    consistent with it.
    """
    run, _ = _trained_run(layout)
    model = load_model(run / "best.pt")
    image = _image(layout)
    h, w = image.shape

    out = window_prediction(model, image, 10_000, 10_000)
    assert out["x0"] == w - out["patch_size"]
    assert out["y0"] == h - out["patch_size"]

    n = out["patch_size"]
    # image_x/y are rounded to 2 places and x/y to 4, so recomputing from the
    # rounded fractions lands within a rounding tolerance, not bit-exact.
    for c in out["corners"]:
        assert abs(c["image_x"] - (out["x0"] + c["x"] * n)) < 0.05
        assert abs(c["image_y"] - (out["y0"] + c["y"] * n)) < 0.05
    # Stability is a magnitude of score change: non-negative, and bounded by 1.
    assert 0.0 <= out["stability"]["max"] <= 1.0


def test_the_model_cache_notices_a_rewritten_checkpoint(layout):
    """**`best.pt` is rewritten on every improving epoch.**

    So a cache keyed on the path alone serves epoch 5's weights forever: you would
    watch a run improve and its kernels would never change. The mtime is what
    makes the cache invalidate itself -- the same reason it is in `TableKey`.
    """
    run, _ = _trained_run(layout)
    cache = ModelCache()
    checkpoint = run / "best.pt"

    first = cache.get(checkpoint)
    assert cache.get(checkpoint) is first, "el mismo .pt no se recarga"

    # Touch it the way an improving epoch would: same path, new content.
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    torch.save(payload, checkpoint)

    assert cache.get(checkpoint) is not first, "un checkpoint reescrito es otro modelo"
