"""C — the network. Config pure, zero data, and that is testable without data."""

from __future__ import annotations

import pytest
import torch

from itf.models import NetworkConfig, build_model, flat_features, spatial_trace

BASE = {
    "input_size": 40,
    "in_channels": 1,
    "backbone": [
        {"filters": 32, "kernel": 3, "stride": 1, "pool": 2},
        {"filters": 64, "kernel": 3, "stride": 1, "pool": 2},
    ],
    "head": {"hidden": [128]},
}


def test_the_head_shape_is_four_corners_by_three():
    """`(B, 4, 3)` == `[exists, x, y]` per corner. Not a softmax over 4 classes."""
    model = build_model(BASE)
    out = model(torch.zeros(2, 1, 40, 40))
    assert out.shape == (2, 4, 3)


def test_coordinates_are_squashed_and_the_logit_is_not():
    """Channel 0 stays a RAW logit; 1:3 are sigmoid'd into [0,1] like the labels.

    The loss applies the sigmoid to the logit itself (BCEWithLogits) for
    numerical stability, so squashing it here would apply it twice.
    """
    model = build_model(BASE)
    # Big weights, so an already-squashed logit would be pinned inside [0,1].
    with torch.no_grad():
        model.head.out.weight.mul_(50)
        model.head.out.bias.fill_(8.0)
    out = model(torch.rand(4, 1, 40, 40))
    assert out[..., 0].abs().max() > 1.0, "el logit no puede venir aplastado"
    assert 0.0 <= out[..., 1:3].min() and out[..., 1:3].max() <= 1.0


def test_the_spatial_trace_matches_what_the_tensor_actually_does():
    """The seam: the trace is analytic, the Linear is sized empirically.

    `spatial_trace` computes 40→20→10 with arithmetic (it feeds the Redes screen,
    where there are no weights); `_infer_flat_features` measures it by pushing a
    dummy tensor through (it sizes the head). **Two ways to compute one number is
    exactly what contract ⑤ is about** -- if they drift, the screen lies about the
    network you are building, and nothing raises.
    """
    for pool in (2, 1):
        for input_size in (40, 60, 28):
            config = NetworkConfig.from_dict(
                {**BASE, "input_size": input_size, "backbone": [{**b, "pool": pool} for b in BASE["backbone"]]}
            )
            model = build_model(config)
            measured = model._infer_flat_features(config)
            assert flat_features(config) == measured, f"traza != tensor con {input_size}px, pool {pool}"


def test_the_trace_reads_like_the_screen_shows_it():
    trace = spatial_trace(NetworkConfig.from_dict(BASE))
    assert [t["out"] for t in trace] == [20, 10]  # 40 -> 20 -> 10
    assert [t["channels"] for t in trace] == [32, 64]


def test_a_layer_that_does_not_fit_says_which_one_and_why():
    """400 with WHICH layer, at what size, and how to fix it -- not a stack trace
    from inside a job (api.md §3)."""
    config = NetworkConfig.from_dict(
        {**BASE, "input_size": 8, "backbone": [{"filters": 8, "kernel": 3, "pool": 2}] * 4}
    )
    with pytest.raises(ValueError, match="la capa 4 no cabe"):
        spatial_trace(config)


def test_border_features_changes_the_head_and_the_forward():
    """Contract ②, from C's side: the dataset offers, the network decides."""
    without = build_model(BASE)
    with_border = build_model({**BASE, "border_features": True})

    x = torch.zeros(2, 1, 40, 40)
    border = torch.tensor([[1.0, 0, 0, 1], [0, 1, 1, 0]])

    assert without(x).shape == with_border(x, border).shape == (2, 4, 3)
    # The head is 4 inputs wider, and that is the whole mechanism.
    assert with_border.head.mlp[0].in_features == without.head.mlp[0].in_features + 4


def test_a_network_that_wants_borders_refuses_to_run_without_them():
    """Silently passing zeros here is exactly the ② trap: the model cannot tell
    "unknown" from "touches no border"."""
    model = build_model({**BASE, "border_features": True})
    with pytest.raises(ValueError, match="border_features"):
        model(torch.zeros(1, 1, 40, 40))


def test_the_conv_is_always_element_zero_of_its_block():
    """A contract, not an accident: `kernels()` reads `block[0].weight`.

    It has to keep holding with `batchnorm: true`, which is the case that would
    break it if the block were ever reordered -- and the kernel view would then
    silently show a BatchNorm's weights as if they were a kernel.
    """
    model = build_model({**BASE, "backbone": [{"filters": 8, "kernel": 3, "batchnorm": True, "pool": 2}]})
    kernels = model.kernels()
    assert len(kernels) == 1
    assert kernels[0].shape == (8, 1, 3, 3)


def test_feature_maps_come_out_one_per_block():
    """V2's input. No hooks: the backbone is a Sequential of Sequentials."""
    model = build_model(BASE)
    maps = model.feature_maps(torch.zeros(1, 1, 40, 40))
    assert [m.shape[-1] for m in maps] == [20, 10]
    assert [m.shape[1] for m in maps] == [32, 64]


def test_the_head_says_what_to_type_when_hidden_is_not_a_list():
    """These YAMLs are hand-written, so `hidden: 32` is a plausible slip.

    Without the check it surfaces as "'int' object is not iterable" from a
    for-loop, and the API answers 500 instead of a 400 that names the fix.
    """
    with pytest.raises(ValueError, match=r"hidden: \[32\]"):
        build_model({**BASE, "head": {"hidden": 32}})
