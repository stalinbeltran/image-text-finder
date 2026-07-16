"""Same seed + same config => same weights.

Rule 1 of protocolo.md §7, and it is not a contract of organizacion.md §2 -- it
is the assumption the other five comparison rules stand on. "Recipe X beats Y"
means nothing if a run does not repeat itself.

The sibling project has this test; ITF never did. It should pass as soon as the
loop exists (`manual_seed` plus `num_workers=0` cover the shuffle), but *should
pass* is not *passes* -- which is the whole reason to write it down. tests.md §5.
"""

from __future__ import annotations

import pytest


@pytest.mark.xfail(strict=True, reason="reproducibilidad: sin bucle todavía, plan-ui.md fase 4")
def test_same_seed_and_config_produce_the_same_weights(tmp_path):
    import torch

    from itf.models import build_model

    cfg = {
        "input_size": 40,
        "in_channels": 1,
        "border_features": False,
        "backbone": [{"filters": 8, "kernel": 3, "stride": 1, "padding": 1, "pool": 2}],
        "head": {"hidden": 32},
    }

    def weights(seed: int) -> list[torch.Tensor]:
        torch.manual_seed(seed)
        return [p.detach().clone() for p in build_model(cfg).parameters()]

    for a, b in zip(weights(7), weights(7)):
        assert torch.equal(a, b)

    # The control: if a different seed gave the same weights, the seed is not
    # wired up and the test above would be passing for the wrong reason.
    assert any(not torch.equal(a, b) for a, b in zip(weights(7), weights(8)))
