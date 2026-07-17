"""Same seed + same config => same weights.

Rule 1 of protocolo.md §7, and it is not a contract of organizacion.md §2 -- it
is the assumption the other five comparison rules stand on. "Recipe X beats Y"
means nothing if a run does not repeat itself.

The sibling project has this test; ITF never did. It should pass as soon as the
loop exists (`manual_seed` plus `num_workers=0` cover the shuffle), but *should
pass* is not *passes* -- which is the whole reason to write it down. tests.md §5
puts it in fase 4, with the rest of E.
"""

from __future__ import annotations

import torch

from conftest import write_tiny_source

CONFIG = {
    "input_size": 40,
    "in_channels": 1,
    "border_features": False,
    "backbone": [{"filters": 8, "kernel": 3, "stride": 1, "padding": 1, "pool": 2}],
    # A LIST of layer widths, one entry per hidden layer.
    "head": {"hidden": [32]},
}


def test_the_same_seed_gives_the_same_initial_weights():
    """The cheap half: the initialisation. It runs in milliseconds."""
    from itf.models import build_model

    def weights(seed: int) -> list[torch.Tensor]:
        torch.manual_seed(seed)
        return [p.detach().clone() for p in build_model(CONFIG).parameters()]

    for a, b in zip(weights(7), weights(7)):
        assert torch.equal(a, b)

    # The control: if a different seed gave the same weights, the seed is not
    # wired up and the test above would be passing for the wrong reason.
    assert any(not torch.equal(a, b) for a, b in zip(weights(7), weights(8)))


def test_the_same_seed_and_recipe_train_to_the_same_weights(tmp_path):
    """The half that matters, and the one the protocol actually rests on.

    Initialisation repeating is not the claim -- **training** repeating is. Between
    the two sit the shuffle, the batching and the optimiser, which is where a
    forgotten seed hides. Rule 1 of protocolo.md §7 is the floor of every other
    rule: comparing two recipes means nothing if one recipe does not equal
    itself.
    """
    from itf.patches import PatchExtractConfig, SplitConfig, extract_dataset
    from itf.training.loop import RunSpec, train
    from itf.training.recipe import Recipe

    source = write_tiny_source(tmp_path / "src", num_samples=10)
    data = tmp_path / "b"
    extract_dataset(
        PatchExtractConfig(
            source=str(source),
            out=str(data),
            patch_size=40,
            stride=20,
            split=SplitConfig(0.6, 0.2, 0.2),
        )
    )

    provenance = {
        "patch_dataset": {"name": "b", "fingerprint": "sha256:" + "0" * 64},
        "network": {"name": "cnn-a", "value": CONFIG},
        "recipe": {"name": "r", "value": {}},
        "sweep": None,
        "git_commit": "0" * 40,
        "environment": {"python": "3.12.10", "torch": "2.13.0+cpu", "platform": "win32"},
    }

    def trained(out: str, seed: int) -> dict:
        train(
            RunSpec(
                data=str(data),
                out=str(tmp_path / out),
                network=CONFIG,
                provenance=provenance,
                # `num_workers=0` is not incidental: with workers, the order the
                # batches arrive in is up to the OS, and the run stops being a
                # function of its seed.
                recipe=Recipe(epochs=2, batch_size=16, lr=1e-2, seed=seed),
                num_workers=0,
            )
        )
        return torch.load(tmp_path / out / "last.pt", weights_only=False)["model"]

    first = trained("a", seed=7)
    again = trained("b-run", seed=7)
    other = trained("c", seed=8)

    assert first.keys() == again.keys()
    for key in first:
        assert torch.equal(first[key], again[key]), f"'{key}' no se repitió con la misma semilla"

    # The control: the seed of D is the REPLICATION AXIS (organizacion.md §1-D).
    # If it changed nothing, the equality above would prove nothing either -- and
    # measuring the noise floor, which is what tells a real difference from a
    # coin flip, would be impossible.
    assert any(not torch.equal(first[k], other[k]) for k in first)
