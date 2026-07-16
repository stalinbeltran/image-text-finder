"""D — the recipe and the loop. Invariants, never results (tests.md §6).

The heart of this file is the trap tests. Almost every trap in this project was
a DEFAULT -- nobody chose them, they appeared by not choosing (organizacion.md
§3) -- so the tests that matter are the ones asserting a value was chosen ON
PURPOSE. Building from scratch does not protect against them: it invites them.
"""

from __future__ import annotations

import pytest
import torch

from itf.models import build_model
from itf.training.losses import CornerLoss
from itf.training.loop import RunSpec, _make_optimizer, train
from itf.training.recipe import Recipe, RecipeStore

from conftest import write_tiny_source

NETWORK = {
    "input_size": 40,
    "in_channels": 1,
    "border_features": False,
    "backbone": [{"filters": 4, "kernel": 3, "stride": 1, "pool": 2}],
    "head": {"hidden": [8]},
}


# ── the two default traps ─────────────────────────────────────────────────────


def test_sgd_gets_its_momentum():
    """The trap: torch's SGD defaults to momentum=0.

    The old `_make_optimizer` passed only `lr` and `weight_decay`, so `sgd` ran
    with no inertia. Sweeping `optimizer` compared Adam against a crippled SGD --
    it always lost, and not for being SGD. Nobody wrote momentum=0; it is what
    you get by not passing it.
    """
    model = build_model(NETWORK)
    opt = _make_optimizer(Recipe(optimizer="sgd", momentum=0.9), model.parameters())
    assert opt.param_groups[0]["momentum"] == 0.9

    # And it must actually be settable, or the field is decoration.
    opt = _make_optimizer(Recipe(optimizer="sgd", momentum=0.0), model.parameters())
    assert opt.param_groups[0]["momentum"] == 0.0


def test_adam_does_not_get_a_momentum_it_cannot_take():
    """The control: `momentum` only applies to sgd/rmsprop. Adam would TypeError."""
    model = build_model(NETWORK)
    opt = _make_optimizer(Recipe(optimizer="adam"), model.parameters())
    assert "momentum" not in opt.param_groups[0]


def test_smooth_l1_beta_actually_leaves_the_quadratic_branch():
    """The trap: torch's beta defaults to 1.0 and the coords live in [0,1].

    So |error| < 1 ALWAYS, the loss never leaves the quadratic branch, and the
    position term is plain MSE -- Huber's robustness never once activates. It was
    like that for months with nobody having decided it.

    This test does not check the beta field; it checks the CONSEQUENCE. With
    beta=1.0 the loss is quadratic in the error, so doubling a 0.4 error roughly
    quadruples it. With beta=0.05 that error is deep in the linear branch, so
    doubling it roughly doubles it. That difference is the whole point.
    """

    def loss_at(error: float, beta: float) -> float:
        pred = torch.zeros(1, 4, 3)
        target = torch.zeros(1, 4, 3)
        target[..., 0] = 1.0  # every corner exists, so the mask lets it through
        pred[..., 1:3] = error
        fn = CornerLoss(lambda_pos=1.0, smooth_l1_beta=beta)
        return float(fn(pred, target)["pos_loss"])

    quadratic = loss_at(0.4, beta=1.0) / loss_at(0.2, beta=1.0)
    linear = loss_at(0.4, beta=0.05) / loss_at(0.2, beta=0.05)

    # beta=1.0: both errors are inside |x| < beta, so smoothL1 is 0.5x²/beta and
    # doubling the error quadruples the loss. That is MSE, exactly.
    assert quadratic == pytest.approx(4.0, rel=0.01), "con beta=1.0 debería ser MSE puro"

    # beta=0.05: both errors are past the knee, so smoothL1 is |x| - 0.5·beta and
    # the ratio is (0.4-0.025)/(0.2-0.025) = 2.142857 -- NOT 2.0, because of that
    # offset. Slightly super-linear, and nowhere near quadratic: that gap is what
    # Huber's robustness to outliers actually buys.
    assert linear == pytest.approx((0.4 - 0.025) / (0.2 - 0.025), rel=0.01)
    assert linear < 2.5, "con beta=0.05 el Huber tiene que estar en la rama lineal"


def test_the_default_recipe_does_not_ship_the_traps():
    """The defaults must be the CHOSEN values, not torch's.

    This is the test that would have caught both traps at birth: it asserts that
    what you get by not choosing is what someone chose.
    """
    recipe = Recipe()
    assert recipe.smooth_l1_beta != 1.0, "1.0 es el default de PyTorch y anula el Huber"
    assert 0.01 <= recipe.smooth_l1_beta <= 0.2, "~0.05-0.1 = 2-4 px en un patch de 40"
    assert recipe.momentum > 0, "un SGD sin momentum sesga cualquier comparación de optimizadores"


# ── the recipe as an entity ───────────────────────────────────────────────────


def test_the_recipe_refuses_fields_that_are_not_hers():
    """A typo'd `momentun` silently dropped is a recipe that lies about itself."""
    with pytest.raises(ValueError, match="no son de la receta"):
        Recipe.from_dict({"lr": 1e-3, "momentun": 0.9})


def test_the_recipe_refuses_a_monitor_that_does_not_exist():
    with pytest.raises(ValueError, match="monitor desconocido"):
        Recipe(monitor="val_accuracy")


def test_the_recipe_survives_a_round_trip_through_yaml(tmp_path):
    """It is a noun: it has a name, a store, and it comes back the same."""
    store = RecipeStore(tmp_path / "recipes")
    recipe = Recipe(lr=3e-4, optimizer="sgd", momentum=0.95, lambda_pos=2.0, smooth_l1_beta=0.08)
    store.save("baseline", recipe)
    assert store.names() == ["baseline"]
    assert store.get("baseline") == recipe


# ── the loop ──────────────────────────────────────────────────────────────────


def _tiny_patch_dataset(tmp_path):
    from itf.patches import PatchExtractConfig, SplitConfig, extract_dataset

    source = write_tiny_source(tmp_path / "src", num_samples=10)
    out = tmp_path / "b"
    extract_dataset(
        PatchExtractConfig(
            source=str(source),
            out=str(out),
            patch_size=40,
            stride=20,
            split=SplitConfig(0.6, 0.2, 0.2),
        )
    )
    return out


def test_training_reduces_the_loss(tmp_path):
    """An invariant -- the gradients flow -- not a number.

    A test asserting `f1 > 0.75` would not be a test: it is a research claim in
    disguise, it breaks for legitimate reasons, and it teaches people to tune the
    threshold until it passes. Results go in the protocol (tests.md §6).
    """
    data = _tiny_patch_dataset(tmp_path)
    summary = train(
        RunSpec(
            data=str(data),
            out=str(tmp_path / "run"),
            network=NETWORK,
            recipe=Recipe(epochs=3, batch_size=32, lr=1e-2),
        )
    )
    losses = [h["train_loss"] for h in _history(tmp_path / "run")]
    assert len(losses) == 3
    assert losses[-1] < losses[0]
    assert summary["epochs_run"] == 3


def _history(run_dir):
    import json

    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def test_training_refuses_a_dataset_without_val(tmp_path):
    """protocolo.md §1.3, and it is the hard half of the rule.

    Without val, `monitor` fell back to the TRAIN loss with no warning and
    `best.pt` became the most overfitted checkpoint. Building such a dataset
    warns (fase 2); training on it must refuse, because this is where the damage
    happens.
    """
    from itf.patches import PatchExtractConfig, SplitConfig, extract_dataset

    source = write_tiny_source(tmp_path / "src", num_samples=2)
    data = tmp_path / "sin-val"
    extract_dataset(
        PatchExtractConfig(
            source=str(source), out=str(data), patch_size=40, stride=20, split=SplitConfig(1.0, 0.0, 0.0)
        )
    )
    with pytest.raises(ValueError, match="no tiene val"):
        train(RunSpec(data=str(data), out=str(tmp_path / "run"), network=NETWORK, recipe=Recipe(epochs=1)))


def test_the_loop_validates_before_a_single_batch(tmp_path):
    """The safety net of ① and ②: `itf-train` does not go through the API.

    Without this call the CLI walks straight past the 400 the API would have
    given, and the mismatch reappears mid-epoch as `mat1 and mat2 shapes cannot
    be multiplied`. Note the test is FAST -- it never trains -- which is the sign
    the check sits in the right place.
    """
    from itf.training.loop import IncompatibleError

    data = _tiny_patch_dataset(tmp_path)
    with pytest.raises(IncompatibleError) as exc:
        train(
            RunSpec(
                data=str(data),
                out=str(tmp_path / "run"),
                network={**NETWORK, "input_size": 60},
                recipe=Recipe(epochs=1),
            )
        )
    assert [p["code"] for p in exc.value.problems] == ["patch_size_mismatch"]
    assert not (tmp_path / "run").exists(), "no debería haber creado el run"


def test_early_stopping_cuts_and_says_so(tmp_path):
    """`patience` watches ITS OWN val curve. Distinct from the sweep's pruning,
    which compares between runs (contract ⑨)."""
    data = _tiny_patch_dataset(tmp_path)
    summary = train(
        RunSpec(
            data=str(data),
            out=str(tmp_path / "run"),
            network=NETWORK,
            # lr=0 => nothing ever improves => it must cut at the first chance.
            recipe=Recipe(epochs=10, lr=0.0, patience=2),
        )
    )
    assert summary["stopped_early"] is True
    assert summary["epochs_run"] < 10


def test_the_run_records_its_state_explicitly(tmp_path):
    """`status.json`, not "guess from which files exist".

    Inferring it is what left a crashed run reading "running" forever -- and in a
    20-point sweep, the dead being indistinguishable from the living.
    """
    import json

    data = _tiny_patch_dataset(tmp_path)
    run = tmp_path / "run"
    train(RunSpec(data=str(data), out=str(run), network=NETWORK, recipe=Recipe(epochs=1)))
    assert json.loads((run / "status.json").read_text())["state"] == "done"


def test_the_checkpoint_carries_the_network_config(tmp_path):
    """Contract ④'s half that fase 3 can already prove: the `.pt` describes itself."""
    data = _tiny_patch_dataset(tmp_path)
    run = tmp_path / "run"
    train(RunSpec(data=str(data), out=str(run), network=NETWORK, recipe=Recipe(epochs=1)))
    ckpt = torch.load(run / "best.pt", weights_only=False)
    assert ckpt["config"]["model"] == NETWORK
