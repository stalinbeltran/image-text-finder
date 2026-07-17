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


def _provenance(**overrides) -> dict:
    """The block every run is born with (contract ③).

    Written out rather than built with `build_provenance` because these tests are
    about the LOOP: calling the real builder would shell out to git twice per
    run for no gain, and the shape has its own test in `test_contracts.py`.
    """
    return {
        "patch_dataset": {"name": "tiny-40", "fingerprint": "sha256:" + "0" * 64},
        "network": {"name": "cnn-a", "value": NETWORK},
        "recipe": {"name": "adam", "value": {}},
        "sweep": None,
        "git_commit": "0" * 40,
        "environment": {"python": "3.12.10", "torch": "2.13.0+cpu", "platform": "win32"},
        **overrides,
    }


def _spec(**kwargs) -> RunSpec:
    """A RunSpec with the boring parts filled in.

    `provenance` has no default in `RunSpec` on purpose: every run is born with
    it (formatos.md §4.2), so there is no way to spell a run without one -- and
    that includes the runs a test makes.
    """
    return RunSpec(**{"network": NETWORK, "provenance": _provenance(), **kwargs})


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
        _spec(data=str(data), out=str(tmp_path / "run"), recipe=Recipe(epochs=3, batch_size=32, lr=1e-2))
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
        train(_spec(data=str(data), out=str(tmp_path / "run"), recipe=Recipe(epochs=1)))


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
            _spec(
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
        _spec(
            data=str(data),
            out=str(tmp_path / "run"),
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
    train(_spec(data=str(data), out=str(run), recipe=Recipe(epochs=1)))
    assert json.loads((run / "status.json").read_text())["state"] == "done"


def test_the_checkpoint_carries_the_network_config(tmp_path):
    """Contract ④'s half that fase 3 can already prove: the `.pt` describes itself."""
    data = _tiny_patch_dataset(tmp_path)
    run = tmp_path / "run"
    train(_spec(data=str(data), out=str(run), recipe=Recipe(epochs=1)))
    ckpt = torch.load(run / "best.pt", weights_only=False)
    assert ckpt["config"]["model"] == NETWORK


# ── what fase 4 added: provenance, the stop, and never overwriting ────────────


def test_the_frozen_config_carries_the_provenance_and_keeps_x_out_of_it(tmp_path):
    """Contract ③ and contract ⑩ in one file, which is where they meet.

    `data` is deliberately NOT in here any more: it was an absolute path, and
    following it is what broke the provenance the moment B was rebuilt or moved
    (organizacion.md §2-③). The name and the fingerprint say the same thing and
    survive.
    """
    import json as _json

    data = _tiny_patch_dataset(tmp_path)
    run = tmp_path / "run"
    train(_spec(data=str(data), out=str(run), recipe=Recipe(epochs=1), device="cpu"))

    config = _json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert config["provenance"] == _provenance()
    assert config["execution"] == {"device": "cpu", "num_workers": 0}
    assert "device" not in config["recipe"], "device es X: fuera de la identidad de D (⑩)"
    assert "data" not in config, "la referencia a B es el nombre + la huella, no una ruta"


def test_a_run_cannot_be_born_without_provenance():
    """There is no way to spell it, and that is the point.

    A run that cannot say where it came from is exactly the hole contract ③
    describes. Making the field optional would make the hole reachable by
    accident -- which is how every trap in this project happened.
    """
    with pytest.raises(TypeError):
        RunSpec(data="d", out="o", network=NETWORK)  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="le faltan campos"):
        RunSpec(data="d", out="o", network=NETWORK, provenance={"network": {"name": "cnn-a"}})


def test_the_loop_refuses_to_overwrite_a_finished_run(tmp_path):
    """The trap, and it is why `metrics.jsonl` is the tell.

    `mkdir(exist_ok=True)` plus truncating destroyed a finished run without a
    word, and a sweep that auto-generates names is who steps on it. A directory
    with only a reservation in it (config + status) is a name being claimed; one
    with metrics is a RESULT.
    """
    from itf.training.registry import RunExists

    data = _tiny_patch_dataset(tmp_path)
    run = tmp_path / "run"
    train(_spec(data=str(data), out=str(run), recipe=Recipe(epochs=1)))
    before = (run / "metrics.jsonl").read_text(encoding="utf-8")

    with pytest.raises(RunExists):
        train(_spec(data=str(data), out=str(run), recipe=Recipe(epochs=1)))

    assert (run / "metrics.jsonl").read_text(encoding="utf-8") == before


def test_the_summary_never_writes_an_infinity(tmp_path):
    """`best` is None when nothing was measured — not ±inf, and this bites.

    An infinite sentinel is not a measurement, it is the absence of one
    (formatos.md §2). And it does not survive the trip out: `json.dumps` writes
    `Infinity`, which is **not valid JSON and no browser can parse it**, so one
    run whose monitor never fired would take `GET /runs` down for every other run
    on the screen.

    The path is real: `val_pos_err_px` is None on a val split with no corners, so
    the monitor never produces a number and `best` never leaves its sentinel.
    """
    import json as _json
    import math

    data = _tiny_patch_dataset(tmp_path)
    run = tmp_path / "run"
    summary = train(_spec(data=str(data), out=str(run), recipe=Recipe(epochs=1)))

    text = (run / "summary.json").read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    # And the value that IS there is a real measurement.
    assert summary["best"] is None or math.isfinite(summary["best"])


def test_the_stop_is_cooperative_and_keeps_the_epoch(tmp_path):
    """Asked at the end of an epoch, so nothing is lost and nothing is killed.

    Cutting mid-batch would leave a half-written `last.pt`. And the run closes as
    `cancelled`, not `done`: it has real weights, so calling it done would let it
    into a comparison as though it had finished (protocolo.md §7).
    """
    import json as _json

    data = _tiny_patch_dataset(tmp_path)
    run = tmp_path / "run"
    summary = train(
        _spec(data=str(data), out=str(run), recipe=Recipe(epochs=10, batch_size=32)),
        should_stop=lambda: True,  # stop at the first safe point
    )

    assert summary["cancelled"] is True
    assert summary["epochs_run"] == 1, "la época en curso se termina, no se aborta"
    assert _json.loads((run / "status.json").read_text())["state"] == "cancelled"
    assert (run / "best.pt").exists()
    assert len(_history(run)) == 1, "la época que corrió dejó su métrica"
