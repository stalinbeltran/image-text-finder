"""One test per contract of organizacion.md §2.

`pytest tests/test_contracts.py -v` must read as the status report of that section:
each name says which boundary it guards, so a failure needs no code reading.

Every test here is `xfail(strict=True)` because the tree has no `src/` yet
(D18, plan-ui.md §0). That is the point, and `strict` is what makes it work:

  - while the contract does not exist  -> the test fails  -> expected xfail -> suite green
  - the day its phase implements it    -> the test passes -> strict XPASS   -> suite RED

So the phase cannot forget to remove the marker, and this file cannot drift away
from the design. See tests.md §2.

Imports live inside the test bodies on purpose: at module level a missing `itf`
would be a collection error, which is a broken suite, not an expected xfail.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from conftest import write_tiny_source as _write_tiny_source

SRC = Path(__file__).resolve().parents[1] / "src"


# --------------------------------------------------------------------------- #
# The two dicts the ①/② validator compares. It never sees an object: that is
# what lets it run in milliseconds and be called from both train() and the API.
# --------------------------------------------------------------------------- #


def _manifest(*, patch_size: int = 40, has_border: bool = True, channels: int = 1) -> dict:
    """What B offers, as the validator sees it (formatos.md §4.1)."""
    return {
        "format_version": 1,
        "fingerprint": "sha256:" + "0" * 64,
        "has_border": has_border,
        "config": {"patch_size": patch_size},
        "patch_shape": [patch_size, patch_size, channels],
        "corner_order": ["TL", "TR", "BR", "BL"],
        "border_order": ["top", "right", "bottom", "left"],
    }


def _network(*, input_size: int = 40, border_features: bool = False, in_channels: int = 1) -> dict:
    """What C needs, as the validator sees it (organizacion.md §1-C)."""
    return {
        "input_size": input_size,
        "in_channels": in_channels,
        "border_features": border_features,
        "backbone": [{"filters": 8, "kernel": 3, "stride": 1, "padding": 1, "pool": 2}],
        "head": {"hidden": [32]},
    }


# --------------------------------------------------------------------------- #
# ① B ↔ C — patch_size == input_size   (the critical one)
# --------------------------------------------------------------------------- #


def test_contract_01_rejects_patch_size_mismatch():
    """A network that expects 40 must not be trainable on 60-pixel patches.

    The failure must land here -- pure, milliseconds, no torch -- and not half an
    hour later inside the job thread as `mat1 and mat2 shapes cannot be
    multiplied`. That the test is fast IS the proof the validation sits in the
    right layer (tests.md §3).
    """
    from itf.validation import check_compatible

    problems = check_compatible(_manifest(patch_size=60), _network(input_size=40))

    assert [p["code"] for p in problems] == ["patch_size_mismatch"]
    # R4 of api.md: an error says why and how to fix it, and carries both numbers.
    assert "60" in problems[0]["message"] and "40" in problems[0]["message"]
    assert problems[0]["hint"]

    # The negative control: the matching pair must pass clean.
    assert check_compatible(_manifest(patch_size=40), _network(input_size=40)) == []


def test_contract_01_post_runs_refuses_the_mismatch_before_creating_the_job(itf_api):
    """The half fase 3 could not close: **the same rule, over HTTP**.

    The validator held the rule and `itf-train` called it, so the CLI door was
    shut -- but `POST /runs` did not exist yet, so nothing forced the route to
    ask (tests.md §3). This is that test, and what it really asserts is the
    TIMING: the 400 comes back with no job and no run on disk. A 400 on the way
    in is worth a thousand stack traces inside a job thread half an hour later,
    which is what `mat1 and mat2 shapes cannot be multiplied` was.
    """
    client, layout = itf_api
    layout.write_patch_dataset("grande-60", patch_size=60)
    layout.write_network("cnn-40", input_size=40)
    layout.write_recipe("rapida")

    response = client.post(
        "/runs",
        json={
            "name": "nunca-nace",
            "patch_dataset": "grande-60",
            "network": "cnn-40",
            "recipe": "rapida",
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "patch_size_mismatch"
    assert "60" in detail["message"] and "40" in detail["message"]
    assert detail["hint"]
    # Nothing was created. This is the assertion that matters -- a 400 after the
    # run directory exists would leave a corpse behind on every mistake.
    assert not (layout.runs / "nunca-nace").exists()
    assert client.get("/runs").json()["runs"] == []


def test_contract_01_post_runs_accepts_the_matching_pair(itf_api):
    """The control: the 400 must come from the MISMATCH, not from /runs refusing.

    Without this, a `POST /runs` that rejected everything would pass the test
    above and look like a working contract.
    """
    client, layout = itf_api
    layout.write_patch_dataset("justo-40", patch_size=40)
    layout.write_network("cnn-40", input_size=40)
    layout.write_recipe("rapida")

    response = client.post(
        "/runs",
        json={
            "name": "nace",
            "patch_dataset": "justo-40",
            "network": "cnn-40",
            "recipe": "rapida",
        },
    )

    # 202, not 200 (R3): training is minutes, so the answer is a job. The job
    # itself then fails on this manifest-only dataset -- there is no `.npz` to
    # load -- and that is fine and even to the point: the compatible pair gets
    # PAST the gate, which is all this asserts.
    assert response.status_code == 202
    assert response.json()["kind"] == "train"
    assert (layout.runs / "nace" / "config.json").exists()


# --------------------------------------------------------------------------- #
# ② B ↔ C — border_features
# --------------------------------------------------------------------------- #


def test_contract_02_border_backfill_is_harmless_when_the_network_ignores_it():
    """Zeros for an absent `border` are legal only because nobody reads them.

    formatos.md §2: filling an absent field is legal iff the consumer does not
    use it. This is that half.
    """
    from itf.validation import check_compatible

    assert check_compatible(_manifest(has_border=False), _network(border_features=False)) == []


def test_contract_02_refuses_border_features_when_the_dataset_lacks_them():
    """The case the old suite never had, and the one that matters.

    The old tests proved the backfill *happens*; they never proved the system
    *refuses* (tests.md §1.1). Zero means "touches no border", not "unknown": a
    network trained on fabricated zeros meets real flags at inference and sees a
    distribution it never trained on, exactly at the image borders. Nothing
    raises -- predictions just get worse. So this must be an error, not a fill.
    """
    from itf.validation import check_compatible

    problems = check_compatible(_manifest(has_border=False), _network(border_features=True))

    assert [p["code"] for p in problems] == ["border_features_unavailable"]
    assert problems[0]["hint"], "must say: rebuild it, or train with border_features: false"


def test_contract_02_post_runs_refuses_border_features_over_http(itf_api):
    """The same refusal at the API's door, which is the other half of tests.md §3.

    Same validator, same call site as ①: they are one question -- *can this C
    train on this B?* -- so `POST /runs` asks once and both contracts are shut.
    """
    client, layout = itf_api
    layout.write_patch_dataset("sin-bordes", patch_size=40, has_border=False)
    layout.write_network("con-bordes", input_size=40, border_features=True)
    layout.write_recipe("rapida")

    response = client.post(
        "/runs",
        json={
            "name": "nunca-nace",
            "patch_dataset": "sin-bordes",
            "network": "con-bordes",
            "recipe": "rapida",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "border_features_unavailable"
    assert not (layout.runs / "nunca-nace").exists()


# --------------------------------------------------------------------------- #
# ③ B + C + D → E — provenance: name AND value
# --------------------------------------------------------------------------- #


def test_contract_03_run_records_network_and_recipe_by_name_and_by_value():
    """Name to group, value to reproduce -- both, and that is not redundancy.

    With only the value you must diff dicts by hand to ask "which runs used
    network X?", which is the question a sweep asks constantly. Shape fixed by
    D2 (formatos.md §4.2.1).
    """
    from itf.training.provenance import build_provenance

    prov = build_provenance(
        patch_dataset={"name": "tiny-40", "fingerprint": "sha256:" + "a" * 64},
        network={"name": "cnn-a", "value": _network()},
        recipe={"name": "adam-lr1e-3", "value": {"lr": 1e-3, "epochs": 2}},
        sweep=None,
    )

    assert set(prov) == {
        "patch_dataset",
        "network",
        "recipe",
        "sweep",
        "git_commit",
        "environment",
    }
    assert prov["network"]["name"] == "cnn-a"
    assert prov["network"]["value"] == _network()
    assert prov["patch_dataset"]["fingerprint"].startswith("sha256:")
    # environment closes the hole git_commit leaves: the commit fixes the code,
    # not the interpreter (D2).
    assert set(prov["environment"]) >= {"python", "torch", "platform"}
    # Nothing is silently filled (formatos.md §2).
    assert prov["git_commit"], "sin git se escribe la razón, no null"


def test_contract_03_an_unknowable_commit_writes_the_right_reason(tmp_path):
    """"Sin git se escribe la razón" — and it has to be the TRUE one.

    formatos.md §2 asks for *the* reason, not for *a* reason: a run whose
    provenance says "not a git repo" about a repo that simply has no commits yet
    sends whoever reads it looking in the wrong place. The two cases are
    indistinguishable by exit code (`git rev-parse HEAD` fails the same way for
    both), so they have to be told apart on purpose.
    """
    import subprocess

    from itf.training.provenance import git_commit

    plain = tmp_path / "sin-git"
    plain.mkdir()
    assert "no es un repositorio git" in git_commit(plain)

    empty = tmp_path / "vacio"
    empty.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=empty, check=True, capture_output=True)
    assert "no tiene ningún commit" in git_commit(empty)


def test_contract_03_delete_of_a_patch_dataset_in_use_returns_409(itf_api):
    """Deleting a B that runs point at must say which ones, not break them quietly.

    The failure mode it replaces: the directory goes, the provenance chain
    returns None in silence, and the Predict tab loses its dataset without ever
    saying why. Nothing raises.
    """
    client, layout = itf_api
    layout.write_patch_dataset("tiny-40")
    layout.write_run("cnn-a-01", patch_dataset="tiny-40")
    layout.write_run("cnn-a-02", patch_dataset="tiny-40")

    response = client.delete("/patch-datasets/tiny-40")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "patch_dataset_in_use"
    assert detail["used_by"] == ["cnn-a-01", "cnn-a-02"], "el 409 tiene que decir QUÉ runs lo referencian"
    assert detail["hint"]
    # And it must not have deleted it on the way out.
    assert (layout.patch_datasets / "tiny-40" / "manifest.json").exists()


def test_contract_03_delete_of_an_unused_patch_dataset_succeeds(itf_api):
    """The control: the 409 must come from being IN USE, not from delete failing.

    Without this, a `delete` that always refused would pass the test above.
    """
    client, layout = itf_api
    layout.write_patch_dataset("libre-40")
    layout.write_run("cnn-a-01", patch_dataset="otro-dataset")

    assert client.delete("/patch-datasets/libre-40").status_code == 204
    assert not (layout.patch_datasets / "libre-40").exists()


# --------------------------------------------------------------------------- #
# ④ E → F — the checkpoint describes itself
# --------------------------------------------------------------------------- #


def test_contract_04_checkpoint_rebuilds_the_network_without_the_yaml(tmp_path):
    """A `.pt` is portable: F never needs C's YAML nor B.

    The cleanest dependency in the project, and the best property of the format.
    Note this test needs no training -- a checkpoint of untrained weights proves
    the same thing, which is why it is cheap.
    """
    import torch

    from itf.inference import load_model
    from itf.models import build_model

    cfg = _network()
    model = build_model(cfg)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "config": {"model": cfg}, "epoch": 0}, ckpt)

    rebuilt = load_model(ckpt)  # no YAML, no dataset, no config argument

    x = torch.zeros(1, cfg["in_channels"], cfg["input_size"], cfg["input_size"])
    assert rebuilt(x).shape == (1, 4, 3)


def test_contract_04_a_checkpoint_that_cannot_describe_itself_is_refused(tmp_path):
    """The control, and it is what gives the contract teeth.

    "It rebuilds" would also pass if `load_model` guessed an architecture. It must
    not: guessing loads the weights into the wrong shape, or worse into a
    right-shaped wrong network, and the failure is silent (formatos.md §2 --
    ausente ≠ cero, applied to a whole config). A bare `state_dict` is the case
    that turns up in real life, saved by a well-meaning script.
    """
    import pytest as _pytest
    import torch

    from itf.inference import load_model
    from itf.models import build_model

    bare = tmp_path / "bare.pt"
    torch.save(build_model(_network()).state_dict(), bare)

    with _pytest.raises(ValueError, match="no lleva su config de red"):
        load_model(bare)

    # And not a traceback either when it is not even a mapping.
    not_ours = tmp_path / "not-ours.pt"
    torch.save(torch.zeros(3), not_ours)
    with _pytest.raises(ValueError, match="no lleva su config de red"):
        load_model(not_ours)


# --------------------------------------------------------------------------- #
# ⑤ B ↔ F — the shared window   (the one that breaks silently)
# --------------------------------------------------------------------------- #


def test_contract_05_extraction_and_inference_see_the_same_window(tmp_path):
    """Not "is `positions` correct?" but "do B and F see the same window?".

    The old suite tested the half that could not break (both sides imported the
    same private function) and left the half that could: the border flags were
    duplicated literally in extract.py and predict.py. Identical, so it worked --
    until someone touched one. tests.md §1.2.

    The design removes the seam by putting the window in `itf.geometry`. This
    test is what stops the duplication from coming back, because re-typing those
    six lines is exactly what feels natural when writing inference.
    """
    import numpy as np

    from itf import geometry
    from itf.inference import predict as inference_mod
    from itf.patches import extract as extract_mod
    from itf.patches.extract import PatchExtractConfig, extract_dataset

    # 1. Same function, not a copy. This is the anti-duplication guard.
    assert extract_mod.windows is geometry.windows
    assert inference_mod.windows is geometry.windows

    # 2. And what the extractor actually wrote matches what geometry says.
    source = _write_tiny_source(tmp_path / "src", num_samples=2, width=120, height=100)
    out = tmp_path / "out"
    extract_dataset(PatchExtractConfig(source=str(source), out=str(out), patch_size=40, stride=20))

    data = np.load(out / "patches.npz")
    first = data["sample_idx"] == 0
    written = {
        (int(x), int(y)): tuple(int(b) for b in flags)
        for (x, y), flags in zip(data["patch_xy"][first], data["border"][first])
    }
    expected = {(w.x0, w.y0): tuple(w.border) for w in geometry.windows(120, 100, 40, 20)}
    assert written == expected


# --------------------------------------------------------------------------- #
# ⑦ G → all — the direction of dependencies
# --------------------------------------------------------------------------- #


def _itf_imports(package: str) -> list[str]:
    """Every `itf.*` module that `itf.<package>` imports, read from the AST."""
    files = sorted((SRC / "itf" / package).rglob("*.py"))
    assert files, f"itf.{package} no existe"
    found = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("itf"):
                found.append(f"{path.name}: {node.module}")
            elif isinstance(node, ast.Import):
                found += [f"{path.name}: {a.name}" for a in node.names if a.name.startswith("itf")]
    return found


def test_contract_07_the_network_does_not_know_the_source_loader_exists():
    """C must not know that A exists.

    The old violation: `models/builder.py` imported `NUM_BORDERS` from
    `datasets.loader` -- the network importing the source-dataset loader just to
    learn there are 4 borders. Nobody decided that; the constant was there and
    the import worked, and that is how every trap in this project was born.

    `itf.geometry` is allowed, and is the FIX rather than an exception to it
    (tests.md §4): the vocabulary is what everyone shares, C included. Writing 4
    by hand in `models` to avoid the import would be the same bug with more
    steps -- two constants that can drift.
    """
    # `itf.models.*` is itself -- a package importing its own submodules says
    # nothing about layering. What the contract is about is what it reaches
    # OUTSIDE its own walls, and the only legal answer there is `itf.geometry`.
    allowed = ("itf.geometry", "itf.models")
    offenders = [imp for imp in _itf_imports("models") if not imp.split(": ")[1].startswith(allowed)]
    assert offenders == [], "itf.models solo puede importar de itf.geometry (y de sí mismo)"


def test_contract_07_the_validator_is_pure():
    """`itf.validation` imports nothing from itf, and that is load-bearing.

    It compares DICTS, not objects: B's manifest against C's config. That is what
    lets it run in milliseconds without torch, and therefore what lets it be
    called from both `train()` and the API. If it needed to import the model to
    validate the model, the validation would be in the wrong layer -- and the
    speed of its test is what would tell you (tests.md §3).
    """
    assert _itf_imports("validation") == []


# --------------------------------------------------------------------------- #
# ⑧ H ↔ B, C — comparability
# --------------------------------------------------------------------------- #


def test_contract_08_fingerprint_tracks_content_not_name(tmp_path):
    """A B rebuilt under the same name is not the same B, and nothing else notices.

    Without a content fingerprint a half-finished sweep becomes incomparable *in
    silence*: `data` is a path, and a rebuilt path still points the same way.
    """
    import json as _json

    from itf.patches.extract import PatchExtractConfig, extract_dataset

    source = _write_tiny_source(tmp_path / "src")

    def fingerprint_of(out: Path, **overrides) -> str:
        cfg = {"patch_size": 40, "stride": 20, "seed": 1, **overrides}
        extract_dataset(PatchExtractConfig(source=str(source), out=str(out), **cfg))
        return _json.loads((out / "manifest.json").read_text(encoding="utf-8"))["fingerprint"]

    same = fingerprint_of(tmp_path / "a")
    renamed = fingerprint_of(tmp_path / "b")  # same content, different name
    rebuilt = fingerprint_of(tmp_path / "c", stride=10)  # different content

    assert same == renamed, "la huella es del contenido, no del nombre ni de la ruta"
    assert same != rebuilt, "otro stride es otro dataset, y la huella tiene que decirlo"
    assert same.startswith("sha256:")


def test_contract_08_split_seed_alone_decides_the_split(tmp_path):
    """B's seed fixes the split; D's seed is the replication axis. Two things.

    Confusing them makes every sweep point evaluate on a different split, so the
    sweep measures split noise instead of recipe quality (protocolo.md §4). And
    the split is per image, never per patch: overlapping windows of one image in
    two splits would contaminate val.
    """
    import json as _json

    import numpy as np

    from itf.patches.extract import PatchExtractConfig, extract_dataset

    source = _write_tiny_source(tmp_path / "src")

    def split_of(out: Path, seed: int) -> dict:
        extract_dataset(
            PatchExtractConfig(source=str(source), out=str(out), patch_size=40, stride=20, seed=seed)
        )
        return _json.loads((out / "split.json").read_text(encoding="utf-8"))

    assert split_of(tmp_path / "a", seed=7) == split_of(tmp_path / "b", seed=7)
    assert split_of(tmp_path / "a", seed=7) != split_of(tmp_path / "c", seed=8)

    # And the invariant the split rests on: no image lands in two splits.
    data = np.load(tmp_path / "a" / "patches.npz")
    by_split = [set(data["sample_idx"][data["split"] == s].tolist()) for s in (0, 1, 2)]
    assert by_split[0] & by_split[1] == set()
    assert by_split[0] & by_split[2] == set()
    assert by_split[1] & by_split[2] == set()


# --------------------------------------------------------------------------- #
# ⑨ H ↔ D — the objective, and the λ trap
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(strict=True, reason="contrato ⑨: sin implementar, plan-ui.md fase 7")
def test_contract_09_sweep_rejects_loss_objective_when_lambda_pos_varies():
    """`loss = cls + λ·pos`. Sweep λ and rank by loss and λ=0 "wins".

    That is: the optimum is to not predict positions at all. And the failure is
    silent -- it produces a winner with a good face. So this is a 400, not a
    warning (api.md §3).
    """
    from fastapi.testclient import TestClient

    from itf.api.app import app

    client = TestClient(app)
    response = client.post(
        "/sweeps",
        json={
            "name": "trampa",
            "patch_dataset": "tiny-40",
            "network": "cnn-a",
            "space": {"lambda_pos": {"type": "float", "low": 0.0, "high": 5.0}},
            "objective": "loss",
            "strategy": "random",
            "budget": {"points": 4, "epochs": 3},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "objective_varies_with_space"


# --------------------------------------------------------------------------- #
# ⑩ X ↔ D — the device is not a recipe
# --------------------------------------------------------------------------- #


def test_contract_10_device_is_not_part_of_the_recipe_identity():
    """The same recipe on CPU and on GPU is the same recipe.

    If `device` lives inside the recipe, the two produce different `config.json`
    and look like two recipes. Everything trained on CPU today becomes
    incomparable with tomorrow's GPU -- which is why X must come out *before* the
    GPU arrives, not after.

    The trap next door: `batch_size` is D, not X, however much VRAM invites
    raising it. Bigger batch on GPU is a new point of the space, not the same
    point going faster.
    """
    from itf.training.recipe import Recipe

    fields = set(Recipe.__dataclass_fields__)
    assert "device" not in fields
    assert "num_workers" not in fields
    # And the one that looks like X and is not:
    assert "batch_size" in fields, "batch_size cambia los pesos: es D (contrato ⑩)"
