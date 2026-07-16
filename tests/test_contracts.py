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
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

SRC = Path(__file__).resolve().parents[1] / "src"


# --------------------------------------------------------------------------- #
# Fixtures: tiny synthetic sources, built in the test (tests.md §7).
# Never touch a real data/ or runs/.
# --------------------------------------------------------------------------- #


def _write_tiny_source(root: Path, *, num_samples: int = 6, width: int = 120, height: int = 100) -> Path:
    """Write a minimal source dataset (A) in the format of SAMPLE_FORMAT.md.

    Only the fields formatos.md §4.5 declares we consume: `index`, `image`,
    `labels.{width,height,has_overlap}` and `blocks[].{block_id,kind,angle,quad}`.
    One axis-aligned paragraph per image, placed differently in each so the
    corner labels are not all identical.
    """
    (root / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(num_samples):
        img = Image.new("L", (width, height), color=255)
        x0, y0 = 10 + (i % 3) * 6, 12 + (i % 2) * 8
        x1, y1 = x0 + 62, y0 + 44
        ImageDraw.Draw(img).rectangle([x0, y0, x1, y1], fill=0)
        img.save(root / "images" / f"{i:06d}.png")
        records.append(
            {
                "index": i,
                "image": f"images/{i:06d}.png",
                "labels": {
                    "image_id": f"tiny/{i:06d}",
                    "width": width,
                    "height": height,
                    "has_overlap": False,
                    "blocks": [
                        {
                            "block_id": "b0",
                            "kind": "paragraph",
                            "text": "x",
                            "angle": 0.0,
                            "box": [x0, y0, x1 - x0, y1 - y0],
                            # clockwise from TL, as CORNER_NAMES expects
                            "quad": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                        }
                    ],
                },
            }
        )
    (root / "labels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return root


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
        "head": {"hidden": 32},
    }


# --------------------------------------------------------------------------- #
# ① B ↔ C — patch_size == input_size   (the critical one)
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(strict=True, reason="contrato ①: sin implementar, plan-ui.md fase 4")
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


# --------------------------------------------------------------------------- #
# ② B ↔ C — border_features
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(strict=True, reason="contrato ②: sin implementar, plan-ui.md fase 4")
def test_contract_02_border_backfill_is_harmless_when_the_network_ignores_it():
    """Zeros for an absent `border` are legal only because nobody reads them.

    formatos.md §2: filling an absent field is legal iff the consumer does not
    use it. This is that half.
    """
    from itf.validation import check_compatible

    assert check_compatible(_manifest(has_border=False), _network(border_features=False)) == []


@pytest.mark.xfail(strict=True, reason="contrato ②: el caso que faltaba, formatos.md §2 / plan-ui.md fase 4")
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


# --------------------------------------------------------------------------- #
# ③ B + C + D → E — provenance: name AND value
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(strict=True, reason="contrato ③ / D2: sin implementar, plan-ui.md fase 4")
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


@pytest.mark.xfail(strict=True, reason="contrato ③: sin implementar, plan-ui.md fase 2")
def test_contract_03_delete_of_a_patch_dataset_in_use_returns_409():
    """Deleting a B that runs point at must say which ones, not break them quietly.

    Today's failure mode: the directory goes, `_run_source()` returns None
    silently, and the Predict tab loses its dataset without saying why.
    """
    from fastapi.testclient import TestClient

    from itf.api.app import app

    client = TestClient(app)
    response = client.delete("/patch-datasets/tiny-40")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "patch_dataset_in_use"
    assert detail["used_by"], "el 409 tiene que decir QUÉ runs lo referencian"


# --------------------------------------------------------------------------- #
# ④ E → F — the checkpoint describes itself
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(strict=True, reason="contrato ④: sin implementar, plan-ui.md fase 4")
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


# --------------------------------------------------------------------------- #
# ⑤ B ↔ F — the shared window   (the one that breaks silently)
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(strict=True, reason="contrato ⑤: sin implementar, plan-ui.md fase 6")
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


@pytest.mark.xfail(strict=True, reason="contrato ⑦: sin implementar, plan-ui.md fase 3")
def test_contract_07_models_and_validation_import_nothing_from_itf():
    """C must not know that A exists, and the validator must stay pure.

    The old violation: `models/builder.py` imported `NUM_BORDERS` from
    `datasets.loader` -- the network importing the source loader, just to learn
    there are 4 borders. Nobody decided that; the constant was there and the
    import worked. It is project vocabulary (G) and wants its own module.

    `itf.validation` imports nothing because it compares dicts, not objects --
    which is what lets it run in milliseconds and be called from both `train()`
    and the API (tests.md §4).
    """
    offenders = []
    for package in ("models", "validation"):
        files = sorted((SRC / "itf" / package).rglob("*.py"))
        assert files, f"itf.{package} no existe todavía (plan-ui.md fase 3)"
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("itf"):
                    offenders.append(f"{path.name}: from {node.module} import ...")
                elif isinstance(node, ast.Import):
                    offenders += [
                        f"{path.name}: import {a.name}"
                        for a in node.names
                        if a.name.startswith("itf")
                    ]
    assert offenders == []


# --------------------------------------------------------------------------- #
# ⑧ H ↔ B, C — comparability
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(strict=True, reason="contrato ⑧: sin implementar, plan-ui.md fase 2")
def test_contract_08_fingerprint_tracks_content_not_name(tmp_path):
    """A B rebuilt under the same name is not the same B, and today nothing notices.

    Without a content fingerprint a half-finished sweep is incomparable *in
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


@pytest.mark.xfail(strict=True, reason="contrato ⑧: sin implementar, plan-ui.md fase 2")
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


@pytest.mark.xfail(strict=True, reason="contrato ⑩: sin implementar, plan-ui.md fase 4")
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
