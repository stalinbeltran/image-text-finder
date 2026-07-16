"""End-to-end API test: discover -> build patches -> train -> predict.

Uses isolated temp dirs (via ITF_* env vars) so it doesn't touch the repo's
data/ or runs/. Skipped if the source dataset or fastapi isn't available.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

SOURCE_ROOT = "C:/Desarrollo/image-text-sample-generator/data/datasets"
SOURCE_ID = "clean-paragraphs-01/reducido"

pytest.importorskip("fastapi")
pytestmark = pytest.mark.skipif(
    not Path(SOURCE_ROOT, SOURCE_ID, "labels.jsonl").exists(),
    reason="source dataset not available",
)

_TMP = tempfile.mkdtemp(prefix="itf-api-")
os.environ["ITF_DATASETS_ROOT"] = SOURCE_ROOT
os.environ["ITF_PATCH_DATASETS_DIR"] = str(Path(_TMP, "patch-datasets"))
os.environ["ITF_RUNS_DIR"] = str(Path(_TMP, "runs"))
os.environ["ITF_MODELS_DIR"] = str(Path(_TMP, "models"))

from fastapi.testclient import TestClient  # noqa: E402

from itf.api.app import app  # noqa: E402

client = TestClient(app)

SMALL_MODEL = {
    "input_size": 40,
    "in_channels": 1,
    "border_features": True,
    "backbone": [
        {"filters": 8, "kernel": 3, "batchnorm": True, "activation": "relu", "pool": 2},
        {"filters": 16, "kernel": 3, "activation": "relu", "pool": 2},
    ],
    "head": {"hidden": [32]},
}


def _wait(job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}").json()
        if r["status"] in {"done", "error"}:
            return r
        time.sleep(0.2)
    raise AssertionError("job timed out")


def test_end_to_end():
    # 1. discover source datasets
    datasets = client.get("/datasets").json()
    assert any(d["id"] == SOURCE_ID for d in datasets)

    # 2. build a patch dataset
    r = client.post("/patch-datasets", json={
        "source": SOURCE_ID, "name": "apitest", "patch_size": 40, "stride": 20,
        "split": {"train": 0.6, "val": 0.2, "test": 0.2}, "seed": 3,
    })
    job = _wait(r.json()["id"])
    assert job["status"] == "done", job.get("error")
    assert client.get("/patch-datasets/apitest").json()["manifest"]["num_patches"] > 0

    # 3. train a tiny run
    r = client.post("/runs", json={
        "data": "apitest", "name": "apirun", "model": SMALL_MODEL,
        "epochs": 2, "batch_size": 32, "device": "cpu",
    })
    job = _wait(r.json()["id"])
    assert job["status"] == "done", job.get("error")

    run = client.get("/runs/apirun").json()
    assert run["status"] == "done"
    assert len(run["metrics"]) == 2
    assert "best.pt" in run["checkpoints"]

    # 4. predict on a real image from the source dataset
    img = Path(SOURCE_ROOT, SOURCE_ID, "images", "000000.png")
    with img.open("rb") as fh:
        r = client.post("/predict", data={"run": "apirun", "checkpoint": "best", "threshold": "0.5"},
                        files={"file": ("000000.png", fh, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image_size"] == [160, 160]
    assert "corners" in body and "paragraphs" in body
