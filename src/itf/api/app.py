"""FastAPI app exposing the full pipeline (component 3).

Endpoints
    GET  /datasets                 discover source datasets (SAMPLE_FORMAT)
    GET  /datasets/{id}            source dataset metadata
    GET  /patch-datasets           list built patch datasets
    POST /patch-datasets           build one (background job)
    GET  /patch-datasets/{name}    manifest of a built patch dataset
    GET  /models                   list saved model configs
    POST /models                   save a model config
    GET  /models/{name}            a saved model config
    GET  /runs                     list training runs (+ live status)
    POST /runs                     start a training run (background job)
    GET  /runs/{name}              run config + per-epoch metrics
    GET  /jobs                     list background jobs
    GET  /jobs/{id}                a background job
    POST /predict                  run inference on an uploaded image
"""

from __future__ import annotations

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from itf.api import settings
from itf.api.jobs import JOBS
from itf.api.schemas import BuildPatchRequest, SaveModelRequest, TrainRequest
from itf.datasets.loader import SourceDataset
from itf.inference.predict import load_model, predict_image
from itf.patches.extract import PatchExtractConfig, SplitConfig, extract_dataset
from itf.training.loop import RunConfig, train

@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_dirs()
    yield


app = FastAPI(title="image-text-finder API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _resolve_source(source: str) -> Path:
    p = Path(source)
    if p.is_absolute() and (p / "labels.jsonl").exists():
        return p
    cand = settings.DATASETS_ROOT / source
    if (cand / "labels.jsonl").exists():
        return cand
    raise HTTPException(404, f"source dataset not found: {source}")


def _discover_datasets() -> list[dict]:
    out = []
    root = settings.DATASETS_ROOT
    if not root.exists():
        return out
    for labels in root.rglob("labels.jsonl"):
        d = labels.parent
        rel = d.relative_to(root).as_posix()
        meta = {}
        mp = d / "dataset.json"
        if mp.exists():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        out.append({
            "id": rel,
            "name": meta.get("name", d.name),
            "count": meta.get("count"),
            "path": str(d),
        })
    return sorted(out, key=lambda x: x["id"])


def _read_manifest(dir_: Path) -> dict | None:
    mp = dir_ / "manifest.json"
    if mp.exists():
        return json.loads(mp.read_text(encoding="utf-8"))
    return None


def _read_metrics(run_dir: Path) -> list[dict]:
    mp = run_dir / "metrics.jsonl"
    if not mp.exists():
        return []
    return [json.loads(line) for line in mp.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_status(run_dir: Path) -> str:
    if (run_dir / "summary.json").exists():
        return "done"
    if (run_dir / "config.json").exists():
        return "running"
    return "unknown"


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
@app.get("/datasets")
def list_datasets() -> list[dict]:
    return _discover_datasets()


@app.get("/datasets/{dataset_id:path}")
def get_dataset(dataset_id: str) -> dict:
    d = _resolve_source(dataset_id)
    ds = SourceDataset(d)
    samples = ds.samples()
    return {
        "id": dataset_id,
        "meta": ds.meta,
        "num_samples": len(samples),
        "example": {
            "width": samples[0].width, "height": samples[0].height,
            "num_paragraphs": len(samples[0].blocks_of_kind(["paragraph"])),
        } if samples else None,
    }


# --------------------------------------------------------------------------- #
# patch datasets
# --------------------------------------------------------------------------- #
@app.get("/patch-datasets")
def list_patch_datasets() -> list[dict]:
    out = []
    for d in sorted(settings.PATCH_DATASETS_DIR.glob("*")):
        if (d / "patches.npz").exists():
            out.append({"name": d.name, "manifest": _read_manifest(d)})
    return out


@app.post("/patch-datasets")
def build_patch_dataset(req: BuildPatchRequest) -> dict:
    source = _resolve_source(req.source)
    out_dir = settings.PATCH_DATASETS_DIR / req.name
    config = PatchExtractConfig(
        source=str(source), out=str(out_dir),
        patch_size=req.patch_size, stride=req.stride,
        target_kinds=req.target_kinds, drop_overlap=req.drop_overlap,
        split=SplitConfig(**req.split.model_dump()), seed=req.seed,
    )
    job = JOBS.submit("extract", lambda: extract_dataset(config)["manifest"],
                      meta={"name": req.name})
    return job.as_dict()


@app.get("/patch-datasets/{name}")
def get_patch_dataset(name: str) -> dict:
    d = settings.PATCH_DATASETS_DIR / name
    manifest = _read_manifest(d)
    if manifest is None:
        raise HTTPException(404, f"patch dataset not found: {name}")
    return {"name": name, "manifest": manifest}


# --------------------------------------------------------------------------- #
# model configs
# --------------------------------------------------------------------------- #
@app.get("/models")
def list_models() -> list[dict]:
    out = []
    for f in sorted(settings.MODELS_DIR.glob("*.yaml")):
        out.append({"name": f.stem, "model": yaml.safe_load(f.read_text(encoding="utf-8"))})
    return out


@app.post("/models")
def save_model(req: SaveModelRequest) -> dict:
    settings.ensure_dirs()
    f = settings.MODELS_DIR / f"{req.name}.yaml"
    f.write_text(yaml.safe_dump(req.model.model_dump(), sort_keys=False), encoding="utf-8")
    return {"name": req.name, "path": str(f)}


@app.get("/models/{name}")
def get_model(name: str) -> dict:
    f = settings.MODELS_DIR / f"{name}.yaml"
    if not f.exists():
        raise HTTPException(404, f"model config not found: {name}")
    return {"name": name, "model": yaml.safe_load(f.read_text(encoding="utf-8"))}


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
@app.get("/runs")
def list_runs() -> list[dict]:
    out = []
    for d in sorted(settings.RUNS_DIR.glob("*")):
        if not d.is_dir():
            continue
        metrics = _read_metrics(d)
        out.append({
            "name": d.name,
            "status": _run_status(d),
            "epochs_done": len(metrics),
            "last": metrics[-1] if metrics else None,
        })
    return out


@app.post("/runs")
def start_run(req: TrainRequest) -> dict:
    data_dir = settings.PATCH_DATASETS_DIR / req.data
    if not (data_dir / "patches.npz").exists():
        # allow absolute path too
        p = Path(req.data)
        if (p / "patches.npz").exists() or p.suffix == ".npz":
            data_dir = p
        else:
            raise HTTPException(404, f"patch dataset not found: {req.data}")
    out_dir = settings.RUNS_DIR / req.name
    config = RunConfig(
        data=str(data_dir), out=str(out_dir), model=req.model.model_dump(),
        epochs=req.epochs, batch_size=req.batch_size, lr=req.lr,
        optimizer=req.optimizer, weight_decay=req.weight_decay,
        lambda_pos=req.lambda_pos, pos_weight=req.pos_weight,
        seed=req.seed, device=req.device, num_workers=req.num_workers,
    )
    job = JOBS.submit("train", lambda: train(config), meta={"name": req.name})
    return job.as_dict()


@app.get("/runs/{name}")
def get_run(name: str) -> dict:
    d = settings.RUNS_DIR / name
    if not d.exists():
        raise HTTPException(404, f"run not found: {name}")
    config = json.loads((d / "config.json").read_text(encoding="utf-8")) if (d / "config.json").exists() else None
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else None
    return {
        "name": name,
        "status": _run_status(d),
        "config": config,
        "metrics": _read_metrics(d),
        "summary": summary,
        "checkpoints": [p.name for p in d.glob("*.pt")],
    }


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #
@app.get("/jobs")
def list_jobs() -> list[dict]:
    return [j.as_dict() for j in JOBS.list()]


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"job not found: {job_id}")
    return job.as_dict()


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
@app.post("/predict")
def predict(
    run: str = Form(...),
    checkpoint: str = Form("best"),
    threshold: float = Form(0.5),
    stride: int | None = Form(None),
    device: str = Form("cpu"),
    file: UploadFile = File(...),
) -> dict:
    ckpt = settings.RUNS_DIR / run / f"{checkpoint}.pt"
    if not ckpt.exists():
        raise HTTPException(404, f"checkpoint not found: {run}/{checkpoint}.pt")
    model = load_model(ckpt, device=device)
    image = np.asarray(Image.open(io.BytesIO(file.file.read())).convert("L"), dtype=np.uint8)
    result = predict_image(model, image, stride=stride, threshold=threshold, device=device)
    result["run"] = run
    result["checkpoint"] = checkpoint
    return result
