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
import shutil
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from itf.api import settings
from itf.api.jobs import JOBS
from itf.api.schemas import (
    BuildPatchRequest,
    PredictPathRequest,
    RenameRunRequest,
    RetrainRequest,
    SaveModelRequest,
    TrainRequest,
)
from itf.datasets.loader import SourceDataset
from itf.inference.predict import load_model, predict_image
from itf.patches.extract import PatchExtractConfig, SplitConfig, extract_dataset
from itf.training.loop import RunConfig, train

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff"}

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
# model cache + image serving (batch inference over folders / subsets)
# --------------------------------------------------------------------------- #
_MODEL_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_MODEL_CACHE_MAX = 4


def _get_model(run: str, checkpoint: str, device: str):
    """Load a checkpoint, caching by (run, checkpoint, device, mtime) so batch
    prediction over many images does not re-read the model from disk each time."""
    ckpt = settings.RUNS_DIR / run / f"{checkpoint}.pt"
    if not ckpt.exists():
        raise HTTPException(404, f"checkpoint not found: {run}/{checkpoint}.pt")
    key = (run, checkpoint, device, ckpt.stat().st_mtime_ns)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = load_model(ckpt, device=device)
        _MODEL_CACHE[key] = model
        while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
            _MODEL_CACHE.popitem(last=False)
    else:
        _MODEL_CACHE.move_to_end(key)
    return model


def _drop_model_cache(run: str) -> None:
    for key in [k for k in _MODEL_CACHE if k[0] == run]:
        _MODEL_CACHE.pop(key, None)


def _resolve_image_path(path: str) -> Path:
    p = Path(path)
    if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(404, f"image not found: {path}")
    return p


def _image_response(p: Path, width: int | None) -> Response:
    """Serve an image, optionally downscaled to ``width`` px (for thumbnails)."""
    with Image.open(p) as im:
        im = im.convert("L")
        if width and width < im.width:
            h = max(1, round(im.height * width / im.width))
            im = im.resize((width, h))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


def _dataset_samples(dataset_id: str) -> list[dict]:
    """Lightweight per-sample listing (index, absolute image path, size, #paragraphs)."""
    d = _resolve_source(dataset_id)
    ds = SourceDataset(d)
    out = []
    for s in ds:
        out.append({
            "index": s.index,
            "name": s.image_path.name,
            "path": str(s.image_path),
            "width": s.width,
            "height": s.height,
            "num_paragraphs": len(s.blocks_of_kind(["paragraph"])),
        })
    return out


def _split_map(patch_dataset: str) -> dict[int, str]:
    """sample index -> split name ('train'|'val'|'test') for a built patch dataset."""
    sp = settings.PATCH_DATASETS_DIR / patch_dataset / "split.json"
    if not sp.exists():
        raise HTTPException(404, f"split not found for patch dataset: {patch_dataset}")
    data = json.loads(sp.read_text(encoding="utf-8"))
    return {int(i): split for split, idxs in data.items() for i in idxs}


def _run_source(run: str) -> tuple[str | None, str | None]:
    """Resolve (patch_dataset_name, source_dataset_id) a run was trained on."""
    cfg_path = settings.RUNS_DIR / run / "config.json"
    if not cfg_path.exists():
        return None, None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    patch_dir = Path(cfg.get("data", ""))
    patch_name = patch_dir.name if patch_dir.name else None
    source_id = None
    manifest = _read_manifest(patch_dir) if patch_dir.exists() else None
    if manifest:
        src = Path(manifest.get("config", {}).get("source", ""))
        root = settings.DATASETS_ROOT
        try:
            source_id = src.resolve().relative_to(root).as_posix()
        except ValueError:
            source_id = src.name or None
    return patch_name, source_id


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
@app.get("/datasets")
def list_datasets() -> list[dict]:
    return _discover_datasets()


# NOTE: the ``/samples`` route must be declared before the greedy
# ``{dataset_id:path}`` catch-all, or the latter would swallow ".../samples".
@app.get("/datasets/{dataset_id:path}/samples")
def list_dataset_samples(dataset_id: str, patch_dataset: str | None = None) -> dict:
    """Per-sample thumbnails source. When ``patch_dataset`` is given, each sample
    is annotated with its split (train/val/test) so subsets can be filtered."""
    samples = _dataset_samples(dataset_id)
    splits: dict[str, int] = {}
    if patch_dataset:
        smap = _split_map(patch_dataset)
        for s in samples:
            s["split"] = smap.get(s["index"])
        for s in samples:
            if s.get("split"):
                splits[s["split"]] = splits.get(s["split"], 0) + 1
    return {"dataset": dataset_id, "count": len(samples),
            "splits": splits or None, "samples": samples}


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
# images & folders (batch inference sources + thumbnails)
# --------------------------------------------------------------------------- #
@app.get("/image")
def serve_image(path: str = Query(...), w: int | None = Query(None)) -> Response:
    """Serve any local image by absolute path, optionally downscaled to ``w`` px."""
    return _image_response(_resolve_image_path(path), w)


@app.get("/folder")
def list_folder(path: str = Query(...)) -> dict:
    """List image files in an arbitrary local folder (non-recursive)."""
    d = Path(path)
    if not d.is_dir():
        raise HTTPException(404, f"folder not found: {path}")
    items = [
        {"name": f.name, "path": str(f)}
        for f in sorted(d.iterdir())
        if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
    ]
    return {"path": str(d), "count": len(items), "images": items}


@app.get("/runs/{name}/source")
def get_run_source(name: str) -> dict:
    """Which patch dataset + source dataset (with subsets) a run was trained on."""
    if not (settings.RUNS_DIR / name).exists():
        raise HTTPException(404, f"run not found: {name}")
    patch_dataset, source = _run_source(name)
    return {"run": name, "patch_dataset": patch_dataset, "source": source}


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


@app.patch("/runs/{name}")
def rename_run(name: str, req: RenameRunRequest) -> dict:
    d = settings.RUNS_DIR / name
    if not d.exists():
        raise HTTPException(404, f"run not found: {name}")
    if _run_status(d) == "running":
        raise HTTPException(409, f"cannot rename an active run: {name}")
    target = settings.RUNS_DIR / req.new_name
    if target.exists():
        raise HTTPException(409, f"run already exists: {req.new_name}")
    _drop_model_cache(name)
    d.rename(target)
    return {"renamed": name, "to": req.new_name}


@app.delete("/runs/{name}")
def delete_run(name: str, force: bool = False) -> dict:
    d = settings.RUNS_DIR / name
    if not d.exists():
        raise HTTPException(404, f"run not found: {name}")
    if _run_status(d) == "running" and not force:
        raise HTTPException(409, f"run '{name}' looks active; pass force=true to delete it")
    _drop_model_cache(name)
    shutil.rmtree(d)
    return {"deleted": name}


@app.post("/runs/{name}/retrain")
def retrain_run(name: str, req: RetrainRequest) -> dict:
    """Start a new run reusing an existing run's config/model, with overrides."""
    cfg_path = settings.RUNS_DIR / name / "config.json"
    if not cfg_path.exists():
        raise HTTPException(404, f"run config not found: {name}")
    out_dir = settings.RUNS_DIR / req.name
    if out_dir.exists():
        raise HTTPException(409, f"run already exists: {req.name}")
    base = json.loads(cfg_path.read_text(encoding="utf-8"))
    overrides = {k: v for k, v in {
        "epochs": req.epochs, "batch_size": req.batch_size, "lr": req.lr,
        "optimizer": req.optimizer, "lambda_pos": req.lambda_pos, "device": req.device,
    }.items() if v is not None}
    base.update(overrides)
    base["out"] = str(out_dir)
    config = RunConfig.from_dict(base)
    job = JOBS.submit("train", lambda: train(config), meta={"name": req.name})
    return job.as_dict()


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
    model = _get_model(run, checkpoint, device)
    image = np.asarray(Image.open(io.BytesIO(file.file.read())).convert("L"), dtype=np.uint8)
    result = predict_image(model, image, stride=stride, threshold=threshold, device=device)
    result["run"] = run
    result["checkpoint"] = checkpoint
    return result


@app.post("/predict-path")
def predict_path(req: PredictPathRequest) -> dict:
    """Run inference on an image already on disk (no re-upload) — used by the
    thumbnail grid to evaluate folders / dataset subsets image by image."""
    p = _resolve_image_path(req.path)
    model = _get_model(req.run, req.checkpoint, req.device)
    image = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
    result = predict_image(model, image, stride=req.stride,
                           threshold=req.threshold, device=req.device)
    result["run"] = req.run
    result["checkpoint"] = req.checkpoint
    result["path"] = str(p)
    return result
