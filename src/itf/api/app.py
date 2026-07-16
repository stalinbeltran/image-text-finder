"""The REST API: one resource per domain (api.md R1).

**This layer owns HTTP and nothing else**: routes, codes, serialisation, turning
domain errors into responses. Everything else lives under `itf` and must work
without the API -- `itf-extract` is the proof.

The rule that keeps it that way, and it is mechanical: **if a function here does
not mention HTTP, it is not the API's.** The old `app.py` was 511 lines with
`_discover_datasets`, `_split_map`, `_run_source`, `_run_status` and the model
cache inside it. None of them mention HTTP. None of them are here.

Names (R2): `/sources` (A), not `/datasets` -- because `/patch-datasets` are
datasets too. And `/models` never comes back: "model" is the ambiguous word, it
means C or E depending on who is speaking (glosario.md §1).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from itf.api.errors import bad_request, conflict, not_found
from itf.api.jobs import JobQueue
from itf.api.schemas import BuildPatchDatasetBody
from itf.datasets import SourceDataset, discover_sources
from itf.patches import SPLIT_NAMES, PatchExtractConfig, SplitConfig, extract_dataset
from itf.patches.store import PatchDatasetNotFound, PatchDatasetStore
from itf.settings import Settings
from itf.training import RunStore


@dataclass(frozen=True)
class Context:
    """Everything a route needs, built once and passed in.

    Explicit rather than module-level singletons: it is what lets a test point
    the API at a temp directory instead of at the real `data/` and `runs/`
    (tests.md §7 -- a test never touches those).
    """

    settings: Settings
    patch_datasets: PatchDatasetStore
    runs: RunStore
    jobs: JobQueue


def get_context(request: Request) -> Context:
    return request.app.state.context


# ── A: /sources ───────────────────────────────────────────────────────────────


def _source_path(c: Context, source_id: str) -> Path:
    """Resolve a source id to a directory, refusing anything outside the root.

    The client sends an ID, never a path: the path is derived here. That is the
    D4 shape -- `GET /image?path=C:\\whatever` is gone, and with it the hole
    where any page you visited could enumerate your disk.

    Resolve FIRST, check AFTER: checked before resolving, `../..` walks straight
    out of the root and the check reads as if it passed.
    """
    root = c.settings.datasets_root.resolve()
    candidate = (root / source_id).resolve()
    if candidate != root and root not in candidate.parents:
        raise not_found(
            "source_not_found",
            f"no existe la fuente '{source_id}'",
            f"mira GET /sources; la raíz es {root}",
        )
    if not (candidate / "labels.jsonl").exists():
        raise not_found(
            "source_not_found",
            f"no existe la fuente '{source_id}'",
            f"mira GET /sources; la raíz es {root}",
        )
    return candidate


def _sample(c: Context, source_id: str, index: int):
    ds = SourceDataset(_source_path(c, source_id))
    for s in ds.samples():
        if s.index == index:
            return s
    raise not_found("sample_not_found", f"la fuente '{source_id}' no tiene la imagen {index}")


def register_sources(app: FastAPI) -> None:
    @app.get("/sources")
    def list_sources(c: Context = Depends(get_context)) -> dict:
        out = []
        for path in discover_sources(c.settings.datasets_root):
            ds = SourceDataset(path)
            samples = ds.samples()
            out.append(
                {
                    "id": path.relative_to(c.settings.datasets_root).as_posix(),
                    "source_id": ds.id,
                    "num_samples": len(samples),
                    "num_overlapping": sum(1 for s in samples if s.has_overlap),
                }
            )
        return {"sources": out, "root": str(c.settings.datasets_root)}

    @app.get("/sources/{source_id:path}/samples")
    def list_samples(
        source_id: str, patch_dataset: str | None = None, c: Context = Depends(get_context)
    ) -> dict:
        """The images of a source, optionally annotated with their split.

        `?patch_dataset=` is the legitimate A×B crossing (contract ⑥): it is what
        lets Predecir say "only the test images". A view, not a structural
        coupling -- the split comes from B's `split.json`, and A knows nothing.
        """
        ds = SourceDataset(_source_path(c, source_id))
        split_of: dict[int, str] = {}
        if patch_dataset:
            try:
                split_of = c.patch_datasets.split_map(patch_dataset)
            except (PatchDatasetNotFound, ValueError):
                raise not_found(
                    "patch_dataset_not_found",
                    f"no existe el dataset de patches '{patch_dataset}'",
                )
        return {
            "samples": [
                {
                    "index": s.index,
                    "width": s.width,
                    "height": s.height,
                    "has_overlap": s.has_overlap,
                    "num_blocks": len(s.blocks),
                    "split": split_of.get(s.index),
                }
                for s in ds.samples()
            ]
        }

    @app.get("/sources/{source_id:path}/samples/{index}/geometry")
    def sample_geometry(source_id: str, index: int, c: Context = Depends(get_context)) -> dict:
        """The ground truth of one image: the quads, in pixels.

        Numbers, not a rendered image -- the browser draws them. Same reasoning
        as `map_payload`: the server sends data, the client decides the colour.
        """
        s = _sample(c, source_id, index)
        return {
            "index": s.index,
            "width": s.width,
            "height": s.height,
            "has_overlap": s.has_overlap,
            "blocks": [
                {"block_id": b.block_id, "kind": b.kind, "angle": b.angle, "quad": b.quad.tolist()}
                for b in s.blocks
            ],
        }

    @app.get("/sources/{source_id:path}/samples/{index}/image")
    def sample_image(
        source_id: str,
        index: int,
        w: int | None = Query(default=None, gt=0),
        c: Context = Depends(get_context),
    ) -> Response:
        """The image itself. `?w=` for thumbnails."""
        s = _sample(c, source_id, index)
        with Image.open(s.image_path) as img:
            img = img.convert("L")
            if w and w < img.width:
                img = img.resize((w, round(img.height * w / img.width)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")


# ── B: /patch-datasets ────────────────────────────────────────────────────────


def register_patch_datasets(app: FastAPI) -> None:
    @app.get("/patch-datasets")
    def list_patch_datasets(c: Context = Depends(get_context)) -> dict:
        return {
            "patch_datasets": [
                {"name": name, "manifest": c.patch_datasets.manifest(name)}
                for name in c.patch_datasets.names()
            ]
        }

    @app.get("/patch-datasets/{name}")
    def get_patch_dataset(name: str, c: Context = Depends(get_context)) -> dict:
        try:
            manifest = c.patch_datasets.manifest(name)
        except (PatchDatasetNotFound, ValueError):
            raise not_found("patch_dataset_not_found", f"no existe el dataset de patches '{name}'")
        return {
            "name": name,
            "manifest": manifest,
            "fingerprint": manifest.get("fingerprint"),
            # What makes the 409 possible, and what makes it informative.
            "used_by": c.runs.using_patch_dataset(name),
        }

    @app.post("/patch-datasets", status_code=202)
    def build_patch_dataset(body: BuildPatchDatasetBody, c: Context = Depends(get_context)) -> dict:
        """→ job (R3: minutes, so 202 and a job, not 200 and a lie).

        Refuses to overwrite. That is the trap `POST /runs` had: `mkdir(exist_ok=
        True)` plus truncating destroys results with no warning, and a sweep that
        auto-generates names is exactly who steps on it.
        """
        try:
            target = c.patch_datasets.path(body.name)
        except ValueError as exc:
            raise bad_request("invalid_name", str(exc), "usa un nombre sin barras ni '..'")
        if c.patch_datasets.exists(body.name):
            raise conflict(
                "patch_dataset_exists",
                f"ya existe un dataset de patches llamado '{body.name}'",
                "elige otro nombre, o borra el que hay primero",
            )
        source_path = _source_path(c, body.source)
        config = PatchExtractConfig(
            source=str(source_path),
            out=str(target),
            patch_size=body.patch_size,
            stride=body.stride,
            target_kinds=body.target_kinds,
            drop_overlap=body.drop_overlap,
            split=SplitConfig(**body.split.model_dump()),
            seed=body.seed,
        )
        job = c.jobs.submit(
            "build-patch-dataset",
            lambda: extract_dataset(config)["manifest"],
            detail={"name": body.name, "source": body.source},
        )
        return job.as_dict()

    @app.delete("/patch-datasets/{name}", status_code=204)
    def delete_patch_dataset(name: str, c: Context = Depends(get_context)) -> Response:
        """409 if any run references it, and the body says WHICH (contract ③).

        Deleting a B in use does not break loudly: it leaves every run that
        pointed at it without provenance, in silence. So the check is the
        endpoint's job, and the list of runs IS the answer to "why not".
        """
        try:
            exists = c.patch_datasets.exists(name)
        except ValueError as exc:
            raise bad_request("invalid_name", str(exc))
        if not exists:
            raise not_found("patch_dataset_not_found", f"no existe el dataset de patches '{name}'")

        used_by = c.runs.using_patch_dataset(name)
        if used_by:
            raise conflict(
                "patch_dataset_in_use",
                f"'{name}' lo usan {len(used_by)} run(s): {', '.join(used_by)}",
                "borra esos runs primero, o quédate el dataset: sin él pierden su procedencia",
                used_by=used_by,
            )
        c.patch_datasets.delete(name)
        return Response(status_code=204)

    @app.get("/patch-datasets/{name}/patches/{index}")
    def get_patch(name: str, index: int, c: Context = Depends(get_context)) -> dict:
        """One patch: pixels, label, border flags and provenance.

        The real input of the CNN (contract ①), which is why this -- and not a
        whole image -- is what the feature-map view takes. Returns `sample_idx`
        and `patch_xy` too: they have been in the `.npz` all along with nobody
        reading them, and they are the patch's provenance (V15).
        """
        try:
            manifest = c.patch_datasets.manifest(name)
            data = c.patch_datasets.arrays(name)
        except (PatchDatasetNotFound, ValueError):
            raise not_found("patch_dataset_not_found", f"no existe el dataset de patches '{name}'")
        with data:
            total = int(data["X"].shape[0])
            if not 0 <= index < total:
                raise not_found("patch_not_found", f"'{name}' tiene {total} patches; pediste el {index}")
            return {
                "index": index,
                "patch": data["X"][index, :, :, 0].astype(int).tolist(),
                "label": data["y"][index].tolist(),
                "border": data["border"][index].astype(int).tolist(),
                "sample_idx": int(data["sample_idx"][index]),
                "patch_xy": data["patch_xy"][index].astype(int).tolist(),
                "split": SPLIT_NAMES[int(data["split"][index])],
                "corner_order": manifest["corner_order"],
                "border_order": manifest["border_order"],
            }


# ── X: /jobs ──────────────────────────────────────────────────────────────────


def register_jobs(app: FastAPI) -> None:
    @app.get("/jobs")
    def list_jobs(c: Context = Depends(get_context)) -> dict:
        return {"jobs": [j.as_dict() for j in c.jobs.list()], "max_workers": c.jobs.max_workers}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, c: Context = Depends(get_context)) -> dict:
        job = c.jobs.get(job_id)
        if job is None:
            raise not_found("job_not_found", f"no existe el job '{job_id}'")
        return job.as_dict()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="image-text-finder", version="0.1.0")
    app.state.context = Context(
        settings=settings,
        patch_datasets=PatchDatasetStore(settings.patch_datasets_root),
        runs=RunStore(settings.runs_root),
        # On CPU the limit is 1: torch already uses every core inside one run, so
        # N at once just fight each other and each holds its dataset in RAM.
        jobs=JobQueue(max_workers=1),
    )
    # D4: closed to the front's origin, never `*`.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_sources(app)
    register_patch_datasets(app)
    register_jobs(app)
    return app


app = create_app()
