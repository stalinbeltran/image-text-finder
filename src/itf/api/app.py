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
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from itf.api.errors import bad_request, conflict, not_found, problem
from itf.api.jobs import JobQueue
from itf.api.schemas import (
    BuildPatchDatasetBody,
    CreateRunBody,
    CreateSweepBody,
    FeatureMapsBody,
    NamedNetworkBody,
    NetworkBody,
    PredictBody,
    RecipeBody,
    RenameRunBody,
    ResizeSourceBody,
    WindowBody,
)
from itf.datasets import SourceDataset
from itf.datasets.index import SourceIndex, count_lines, load_index, peek_index
from itf.datasets.resize import ResizeRequest, check_resize, resize_source, source_sizes
from itf.datasets.roots import list_ids, source_roots, split_id
from itf.diagnostics import (
    DEFAULT_CHECKPOINT,
    NotMeasurable,
    TableCache,
    coactivation,
    error_map,
    open_diagnostics,
    pr,
    rows,
)
from itf.inference import (
    ModelCache,
    NotInspectable,
    border_test,
    deconvolution,
    feature_maps,
    kernels,
    occlusion,
    predict_image,
    window_prediction,
)
from itf.models import (
    NetworkConfig,
    NetworkNotFound,
    NetworkStore,
    build_model,
    count_params,
    flat_features,
    spatial_trace,
)
from itf.patches import SPLIT_NAMES, PatchExtractConfig, SplitConfig, extract_dataset
from itf.patches.rows import load_rows
from itf.patches.store import PatchDatasetNotFound, PatchDatasetStore
from itf.settings import Settings
from itf.sweeps import (
    SweepExists,
    SweepNotFound,
    SweepSpec,
    SweepStore,
    check_sweep,
)
from itf.training import (
    LIVE_STATES,
    Recipe,
    RecipeNotFound,
    RecipeStore,
    RunExists,
    RunNotFound,
    RunStore,
    build_provenance,
)
from itf.training.loop import RunSpec, frozen_config, train
from itf.validation import check_run


@dataclass(frozen=True)
class Context:
    """Everything a route needs, built once and passed in.

    Explicit rather than module-level singletons: it is what lets a test point
    the API at a temp directory instead of at the real `data/` and `runs/`
    (tests.md §7 -- a test never touches those).
    """

    settings: Settings
    patch_datasets: PatchDatasetStore
    networks: NetworkStore
    recipes: RecipeStore
    runs: RunStore
    sweeps: SweepStore
    jobs: JobQueue
    diagnostics: TableCache
    #: Loaded networks, keyed on the checkpoint's mtime. V2 and the predict
    #: sliders repaint live, and rebuilding the network on every tick is what
    #: makes a live view feel broken.
    models: ModelCache


def get_context(request: Request) -> Context:
    return request.app.state.context


# ── A: /sources ───────────────────────────────────────────────────────────────


def _source_roots(c: Context) -> tuple[tuple[str, Path], ...]:
    """The two source roots (D19), resolved. The mapping itself is `itf.datasets.roots`."""
    return source_roots(c.settings.datasets_root.resolve(), c.settings.derived_sources_root.resolve())


def _source_path(c: Context, source_id: str) -> Path:
    """Resolve a source id to a directory, refusing anything outside the root.

    The client sends an ID, never a path: the path is derived here. That is the
    D4 shape -- `GET /image?path=C:\\whatever` is gone, and with it the hole
    where any page you visited could enumerate your disk.

    Resolve FIRST, check AFTER: checked before resolving, `../..` walks straight
    out of the root and the check reads as if it passed.
    """
    root, rel = split_id(source_id, _source_roots(c))
    candidate = (root / rel).resolve()
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


def _source_index(c: Context, ds: SourceDataset) -> SourceIndex:
    """The offset index of a source, built on first use and then cached on disk.

    Building it reads `labels.jsonl` whole (30 s on a 522 MB source), so this is
    the one place allowed to pay that -- and it pays it once per version of the
    file, not once per request.
    """
    return load_index(ds.labels_path, c.settings.sources_index_cache_root)


def _sample(c: Context, source_id: str, index: int):
    """One image. **One seek and one 26 KB parse**, not a pass over the file.

    This used to iterate `ds.samples()` until it found the index, which parses
    the entire source -- and it is called once per thumbnail, per predict and
    per scrubber drag. On a 20 000-image source a single gallery was minutes of
    CPU, which is why the screen looked dead rather than slow.
    """
    ds = SourceDataset(_source_path(c, source_id))
    entry = _source_index(c, ds).get(index)
    if entry is None:
        raise not_found("sample_not_found", f"la fuente '{source_id}' no tiene la imagen {index}")
    return ds.sample_at(entry.offset)


def register_sources(app: FastAPI) -> None:
    @app.get("/sources")
    def list_sources(c: Context = Depends(get_context)) -> dict:
        """Every source, from both roots. Derived ones say so (D19)."""
        out = []
        for source_id, path in list_ids(_source_roots(c)):
            ds = SourceDataset(path)
            # The picker must stay cheap. Counting images is a scan for
            # newlines (~1 s on 522 MB); knowing how many OVERLAP means parsing
            # every line, so it is reported only when the index already exists.
            # `null` and not `0`: absent is not zero (formatos.md §2) -- zero
            # overlapping images is a fact about the data, and this is not it.
            cached = peek_index(ds.labels_path, c.settings.sources_index_cache_root)
            out.append(
                {
                    "id": source_id,
                    "source_id": ds.id,
                    "num_samples": len(cached) if cached else count_lines(ds.labels_path),
                    "num_overlapping": cached.num_overlapping if cached else None,
                    # Absent means an ORIGINAL, not "a derived one we lost track
                    # of" (formatos.md §4.6). The screen can say which parent and
                    # which scale instead of guessing from a name.
                    "derived": ds.meta.get("derived"),
                }
            )
        return {
            "sources": out,
            "root": str(c.settings.datasets_root),
            "derived_root": str(c.settings.derived_sources_root),
        }

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
        # From the index, not from `ds.samples()`: a listing needs five scalars
        # per image and none of the geometry, and the geometry is 99 % of the
        # bytes. `GET .../geometry` is where the quads are, one image at a time.
        return {
            "samples": [
                {
                    "index": s.index,
                    "width": s.width,
                    "height": s.height,
                    "has_overlap": s.has_overlap,
                    "num_blocks": s.num_blocks,
                    "split": split_of.get(s.index),
                }
                for s in _source_index(c, ds).entries
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

    @app.post("/sources/{source_id:path}/resize", status_code=202)
    def resize_source_endpoint(
        source_id: str, body: ResizeSourceBody, c: Context = Depends(get_context)
    ) -> dict:
        """→ job. The only write in all of `/sources`, and it writes elsewhere (D19).

        The parent is never touched: A is external and read-only, so the output
        goes to the local derived root and comes back with a `derived/` id.

        Every refusal happens HERE, before the job exists. `check_resize` is pure
        and reads only `labels.jsonl` -- no image is opened -- so "this would
        upscale 3 of your 500 samples" costs milliseconds instead of being
        discovered by sample 400 with 399 files already on disk.
        """
        src = _source_path(c, source_id)
        req = ResizeRequest(name=body.name, to_width=body.width, to_height=body.height)

        if "/" in body.name or "\\" in body.name or body.name in (".", ".."):
            raise bad_request(
                "invalid_name",
                f"'{body.name}' no vale como nombre de fuente",
                "usa un nombre sin barras ni '..'",
            )

        problems = check_resize(source_sizes(src), req)
        if problems:
            first = problems[0]
            raise bad_request(first["code"], first["message"], first["hint"], problems=problems)

        dst = c.settings.derived_sources_root / body.name
        if dst.exists():
            raise conflict(
                "source_exists",
                f"ya existe una fuente derivada llamada '{body.name}'",
                "elige otro nombre, o borra la que hay primero",
            )

        # Cooperative cancellation, the queue's shape: `cancel` ASKS, the work
        # function checks between samples and cuts where nothing is half-written.
        stop = threading.Event()
        job = c.jobs.submit(
            "resize-source",
            lambda: resize_source(src, dst, req, source_id=source_id, should_stop=stop.is_set),
            detail={"name": body.name, "source": source_id, "request": req.as_json()},
            cancel=stop.set,
        )
        return job.as_dict()


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
            rows = load_rows(c.patch_datasets.npz_path(name), c.settings.patch_rows_cache_root)
        except (PatchDatasetNotFound, ValueError):
            raise not_found("patch_dataset_not_found", f"no existe el dataset de patches '{name}'")
        total = len(rows)
        if not 0 <= index < total:
            raise not_found("patch_not_found", f"'{name}' tiene {total} patches; pediste el {index}")
        row = rows.row(index)
        return {
            "index": index,
            "patch": row["X"][:, :, 0].astype(int).tolist(),
            "label": row["y"].tolist(),
            "border": row["border"].astype(int).tolist(),
            "sample_idx": int(row["sample_idx"]),
            "patch_xy": row["patch_xy"].astype(int).tolist(),
            "split": SPLIT_NAMES[int(row["split"])],
            "corner_order": manifest["corner_order"],
            "border_order": manifest["border_order"],
        }


# ── C: /networks ──────────────────────────────────────────────────────────────


def _describe_network(config: dict) -> dict:
    """Spatial trace + parameter count. Pure, synchronous, cheap.

    `POST /networks/validate` makes the check a FUNCTION OF THE API rather than a
    side effect of training: the old flow only told you a layer did not fit by
    exploding inside a job.
    """
    cfg = NetworkConfig.from_dict(config)
    trace = spatial_trace(cfg)
    model = build_model(cfg)
    return {
        "valid": True,
        "trace": trace,
        "num_params": count_params(model),
        # The FLATTENED CONV features -- not what the head receives. With
        # `border_features` the head gets flat + 4, and reporting that number
        # under the name `flat_features` would quietly be wrong by 4.
        "flat_features": flat_features(cfg),
    }


def register_networks(app: FastAPI) -> None:
    @app.get("/networks")
    def list_networks(c: Context = Depends(get_context)) -> dict:
        return {"networks": [{"name": n, "config": c.networks.get(n)} for n in c.networks.names()]}

    @app.get("/networks/{name}")
    def get_network(name: str, c: Context = Depends(get_context)) -> dict:
        try:
            return {"name": name, "config": c.networks.get(name)}
        except (NetworkNotFound, ValueError):
            raise not_found("network_not_found", f"no existe la red '{name}'")

    @app.post("/networks/validate")
    def validate_network(body: NetworkBody) -> dict:
        """Does this architecture even fit? Synchronous, no saving, milliseconds.

        Feeds the Redes screen live. A network that does not fit answers 400 with
        WHICH layer, at what size, and how to fix it.
        """
        try:
            return _describe_network(body.model_dump())
        except ValueError as exc:
            raise bad_request("layer_does_not_fit", str(exc), "baja el pool o el stride, o sube input_size")

    @app.post("/networks", status_code=201)
    def create_network(body: NamedNetworkBody, c: Context = Depends(get_context)) -> dict:
        payload = body.model_dump()
        name = payload.pop("name")
        try:
            if c.networks.exists(name):
                raise conflict("network_exists", f"ya existe una red llamada '{name}'", "elige otro nombre")
        except ValueError as exc:
            raise bad_request("invalid_name", str(exc))
        # It must not be saveable if it does not build: a stored network that
        # explodes at training time is the failure this endpoint exists to stop.
        try:
            _describe_network(payload)
        except ValueError as exc:
            raise bad_request("layer_does_not_fit", str(exc), "baja el pool o el stride, o sube input_size")
        c.networks.save(name, payload)
        return {"name": name, "config": c.networks.get(name)}

    @app.delete("/networks/{name}", status_code=204)
    def delete_network(name: str, c: Context = Depends(get_context)) -> Response:
        try:
            c.networks.delete(name)
        except (NetworkNotFound, ValueError):
            raise not_found("network_not_found", f"no existe la red '{name}'")
        return Response(status_code=204)


# ── D: /recipes ───────────────────────────────────────────────────────────────


def register_recipes(app: FastAPI) -> None:
    @app.get("/recipes")
    def list_recipes(c: Context = Depends(get_context)) -> dict:
        return {"recipes": [{"name": n, "recipe": c.recipes.get(n).as_dict()} for n in c.recipes.names()]}

    @app.get("/recipes/{name}")
    def get_recipe(name: str, c: Context = Depends(get_context)) -> dict:
        try:
            return {"name": name, "recipe": c.recipes.get(name).as_dict()}
        except (RecipeNotFound, ValueError):
            raise not_found("recipe_not_found", f"no existe la receta '{name}'")

    @app.post("/recipes", status_code=201)
    def create_recipe(body: RecipeBody, c: Context = Depends(get_context)) -> dict:
        payload = body.model_dump()
        name = payload.pop("name")
        try:
            if c.recipes.exists(name):
                raise conflict("recipe_exists", f"ya existe una receta llamada '{name}'", "elige otro nombre")
        except ValueError as exc:
            raise bad_request("invalid_name", str(exc))
        try:
            recipe = Recipe.from_dict(payload)
        except ValueError as exc:
            raise bad_request("invalid_recipe", str(exc))
        c.recipes.save(name, recipe)
        return {"name": name, "recipe": recipe.as_dict()}

    @app.delete("/recipes/{name}", status_code=204)
    def delete_recipe(name: str, c: Context = Depends(get_context)) -> Response:
        try:
            c.recipes.delete(name)
        except (RecipeNotFound, ValueError):
            raise not_found("recipe_not_found", f"no existe la receta '{name}'")
        return Response(status_code=204)


# ── E: /runs ──────────────────────────────────────────────────────────────────


def _run_or_404(c: Context, name: str) -> None:
    try:
        exists = c.runs.exists(name)
    except ValueError as exc:
        raise bad_request("invalid_name", str(exc))
    if not exists:
        raise not_found("run_not_found", f"no existe el run '{name}'", "mira GET /runs")


#: Everything a half-written or half-deleted run can throw at a reader. A run is
#: being written WHILE this list is being read -- `write_text` is not atomic, and
#: a delete can land between `names()` and the reads below -- so a torn file is
#: normal, not corruption.
_UNREADABLE = (RunNotFound, json.JSONDecodeError, OSError)


def _run_summary_row(c: Context, name: str) -> dict:
    """One row of `GET /runs`. **No metrics** (R5): state, provenance, aggregates.

    `seconds_per_epoch` is what Entrenar estimates the cost of a new run with. It
    is an aggregate, so it is computed here rather than by shipping the epoch
    records to the browser to average (R6) -- and it is `null`, never 0, when the
    run has not finished an epoch: 0 would read as "instant".

    **Every read here is guarded, not just the config.** One unreadable run must
    degrade to one bad row, never to a 500 -- because a 500 here is not "one run
    is broken", it is *the Runs screen shows nothing at all, for every run*.
    """
    row: dict = {
        "name": name,
        "state": "error",
        "provenance": None,
        "seconds_per_epoch": None,
        "summary": None,
    }
    try:
        row["state"] = c.runs.status(name).get("state")
        row["seconds_per_epoch"] = c.runs.seconds_per_epoch(name)
        row["summary"] = c.runs.summary(name)
        row["provenance"] = c.runs.config(name).get("provenance")
    except _UNREADABLE as exc:
        row["error"] = f"el run no se puede leer entero: {type(exc).__name__}"
        return row
    if row["provenance"] is None:
        # **Said out loud, not degraded.** A run with no provenance is not a
        # legacy case to read around: D3 killed the degrading reader, so every
        # run born from fase 4 on has the block (formatos.md §4.2). One from
        # before it -- fase 3's own verification left exactly one -- cannot say
        # which C or D it came from, and nothing can recover that. Showing it as
        # a run like any other is what would be dishonest: it is not comparable
        # with anything.
        row["error"] = (
            "este run no tiene procedencia: es anterior a la fase 4, así que no puede decir "
            "de qué red ni de qué receta salió. No es comparable: bórralo y reentrénalo."
        )
    return row


def register_runs(app: FastAPI) -> None:
    @app.get("/runs")
    def list_runs(c: Context = Depends(get_context)) -> dict:
        return {"runs": [_run_summary_row(c, name) for name in c.runs.names()]}

    @app.post("/runs", status_code=202)
    def create_run(body: CreateRunBody, c: Context = Depends(get_context)) -> dict:
        """→ job (R3: training is minutes to hours).

        **This is where the API earns its salary** (api.md §4). Everything it
        refuses here, it refuses in milliseconds and BEFORE the job exists. The
        old code validated nothing: a `patch_size` mismatch showed up half an
        hour later, inside the job thread, as `mat1 and mat2 shapes cannot be
        multiplied` -- a message that says nothing about the actual problem.
        """
        try:
            taken = c.runs.exists(body.name)
        except ValueError as exc:
            raise bad_request("invalid_name", str(exc), "usa un nombre sin barras ni '..'")
        if taken:
            # The trap, and it is not hypothetical: `mkdir(exist_ok=True)` plus
            # truncating `metrics.jsonl` destroyed a finished run without a word,
            # and a sweep that auto-generates names is exactly who steps on it.
            raise conflict(
                "run_exists",
                f"ya existe un run llamado '{body.name}'",
                "elige otro nombre, o borra ese run primero: no se sobrescribe nunca",
            )

        try:
            network = c.networks.get(body.network)
        except (NetworkNotFound, ValueError):
            raise not_found("network_not_found", f"no existe la red '{body.network}'")
        try:
            recipe = c.recipes.get(body.recipe)
        except RecipeNotFound:
            raise not_found("recipe_not_found", f"no existe la receta '{body.recipe}'")
        except ValueError as exc:
            # The recipe exists but does not parse -- a hand-edited YAML with a
            # typo'd field. "It does not exist" would send you looking in the
            # wrong place (R4: an error says why).
            raise bad_request(
                "invalid_recipe",
                f"la receta '{body.recipe}' no es válida: {exc}",
                f"arregla {c.recipes.path(body.recipe).name}, o guárdala otra vez desde Recetas",
            )
        try:
            manifest = c.patch_datasets.manifest(body.patch_dataset)
        except (PatchDatasetNotFound, ValueError):
            raise not_found(
                "patch_dataset_not_found",
                f"no existe el dataset de patches '{body.patch_dataset}'",
            )

        # `format_version` is the FILE's, not the network's (formatos.md §4.3).
        # Frozen inside the run it would fossilise in the checkpoint and in the
        # provenance, where it means nothing.
        network.pop("format_version", None)

        # Contracts ① and ②, in one call, before the job exists -- and before the
        # name is reserved, so a refusal leaves nothing behind. A pure function of
        # two dicts: no torch, milliseconds (organizacion.md §2). `itf-train` asks
        # the very same function, which is what keeps the two doors identical.
        problems = check_run(manifest, network)
        if problems:
            # `code` is the first problem's, so the UI can switch on the usual
            # single-reason case exactly as R4 describes; `problems` carries all
            # of them, because a dataset that mismatches on both `patch_size` and
            # `border_features` should be one 400 with two lines, not two round
            # trips (itf.validation returns them all for that reason).
            raise bad_request(
                problems[0]["code"],
                "; ".join(p["message"] for p in problems),
                problems[0]["hint"],
                problems=problems,
            )

        spec = RunSpec(
            data=str(c.patch_datasets.path(body.patch_dataset)),
            out=str(c.runs.path(body.name)),
            network=network,
            provenance=build_provenance(
                patch_dataset={
                    "name": body.patch_dataset,
                    "fingerprint": manifest["fingerprint"],
                },
                network={"name": body.network, "value": network},
                recipe={"name": body.recipe, "value": recipe.as_dict()},
                sweep=None,
            ),
            recipe=recipe,
            device=body.device,
            num_workers=body.num_workers,
        )

        try:
            # The same gate `itf-train` passes through. Reserving BEFORE queueing
            # is what makes the name taken from this instant: two POSTs racing for
            # one name give a 202 and a 409, never two runs writing one directory.
            c.runs.create(body.name, frozen_config(spec))
        except RunExists:
            raise conflict("run_exists", f"ya existe un run llamado '{body.name}'")

        def run_training() -> dict:
            # `marking_failures` is what stops a run that dies before its first
            # epoch from sitting at `queued` forever: the job would know, but the
            # job's state lives in memory and the run's lives on disk.
            with c.runs.marking_failures(body.name):
                return train(spec, should_stop=lambda: c.runs.stop_requested(body.name))

        # The job's cooperative-stop signal is the run's own `stop.json`: cancelling
        # the job is exactly asking the run to stop, and both the API's own
        # `POST /runs/{name}/stop` and a sweep parent reach for the same file.
        job = c.jobs.submit(
            "train",
            run_training,
            detail={"run": body.name},
            cancel=lambda: c.runs.request_stop(body.name),
        )
        return job.as_dict()

    @app.get("/runs/{name}")
    def get_run(name: str, c: Context = Depends(get_context)) -> dict:
        """State, config, provenance and checkpoints. **No metrics** (R5).

        Metrics come from `/runs/{name}/metrics?since=N`. Bundling them here is
        what made watching a run cost more with every epoch: the UI polls this in
        a loop, and the old version returned the whole history each time.
        """
        _run_or_404(c, name)
        try:
            config = c.runs.config(name)
        except (RunNotFound, json.JSONDecodeError) as exc:
            raise not_found(
                "run_config_unreadable",
                f"el run '{name}' no tiene un config.json legible: {exc}",
                "un run sin config es un run corrupto: bórralo",
            )
        return {
            "name": name,
            "state": c.runs.status(name),
            "config": config,
            # Contract ③ (D2). By NAME, which is what lets you ask "which runs
            # used network X?" without diffing dictionaries by hand.
            "provenance": config.get("provenance"),
            "checkpoints": c.runs.checkpoints(name),
            "summary": c.runs.summary(name),
            "seconds_per_epoch": c.runs.seconds_per_epoch(name),
        }

    @app.get("/runs/{name}/metrics")
    def get_run_metrics(
        name: str, since: int = Query(default=0, ge=0), c: Context = Depends(get_context)
    ) -> dict:
        """Incremental polling (R5): `{records, next}`. Never the whole history."""
        _run_or_404(c, name)
        return c.runs.metrics(name, since=since)

    @app.post("/runs/{name}/stop", status_code=202)
    def stop_run(name: str, c: Context = Depends(get_context)) -> dict:
        """Cooperative cancellation: it cuts at the end of the current epoch.

        202, not 200: the run is still going when this answers. Nothing is
        killed -- the epoch finishes, its metrics and checkpoint land, and the
        run closes as `cancelled`.
        """
        _run_or_404(c, name)
        state = c.runs.status(name).get("state")
        if state not in LIVE_STATES:
            raise conflict(
                "run_not_running",
                f"el run '{name}' ya está '{state}': no hay nada que parar",
            )
        c.runs.request_stop(name)
        return {"name": name, "state": state, "stop_requested": True}

    @app.patch("/runs/{name}")
    def rename_run(name: str, body: RenameRunBody, c: Context = Depends(get_context)) -> dict:
        _run_or_404(c, name)
        if c.runs.is_live(name):
            # Moving the directory out from under the loop would leave it writing
            # metrics into a path that no longer exists.
            raise conflict(
                "run_running",
                f"el run '{name}' está corriendo",
                "párala primero con POST /runs/{name}/stop",
            )
        try:
            c.runs.rename(name, body.name)
        except ValueError as exc:
            raise bad_request("invalid_name", str(exc))
        except RunExists:
            raise conflict("run_exists", f"ya existe un run llamado '{body.name}'")
        return {"name": body.name}

    @app.delete("/runs/{name}", status_code=204)
    def delete_run(name: str, c: Context = Depends(get_context)) -> Response:
        _run_or_404(c, name)
        if c.runs.is_live(name):
            raise conflict(
                "run_running",
                f"el run '{name}' está corriendo",
                "párala primero con POST /runs/{name}/stop",
            )
        # The other half of the check (fase 7): a run a sweep still references
        # cannot go, or the sweep loses a point of its space. The run names its
        # own parent in `provenance.sweep`; if that sweep is still on disk, refuse.
        try:
            parent = ((c.runs.config(name).get("provenance") or {}).get("sweep"))
        except _UNREADABLE:
            parent = None
        if parent and c.sweeps.exists(parent):
            raise conflict(
                "run_in_sweep",
                f"el run '{name}' es un punto del barrido '{parent}'",
                f"borra el barrido '{parent}' primero, o quédate el run",
            )
        c.runs.delete(name)
        return Response(status_code=204)


# ── E×B: /runs/{name}/diagnostics ─────────────────────────────────────────────


#: `NotMeasurable.code` → HTTP status. The default is 409 because that is what
#: most of these are: the request is well formed and the state says no -- the run
#: never trained, its dataset was rebuilt underneath it. 400 is for a request that
#: could never work whatever the state (an unknown split, an unknown corner), and
#: 404 for a name that is not there.
_DIAGNOSTIC_STATUS: dict[str, int] = {
    "run_not_found": 404,
    "patch_dataset_missing": 404,
    "unknown_split": 400,
    "unknown_corner": 400,
    "unknown_outcome": 400,
    "unknown_order": 400,
}


def _diagnostics(c: Context, name: str, split: str, checkpoint: str):
    """Open the table, turning every refusal into its status code (R4).

    The whole HTTP contribution of this resource. Everything that decides
    anything -- following the provenance, checking the fingerprint, computing or
    reusing the table -- is in `itf.diagnostics`, because none of it mentions
    HTTP (api.md §0).
    """
    try:
        return open_diagnostics(
            runs=c.runs,
            patch_datasets=c.patch_datasets,
            cache=c.diagnostics,
            run=name,
            split=split,
            checkpoint=checkpoint,
        )
    except NotMeasurable as exc:
        raise problem(
            _DIAGNOSTIC_STATUS.get(exc.code, 409), exc.code, str(exc), exc.hint or None
        )


def register_diagnostics(app: FastAPI) -> None:
    """The substrate of ui.md §3. **A cache, not an entity** (D1).

    So: no `POST` that creates one, no id, no listing, nothing to name. Every
    route is a `GET` over `(run, split)`, idempotent, and the table behind them is
    computed on the first one and invalidated by its own key.

    **Synchronous, not a job** (R3), and that is a statement: one pass over val is
    ~10⁴ forwards in batches -- seconds. If it ever needs a job, the table stopped
    being cheap and V8's free threshold sweep went with it.
    """

    @app.get("/runs/{name}/diagnostics/pr")
    def diagnostics_pr(
        name: str,
        split: str = "val",
        corner: str | None = None,
        checkpoint: str = DEFAULT_CHECKPOINT,
        c: Context = Depends(get_context),
    ) -> dict:
        """V8 — score histogram + PR curve. Every threshold, zero forwards."""
        diag = _diagnostics(c, name, split, checkpoint)
        try:
            return pr(diag, corner)
        except NotMeasurable as exc:
            raise problem(_DIAGNOSTIC_STATUS.get(exc.code, 409), exc.code, str(exc), exc.hint)

    @app.get("/runs/{name}/diagnostics/error-map")
    def diagnostics_error_map(
        name: str,
        split: str = "val",
        corner: str | None = None,
        bins: int | None = Query(default=None, ge=1),
        checkpoint: str = DEFAULT_CHECKPOINT,
        c: Context = Depends(get_context),
    ) -> dict:
        """V7 — where in the patch the corner was, and how far off we were.

        `bins` is the map's resolution. It is a query parameter and not a fixed
        40 because the readable resolution follows the amount of data, not the
        patch (see `DEFAULT_ERROR_MAP_BINS`); `bins=patch_size` is the
        full-resolution map ui.md §4.1 describes.
        """
        diag = _diagnostics(c, name, split, checkpoint)
        try:
            return error_map(diag, corner, bins)
        except NotMeasurable as exc:
            raise problem(_DIAGNOSTIC_STATUS.get(exc.code, 409), exc.code, str(exc), exc.hint)

    @app.get("/runs/{name}/diagnostics/patches")
    def diagnostics_patches(
        name: str,
        split: str = "val",
        corner: str | None = None,
        outcome: str = "all",
        order: str = "error",
        threshold: float = Query(default=0.5, ge=0.0, le=1.0),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=24, ge=1, le=200),
        checkpoint: str = DEFAULT_CHECKPOINT,
        c: Context = Depends(get_context),
    ) -> dict:
        """V6 — the worst-first gallery. **Filtered and paginated** (R6).

        `limit` is capped rather than trusted: this is the one diagnostics route
        that serves rows, and an unbounded one would happily ship the whole 10⁵-row
        table -- which is the exact thing R6 exists to prevent.
        """
        diag = _diagnostics(c, name, split, checkpoint)
        try:
            return rows(
                diag,
                corner=corner,
                outcome=outcome,
                order=order,
                threshold=threshold,
                offset=offset,
                limit=limit,
            )
        except NotMeasurable as exc:
            raise problem(_DIAGNOSTIC_STATUS.get(exc.code, 409), exc.code, str(exc), exc.hint)

    @app.get("/runs/{name}/diagnostics/coactivation")
    def diagnostics_coactivation(
        name: str,
        split: str = "val",
        threshold: float = Query(default=0.5, ge=0.0, le=1.0),
        checkpoint: str = DEFAULT_CHECKPOINT,
        c: Context = Depends(get_context),
    ) -> dict:
        """V9 — given the truth was TL, which heads fired? 4×4, over the cache.

        `threshold` is a query parameter, not part of the table's key: the scores
        are stored raw, so re-deciding what "fired" means costs a comparison, not
        a pass over the model. Same free sweep as V8.
        """
        diag = _diagnostics(c, name, split, checkpoint)
        try:
            return coactivation(diag, threshold)
        except NotMeasurable as exc:
            raise problem(_DIAGNOSTIC_STATUS.get(exc.code, 409), exc.code, str(exc), exc.hint)


# ── F: /runs/{name}/kernels, /feature-maps, /predict ──────────────────────────


#: `NotInspectable.code` → HTTP status. `patch_size_mismatch` is **400**: it is
#: contract ① reaching F, and like its `POST /runs` twin the request can never
#: work -- a 40-px network will not take a 60-px patch whatever the state is.
#: `kernels_not_projectable` is 409: the request is fine and this network's shape
#: is what says no.
_INSPECT_STATUS: dict[str, int] = {
    "patch_size_mismatch": 400,
    "invalid_patch": 400,
    "border_required": 400,
    "mask_too_big": 400,
    "patch_not_found": 404,
    # The request is fine; this network's shape is what says no -- a `border_test`
    # over a network that ignores the flags would measure nothing (like ①'s twin).
    "border_not_used": 409,
}


def _model_of(c: Context, name: str, checkpoint: str):
    """The run's trained network, from the cache. Every refusal, a status code.

    This function mentions HTTP in every branch, which is what makes it the
    API's: the loading, the caching and the mtime key are all in `itf.inference`
    (api.md §0 -- the model cache used to live in this file and was never the
    API's).
    """
    _run_or_404(c, name)
    path = c.runs.path(name) / checkpoint
    if not path.exists():
        state = c.runs.status(name).get("state")
        raise conflict(
            "run_has_no_checkpoint",
            f"el run '{name}' no tiene '{checkpoint}' (está '{state}'): todavía no hay ningún "
            f"modelo que mirar",
            "espera a que termine una época, o elige otro run",
        )
    try:
        return c.models.get(path, device="cpu")
    except ValueError as exc:
        # Contract ④: a checkpoint that cannot describe itself. `load_model` says
        # so with a reason rather than an AttributeError.
        raise conflict("checkpoint_unreadable", str(exc), "reentrena el run: ese .pt no es nuestro")


def _inspect_problem(exc: NotInspectable):
    return problem(_INSPECT_STATUS.get(exc.code, 409), exc.code, str(exc), exc.hint or None)


def _patch_from_body(c: Context, body: FeatureMapsBody) -> tuple[np.ndarray, list[int] | None]:
    """Resolve a patch-view body to `(pixels, border)`. One place, three views.

    V2, V4 and V10 all take the same input -- a patch, by `{patch_dataset, index}`
    or inline `{patch, border}` (contract ①) -- so the resolution lives here rather
    than copied into each. `{patch_dataset, index}` is the path the UI uses by
    default because the border flags then come from the dataset and are the real
    ones. Sending both is a question, not a convenience: which one did you mean?
    """
    if body.patch is None:
        if not body.patch_dataset or body.index is None:
            raise bad_request(
                "patch_required",
                "necesito un patch: o `{patch_dataset, index}`, o `patch` con sus píxeles",
                "la entrada de esta vista es un patch, no una imagen (contrato ①)",
            )
        try:
            rows = load_rows(
                c.patch_datasets.npz_path(body.patch_dataset), c.settings.patch_rows_cache_root
            )
        except (PatchDatasetNotFound, ValueError):
            raise not_found(
                "patch_dataset_not_found",
                f"no existe el dataset de patches '{body.patch_dataset}'",
            )
        total = len(rows)
        if not 0 <= body.index < total:
            raise not_found(
                "patch_not_found",
                f"'{body.patch_dataset}' tiene {total} patches; pediste el {body.index}",
            )
        row = rows.row(body.index)
        patch = row["X"][:, :, 0].tolist()
        border = row["border"].astype(int).tolist()
    elif body.patch_dataset or body.index is not None:
        raise bad_request(
            "patch_ambiguous",
            "manda `{patch_dataset, index}` o `patch`, no las dos cosas",
            "no puedo saber cuál de las dos querías mirar",
        )
    else:
        patch = body.patch
        border = body.border
    return np.asarray(patch, dtype=np.float32), border


def register_inference(app: FastAPI) -> None:
    """V1, V2 and V11. **Synchronous** (R3): one forward pass, or none at all.

    All three are introspection of an E, so they hang off `/runs/{name}`. The
    kernels need only the weights; the feature maps need a patch (contract ①);
    predict needs a whole image, which is the one question that is F's alone.
    """

    @app.get("/runs/{name}/kernels")
    def get_kernels(
        name: str, checkpoint: str = DEFAULT_CHECKPOINT, c: Context = Depends(get_context)
    ) -> dict:
        """V1 — the learned kernels of layer 1. **Layer 1 only** (D13).

        Free: no input, no forward pass, just the weights. And it is the phase's
        own verification -- if a trained run's layer 1 does not look like oriented
        edge detectors, the network did not learn, and that is information rather
        than a bug in the view.
        """
        model = _model_of(c, name, checkpoint)
        try:
            return kernels(model)
        except NotInspectable as exc:
            raise _inspect_problem(exc)

    @app.post("/runs/{name}/feature-maps")
    def post_feature_maps(
        name: str,
        body: FeatureMapsBody,
        checkpoint: str = DEFAULT_CHECKPOINT,
        c: Context = Depends(get_context),
    ) -> dict:
        """V2 — the activations of every layer over ONE PATCH (contract ①).

        A `POST` for a read, and it is the exception R3 allows: the body may carry
        the pixels of a patch that is stored nowhere, and a 40×40 matrix does not
        go in a query string.
        """
        model = _model_of(c, name, checkpoint)
        patch, border = _patch_from_body(c, body)
        try:
            return feature_maps(model, patch, border)
        except NotInspectable as exc:
            raise _inspect_problem(exc)

    @app.post("/runs/{name}/occlusion")
    def post_occlusion(
        name: str,
        body: FeatureMapsBody,
        checkpoint: str = DEFAULT_CHECKPOINT,
        c: Context = Depends(get_context),
    ) -> dict:
        """V4 — occlusion sensitivity over ONE PATCH (contract ①).

        Same input shape as the feature maps -- a patch, by index or inline -- and
        the same forward-only cost, just ~361 of them in one batch. Synchronous
        (R3): under a second (ui.md §4.1).
        """
        model = _model_of(c, name, checkpoint)
        patch, border = _patch_from_body(c, body)
        try:
            return occlusion(model, patch, border)
        except NotInspectable as exc:
            raise _inspect_problem(exc)

    @app.post("/runs/{name}/deconvolution")
    def post_deconvolution(
        name: str,
        body: FeatureMapsBody,
        checkpoint: str = DEFAULT_CHECKPOINT,
        c: Context = Depends(get_context),
    ) -> dict:
        """V16 — deconvolución sobre UN PATCH (contrato ①).

        Misma entrada que los feature maps —un patch, por índice o inline— y el
        mismo sitio donde resolverlo. Síncrono (R3): un forward y un backward por
        capa, del orden de V4.

        `border` se acepta (el cuerpo es el mismo) pero **no se pasa**: la
        deconvolución no sale del backbone y `border_features` solo toca la cabeza.
        """
        model = _model_of(c, name, checkpoint)
        patch, _border = _patch_from_body(c, body)
        try:
            return deconvolution(model, patch)
        except NotInspectable as exc:
            raise _inspect_problem(exc)

    @app.post("/runs/{name}/border-test")
    def post_border_test(
        name: str,
        body: FeatureMapsBody,
        checkpoint: str = DEFAULT_CHECKPOINT,
        c: Context = Depends(get_context),
    ) -> dict:
        """V10 — flip each of the 4 border flags over ONE PATCH.

        Refuses (409 `border_not_used`) if the network ignores the flags: flipping
        them would change nothing and the view would read as "the border does not
        matter", a claim about the data when the truth is the architecture never
        looks at it.
        """
        model = _model_of(c, name, checkpoint)
        patch, border = _patch_from_body(c, body)
        if border is None:
            raise bad_request(
                "border_required",
                "esta sonda necesita los 4 flags de borde del patch, y no me los diste",
                "pide el patch por índice y salen de su dataset, o manda `border`",
            )
        try:
            return border_test(model, patch, border)
        except NotInspectable as exc:
            raise _inspect_problem(exc)

    @app.post("/runs/{name}/window")
    def post_window(
        name: str,
        body: WindowBody,
        checkpoint: str = DEFAULT_CHECKPOINT,
        c: Context = Depends(get_context),
    ) -> dict:
        """V5 — the scrubber: one off-grid 40×40 crop of an image → 4 heads live.

        The one probe whose input is a whole image (like F), because dragging a
        window over one is exactly what it is. Synchronous (R3): five forwards, so
        the drag repaints without a job. The border flags of the crop come from
        `window_at` (contract ⑤), so an off-grid crop is flagged as its trained-on
        twin would be.
        """
        model = _model_of(c, name, checkpoint)
        sample = _sample(c, body.source, body.index)
        with Image.open(sample.image_path) as img:
            image = np.asarray(img.convert("L"), dtype=np.uint8)
        try:
            payload = window_prediction(model, image, body.x0, body.y0, device="cpu")
        except ValueError as exc:
            # `window_at`/`_as_gray` refuse when the patch is bigger than the image.
            raise bad_request("image_too_small", str(exc), "elige una imagen mayor que el patch")
        payload["source"] = body.source
        payload["index"] = body.index
        return payload

    @app.post("/runs/{name}/predict")
    def post_predict(
        name: str,
        body: PredictBody,
        checkpoint: str = DEFAULT_CHECKPOINT,
        c: Context = Depends(get_context),
    ) -> dict:
        """F — a whole image → **the three stages** (V11).

        `raw` (pre-NMS), `corners` (post-NMS) and `paragraphs` (post-pairing) all
        come back, because the failure is born in one of the three and they are
        fixed in different places: stage 1 is the model, stages 2 and 3 are knobs.
        Returning only the last one is what made "the paragraph came out wrong"
        undiagnosable.

        Synchronous (R3): ~700 windows in batches is a fraction of a second, which
        is what lets the knobs be sliders instead of a form.
        """
        model = _model_of(c, name, checkpoint)
        sample = _sample(c, body.source, body.index)
        with Image.open(sample.image_path) as img:
            image = np.asarray(img.convert("L"), dtype=np.uint8)

        try:
            payload = predict_image(
                model,
                image,
                stride=body.stride,
                threshold=body.threshold,
                nms_radius=body.nms_radius,
                min_size=body.min_size,
                device="cpu",
            )
        except ValueError as exc:
            # `positions` refuses when the patch is bigger than the image: a
            # 40-px window over a 32-px thumbnail has nowhere to sit.
            raise bad_request("image_too_small", str(exc), "elige una imagen mayor que el patch")
        payload["source"] = body.source
        payload["index"] = body.index
        return payload


# ── H: /sweeps ────────────────────────────────────────────────────────────────


def _sweep_progress(c: Context, spec: SweepSpec) -> dict:
    """The trials table, read from the worker's JSON snapshot -- never optuna.

    The worker thread holds the SQLite study open for the whole sweep; a second
    connection from this (the API) thread races it on schema creation and write
    locks. So the worker writes `progress.json` after each trial and the API reads
    that. It also keeps optuna out of this file entirely, which is what lets the
    ⑨ 400 be answered without the engine installed.
    """
    return c.sweeps.read_progress(spec)


def _submit_sweep_job(c: Context, spec: SweepSpec, resumed: bool = False):
    """Queue the job that runs (or resumes) a sweep. One place, two callers.

    `POST /sweeps` calls it to start a sweep; the startup resumer calls it to pick
    one back up. They MUST build the job the same way -- the whole point of resume
    is that `run_sweep` cannot tell "started" from "resumed": it counts the trials
    already done and runs the rest, and the study on disk is what remembers them.
    """

    def run_the_sweep() -> dict:
        from itf.sweeps.runner import run_sweep

        return run_sweep(
            spec,
            runs=c.runs,
            patch_datasets=c.patch_datasets,
            networks=c.networks,
            recipes=c.recipes,
            sweeps=c.sweeps,
            should_stop=lambda: c.sweeps.stop_requested(spec.name),
        )

    return c.jobs.submit(
        "sweep",
        run_the_sweep,
        detail={"sweep": spec.name, "resumed": resumed},
        cancel=lambda: c.sweeps.request_stop(spec.name),
    )


def _why_not_resumable(c: Context, spec: SweepSpec) -> tuple[str, str] | None:
    """`(code, message)` if there is nothing left to run, else `None`.

    One question, two callers with deliberately different manners: the startup
    resumer *skips* what this rejects, `POST /resume` *refuses out loud* with the
    reason (R4). Asking it once is what stops the button and the restart from
    drifting apart about what "unfinished" means.

    **Two things are NOT asked here**, and both omissions are the point:

    - **The stop request.** A restart must not undo an explicit "stop" -- you
      asked for it, and the API coming back up is not you changing your mind. But
      `POST /resume` **is** you changing your mind, so it clears the request.
    - **Whether it is already running.** At startup nothing has been submitted
      yet, and a sweep whose job record was lost reads as `queued` by default --
      refusing on that would silently stop resuming it, which is the behaviour
      this function exists to preserve. It is the endpoint's question, so the
      endpoint asks it.
    """
    progress = _sweep_progress(c, spec)
    if progress["completed"] >= spec.budget.points:
        return (
            "sweep_already_complete",
            f"el barrido '{spec.name}' ya cumplió su presupuesto "
            f"({progress['completed']}/{spec.budget.points} puntos): no queda nada que reanudar",
        )
    return None


def resume_sweeps(c: Context) -> list[str]:
    """Re-enqueue every sweep that a restart left unfinished. Returns their names.

    This is the payoff of the persistence (plan-ui.md fase 7 verification): a CPU
    sweep runs for hours, so it must survive the API going down. The durable state
    is on disk -- `spec.json`, `optuna.db`, and the runs -- so resuming is just
    submitting the job again. A sweep that already met its budget, or that was
    asked to stop, is left alone.
    """
    resumed = []
    for name in c.sweeps.names():
        try:
            spec = c.sweeps.spec(name)
        except (SweepNotFound, ValueError):
            continue
        # A restart must not undo an explicit stop (see `_why_not_resumable`).
        if c.sweeps.stop_requested(name):
            continue
        if _why_not_resumable(c, spec):
            continue
        _submit_sweep_job(c, spec, resumed=True)
        resumed.append(name)
    return resumed


def _sweep_state(c: Context, name: str) -> str:
    """The sweep's own state, read off its job (the durable record is on disk).

    `queued` before a worker picks it up, `running` while it trains points, then
    `done`/`error`/`cancelled`/`interrupted` -- the last of which is what a
    restart leaves, and what the resumer looks for.
    """
    for job in c.jobs.list():  # newest first
        if job.kind == "sweep" and job.detail.get("sweep") == name:
            return job.state
    return "queued"


def register_sweeps(app: FastAPI) -> None:
    """H — the sweep: a space of D with B and C fixed → many E.

    `POST /sweeps` → job (R3: hours). Everything it refuses, it refuses in
    milliseconds and before the sweep is reserved -- **the ⑨ check first of all**,
    because that is the one that would otherwise produce a winner with a good face.
    """

    @app.get("/sweeps")
    def list_sweeps(c: Context = Depends(get_context)) -> dict:
        out = []
        for name in c.sweeps.names():
            spec = c.sweeps.spec(name)
            progress = _sweep_progress(c, spec)
            out.append(
                {
                    "name": name,
                    "state": _sweep_state(c, name),
                    "objective": spec.objective,
                    "patch_dataset": spec.patch_dataset,
                    "network": spec.network,
                    "completed": progress["completed"],
                    "points": spec.budget.points,
                    "best": progress["best"],
                }
            )
        return {"sweeps": out}

    @app.post("/sweeps", status_code=202)
    def create_sweep(body: CreateSweepBody, c: Context = Depends(get_context)) -> dict:
        payload = body.model_dump()

        # ⑨ (and the rest of the spec's shape) FIRST, before touching the stores:
        # ranking by `loss` while sweeping `lambda_pos` is a 400, and it must be
        # answered whether or not the named B and C happen to exist.
        problems = check_sweep(payload)
        if problems:
            raise bad_request(
                problems[0]["code"],
                "; ".join(p["message"] for p in problems),
                problems[0]["hint"],
                problems=problems,
            )

        name = payload["name"]
        try:
            if c.sweeps.exists(name):
                raise conflict(
                    "sweep_exists",
                    f"ya existe un barrido llamado '{name}'",
                    "elige otro nombre, o borra ese barrido primero: no se sobrescribe nunca",
                )
        except ValueError as exc:
            raise bad_request("invalid_name", str(exc), "usa un nombre sin barras ni '..'")

        # The fixed B and C must exist and be compatible NOW: a sweep whose B and C
        # cannot train together should fail at creation, not once per point inside
        # the job thread (the same lesson as contract ①).
        try:
            manifest = c.patch_datasets.manifest(body.patch_dataset)
        except (PatchDatasetNotFound, ValueError):
            raise not_found(
                "patch_dataset_not_found",
                f"no existe el dataset de patches '{body.patch_dataset}'",
            )
        try:
            network = c.networks.get(body.network)
        except (NetworkNotFound, ValueError):
            raise not_found("network_not_found", f"no existe la red '{body.network}'")
        if body.recipe is not None:
            try:
                c.recipes.get(body.recipe)
            except (RecipeNotFound, ValueError):
                raise not_found("recipe_not_found", f"no existe la receta base '{body.recipe}'")
        network.pop("format_version", None)
        run_problems = check_run(manifest, network)
        if run_problems:
            raise bad_request(
                run_problems[0]["code"],
                "; ".join(p["message"] for p in run_problems),
                run_problems[0]["hint"],
                problems=run_problems,
            )

        spec = SweepSpec.from_dict(payload)
        try:
            c.sweeps.create(spec)
        except SweepExists:
            raise conflict("sweep_exists", f"ya existe un barrido llamado '{name}'")

        return _submit_sweep_job(c, spec).as_dict()

    def _spec_or_404(c: Context, name: str) -> SweepSpec:
        try:
            return c.sweeps.spec(name)
        except (SweepNotFound, ValueError):
            raise not_found("sweep_not_found", f"no existe el barrido '{name}'", "mira GET /sweeps")

    @app.get("/sweeps/{name}")
    def get_sweep(name: str, c: Context = Depends(get_context)) -> dict:
        """Spec + progress (V12/V13 read the `trials` table it carries)."""
        spec = _spec_or_404(c, name)
        return {"spec": spec.as_dict(), "state": _sweep_state(c, name), **_sweep_progress(c, spec)}

    @app.get("/sweeps/{name}/trials")
    def get_sweep_trials(name: str, c: Context = Depends(get_context)) -> dict:
        """The points table, ordered by optuna (V12 Pareto, V13 parallel)."""
        spec = _spec_or_404(c, name)
        progress = _sweep_progress(c, spec)
        return {"trials": progress["trials"], "best": progress["best"], "objective": spec.objective}

    @app.post("/sweeps/{name}/stop", status_code=202)
    def stop_sweep(name: str, c: Context = Depends(get_context)) -> dict:
        """Cooperative: it cuts between trials, and the running point at its epoch."""
        spec = _spec_or_404(c, name)
        c.sweeps.request_stop(name)
        return {"name": spec.name, "stop_requested": True}

    @app.post("/sweeps/{name}/resume", status_code=202)
    def resume_sweep(name: str, c: Context = Depends(get_context)) -> dict:
        """Pick a stopped or interrupted sweep back up. **The counterpart of `/stop`.**

        The machinery already existed and had exactly one trigger: the `lifespan`,
        which resumes unfinished sweeps when the API starts. So a stopped sweep
        could only be continued by **restarting the backend** -- the capability was
        real and unreachable, which reads to a user as "there is no way to
        continue" (it did: 2026-07-19).

        Nothing new happens here. It goes through `_submit_sweep_job`, the same
        door `POST /sweeps` and the resumer use, because `run_sweep` cannot tell
        "started" from "resumed": it counts the terminal trials in `optuna.db` and
        runs the rest. **What is durable is on disk**, so resuming is submitting
        the job again.

        **It clears the stop request**, and that is the difference from the startup
        resumer, which deliberately leaves stopped sweeps alone (see
        `_why_not_resumable`). Without clearing it the job would start and stop
        again at the first trial boundary -- a button that looks like it worked and
        does nothing, which is worse than no button.
        """
        spec = _spec_or_404(c, name)

        # **Ask the queue, not `_sweep_state`** (formatos.md §2: ausente ≠ cero).
        # `_sweep_state` answers `queued` when it finds NO job for the sweep --
        # a sensible default for a display, and a wrong premise for a decision:
        # a stopped sweep whose job record is gone is exactly the case this
        # endpoint exists for, and it would be refused for "already queued".
        # Caught by `test_resume_clears_the_stop_request`.
        live = next(
            (
                j
                for j in c.jobs.list()
                if j.kind == "sweep"
                and j.detail.get("sweep") == name
                and j.state in ("queued", "running")
            ),
            None,
        )
        if live is not None:
            raise conflict(
                "sweep_already_running",
                f"el barrido '{spec.name}' ya está {live.state}",
                "míralo en la tabla de puntos; si quieres pararlo, usa /stop",
            )
        if why := _why_not_resumable(c, spec):
            code, message = why
            raise conflict(code, message, "crea un barrido nuevo si quieres más puntos")

        c.sweeps.clear_stop(name)
        job = _submit_sweep_job(c, spec, resumed=True)
        return {"name": spec.name, "job": job.id, "resumed": True}


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

    @app.post("/jobs/{job_id}/cancel", status_code=202)
    def cancel_job(job_id: str, c: Context = Depends(get_context)) -> dict:
        """Cooperative cancel: marks the stop, the work cuts at its safe point.

        202, not 200: a running job is still going when this answers. Nothing is
        killed -- a training finishes its epoch, a sweep its trial. A job that has
        already finished answers 409: there is nothing to cancel.
        """
        job = c.jobs.cancel(job_id)
        if job is None:
            raise not_found("job_not_found", f"no existe el job '{job_id}'")
        if job.state in {"done", "error", "interrupted"}:
            raise conflict(
                "job_not_cancellable",
                f"el job '{job_id}' ya está '{job.state}': no hay nada que cancelar",
            )
        return job.as_dict()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # On startup the queue's workers are already up, so a re-enqueued sweep
        # begins running. A sweep that met its budget or was asked to stop is left
        # alone. This is the restart survival of fase 7 (plan-ui.md).
        resume_sweeps(app.state.context)
        yield

    app = FastAPI(title="image-text-finder", version="0.1.0", lifespan=lifespan)
    app.state.context = Context(
        settings=settings,
        patch_datasets=PatchDatasetStore(settings.patch_datasets_root),
        networks=NetworkStore(settings.networks_root),
        recipes=RecipeStore(settings.recipes_root),
        runs=RunStore(settings.runs_root),
        sweeps=SweepStore(settings.sweeps_root),
        # On CPU the limit is 1: torch already uses every core inside one run, so
        # N at once just fight each other and each holds its dataset in RAM.
        # `persist_dir` is what lets `GET /jobs` survive a restart and what marks
        # a job that was live when the process died as `interrupted` (fase 7).
        jobs=JobQueue(max_workers=1, persist_dir=settings.jobs_root),
        diagnostics=TableCache(settings.diagnostics_cache_root),
        models=ModelCache(),
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
    register_networks(app)
    register_recipes(app)
    register_runs(app)
    register_diagnostics(app)
    register_inference(app)
    register_sweeps(app)
    register_jobs(app)
    return app


app = create_app()
