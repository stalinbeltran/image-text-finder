"""A' — a derived source: the same A, at another resolution (D19).

**The composition, and nothing else.** The two mechanisms live elsewhere on
purpose: `itf.imageops` moves pixels and does not know what a quad is,
`itf.geometry.scale_quad` moves coordinates and does not know what a file is.
This module is the only place that knows both, which is what leaves the first one
usable on an arbitrary test image later -- the point of the whole exercise, not a
tidiness preference (organizacion.md §1-A').

**We become a second producer of somebody else's format.** `labels.jsonl` belongs
to `image-text-sample-generator` (formatos.md §4.5); until now we only read it.
So the rule here is conservative to the point of dullness: rewrite exactly the
fields we consume, **copy everything else through untouched**, invent nothing. A
derived source that the same parser cannot read is a broken derived source, and
"improving" the schema on the way out is how that happens.

Two things that look like details and are not:

  - **The scale is measured from the output**, never the requested factor, and
    there are two of them. See `itf.imageops`.
  - **Masks resample with NEAREST.** Interpolating a label mask fabricates
    classes that do not exist -- the continuous form of `ausente != cero`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image

from itf.geometry import scale_quad
from itf.imageops import resize as resize_image
from itf.imageops import target_size
from itf.validation import Problem


@dataclass(frozen=True)
class ResizeRequest:
    """What was asked for: one dimension, never two."""

    #: The name of the derived source, a directory under the local sources root.
    name: str
    to_width: int | None = None
    to_height: int | None = None

    def as_json(self) -> dict[str, int]:
        return {"width": self.to_width} if self.to_width is not None else {"height": self.to_height}


class ResizeRefused(Exception):
    """The request cannot be honoured. Carries R4 problems, not a bare message."""

    def __init__(self, problems: list[Problem]):
        super().__init__("; ".join(p["message"] for p in problems))
        self.problems = problems


def source_sizes(root: Path) -> list[tuple[int, int]]:
    """Every sample's ``(width, height)``, read from labels only.

    No images are opened: this is what makes the pre-flight check cheap enough to
    run before the job is created, which is where api.md wants the refusal.
    """
    sizes: list[tuple[int, int]] = []
    with (root / "labels.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                lab = json.loads(line)["labels"]
                sizes.append((int(lab["width"]), int(lab["height"])))
    return sizes


def check_resize(sizes: Iterable[tuple[int, int]], req: ResizeRequest) -> list[Problem]:
    """Every reason this resize must be refused. Empty == go ahead. **Pure.**

    Pure and imageless like `itf.validation`, and for the same reason: the refusal
    has to land *before* anything is reserved or written. A resize that discovers
    on image 400 of 500 that it was upscaling has already produced 399 files
    nobody wants.
    """
    problems: list[Problem] = []

    if (req.to_width is None) == (req.to_height is None):
        problems.append(
            {
                "code": "resize_needs_one_dimension",
                "message": "hay que dar exactamente una dimensión: o `width` o `height`",
                "hint": (
                    "la proporción se mantiene por construcción, así que la otra dimensión se "
                    "deriva; dar las dos sería pedir una deformación, que esto no hace"
                ),
            }
        )
        # Without a well-formed request the size check below is meaningless.
        return problems

    sizes = list(sizes)
    if not sizes:
        problems.append(
            {
                "code": "source_empty",
                "message": "la fuente no tiene ninguna muestra",
                "hint": "comprueba que labels.jsonl no está vacío",
            }
        )
        return problems

    # Checked against EVERY sample, not the first. A mixed-size source where the
    # target shrinks 9 images and grows 1 must be refused whole: accepting it
    # would leave a single interpolated sample inside an otherwise legitimate
    # dataset, which is the kind of thing nobody finds later (api.md §3).
    grew = []
    for w, h in sizes:
        out_w, out_h = target_size(w, h, to_width=req.to_width, to_height=req.to_height)
        if out_w > w or out_h > h:
            grew.append((w, h, out_w, out_h))

    if grew:
        w, h, out_w, out_h = grew[0]
        problems.append(
            {
                "code": "upscale_not_allowed",
                "message": (
                    f"esto ampliaría {len(grew)} de {len(sizes)} muestras "
                    f"(p. ej. {w}x{h} → {out_w}x{out_h})"
                ),
                "hint": (
                    "solo se reduce: ampliar un render sintético no añade información, añade "
                    "interpolador, y un dataset de patches extraído de ahí mediría el "
                    "interpolador y no el modelo (D19). Pide una dimensión menor"
                ),
            }
        )

    return problems


def resize_source(
    src_root: Path,
    dst_root: Path,
    req: ResizeRequest,
    *,
    source_id: str | None = None,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Write a resized copy of the source at ``src_root`` into ``dst_root``.

    Returns the `derived` block that went into `dataset.json` (formatos.md §4.6).

    **`source_id` is the ADDRESSABLE id** -- the one `GET /sources` lists and
    `--source` takes -- and it is required for the provenance to mean anything.
    Measured on the real data: `clear-paragraphs-02-reducidos` and
    `clear-paragraphs-02-8ea1ac04` **declare the same `id` inside their own
    `dataset.json`**, because one is a reduced copy that kept the other's. Those
    two are the 14.5x-area trap of organizacion.md §3, so trusting the declared
    id makes `from` name the wrong parent, silently, in exactly the case the
    project already knows is dangerous. Falls back to the directory name, which
    is at least unique within a root.

    `should_stop` is the cooperative-cancellation shape the queue already uses
    (librerias.md, `jobq`): checked per sample, because per sample is a point
    where stopping leaves nothing half-written.
    """
    src_root, dst_root = Path(src_root), Path(dst_root)

    problems = check_resize(source_sizes(src_root), req)
    if problems:
        raise ResizeRefused(problems)

    if dst_root.exists():
        raise ResizeRefused(
            [
                {
                    "code": "source_exists",
                    "message": f"ya existe una fuente en {dst_root}",
                    "hint": (
                        "elige otro `name` o borra la existente: no se sobrescribe en silencio "
                        "(organizacion.md §3, trampa nº5)"
                    ),
                }
            ]
        )

    records = [
        json.loads(line)
        for line in (src_root / "labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    dst_root.mkdir(parents=True)
    out_lines: list[str] = []
    scales: list[tuple[float, float]] = []
    sizes: list[tuple[int, int]] = []

    try:
        for i, rec in enumerate(records):
            if should_stop is not None and should_stop():
                raise KeyboardInterrupt("resize cancelado")
            out_lines.append(json.dumps(_resize_record(rec, src_root, dst_root, req, scales, sizes)))
            if progress is not None:
                progress(i + 1, len(records))
    except BaseException:
        # A half-written source is worse than no source: it loads, it parses, and
        # it silently has fewer samples than the parent. Nothing partial survives.
        shutil.rmtree(dst_root, ignore_errors=True)
        raise

    (dst_root / "labels.jsonl").write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    derived = {
        # The addressable id: what you would type to get this source back.
        "from": source_id or src_root.name,
        # What the parent calls ITSELF. Kept because it is the generator's word,
        # and recorded separately because it is demonstrably not unique -- see
        # the docstring. Two fields that can disagree, on purpose: collapsing
        # them is what lost the information in the first place.
        "from_declared_id": _parent_id(src_root),
        "op": "resize",
        "request": req.as_json(),
        # The realised size and scale. With mixed-size sources these are per
        # sample, so what goes in the manifest is the range -- reporting one pair
        # would be reporting the first sample and calling it the dataset.
        "size": list(sizes[0]) if len(set(sizes)) == 1 else None,
        "size_range": [list(min(sizes)), list(max(sizes))] if len(set(sizes)) > 1 else None,
        "scale": list(scales[0]) if len(set(scales)) == 1 else None,
        "resample": "lanczos",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta = _parent_meta(src_root) | {"id": req.name, "derived": derived}
    (dst_root / "dataset.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return derived


#: Every pixel-valued key in SAMPLE_FORMAT.md §3.1, at any nesting depth.
#: `quad` is (4,2) clockwise from TL; `box` is [x, y, w, h] axis-aligned.
_GEOMETRY_KEYS = ("quad", "box")


def _resize_record(
    rec: dict[str, Any],
    src_root: Path,
    dst_root: Path,
    req: ResizeRequest,
    scales: list[tuple[float, float]],
    sizes: list[tuple[int, int]],
) -> dict[str, Any]:
    """One sample: image, optional mask, and **all** the geometry that must follow.

    **The record is copied, not rebuilt**: fields the generator writes and we do
    not consume ride through untouched, because rebuilding from our parsed
    `Sample` would silently drop them.

    **But "untouched" cannot include pixels**, and that is the sharp edge here.
    The format nests geometry we never read -- `blocks[].box`,
    `blocks[].lines[].quad`, `lines[].words[].box` (SAMPLE_FORMAT.md §3.1) --
    and passing those through unscaled would produce a dataset whose `quad` is at
    the new resolution and whose `box` is at the old one. It would load fine. It
    would draw wrong. **A resize is all-or-nothing**: if it cannot move every
    coordinate, it must not move any, so `box` and `quad` are rescaled wherever
    they appear, at any depth.
    """
    out = dict(rec)

    img_rel = rec["image"]
    with Image.open(src_root / img_rel) as img:
        result = resize_image(img, to_width=req.to_width, to_height=req.to_height)
        _write_beside(dst_root, img_rel, result.image)

    sx, sy = result.scale
    scales.append(result.scale)
    sizes.append(result.size)

    mask_rel = rec.get("mask")
    if mask_rel:
        with Image.open(src_root / mask_rel) as mask:
            # NEAREST: a mask is labels, and interpolating labels invents classes
            # that were never in the palette. Same size as the image by format.
            resized_mask = resize_image(mask, to_width=result.size[0], resample=Image.NEAREST)
            _write_beside(dst_root, mask_rel, resized_mask.image)

    lab = _scale_geometry(rec["labels"], sx, sy)
    lab["width"], lab["height"] = result.size
    out["labels"] = lab
    return out


def _scale_geometry(node: Any, sx: float, sy: float) -> Any:
    """Deep-copy ``node``, scaling every `box`/`quad` it contains. Everything else rides.

    Recursive because the format is: blocks hold lines hold words, and each level
    carries its own pair. A version of this that only walked `labels.blocks` would
    be right on today's `clear-paragraphs` (no `lines`) and quietly wrong on
    `mixed-layout`, which is the worst possible place to be right by accident.
    """
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "quad":
                out[key] = scale_quad(value, sx, sy).tolist()
            elif key == "box":
                x, y, w, h = value
                # [x, y, w, h]: the origin scales, and so does the extent.
                out[key] = [x * sx, y * sy, w * sx, h * sy]
            else:
                out[key] = _scale_geometry(value, sx, sy)
        return out
    if isinstance(node, list):
        return [_scale_geometry(v, sx, sy) for v in node]
    return node


def _write_beside(dst_root: Path, rel: str, img: Image.Image) -> None:
    """Write ``img`` at the same relative path the parent used.

    Same layout as the parent, deliberately: the derived source is meant to be
    indistinguishable to `SourceDataset`, and relative paths are part of that.
    """
    out_path = dst_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _parent_meta(src_root: Path) -> dict[str, Any]:
    meta_path = src_root / "dataset.json"
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}


def _parent_id(src_root: Path) -> str:
    return str(_parent_meta(src_root).get("id", src_root.name))
