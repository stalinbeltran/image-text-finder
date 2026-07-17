"""Reads the datasets that `image-text-sample-generator` produces.

A source (domain A) is the ground truth: images plus paragraph geometry. Nothing
here depends on the CNN, and nothing here knows what a patch is -- it does not
know what `n` is, and must not.

**The boundary is not ours.** `labels.jsonl` is specified by the generator, in
its `SAMPLE_FORMAT.md`. We do not specify it: we specify what we CONSUME, which
is `index`, `labels.{width,height,has_overlap}` and
`blocks[].{block_id,kind,angle,quad}` (formatos.md §4.5). If it changes over
there, this file breaks over here, and the only thing that would catch it is a
test with a sample `labels.jsonl` -- so the parsing stays narrow on purpose.

This module used to own `CORNER_NAMES` and friends, which made domain C import
from domain A (contract ⑦). They live in `itf.geometry` now.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Block:
    """One ground-truth block: a paragraph, a word, a letter..."""

    block_id: str
    kind: str
    angle: float
    #: (4, 2) float32, in pixels, clockwise from top-left == CORNER_NAMES order.
    quad: np.ndarray

    @classmethod
    def from_json(cls, obj: dict) -> "Block":
        return cls(
            block_id=obj["block_id"],
            kind=obj["kind"],
            # May be rotated. Nothing downstream handles rotation yet, and the
            # paragraph metric will have to decide what to do about it (D7).
            angle=float(obj.get("angle", 0.0)),
            quad=np.asarray(obj["quad"], dtype=np.float32).reshape(4, 2),
        )


@dataclass
class Sample:
    """One IMAGE and its geometry.

    Note the word: a `Sample` is an image, not a training example. The training
    example of this CNN is the PATCH. `num_samples: 200` means 200 images; it is
    the single most expensive naming collision in the project, because reading it
    as "200 training examples" is exactly the mistake that made "980 val patches"
    sound like a large sample when it was 20 images (glosario.md §1).
    """

    index: int
    width: int
    height: int
    has_overlap: bool
    image_path: Path
    mask_path: Path | None
    blocks: list[Block]

    def blocks_of_kind(self, kinds: Sequence[str]) -> list[Block]:
        wanted = set(kinds)
        return [b for b in self.blocks if b.kind in wanted]

    def load_image(self) -> np.ndarray:
        """Greyscale image as ``(H, W)`` uint8."""
        with Image.open(self.image_path) as img:
            return np.asarray(img.convert("L"), dtype=np.uint8)


class SourceDataset:
    """A directory following the generator's SAMPLE_FORMAT.md."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        labels = self.root / "labels.jsonl"
        if not labels.exists():
            raise FileNotFoundError(f"no encuentro labels.jsonl en {self.root}")
        self.labels_path = labels
        meta_path = self.root / "dataset.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    @property
    def id(self) -> str:
        return self.meta.get("id", self.root.name)

    def __iter__(self) -> Iterator[Sample]:
        with self.labels_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield self._parse(json.loads(line))

    def samples(self) -> list[Sample]:
        return list(self)

    def _parse(self, rec: dict) -> Sample:
        lab = rec["labels"]
        mask_rel = rec.get("mask")
        return Sample(
            index=int(rec["index"]),
            width=int(lab["width"]),
            height=int(lab["height"]),
            has_overlap=bool(lab.get("has_overlap", False)),
            image_path=self.root / rec["image"],
            mask_path=(self.root / mask_rel) if mask_rel else None,
            blocks=[Block.from_json(b) for b in lab["blocks"]],
        )


def discover_sources(root: str | Path) -> list[Path]:
    """Every source dataset under ``root``, deepest-first, sorted.

    A directory is a source iff it holds a `labels.jsonl`. Generators nest them
    (``clean-paragraphs-01/reducido``), so this walks rather than listing.
    """
    base = Path(root)
    if not base.is_dir():
        return []
    return sorted(p.parent for p in base.rglob("labels.jsonl"))
