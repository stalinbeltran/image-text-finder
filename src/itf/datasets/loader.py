"""Loader for datasets produced by image-text-sample-generator.

Reads the ``labels.jsonl`` file described in SAMPLE_FORMAT.md and exposes each
sample together with its image and the paragraph-block geometry we need for
corner extraction. Only what the patch extractor consumes is parsed here; the
richer spec (``specs.jsonl``, ``dataset.json``) is left untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image

# Corner order is fixed project-wide: the quad in SAMPLE_FORMAT is clockwise
# from the top-left corner, i.e. [TL, TR, BR, BL].
CORNER_NAMES = ("TL", "TR", "BR", "BL")
NUM_CORNERS = len(CORNER_NAMES)


@dataclass(frozen=True)
class Block:
    """A single ground-truth block (paragraph, word, letter, ...)."""

    block_id: str
    kind: str
    angle: float
    # quad corners in pixels, clockwise from top-left: [[x,y] x4] == [TL,TR,BR,BL]
    quad: np.ndarray  # shape (4, 2), float32

    @classmethod
    def from_json(cls, obj: dict) -> "Block":
        return cls(
            block_id=obj["block_id"],
            kind=obj["kind"],
            angle=float(obj.get("angle", 0.0)),
            quad=np.asarray(obj["quad"], dtype=np.float32).reshape(4, 2),
        )


@dataclass
class Sample:
    """One training sample: an image plus its block geometry."""

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
        """Grayscale image as ``(H, W)`` uint8."""
        img = Image.open(self.image_path).convert("L")
        return np.asarray(img, dtype=np.uint8)


class SourceDataset:
    """A dataset directory following SAMPLE_FORMAT.md."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        labels = self.root / "labels.jsonl"
        if not labels.exists():
            raise FileNotFoundError(f"labels.jsonl not found under {self.root}")
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
                if not line:
                    continue
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
