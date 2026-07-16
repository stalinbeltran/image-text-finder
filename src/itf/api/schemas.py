"""Request bodies. HTTP's shape, not the domain's."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SplitBody(BaseModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


class BuildPatchDatasetBody(BaseModel):
    """`POST /patch-datasets`.

    Note what is NOT here: nothing of C, nothing of D. This builds a B, and a B
    is defined by its extraction parameters and nothing else.
    """

    name: str = Field(min_length=1)
    #: The id of a source (A), relative to DATASETS_ROOT -- not an absolute path.
    #: The path is resolved inside the domain; the client never supplies one (D4).
    source: str = Field(min_length=1)
    patch_size: int = Field(default=40, gt=0)
    #: The stride of EXTRACTION, part of B's identity. Not the stride of
    #: inference, which is a knob of F chosen per call (glosario.md).
    stride: int = Field(default=20, gt=0)
    target_kinds: list[str] = Field(default_factory=lambda: ["paragraph"])
    drop_overlap: bool = False
    split: SplitBody = Field(default_factory=SplitBody)
    #: The seed of B: fixes the split. Not the seed of D.
    seed: int = 1
