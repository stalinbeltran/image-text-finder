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


class NetworkBody(BaseModel):
    """`POST /networks` / `POST /networks/validate`. C, and only C.

    No `lr`, no `epochs`, no `device`. The old `ModelConfigForm` was C + D + X in
    one form -- its own comment admitted it -- and `lockArchitecture` was a
    boolean papering over the C/D border instead of resolving it.
    """

    input_size: int = Field(default=40, gt=0)
    in_channels: int = Field(default=1, gt=0)
    backbone: list[dict] = Field(default_factory=list)
    head: dict = Field(default_factory=dict)
    #: The dataset offers the border flags; the network decides (contract ②).
    border_features: bool = False


class NamedNetworkBody(NetworkBody):
    name: str = Field(min_length=1)


class CreateRunBody(BaseModel):
    """`POST /runs`. **Names, never values** (api.md R7).

    Want a bespoke network or recipe? Save it first. It looks rigid and it is
    deliberate: it is what makes contract ③ hold by itself -- every run can say
    which C and which D it came from, which is exactly what a sweep needs in
    order to group, and what was impossible when the run copied the value and
    lost the identity.

    `device` and `num_workers` are here and NOT in the recipe: they are X
    (contract ⑩). They cost time; they do not change the weights.
    """

    name: str = Field(min_length=1)
    #: The name of a B, resolved inside the domain (D4: the client sends an id).
    patch_dataset: str = Field(min_length=1)
    #: The name of a C.
    network: str = Field(min_length=1)
    #: The name of a D.
    recipe: str = Field(min_length=1)
    device: str = "cpu"
    num_workers: int = Field(default=0, ge=0)


class RenameRunBody(BaseModel):
    """`PATCH /runs/{name}` — renaming, and nothing else.

    The config cannot be patched: a run's config is what it was trained with, and
    editing it after the fact would make the record lie about the weights.
    """

    name: str = Field(min_length=1)


class RecipeBody(BaseModel):
    """`POST /recipes`. D, and only D.

    **`device` and `num_workers` are absent on purpose** (contract ⑩): a recipe
    carrying the device stops being comparable between CPU and GPU. They go in
    Entrenar, as execution options.

    The fields mirror `itf.training.recipe.Recipe`. Validation of the enums lives
    there, not here: the API owns HTTP, the domain owns meaning.
    """

    name: str = Field(min_length=1)
    lr: float = 1e-3
    optimizer: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    batch_size: int = Field(default=64, gt=0)
    grad_clip: float = 0.0
    epochs: int = Field(default=20, gt=0)
    scheduler: str = "none"
    warmup_epochs: int = 0
    patience: int = 0
    min_delta: float = 0.0
    lambda_pos: float = 1.0
    pos_weight: float | None = None
    smooth_l1_beta: float = 0.05
    monitor: str = "val_loss"
    seed: int = 1
