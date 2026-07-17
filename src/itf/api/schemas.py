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


class FeatureMapsBody(BaseModel):
    """`POST /runs/{name}/feature-maps` (V2).

    **The input is a patch** (contract ①): the patch is the real input of the CNN.
    A whole image is F's question and goes to `/predict`. Two ways to name one,
    and the pair is what makes the view usable:

    - `{patch_dataset, index}` — a patch of a B, by index. Its border flags come
      from the dataset, so they are the real ones.
    - `{patch: [[…]], border: […]}` — the pixels inline, for a patch that is not
      stored anywhere (V5's scrubber crops one out of an image).

    `border` is optional **only because a network without `border_features` never
    reads it**. If the network uses it and it is missing, the domain refuses --
    zeros would mean "touches no edge", which is a claim, not a default
    (formatos.md §2).
    """

    patch_dataset: str | None = None
    index: int | None = None
    #: Raw pixels, `(n, n)`, 0-255. `n` must equal the network's `input_size`.
    patch: list[list[float]] | None = None
    #: The 4 flags in BORDER_NAMES order: top, right, bottom, left.
    border: list[int] | None = None


class PredictBody(BaseModel):
    """`POST /runs/{name}/predict` (F, V11).

    `source` + `index` names an image of A; the path is resolved inside the domain
    and never sent by the client (D4).

    **The four knobs are F, not D** (organizacion.md §1-D): they are chosen per
    call over a model that is already trained, so they are sliders with live
    repaint rather than fields of a form you submit (ui.md §2). Sweeping them
    costs a forward pass; putting them in D would cost afternoons.
    """

    source: str = Field(min_length=1)
    index: int
    #: `p(exists)` above which a detection counts.
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: The stride of INFERENCE, chosen per call. **Not B's stride**, which is part
    #: of the dataset's identity -- the shared name is a dangerous coincidence
    #: (contract ⑤, glosario.md). Defaults to half the patch.
    stride: int | None = Field(default=None, gt=0)
    #: Defaults to `stride / 2`: two windows are `stride` apart, so anything
    #: closer than half of that is the same corner seen twice.
    nms_radius: float | None = Field(default=None, ge=0.0)
    #: Boxes thinner than this are two corners that happened to pair, not a
    #: paragraph.
    min_size: float = Field(default=4.0, ge=0.0)


class WindowBody(BaseModel):
    """`POST /runs/{name}/window` (V5, the scrubber).

    `source` + `index` names an image of A (the path is resolved inside the domain,
    D4); `(x0, y0)` is the top-left of the 40×40 crop being dragged over it. Unlike
    V2's patch, this window is **off-grid** -- it can sit anywhere -- and its border
    flags come from `geometry.window_at`, the same formula B extracted with, so a
    crop against an edge is flagged exactly as its trained-on twin (contract ⑤).

    A `POST` for a read, the same exception R3 allows the feature maps: it is driven
    by a live drag and the crop position is the whole payload.
    """

    source: str = Field(min_length=1)
    index: int
    x0: int = Field(ge=0)
    y0: int = Field(ge=0)


class CreateSweepBody(BaseModel):
    """`POST /sweeps` (H). **Names for what is fixed, a space for what varies.**

    `patch_dataset` and `network` are fixed by name (contract ⑧: every point must
    share the same B and C, or the sweep measures the ruler). `recipe` is an
    OPTIONAL base -- the fields not in `space` take its values.

    `space` is `{recipe_field: distribution}`, where a distribution is
    `{"type": "float"|"int", "low", "high", "log"?}` or
    `{"type": "categorical", "choices": [...]}`.

    `objective` is what ranks the points: `f1` | `pos_err_px` | `loss`. **It may
    not be `loss` while `lambda_pos` is in the space** (contract ⑨) -- that is a
    400, not a warning, because it produces a winner with a good face (λ=0).
    `device`/`num_workers` are absent for the same reason recipes lack them (⑩).
    """

    name: str = Field(min_length=1)
    patch_dataset: str = Field(min_length=1)
    network: str = Field(min_length=1)
    #: Optional base recipe; the fields not in `space` take its values.
    recipe: str | None = None
    space: dict[str, dict] = Field(default_factory=dict)
    objective: str = "f1"
    strategy: str = "tpe"
    budget: dict = Field(default_factory=dict)
    #: The sampler's seed (which points get tried). Distinct from B's split seed
    #: and D's replication seed.
    seed: int = 0


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
