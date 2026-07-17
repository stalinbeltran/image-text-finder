"""D — Receta: the hyperparameters that DEFINE the result.

A recipe is a noun in this project — it has a name, a store, and gets compared
and reused — and that is a decision, not an accident: **a sweep is literally "a
list of D with B and C fixed"**, so without an identity for D there is no H.

The membership rule, in order (CLAUDE.md):

  1. Does it change the SHAPE of the model? -> C, even if it sounds like training
     (`dropout`, `batchnorm`, `border_features`).
  2. Does it change the resulting WEIGHTS? -> D, even if it sounds like
     architecture (`lambda_pos`, `pos_weight`).
  3. Does it only change how LONG it takes? -> X. Never inside D's identity.
  4. Is it tunable POST-HOC without retraining? -> F (`threshold`, inference
     stride, NMS). Sweeping those costs minutes, not hours; do not put them here.

**`device` and `num_workers` are not here, and their absence is the point**
(contract ⑩). They lived inside `RunConfig` and got frozen into `config.json`, so
the same recipe trained on CPU and on GPU produced two different files and looked
like two recipes. X had to come out BEFORE the GPU arrived, or everything trained
on CPU today would be incomparable with tomorrow.

The trap next door: **`batch_size` IS D**, however much VRAM invites raising it.
It changes the result, and it is coupled to `lr`. A bigger batch on GPU is a NEW
POINT of the space, not the same point going faster.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

FORMAT_VERSION = 1

OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop")
SCHEDULERS = ("none", "cosine", "step", "plateau")
MONITORS = ("val_loss", "val_f1", "val_pos_err_px")

#: Which way each monitor is "better". `loss` and `pos_err_px` go down, `f1` up.
MONITOR_DIRECTION = {"val_loss": "min", "val_f1": "max", "val_pos_err_px": "min"}


@dataclass
class Recipe:
    """The full catalogue of organizacion.md §1-D.

    Every field carries its definition, because a hyperparameter without one in
    the catalogue is not finished -- and because these definitions are what the
    Recetas screen shows inline.
    """

    # ── Optimisation: how each step is taken ──────────────────────────────────
    #: Step size. The most influential one. Sweep in LOG scale (1e-4…3e-2), never
    #: linear.
    lr: float = 1e-3
    #: adam | adamw | sgd | rmsprop.
    optimizer: str = "adam"
    #: Inertia of the step; only applies to sgd/rmsprop.
    #:
    #: **A trap, and it was live.** The old `_make_optimizer` passed only `lr` and
    #: `weight_decay`, so SGD ran at momentum=0 -- torch's default. Sweeping
    #: `optimizer` therefore compared Adam against a crippled SGD: it always lost,
    #: and not for being SGD. Nobody chose that; it came from not choosing.
    momentum: float = 0.9
    #: L2 penalty on the weights. Log, 0…1e-2. In `adam` it is coupled L2; in
    #: `adamw` it is decoupled -- NOT the same thing despite the shared name.
    weight_decay: float = 0.0
    #: Samples per gradient step. **Coupled to `lr`**: doubling the batch usually
    #: wants a higher lr. Sweep them together or fix one. This is D (contract ⑩).
    batch_size: int = 64
    #: Clip the gradient norm before the step. 0 = off.
    #: Cheap, and it stabilises high lr: without it the top end of an `lr` sweep
    #: just diverges and you learn nothing about that region.
    grad_clip: float = 0.0

    # ── Duration and schedule ────────────────────────────────────────────────
    epochs: int = 20
    #: How `lr` decays: none | cosine | step | plateau.
    #:
    #: **The most expensive omission.** With `lr` constant, the optimal starting
    #: lr is a different number than it is with a schedule -- so sweeping `lr`
    #: without one optimises for a regime you will not use.
    scheduler: str = "none"
    #: Epochs ramping `lr` from ~0. Only matters with a big batch or a high lr.
    warmup_epochs: int = 0
    #: Early stopping: epochs without improvement before cutting. 0 = off.
    #: Per-run, watching ITS OWN val curve. **Distinct from the sweep's pruning**,
    #: which compares BETWEEN runs (contract ⑨).
    patience: int = 0
    #: How much counts as "improving" for `patience`.
    min_delta: float = 0.0

    # ── Loss: what is specific to this task ──────────────────────────────────
    #: Weight of the position term against the existence term. Arbitrates
    #: "is there a corner?" vs "where exactly?" -- the most interesting knob in
    #: the project, and the one that needs most care when ranking (contract ⑨).
    lambda_pos: float = 1.0
    #: Weight of the positive class in the BCE. The real imbalance measured on
    #: `clear-paragraphs-02` is 20.5% positives => **3.9:1**. Modest, not brutal.
    #: Start a sweep there, and read it from
    #: `manifest.positives_per_corner / num_patches` -- it changes per dataset.
    pos_weight: float | None = None
    #: Where smoothL1 turns from quadratic to linear.
    #:
    #: **A trap, and it was live.** torch's default is 1.0 and the coordinates are
    #: normalised to [0,1], so |error| < 1 ALWAYS: the loss never leaves the
    #: quadratic branch. The position loss was plain MSE and Huber's robustness
    #: never once activated -- with nobody deciding that. ~0.05–0.1 makes it do
    #: something (≈2–4 px in a 40px patch).
    smooth_l1_beta: float = 0.05

    # ── Selection and reproducibility ────────────────────────────────────────
    #: Which metric picks `best.pt`. Was hardcoded to val loss.
    #:
    #: **It cannot be `val_loss` if you sweep λ** (contract ⑨): `loss = cls +
    #: λ·pos`, so each point would be measured with a different loss function and
    #: λ=0 "wins" by definition -- i.e. the optimum is to not predict positions.
    monitor: str = "val_loss"
    #: Seed of D: weight init and shuffle. **Not a hyperparameter to optimise --
    #: it is the REPLICATION AXIS.** Fix everything and vary only this and you
    #: measure the noise floor, which is the only thing that tells you whether a
    #: difference between two recipes is real. Distinct from B's seed, which
    #: fixes the split (contract ⑧).
    seed: int = 1

    def __post_init__(self) -> None:
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer desconocido '{self.optimizer}'; hay: {list(OPTIMIZERS)}")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler desconocido '{self.scheduler}'; hay: {list(SCHEDULERS)}")
        if self.monitor not in MONITORS:
            raise ValueError(f"monitor desconocido '{self.monitor}'; hay: {list(MONITORS)}")

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known - {"format_version"}
        if unknown:
            # Not tolerated on purpose: a typo'd `momentun` that is silently
            # dropped is a recipe that does not do what it says.
            raise ValueError(f"campos que no son de la receta: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    def as_dict(self) -> dict:
        return asdict(self)


class RecipeNotFound(KeyError):
    """No such recipe."""


@dataclass(frozen=True)
class RecipeStore:
    """`configs/recipes/*.yaml`. YAML because a person writes them; versioned in
    git because they are source, not artifacts (formatos.md §4.3)."""

    root: Path

    def path(self, name: str) -> Path:
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"nombre de receta inválido: {name!r}")
        return self.root / f"{name}.yaml"

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.yaml"))

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def get(self, name: str) -> Recipe:
        path = self.path(name)
        if not path.exists():
            raise RecipeNotFound(name)
        return Recipe.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    def save(self, name: str, recipe: Recipe) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": FORMAT_VERSION, **recipe.as_dict()}
        self.path(name).write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    def delete(self, name: str) -> None:
        path = self.path(name)
        if not path.exists():
            raise RecipeNotFound(name)
        path.unlink()
