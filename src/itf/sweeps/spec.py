"""H — the sweep spec, and the rule that keeps its objective honest (contract ⑨).

**Pure: no optuna here.** The spec is a description -- what is fixed (a B and a
C), what varies (a space over D), what it optimises, and the budget -- and its
validation is two dicts, like `itf.validation`. Keeping optuna out of this module
is what lets `POST /sweeps` answer the ⑨ 400 in milliseconds without importing the
engine, and what makes the spec serialisable to `sweeps/<name>/spec.json`
untouched (formatos.md §4.5).

The engine lives in `runner.py`; the organisation lives here. That is the border
librerias.md draws: optuna is *inside* H, not *instead of* it -- its trials are
not our runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from itf.training.recipe import Recipe

#: The objectives a sweep can rank by, and which way each is "better". These are
#: the PATCH metrics `evaluate()` already writes each epoch (they pick `best.pt`
#: too). The real objective is the paragraph F1 (protocolo.md §2), which is
#: λ-independent by construction -- but it does not exist yet (it waits on D7,
#: bbox-vs-rotation), so a sweep ranks by these until it does.
OBJECTIVE_DIRECTION: dict[str, str] = {
    "f1": "maximize",
    "pos_err_px": "minimize",
    "loss": "minimize",
}

#: objective name -> key inside a metric record's `val` block.
OBJECTIVE_METRIC: dict[str, str] = {"f1": "f1", "pos_err_px": "pos_err_px", "loss": "loss"}

STRATEGIES = ("random", "tpe", "grid")

#: The recipe fields that make `loss` scale differently, so ranking by `loss`
#: while sweeping them is the ⑨ trap. `lambda_pos` is the one the contract names:
#: `loss = cls + λ·pos`, so λ=0 "wins" by not predicting positions at all.
_LOSS_SCALING_FIELDS = ("lambda_pos",)


@dataclass(frozen=True)
class Budget:
    """How much CPU the sweep may spend, and whether it prunes."""

    #: Number of points of the space to try. Each is one run.
    points: int
    #: Epochs per point. Short on purpose: a sweep spends its time here, and the
    #: pruner (below) cuts most points long before they reach this.
    epochs: int
    #: Pruning is **the number-one lever on CPU** (organizacion.md §H): most
    #: points look bad by epoch 3, and cutting them there halves the sweep. On by
    #: default; off makes every point run its full `epochs`.
    pruning: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Budget":
        return cls(points=int(d["points"]), epochs=int(d["epochs"]), pruning=bool(d.get("pruning", True)))

    def as_dict(self) -> dict:
        return {"points": self.points, "epochs": self.epochs, "pruning": self.pruning}


@dataclass(frozen=True)
class SweepSpec:
    """A space of D explored with B and C fixed → many E.

    `recipe` is an OPTIONAL base: the fields NOT in `space` take its values (or
    `Recipe()` defaults if absent), and `space` overrides the rest. It keeps a
    sweep from having to restate the whole catalogue to vary one knob.
    """

    name: str
    patch_dataset: str
    network: str
    space: dict[str, dict]
    objective: str
    strategy: str
    budget: Budget
    recipe: str | None = None
    #: The RNG seed of the SAMPLER (which points get tried), distinct from the two
    #: seeds already in the system: B's split seed (⑧) and D's replication seed.
    #: Fixed so a resumed sweep proposes the same points.
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "SweepSpec":
        return cls(
            name=d["name"],
            patch_dataset=d["patch_dataset"],
            network=d["network"],
            space=dict(d["space"]),
            objective=d["objective"],
            strategy=d.get("strategy", "tpe"),
            budget=Budget.from_dict(d["budget"]),
            recipe=d.get("recipe"),
            seed=int(d.get("seed", 0)),
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "patch_dataset": self.patch_dataset,
            "network": self.network,
            "space": self.space,
            "objective": self.objective,
            "strategy": self.strategy,
            "budget": self.budget.as_dict(),
            "recipe": self.recipe,
            "seed": self.seed,
        }

    @property
    def direction(self) -> str:
        return OBJECTIVE_DIRECTION[self.objective]


def _problem(code: str, message: str, hint: str) -> dict:
    return {"code": code, "message": message, "hint": hint}


def check_sweep(spec: dict) -> list[dict]:
    """Everything that makes a sweep spec impossible, before any CPU is spent.

    Returns a list of problems (`code`/`message`/`hint`, R4 of api.md); empty means
    the spec is runnable. The one that gives this its name is **⑨**: ranking by
    `loss` while sweeping `lambda_pos` produces a winner with a good face (λ=0),
    so it is a refusal, not a warning.
    """
    problems: list[dict] = []
    recipe_fields = set(Recipe.__dataclass_fields__)

    objective = spec.get("objective")
    if objective not in OBJECTIVE_DIRECTION:
        problems.append(
            _problem(
                "unknown_objective",
                f"objetivo desconocido '{objective}'",
                f"usa uno de: {sorted(OBJECTIVE_DIRECTION)}",
            )
        )

    strategy = spec.get("strategy", "tpe")
    if strategy not in STRATEGIES:
        problems.append(
            _problem(
                "unknown_strategy",
                f"estrategia desconocida '{strategy}'",
                f"usa una de: {list(STRATEGIES)}",
            )
        )

    space = spec.get("space") or {}
    if not space:
        problems.append(
            _problem(
                "empty_space",
                "el espacio del barrido está vacío: no hay nada que variar",
                "añade al menos un campo de la receta con su rango",
            )
        )
    for name, dist in space.items():
        if name not in recipe_fields:
            problems.append(
                _problem(
                    "unknown_space_field",
                    f"'{name}' no es un campo de la receta, así que no se puede barrer",
                    f"barre uno de: {sorted(recipe_fields)}",
                )
            )
            continue
        problems += _check_distribution(name, dist)

    # ⑨ — the objective must be independent of the space that is being swept.
    if objective == "loss":
        varying = [f for f in _LOSS_SCALING_FIELDS if f in space]
        if varying:
            problems.append(
                _problem(
                    "objective_varies_with_space",
                    (
                        f"no puedes rankear por 'loss' mientras barres {varying}: "
                        f"loss = cls + λ·pos, así que cada punto se mide con una pérdida "
                        f"distinta y λ=0 gana por definición (no predecir posiciones)"
                    ),
                    "rankea por 'f1' o 'pos_err_px' (independientes de λ), o saca lambda_pos del espacio",
                )
            )

    budget = spec.get("budget") or {}
    for key in ("points", "epochs"):
        value = budget.get(key)
        if not isinstance(value, int) or value < 1:
            problems.append(
                _problem(
                    "invalid_budget",
                    f"budget.{key} debe ser un entero >= 1 (es {value!r})",
                    "p. ej. {'points': 20, 'epochs': 10}",
                )
            )
    return problems


def _check_distribution(name: str, dist: Any) -> list[dict]:
    """One space field: a float/int range or a categorical list."""
    if not isinstance(dist, dict) or "type" not in dist:
        return [_problem("invalid_distribution", f"'{name}': falta 'type'", "usa float | int | categorical")]
    kind = dist["type"]
    if kind in ("float", "int"):
        if "low" not in dist or "high" not in dist:
            return [_problem("invalid_distribution", f"'{name}' ({kind}) necesita 'low' y 'high'", "p. ej. {'type':'float','low':1e-4,'high':3e-2,'log':true}")]
        if dist["low"] > dist["high"]:
            return [_problem("invalid_distribution", f"'{name}': low > high", "invierte los límites")]
        return []
    if kind == "categorical":
        if not dist.get("choices"):
            return [_problem("invalid_distribution", f"'{name}' (categorical) necesita 'choices'", "p. ej. {'type':'categorical','choices':['adam','adamw']}")]
        return []
    return [_problem("invalid_distribution", f"'{name}': tipo desconocido '{kind}'", "usa float | int | categorical")]


def build_recipe_dict(base: dict, overrides: dict, epochs: int) -> dict:
    """The recipe for one point: the base, overridden by the sampled point.

    `epochs` comes from the budget, not the base: a sweep decides how long each
    point runs. Everything else the point does not touch keeps the base's value.
    """
    return {**base, **overrides, "epochs": epochs}
