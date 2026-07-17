"""The training loop: B + C + D + X -> E.

Writes a run directory:

    runs/<name>/
    ├── config.json      # frozen: recipe + network by value, plus provenance
    ├── metrics.jsonl    # one line per epoch, append-only, pollable live
    ├── best.pt / last.pt
    ├── summary.json
    └── status.json      # explicit state, so a crash is not "running" forever

Fase 3 got it training from the CLI. Fase 4 added the provenance by name (D2),
the cooperative stop, and the API around it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from itf.geometry import CORNER_NAMES
from itf.metrics import DEFAULT_THRESHOLD, position_error_px, prf1
from itf.models import build_model
from itf.patches.dataset import PatchDataset
from itf.training.losses import CornerLoss
from itf.training.provenance import PROVENANCE_FIELDS
from itf.training.recipe import MONITOR_DIRECTION, Recipe
from itf.training.registry import RunExists, write_json_atomic
from itf.validation import check_compatible

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


class IncompatibleError(ValueError):
    """The network cannot train on this dataset (contracts ① and ②)."""

    def __init__(self, problems: list[dict]):
        self.problems = problems
        super().__init__(
            "; ".join(f"[{p['code']}] {p['message']} -> {p['hint']}" for p in problems)
        )


class NoValidationSplitError(ValueError):
    """The dataset has no val split, so `best.pt` cannot be chosen (protocolo.md §1.3).

    Its own type, not a bare `ValueError`, for one reason: the callers want to
    tell this apart from a bug. It is a **refusal**, not a crash -- the system
    working as designed -- so `itf-train` prints it as a message and exits 2,
    the same way it treats an incompatible network. A traceback would read as
    "the tool broke" when what happened is "your dataset cannot measure".
    """


@dataclass
class RunSpec:
    """What a run needs. Four domains, kept APART rather than flattened.

    The old `RunConfig` flattened B, C, D and X into one dataclass -- the YAML
    separated `model:` from `train:` and the CLI re-flattened them, so the
    separation existed in the file and was lost in the code. Here `device` is a
    field of the RUN, never of the recipe (contract ⑩).
    """

    #: Path to the patch dataset directory (B).
    data: str
    #: Where the run goes (E).
    out: str
    #: The network's config, by value (C).
    network: dict
    #: Where this run came from, by NAME (contract ③). **Required, with no
    #: default**: every run is born with it (formatos.md §4.2), so there is no
    #: way to spell "a run with no provenance" -- which is the hole the contract
    #: describes. Build it with `provenance.build_provenance`.
    provenance: dict
    #: The recipe (D).
    recipe: Recipe = field(default_factory=Recipe)
    #: X. Costs time, does not change the result. Outside D's identity.
    device: str = "cpu"
    num_workers: int = 0

    def __post_init__(self) -> None:
        missing = set(PROVENANCE_FIELDS) - set(self.provenance)
        if missing:
            # Failing with the reason rather than writing a partial block: an
            # absent field is not a zero (formatos.md §2), and a run that cannot
            # say where it came from is not comparable with anything.
            raise ValueError(f"a la procedencia le faltan campos: {sorted(missing)}")


def frozen_config(spec: RunSpec) -> dict:
    """`runs/<name>/config.json` (formatos.md §4.2).

    One function, because two callers write this file: `RunStore.create` reserves
    the name with it before the job starts, and `train` writes it again as it
    begins. Building the dict in each of them separately is how the reservation
    and the run would come to disagree about what was trained.
    """
    return {
        "format_version": 1,
        "network": spec.network,
        "recipe": spec.recipe.as_dict(),
        # X is recorded but sits OUTSIDE the recipe: two runs differing only in
        # `device` have the same recipe identity (contract ⑩).
        "execution": {"device": spec.device, "num_workers": spec.num_workers},
        # Contract ③. Note what is NOT here any more: `data`, the absolute path
        # to B. A path is machine-specific, and following it was what broke the
        # provenance the moment the dataset was rebuilt or moved. The name and
        # the fingerprint say the same thing, and survive.
        "provenance": spec.provenance,
    }


def _make_optimizer(recipe: Recipe, params):
    """Build the optimizer, passing EVERY knob the chosen one takes.

    The trap this replaces: the old version passed only `lr` and `weight_decay`,
    so `sgd` silently ran at momentum=0 -- and any comparison of optimizers was
    biased in Adam's favour before it started.
    """
    key = recipe.optimizer
    kwargs: dict = {"lr": recipe.lr, "weight_decay": recipe.weight_decay}
    if key in {"sgd", "rmsprop"}:
        kwargs["momentum"] = recipe.momentum
    return _OPTIMIZERS[key](params, **kwargs)


def _make_scheduler(recipe: Recipe, optimizer):
    """`lr` is constant without this, which is the most expensive omission."""
    if recipe.scheduler == "none":
        return None
    if recipe.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe.epochs)
    if recipe.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, recipe.epochs // 3), gamma=0.1)
    if recipe.scheduler == "plateau":
        mode = MONITOR_DIRECTION[recipe.monitor]
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=2)
    raise ValueError(f"scheduler desconocido: {recipe.scheduler}")


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, patch_size: int) -> dict:
    """Per-epoch PATCH metrics: cheap, and what picks `best.pt` inside a run.

    Note what these are NOT: the paragraph F1 (protocolo.md §2), which is the
    real objective and the correct thing for a sweep to rank by. These stay
    useful for choosing a checkpoint and for diagnosis -- IF step 2 of the
    protocol confirms they predict the paragraph F1.

    **The definitions of `pos_err_px` and of the P/R/F1 come from `itf.metrics`,
    not from here** (fase 5). The per-patch table behind V7 and V8 measures the
    same two things over the same split, so a second copy of these formulas would
    let the error map and this curve disagree while both are called `pos_err_px`
    -- silently, which is the ⑤-shaped failure.
    """
    model.eval()
    tot_loss = tot_cls = tot_pos = 0.0
    n_batches = 0
    tp = fp = fn = tn = 0
    pos_err_sum = 0.0
    pos_count = 0
    for x, border, y in loader:
        x, border, y = x.to(device), border.to(device), y.to(device)
        pred = model(x, border)
        out = loss_fn(pred, y)
        tot_loss += float(out["loss"])
        tot_cls += float(out["cls_loss"])
        tot_pos += float(out["pos_loss"])
        n_batches += 1

        pred_pos = torch.sigmoid(pred[..., 0]) > DEFAULT_THRESHOLD
        tgt_pos = y[..., 0] > 0.5
        tp += int((pred_pos & tgt_pos).sum())
        fp += int((pred_pos & ~tgt_pos).sum())
        fn += int((~pred_pos & tgt_pos).sum())
        tn += int((~pred_pos & ~tgt_pos).sum())

        if tgt_pos.any():
            err_px = position_error_px(pred[..., 1:3], y[..., 1:3], patch_size)  # (B, 4)
            pos_err_sum += float(err_px[tgt_pos].sum())
            pos_count += int(tgt_pos.sum())

    return {
        "loss": tot_loss / max(n_batches, 1),
        "cls_loss": tot_cls / max(n_batches, 1),
        "pos_loss": tot_pos / max(n_batches, 1),
        "exists_acc": (tp + tn) / max(tp + tn + fp + fn, 1),
        **prf1(tp, fp, fn),
        # None, not 0: no corners means "not measured", and a 0 would read as
        # "perfect" (formatos.md §2 again).
        "pos_err_px": pos_err_sum / pos_count if pos_count else None,
    }


def _monitor_value(recipe: Recipe, val_metrics: dict) -> float | None:
    key = {"val_loss": "loss", "val_f1": "f1", "val_pos_err_px": "pos_err_px"}[recipe.monitor]
    value = val_metrics.get(key)
    return None if value is None else float(value)


def train(
    spec: RunSpec,
    on_epoch: Callable[[int, dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Train one run.

    Args:
        on_epoch: called with each epoch's record, for live reporting.
        should_stop: asked at the end of every epoch. **Cooperative**: the epoch
            finishes, its metrics and checkpoint land, and the run closes as
            `cancelled`. Nothing is killed -- cutting mid-batch would leave a
            half-written `last.pt`. `RunStore.request_stop` is what usually sets
            it.
    """
    recipe = spec.recipe

    # The safety net (api.md §3). The API answers 400 before creating the job,
    # but `itf-train` DOES NOT GO THROUGH THE API -- without this the CLI walks
    # past the gate and the failure reappears mid-epoch as a shape error. It is
    # pure and costs nothing, which is why calling it twice is free.
    manifest_path = Path(spec.data) / "manifest.json"
    if manifest_path.exists():
        problems = check_compatible(
            json.loads(manifest_path.read_text(encoding="utf-8")), spec.network
        )
        if problems:
            raise IncompatibleError(problems)

    torch.manual_seed(recipe.seed)
    np.random.seed(recipe.seed)
    device = torch.device(spec.device)

    train_ds = PatchDataset(spec.data, split="train")
    val_ds = PatchDataset(spec.data, split="val")

    # protocolo.md §1.3. `monitor = val_metrics.get("loss", train_loss)` fell back
    # to the TRAIN loss with no warning, so `best.pt` became the most overfitted
    # checkpoint -- and it happened to `reducido-40`, the README's own example.
    # A dataset with no val is not a measuring tool: building it warns (fase 2),
    # and training on it refuses. Here.
    if len(val_ds) == 0:
        raise NoValidationSplitError(
            f"el dataset '{spec.data}' no tiene val, así que no se puede elegir best.pt "
            f"por '{recipe.monitor}'. Reconstrúyelo con una fracción de val > 0: sin val, "
            f"elegir checkpoint cae en la pérdida de entrenamiento y se queda el más "
            f"sobreajustado, en silencio."
        )

    train_loader = DataLoader(
        train_ds, batch_size=recipe.batch_size, shuffle=True, num_workers=spec.num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=recipe.batch_size, num_workers=spec.num_workers)

    model = build_model(spec.network).to(device)
    patch_size = model.config.input_size
    loss_fn = CornerLoss(
        lambda_pos=recipe.lambda_pos,
        pos_weight=recipe.pos_weight,
        smooth_l1_beta=recipe.smooth_l1_beta,
    ).to(device)
    optimizer = _make_optimizer(recipe, model.parameters())
    scheduler = _make_scheduler(recipe, optimizer)

    out_dir = Path(spec.out)
    # The safety net against the overwrite trap. `RunStore.create` is the gate
    # both callers pass through, but this function can be called directly, and
    # `mkdir(exist_ok=True)` plus truncating `metrics.jsonl` is exactly how the
    # old code destroyed finished runs without a word -- with a sweep that
    # auto-generates names as the one who steps on it. `metrics.jsonl` is the
    # tell: a directory holding only the reservation (config + status) is a name
    # being claimed, but one with metrics in it is a RESULT.
    if (out_dir / "metrics.jsonl").exists():
        raise RunExists(
            f"'{out_dir}' ya tiene métricas: sobrescribirlo machacaría un run terminado. "
            f"Elige otro nombre, o borra ese run primero."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    # **Only if it is not there already.** `RunStore.create` wrote it when it
    # reserved the name, with this same function, so rewriting it is not merely
    # redundant: it truncates a file that pollers are reading, and a `GET` landing
    # in that window answered 404 "config.json ilegible" about a perfectly
    # healthy run. This branch is for the direct `train()` call, which nobody
    # reserved for.
    config_path = out_dir / "config.json"
    if not config_path.exists():
        write_json_atomic(config_path, frozen_config(spec), indent=2)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    status_path = out_dir / "status.json"
    write_json_atomic(status_path, {"state": "running", "epoch": 0})

    direction = MONITOR_DIRECTION[recipe.monitor]
    # **None until something is measured, never ±inf**, and it is formatos.md §2
    # rather than taste. An infinite sentinel is not a measurement; it is the
    # absence of one, and it does not survive the trip out: `json.dumps` writes
    # it as `Infinity`, which is not valid JSON and which **no browser can
    # parse** -- so one run whose monitor never fired would take down `GET /runs`
    # for every other run on the screen. The path is real, not theoretical:
    # `monitor: val_pos_err_px` over a val split with no corners returns None
    # every epoch, and `best` would never leave its sentinel.
    best: float | None = None
    better = (lambda a, b: a < b - recipe.min_delta) if direction == "min" else (lambda a, b: a > b + recipe.min_delta)
    since_improved = 0
    history: list[dict] = []
    stopped_early = False
    cancelled = False

    try:
        for epoch in range(1, recipe.epochs + 1):
            model.train()
            t0 = time.time()
            run_loss = 0.0
            n_batches = 0

            if recipe.warmup_epochs and epoch <= recipe.warmup_epochs:
                for group in optimizer.param_groups:
                    group["lr"] = recipe.lr * epoch / recipe.warmup_epochs

            for x, border, y in train_loader:
                x, border, y = x.to(device), border.to(device), y.to(device)
                optimizer.zero_grad()
                out = loss_fn(model(x, border), y)
                out["loss"].backward()
                if recipe.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip)
                optimizer.step()
                run_loss += out["loss"].detach().item()
                n_batches += 1

            train_loss = run_loss / max(n_batches, 1)
            val_metrics = evaluate(model, val_loader, loss_fn, device, patch_size)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val": val_metrics,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": round(time.time() - t0, 3),
            }
            history.append(record)
            # Append-only: a live run is being read while it is written.
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            if on_epoch:
                on_epoch(epoch, record)

            torch.save(
                {"model": model.state_dict(), "config": {"model": spec.network}, "epoch": epoch},
                out_dir / "last.pt",
            )

            monitor = _monitor_value(recipe, val_metrics)
            if monitor is not None and (best is None or better(monitor, best)):
                best = monitor
                since_improved = 0
                # The checkpoint is SELF-DESCRIBING (contract ④): it carries the
                # network config, so `load_model` rebuilds the net with no YAML
                # and no dataset. The cleanest dependency in the project.
                torch.save(
                    {"model": model.state_dict(), "config": {"model": spec.network}, "epoch": epoch},
                    out_dir / "best.pt",
                )
            else:
                since_improved += 1

            if scheduler is not None and not (recipe.warmup_epochs and epoch <= recipe.warmup_epochs):
                if recipe.scheduler == "plateau":
                    if monitor is not None:
                        scheduler.step(monitor)
                else:
                    scheduler.step()

            # Atomic, because this is the file every poll reads: rewritten
            # every epoch with `write_text`, a reader lands in the truncated
            # window and sees a healthy run as `error`.
            write_json_atomic(status_path, {"state": "running", "epoch": epoch})

            # The end of an epoch is the safe point: metrics written, checkpoint
            # saved. Asked BEFORE `patience`, so a run the user stopped does not
            # report itself as having converged.
            if should_stop is not None and should_stop():
                cancelled = True
                break

            if recipe.patience and since_improved >= recipe.patience:
                stopped_early = True
                break
    except BaseException as exc:
        # Explicit state, so a crash does not read as "running" forever the way
        # inferring it from which files exist did.
        write_json_atomic(status_path, {"state": "error", "error": f"{type(exc).__name__}: {exc}"})
        raise

    summary = {
        "run": out_dir.name,
        "epochs_run": len(history),
        "epochs_requested": recipe.epochs,
        "stopped_early": stopped_early,
        "cancelled": cancelled,
        "monitor": recipe.monitor,
        # None when the monitor never produced a number: "not measured", which is
        # not the same as "infinitely bad" and must not be written as a number.
        "best": best,
        "final": history[-1] if history else {},
        "corner_order": list(CORNER_NAMES),
    }
    # Summary BEFORE status, and atomically: the moment status says `done`, a
    # poller reads `summary.json` — so it must already be there, whole.
    write_json_atomic(out_dir / "summary.json", summary, indent=2)
    # `cancelled`, not `done`: a run cut at epoch 4 of 20 has real weights and a
    # real summary, but calling it done would let it into a comparison as though
    # it had finished (protocolo.md §7).
    write_json_atomic(
        status_path, {"state": "cancelled" if cancelled else "done", "epoch": len(history)}
    )
    return summary
