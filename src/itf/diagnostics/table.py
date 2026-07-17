"""The per-patch table: E × split de B → one row per patch.

**This is the substrate of ui.md §3, and it is a CACHE, not an entity** (D1,
formatos.md §4.4). V6, V7 and V8 are the same pass over the val split; they must
not each re-run the model. And because the table is a pure function of things
that already have identity -- the run, B's fingerprint, the split -- it can be
recomputed exactly, and *what can be recomputed is not stored*: no name, no CRUD,
no listing, no screen. Deleting the cache directory loses nothing.

**What the table deliberately does NOT contain: the threshold.** That is the
whole point of the phase. `score` is the raw `p(exists)`, so choosing a threshold
is filtering a column -- free, post-hoc, and off a model that never runs again
(V8). Had the threshold entered the key, the cache would re-key on every point of
the curve and V8 would cost one full pass per threshold. `threshold` is F, not D
and not part of the measurement (organizacion.md §1-D): **barre D entrenando,
barre F evaluando**, and this file is the "evaluando" half.

What it also does not contain: the ground truth. `exists`, `xy` real,
`sample_idx` and `patch_xy` are already in B's `.npz` and `patch_idx` joins back
to them. A cache that copied them would be a second place for the truth to live,
and the two could then disagree about what the truth was.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from itf.inference import load_model
from itf.metrics import position_error_px
from itf.patches.dataset import PatchDataset


class NotMeasurable(ValueError):
    """This run and this split cannot produce a table, and the reason says why.

    Its own type because every case is a REFUSAL, not a crash: a run that never
    trained, a split with no patches, a B rebuilt under the same name. The API
    turns it into a 4xx with the reason; a traceback would read as "the tool is
    broken" and invite working around it -- and working around this one means
    measuring against the wrong data (fase 3's lesson, `NoValidationSplitError`).
    """

    def __init__(self, code: str, message: str, hint: str = ""):
        self.code = code
        self.hint = hint
        super().__init__(message)


@dataclass(frozen=True)
class TableKey:
    """What the table is a function of. Change any of it and it is another table.

    D1 fixes the key as `(run, huella de B, split, knobs)`. Two notes, both
    learned rather than copied:

    **`knobs` is empty and stays empty.** The only knob that could enter is
    `threshold`, and it must not: it is applied at aggregation time over the
    stored scores, which is precisely what makes V8 free. The field is here to
    say that on purpose rather than by omission -- a future knob that DOES change
    the numbers (an inference `stride`, say) belongs in here, and finding no
    field would be how it quietly did not get added.

    **`checkpoint_mtime_ns` is here and D1 did not ask for it**, because "run"
    only identifies a table if a run is immutable, and a run is not immutable
    while it trains: `best.pt` is rewritten on every improving epoch. Keyed on
    the run alone, opening Diagnóstico at epoch 5 and again at epoch 20 answers
    with epoch 5's table both times -- the cache would be lying, silently, which
    is the whole family of bug this project keeps finding. The old model cache
    keyed on mtime for the same reason (organizacion.md §2-④).
    """

    run: str
    fingerprint: str
    split: str
    checkpoint: str
    checkpoint_mtime_ns: int
    knobs: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "run": self.run,
            "fingerprint": self.fingerprint,
            "split": self.split,
            "checkpoint": self.checkpoint,
            "checkpoint_mtime_ns": self.checkpoint_mtime_ns,
            "knobs": self.knobs,
        }

    def digest(self) -> str:
        # `sort_keys`, or two identical keys hash differently depending on dict
        # insertion order and the cache misses forever without ever being wrong
        # -- the kind of bug that reads as "the diagnóstico is slow".
        raw = json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class PatchTable:
    """The four arrays of formatos.md §4.4, plus the key that produced them.

    `M` is the number of patches in the split, not in the whole dataset.
    """

    #: (M,) int32 — row in B's `.npz`. The join back to pixels and truth.
    patch_idx: np.ndarray
    #: (M, 4) float32 — `p(exists)` per corner. RAW: no threshold applied.
    score: np.ndarray
    #: (M, 4, 2) float32 — predicted position, normalised within the patch.
    xy_pred: np.ndarray
    #: (M, 4) float32 — error in px, **NaN where there is no real corner**.
    err_px: np.ndarray
    key: TableKey

    def __len__(self) -> int:
        return int(self.patch_idx.shape[0])


def compute_table(
    *,
    checkpoint: Path,
    patch_dataset: Path,
    key: TableKey,
    device: str = "cpu",
    batch_size: int = 256,
) -> PatchTable:
    """One pass of the model over one split. ~10⁴ forwards batched: seconds.

    That it is seconds is what lets `/diagnostics` be synchronous (R3 of api.md)
    and what makes the cache a convenience rather than a necessity.
    """
    dataset = PatchDataset(patch_dataset, split=key.split)
    if len(dataset) == 0:
        raise NotMeasurable(
            "split_empty",
            f"el split '{key.split}' de este dataset no tiene ni un patch, así que no hay "
            f"nada que medir.",
            "elige otro split, o reconstruye el dataset con una fracción > 0 para ese split",
        )

    model = load_model(checkpoint, device=device)
    patch_size = model.config.input_size

    scores: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    errors: list[np.ndarray] = []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x, border, y in loader:
            # `shuffle=False` above is load-bearing, not a default: the rows come
            # back in the dataset's order and are zipped against `indices` below.
            # Shuffled, every row would carry another patch's `patch_idx` and the
            # whole table would be quietly misattributed -- V6 showing the wrong
            # thumbnails, V7 the wrong positions, and no exception anywhere.
            x, border, y = x.to(device), border.to(device), y.to(device)
            pred = model(x, border)

            scores.append(torch.sigmoid(pred[..., 0]).cpu().numpy())
            positions.append(pred[..., 1:3].cpu().numpy())

            err = position_error_px(pred[..., 1:3], y[..., 1:3], patch_size)
            # **NaN where no corner exists, never 0** (formatos.md §4.4, and §2
            # underneath it): a 0 there would say "localised it exactly", which is
            # the opposite of "there was nothing to localise". It also keeps the
            # nanmean honest -- fill with zeros and every aggregate gets dragged
            # toward 0 by the ~80 % of corners that do not exist.
            err = torch.where(y[..., 0] > 0.5, err, torch.full_like(err, float("nan")))
            errors.append(err.cpu().numpy())

    return PatchTable(
        patch_idx=dataset.indices.astype(np.int32),
        score=np.concatenate(scores).astype(np.float32),
        xy_pred=np.concatenate(positions).astype(np.float32),
        err_px=np.concatenate(errors).astype(np.float32),
        key=key,
    )


@dataclass(frozen=True)
class TableCache:
    """`data/cache/diagnostics/`, one `.npz` + sidecar per key. Gitignored.

    The sidecar is the reason the key is readable **without opening the `.npz`**
    (formatos.md §4.4). It costs one small file and buys being able to say what a
    cache entry is a table of -- which is what turns a stale hit into a question
    someone can answer instead of a mystery.
    """

    root: Path

    def _paths(self, key: TableKey) -> tuple[Path, Path]:
        stem = f"{key.split}-{key.digest()}"
        directory = self.root / key.run
        return directory / f"{stem}.npz", directory / f"{stem}.json"

    def get(self, key: TableKey) -> PatchTable | None:
        npz_path, _ = self._paths(key)
        if not npz_path.exists():
            return None
        try:
            with np.load(npz_path) as data:
                return PatchTable(
                    patch_idx=data["patch_idx"],
                    score=data["score"],
                    xy_pred=data["xy_pred"],
                    err_px=data["err_px"],
                    key=key,
                )
        except (OSError, ValueError, KeyError):
            # A truncated or half-written cache entry is not an error to report:
            # it is a cache miss. The whole point of a cache that can be
            # recomputed exactly is that throwing it away is always safe.
            return None

    def put(self, table: PatchTable) -> None:
        npz_path, sidecar = self._paths(table.key)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            patch_idx=table.patch_idx,
            score=table.score,
            xy_pred=table.xy_pred,
            err_px=table.err_px,
        )
        sidecar.write_text(json.dumps(table.key.as_dict(), indent=2), encoding="utf-8")

    def get_or_compute(
        self, *, checkpoint: Path, patch_dataset: Path, key: TableKey, device: str = "cpu"
    ) -> PatchTable:
        cached = self.get(key)
        if cached is not None:
            return cached
        table = compute_table(
            checkpoint=checkpoint, patch_dataset=patch_dataset, key=key, device=device
        )
        self.put(table)
        return table
