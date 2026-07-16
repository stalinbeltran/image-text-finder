"""`itf-train` — train from the command line.

Takes NAMES (api.md R7): a network and a recipe that exist in `configs/`. It
looks rigid and it is deliberate -- it is what makes contract ③ hold by itself.
Want something bespoke? Save it first.

`--device` is a flag, not a recipe field: it is X (contract ⑩).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from itf.models import NetworkStore
from itf.settings import Settings
from itf.training.loop import IncompatibleError, NoValidationSplitError, RunSpec, train
from itf.training.recipe import RecipeStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="itf-train", description="Entrena un run (E)")
    parser.add_argument("--name", required=True, help="Nombre del run")
    parser.add_argument("--patch-dataset", required=True, help="Nombre del dataset de patches (B)")
    parser.add_argument("--network", required=True, help="Nombre de la red (C)")
    parser.add_argument("--recipe", required=True, help="Nombre de la receta (D)")
    parser.add_argument("--device", default="cpu", help="X: no cambia el resultado, solo el tiempo")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    settings = Settings.from_env()
    networks = NetworkStore(settings.networks_root)
    recipes = RecipeStore(settings.recipes_root)

    try:
        network = networks.get(args.network)
    except KeyError:
        parser.error(f"no existe la red '{args.network}'. Hay: {networks.names()}")
    try:
        recipe = recipes.get(args.recipe)
    except KeyError:
        parser.error(f"no existe la receta '{args.recipe}'. Hay: {recipes.names()}")

    data = settings.patch_datasets_root / args.patch_dataset
    if not (data / "manifest.json").exists():
        parser.error(f"no existe el dataset de patches '{args.patch_dataset}'")

    out = settings.runs_root / args.name
    # Never overwrite in silence. `mkdir(exist_ok=True)` plus truncating
    # metrics.jsonl destroys results without a word, and a sweep that
    # auto-generates names is exactly who steps on it.
    if out.exists():
        parser.error(f"el run '{args.name}' ya existe: {out}")

    network.pop("format_version", None)
    spec = RunSpec(
        data=str(data),
        out=str(out),
        network=network,
        recipe=recipe,
        device=args.device,
        num_workers=args.num_workers,
    )

    def report(epoch: int, record: dict) -> None:
        val = record["val"]
        err = val["pos_err_px"]
        # `pos_err_px` is None when val held no corners at all -- print "n/a",
        # never 0.0, which would read as a perfect localisation.
        err_text = f"{err:.1f}px" if err is not None else "n/a"
        print(
            f"época {epoch:3d}  train {record['train_loss']:.4f}  "
            f"val {val['loss']:.4f}  f1 {val['f1']:.3f}  "
            f"pos_err {err_text}  {record['seconds']:.1f}s"
        )

    try:
        summary = train(spec, on_epoch=report)
    except IncompatibleError as exc:
        # ① and ②, caught before a single batch runs.
        print("\nLa red y el dataset no son compatibles:\n")
        for p in exc.problems:
            print(f"  [{p['code']}] {p['message']}\n    -> {p['hint']}")
        raise SystemExit(2)
    except NoValidationSplitError as exc:
        # protocolo.md §1.3. A REFUSAL, not a crash: printed like one, because a
        # traceback here reads as "the tool broke" and invites working around it
        # -- and the way around it is exactly the trap (best.pt by train loss).
        print(f"\nEste dataset no sirve para medir:\n\n  {exc}")
        raise SystemExit(2)

    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
