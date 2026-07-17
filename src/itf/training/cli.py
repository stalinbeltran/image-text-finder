"""`itf-train` — train from the command line.

Takes NAMES (api.md R7): a network and a recipe that exist in `configs/`. It
looks rigid and it is deliberate -- it is what makes contract ③ hold by itself.
Want something bespoke? Save it first.

`--device` is a flag, not a recipe field: it is X (contract ⑩).

It reserves the run through `RunStore.create`, the same gate `POST /runs` uses.
That is the point of the gate living in the domain: **the CLI does not go through
the API**, so a check that only lived in the route would guard one of the two
doors -- and the door it left open is the one a sweep script would come through.
"""

from __future__ import annotations

import argparse
import json

from itf.models import NetworkStore
from itf.patches.store import PatchDatasetStore
from itf.settings import Settings
from itf.training.loop import IncompatibleError, NoValidationSplitError, RunSpec, frozen_config, train
from itf.training.provenance import build_provenance
from itf.training.recipe import RecipeStore
from itf.training.registry import RunExists, RunStore
from itf.validation import check_run


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
    patch_datasets = PatchDatasetStore(settings.patch_datasets_root)
    runs = RunStore(settings.runs_root)

    # `ValueError` alongside `KeyError` on purpose: the stores raise it for a name
    # that is not a single directory component (`--network ../evil`). Catching
    # only KeyError turns that into a traceback, while the API answers the same
    # mistake with a 400 -- and two doors that disagree about what a bad name is
    # are two doors waiting to drift.
    try:
        network = networks.get(args.network)
    except (KeyError, ValueError):
        parser.error(f"no existe la red '{args.network}'. Hay: {networks.names()}")
    try:
        recipe = recipes.get(args.recipe)
    except KeyError:
        parser.error(f"no existe la receta '{args.recipe}'. Hay: {recipes.names()}")
    except ValueError as exc:
        # A recipe that exists but does not parse -- a hand-edited YAML with a
        # typo'd field. "It does not exist" would send you looking in the wrong
        # place, which is the one thing R4 is about.
        parser.error(f"la receta '{args.recipe}' no es válida: {exc}")
    try:
        manifest = patch_datasets.manifest(args.patch_dataset)
    except (KeyError, ValueError):
        parser.error(
            f"no existe el dataset de patches '{args.patch_dataset}'. "
            f"Hay: {patch_datasets.names()}"
        )

    # `format_version` is the FILE's, not the network's (formatos.md §4.3):
    # frozen inside the run it would fossilise in the checkpoint and in the
    # provenance, where it means nothing.
    network.pop("format_version", None)

    # Everything that can be refused, refused **before the name is reserved** --
    # the same function `POST /runs` asks, so the CLI and the API refuse exactly
    # the same things. Validating after reserving would leave a dead `runs/<name>/`
    # behind on every mistake, and then fixing the dataset and retrying with the
    # same name would answer "that run already exists".
    problems = check_run(manifest, network)
    if problems:
        print("\nNo se puede entrenar esto, y se ve antes del primer batch:\n")
        for p in problems:
            print(f"  [{p['code']}] {p['message']}\n    -> {p['hint']}\n")
        raise SystemExit(2)

    spec = RunSpec(
        data=str(patch_datasets.path(args.patch_dataset)),
        out=str(runs.path(args.name)),
        network=network,
        provenance=build_provenance(
            patch_dataset={"name": args.patch_dataset, "fingerprint": manifest["fingerprint"]},
            network={"name": args.network, "value": network},
            recipe={"name": args.recipe, "value": recipe.as_dict()},
            sweep=None,
        ),
        recipe=recipe,
        device=args.device,
        num_workers=args.num_workers,
    )

    try:
        runs.create(args.name, frozen_config(spec))
    except RunExists:
        # Never overwrite in silence. A sweep that auto-generates names is
        # exactly who steps on an existing run.
        parser.error(f"el run '{args.name}' ya existe: {runs.path(args.name)}")

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
        # `marking_failures` keeps `status.json` honest whatever happens: the run
        # is reserved as `queued` before this line, and a crash before the first
        # epoch would otherwise leave it saying so forever.
        with runs.marking_failures(args.name):
            summary = train(
                spec,
                on_epoch=report,
                # A CLI run is stoppable too, and by the same mechanism: the stop
                # lives in the run's directory, not in whoever launched it.
                should_stop=lambda: runs.stop_requested(args.name),
            )
    # The two below are now the SAFETY NET, not the usual path: `check_run` above
    # catches both from the manifest, before anything is reserved. They still fire
    # when the manifest and the `.npz` disagree -- a manifest declaring 980 val
    # patches over a split that loads empty -- which is the one case reading the
    # declaration cannot see.
    except IncompatibleError as exc:
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
