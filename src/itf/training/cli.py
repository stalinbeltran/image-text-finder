"""CLI to train a corner-detection CNN.

Usage:
    itf-train --config configs/model.example.yaml --data data/patch-datasets/reducido-40 --out runs/demo
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from itf.training.loop import RunConfig, train


def _build_config(args: argparse.Namespace) -> RunConfig:
    data: dict = {}
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        # Accept either a flat dict or {model: {...}, train: {...}}.
        data.update(raw.get("train", {}))
        if "model" in raw:
            data["model"] = raw["model"]
        for k, v in raw.items():
            if k not in {"model", "train"}:
                data.setdefault(k, v)
    for key in ("data", "out", "epochs", "batch_size", "lr", "optimizer", "seed", "device"):
        val = getattr(args, key)
        if val is not None:
            data[key] = val
    if "data" not in data or "out" not in data:
        raise SystemExit("error: --data and --out are required (via flags or --config)")
    if "model" not in data:
        raise SystemExit("error: config must include a 'model' section")
    return RunConfig.from_dict(data)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="itf-train", description="Train a corner-detection CNN.")
    p.add_argument("--config", help="YAML with model + train sections.")
    p.add_argument("--data", help="Patch-dataset dir (with patches.npz).")
    p.add_argument("--out", help="Run output dir.")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--optimizer")
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    args = p.parse_args(argv)

    config = _build_config(args)
    summary = train(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
