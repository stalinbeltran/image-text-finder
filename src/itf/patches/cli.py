"""CLI to build a patch dataset from a source dataset.

Usage:
    itf-extract --config configs/extract.example.yaml
    itf-extract --source <dir> --out <dir> [--patch-size 40] [--stride 20]
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from itf.patches.extract import PatchExtractConfig, extract_dataset


def _build_config(args: argparse.Namespace) -> PatchExtractConfig:
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    else:
        data = {}
    # CLI flags override the YAML.
    for key in ("source", "out", "patch_size", "stride", "seed"):
        val = getattr(args, key)
        if val is not None:
            data[key] = val
    if args.target_kinds is not None:
        data["target_kinds"] = args.target_kinds
    if args.drop_overlap:
        data["drop_overlap"] = True
    if "source" not in data or "out" not in data:
        raise SystemExit("error: --source and --out are required (via flags or --config)")
    return PatchExtractConfig.from_dict(data)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="itf-extract", description="Build a patch dataset (.npz).")
    p.add_argument("--config", help="YAML config file (see configs/extract.example.yaml).")
    p.add_argument("--source", help="Source dataset dir (with labels.jsonl).")
    p.add_argument("--out", help="Output dir for patches.npz + manifest.")
    p.add_argument("--patch-size", dest="patch_size", type=int)
    p.add_argument("--stride", type=int)
    p.add_argument("--target-kinds", dest="target_kinds", nargs="+")
    p.add_argument("--drop-overlap", dest="drop_overlap", action="store_true")
    p.add_argument("--seed", type=int)
    args = p.parse_args(argv)

    config = _build_config(args)
    summary = extract_dataset(config)
    print(json.dumps(summary["manifest"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
