"""`itf-extract` — build a patch dataset from the command line.

Not a convenience: it is the PROOF of api.md §0. If extraction only worked over
HTTP, the logic would be in the wrong layer. This command touches no FastAPI and
does the same work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from itf.patches.extract import PatchExtractConfig, extract_dataset


def main() -> None:
    parser = argparse.ArgumentParser(prog="itf-extract", description="Construye un dataset de patches (B)")
    parser.add_argument("--config", type=Path, help="YAML con el PatchExtractConfig")
    parser.add_argument("--source", help="Ruta a la fuente (A)")
    parser.add_argument("--out", help="Directorio de salida")
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--stride", type=int, help="Stride de EXTRACCIÓN (parte de la identidad de B)")
    parser.add_argument("--seed", type=int, help="Semilla de B: fija el split por imagen")
    args = parser.parse_args()

    data: dict = {}
    if args.config:
        data = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    for key, value in [
        ("source", args.source),
        ("out", args.out),
        ("patch_size", args.patch_size),
        ("stride", args.stride),
        ("seed", args.seed),
    ]:
        if value is not None:
            data[key] = value

    missing = [k for k in ("source", "out") if not data.get(k)]
    if missing:
        parser.error(f"falta {' y '.join(missing)} (por --config o por bandera)")

    summary = extract_dataset(PatchExtractConfig.from_dict(data))
    manifest = summary["manifest"]
    print(json.dumps({k: manifest[k] for k in ("num_samples", "num_patches", "patches_per_split", "positives_per_corner", "fingerprint")}, indent=2))
    for warning in manifest["warnings"]:
        print(f"\nAVISO [{warning['code']}]: {warning['message']}\n  {warning['hint']}")


if __name__ == "__main__":
    main()
