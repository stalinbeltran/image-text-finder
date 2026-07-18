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

from itf.datasets.roots import list_ids, resolve, source_roots
from itf.patches.extract import PatchExtractConfig, extract_dataset
from itf.settings import Settings


def _resolve_source(value: str, settings: Settings) -> str:
    """Accept an id, a relative path or an absolute path -- and if none resolves,
    **say which sources exist**, with their ids and absolute paths.

    Both source roots, since D19: `--source derived/<name>` reaches a resized
    source. The two-root lookup itself lives in `itf.datasets.roots`, because
    three callers need it and three copies is contract ⑤ with another subject.

    Why this exists: `--source ..\\image-text-sample-generator\\...` only works if
    you are standing in the repo root, and the README documented exactly that. The
    day someone runs it from `web/`, the relative path silently points nowhere and
    the failure surfaces deep inside the loader as a confusing error about a
    missing directory.

    Worse, and this one is measured (protocolo.md §1): there are two sources whose
    names start with `clear-paragraphs-02` and differ by 14.5x in area. Guessing
    the wrong one does NOT fail -- it builds a perfectly valid dataset with a wildly
    different imbalance, and you find out much later, if ever. **Listing the real
    names next to the error is what turns a guess into a choice.**

    The id form is the same one the API takes (D4): the client sends an id, the
    path gets resolved inside. Resolution lives HERE, in the entry point, because
    the domain modules take explicit paths and must stay callable on any directory
    from a test (settings.py).
    """
    roots = source_roots(settings.datasets_root, settings.derived_sources_root)
    found = resolve(value, roots)
    if found is not None:
        return str(found)

    listing = "\n".join(f"    {sid}\n      {path}" for sid, path in list_ids(roots)) or "    (ninguna)"
    raise SystemExit(
        f"itf-extract: no encuentro la fuente '{value}'.\n"
        f"  Se admite un id, una ruta relativa o una ruta ABSOLUTA (la más segura: no\n"
        f"  depende de desde dónde ejecutes).\n"
        f"  Busqué en: {settings.datasets_root}\n"
        f"          y: {settings.derived_sources_root} (las derivadas, con prefijo 'derived/')\n"
        f"  Hay estas fuentes (nómbralas ENTERAS: hay nombres que solo se distinguen\n"
        f"  por el sufijo y NO fallan si te equivocas, solo miden otra cosa):\n{listing}\n"
        f"  Para apuntar a otro sitio: $env:ITF_DATASETS_ROOT = \"D:\\mis\\datasets\""
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="itf-extract", description="Construye un dataset de patches (B)")
    parser.add_argument("--config", type=Path, help="YAML con el PatchExtractConfig")
    parser.add_argument(
        "--source",
        help="La fuente (A): un id de ITF_DATASETS_ROOT, o una ruta relativa o absoluta",
    )
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

    # Resolved to an absolute path BEFORE it reaches the config, so the manifest
    # records which source was really read -- not a relative string whose meaning
    # depends on where the command was run from.
    data["source"] = _resolve_source(str(data["source"]), Settings.from_env())

    summary = extract_dataset(PatchExtractConfig.from_dict(data))
    manifest = summary["manifest"]
    print(json.dumps({k: manifest[k] for k in ("num_samples", "num_patches", "patches_per_split", "positives_per_corner", "fingerprint")}, indent=2))
    for warning in manifest["warnings"]:
        print(f"\nAVISO [{warning['code']}]: {warning['message']}\n  {warning['hint']}")


if __name__ == "__main__":
    main()
