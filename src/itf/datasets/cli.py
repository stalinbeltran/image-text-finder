"""`itf-resize` — build a derived source (A') from the command line.

Same reasoning as `itf-extract`: if the resize only worked over HTTP, the logic
would be in the wrong layer (api.md §0). This touches no FastAPI and does the
same work, through the same pure `check_resize`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from itf.datasets.resize import ResizeRefused, ResizeRequest, resize_source
from itf.datasets.roots import DERIVED_PREFIX, list_ids, resolve, source_roots
from itf.settings import Settings


def _resolve_source(value: str, settings: Settings) -> Path:
    """An id, a relative path or an absolute path -- and if none resolves, list what exists.

    Two roots now (D19), and they are looked up in order: an id may name an
    original or, with the `derived/` prefix, one of ours. Resizing a derived
    source is legal and chains -- `dataset.json` records the immediate parent
    (formatos.md §4.6).

    The listing is not politeness. There are two sources whose names start with
    `clear-paragraphs-02` and differ by 14.5x in area; naming the wrong one does
    not fail, it silently measures something else (protocolo.md §1).
    """
    roots = source_roots(settings.datasets_root, settings.derived_sources_root)
    found = resolve(value, roots)
    if found is not None:
        return found

    listing = "\n".join(f"    {sid}\n      {path}" for sid, path in list_ids(roots)) or "    (ninguna)"
    raise SystemExit(
        f"itf-resize: no encuentro la fuente '{value}'.\n"
        f"  Busqué en: {settings.datasets_root}\n"
        f"          y: {settings.derived_sources_root} (las derivadas, con prefijo "
        f"'{DERIVED_PREFIX}/')\n"
        f"  Hay estas fuentes (nómbralas ENTERAS):\n{listing}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="itf-resize",
        # ASCII only in anything argparse or `print` may emit: the Windows console
        # is cp1252 and a bare `->` costs nothing, while a `→` makes --help
        # itself raise UnicodeEncodeError. Found by running it, not by reading it.
        description="Redimensiona una fuente (A) manteniendo la proporcion -> fuente derivada (A')",
    )
    parser.add_argument("--source", required=True, help="La fuente a redimensionar: un id o una ruta")
    parser.add_argument("--name", required=True, help="Nombre de la fuente derivada")
    # Mutually exclusive at the parser too, and that is NOT the duplication that
    # contract ⑤ warns about: argparse rejects the shape of the command line,
    # `check_resize` owns the rule. The CLI would still be correct without this;
    # it would just say it worse.
    dim = parser.add_mutually_exclusive_group(required=True)
    dim.add_argument("--width", type=int, help="Ancho destino en px (el alto se deriva)")
    dim.add_argument("--height", type=int, help="Alto destino en px (el ancho se deriva)")
    args = parser.parse_args()

    settings = Settings.from_env()
    src = _resolve_source(args.source, settings)
    dst = settings.derived_sources_root / args.name

    req = ResizeRequest(name=args.name, to_width=args.width, to_height=args.height)
    try:
        derived = resize_source(src, dst, req, source_id=args.source, progress=_progress)
    except ResizeRefused as exc:
        lines = "\n".join(f"  - {p['message']}\n    -> {p['hint']}" for p in exc.problems)
        raise SystemExit(f"itf-resize: no se puede.\n{lines}")

    print()
    print(f"Escrita en {dst}")
    print(json.dumps(derived, indent=2, ensure_ascii=False))


def _progress(done: int, total: int) -> None:
    print(f"\r  {done}/{total} muestras", end="", flush=True)
