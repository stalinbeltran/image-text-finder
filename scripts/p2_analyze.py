r"""P2 — ¿este modelo sobreajusta? (protocolo.md §9)

Lee los cinco runs de la réplica (`dirty-20-lambda_pos_1-0005`, seed 1, más los
cuatro `p2-noover-seed{2..5}`) y reporta el **hueco train↔val** por población,
en media ± sd sobre las semillas.

Por qué media ± sd y no el mejor: un run aislado no es un resultado, es una
anécdota (protocolo.md §7). La afirmación que se está poniendo a prueba —"el
hueco es ~0 en todas las poblaciones"— salió de UN run, y los otros ocho del
barrido no valen porque los nueve llevan `seed: 1`.

Por qué separa ciegas y visibles: es la partición de V18. El hueco global puede
salir ~0 y ocultar que una de las dos poblaciones sí memoriza — que es
justamente la pregunta que §5.5 dejó abierta.

No mide el suelo de ruido de protocolo.md §4 (eso pide la F1 de párrafo, que no
existe hasta D7). Mide dispersión entre semillas de lo que sí hay, que es un
subproducto útil: toda diferencia del barrido por debajo de esa sd es un empate.

Uso:  .\.venv\Scripts\python.exe scripts\p2_analyze.py
El diagnóstico se cachea, así que la segunda pasada es gratis.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from itf.diagnostics import evidence_split, open_diagnostics  # noqa: E402
from itf.diagnostics.table import TableCache  # noqa: E402
from itf.metrics import BLIND_EVIDENCE, DEFAULT_THRESHOLD  # noqa: E402
from itf.patches.store import PatchDatasetStore  # noqa: E402
from itf.settings import Settings  # noqa: E402
from itf.training.registry import RunStore  # noqa: E402

#: La semilla 1 es el run del que salió la afirmación. Entra como una réplica
#: más, no como referencia privilegiada: su config es la que se copió.
RUNS = {
    1: "dirty-20-lambda_pos_1-0005",
    2: "p2-noover-seed2",
    3: "p2-noover-seed3",
    4: "p2-noover-seed4",
    5: "p2-noover-seed5",
}
SPLITS = ("train", "val")


def measure(diag) -> dict[str, float]:
    """Los números que se comparan entre train y val, en un solo sitio.

    En un solo sitio a propósito: train y val tienen que medirse con la MISMA
    fórmula o el hueco es un artefacto de haberla escrito dos veces — es la
    trampa de `pos_err_px` otra vez (organizacion.md §3).
    """
    exists = diag.exists
    fired = diag.table.score >= DEFAULT_THRESHOLD
    tp = int((fired & exists).sum())
    fp = int((fired & ~exists).sum())
    fn = int((~fired & exists).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")

    split = evidence_split(diag, blind=BLIND_EVIDENCE)
    return {
        "f1": f1,
        "err_px": float(np.nanmean(np.where(exists, diag.table.err_px, np.nan))),
        "blind_err_px": split["blind"]["err_px"],
        "blind_score": split["blind"]["score_mean"],
        "seen_err_px": split["seen"]["err_px"],
        "seen_score": split["seen"]["score_mean"],
        "blind_share": split["blind"]["corner_share"],
    }


def main() -> int:
    settings = Settings.from_env()
    runs = RunStore(settings.runs_root)
    patch_datasets = PatchDatasetStore(settings.patch_datasets_root)
    cache = TableCache(settings.diagnostics_cache_root)

    missing = [name for name in RUNS.values() if not (settings.runs_root / name).exists()]
    if missing:
        print("Faltan runs, lánzalos primero con .\\scripts\\p2-seeds.ps1:")
        for name in missing:
            print(f"  - {name}")
        return 1

    per_seed: dict[int, dict[str, dict[str, float]]] = {}
    for seed, name in RUNS.items():
        per_seed[seed] = {}
        for split in SPLITS:
            print(f"[{name}] diagnóstico sobre {split}…", flush=True)
            diag = open_diagnostics(
                runs=runs,
                patch_datasets=patch_datasets,
                cache=cache,
                run=name,
                split=split,
            )
            per_seed[seed][split] = measure(diag)

    metrics = ["f1", "err_px", "blind_err_px", "blind_score", "seen_err_px", "seen_score"]

    print()
    print("Por semilla — train / val / hueco (train − val)")
    print(f"{'métrica':<15} " + " ".join(f"{s:>22}" for s in RUNS))
    for metric in metrics:
        cells = []
        for seed in RUNS:
            tr = per_seed[seed]["train"][metric]
            va = per_seed[seed]["val"][metric]
            cells.append(f"{tr:7.4f}/{va:7.4f}/{tr - va:+7.4f}")
        print(f"{metric:<15} " + " ".join(f"{c:>22}" for c in cells))

    print()
    print("Agregado sobre las 5 semillas — media ± sd")
    print(f"{'métrica':<15} {'train':>18} {'val':>18} {'hueco':>18}")
    for metric in metrics:
        train = [per_seed[s]["train"][metric] for s in RUNS]
        val = [per_seed[s]["val"][metric] for s in RUNS]
        gap = [t - v for t, v in zip(train, val)]

        def fmt(xs: list[float]) -> str:
            return f"{statistics.mean(xs):8.4f} ± {statistics.stdev(xs):7.4f}"

        print(f"{metric:<15} {fmt(train):>18} {fmt(val):>18} {fmt(gap):>18}")

    print()
    print("Cómo se lee (fijado ANTES de mirar, protocolo.md §7):")
    print("  · El hueco es creíblemente ~0 si |media| < sd entre semillas: si la")
    print("    dispersión por semilla se come el hueco, no hay sobreajuste que medir.")
    print("  · Si `blind_*` tiene hueco y `seen_*` no, las etiquetas ciegas SÍ se")
    print("    memorizan y §5.5 se corrige — es el resultado que P2 puede refutar.")
    print("  · La sd de `f1` es la banda de ruido del barrido: toda diferencia por")
    print("    debajo es un EMPATE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
