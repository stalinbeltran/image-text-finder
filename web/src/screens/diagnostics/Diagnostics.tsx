import { useEffect, useState } from "react";
import { listRuns, type RunRow, type Split } from "../../api";
import { Empty, ErrorNote, Loading } from "../../components/Async";
import { CORNER_NAMES } from "../../theme/palette";
import { useAsync } from "../../useAsync";
import { ErrorByPosition } from "./ErrorByPosition";
import { Gallery } from "./Gallery";
import { ScoresPr } from "./ScoresPr";

/** Diagnóstico (E × B) — **the corazón** of ui.md §1, and where the app stops
 *  being a control panel and becomes the instrument.
 *
 * This is a research project about visual recognition: the analysis screens are
 * the product, not an extra (ui.md §0).
 *
 * **There is no Evaluaciones screen and there never will be** (D1). The per-patch
 * table behind these three views is a *cache*: a pure function of the run, B's
 * fingerprint and the split, so it can be recomputed exactly — and what can be
 * recomputed is not an entity. It is calculated on the first GET and invalidates
 * itself. Nothing here is named, listed or saved.
 *
 * The **threshold is shared** by all three views on purpose. It is one knob of F,
 * tuned post-hoc over stored scores, and seeing the PR curve, the gallery's
 * filter and V3's meters all move together is what makes "this is free" a thing
 * you can see rather than read.
 */
const SPLITS: Split[] = ["train", "val", "test"];

export function Diagnostics() {
  const runs = useAsync(listRuns, []);
  const [run, setRun] = useState<string | null>(null);
  const [split, setSplit] = useState<Split>("val");
  const [corner, setCorner] = useState<string>("all");
  const [threshold, setThreshold] = useState(0.5);

  // Only runs with provenance can be diagnosed: without it there is no way to
  // know which B to measure against, and guessing one would measure the model
  // over data it may never have seen. fase 3 left exactly one such run.
  const usable = (runs.data?.runs ?? []).filter(
    (r) => r.provenance && (r.state === "done" || r.state === "cancelled" || r.state === "running")
  );

  useEffect(() => {
    if (!run && usable.length) setRun(usable[0].name);
  }, [run, usable]);

  const selected = usable.find((r) => r.name === run) ?? null;

  return (
    <section className="screen screen--wide">
      <h1 className="screen__title">Diagnóstico</h1>
      <p className="screen__lede">
        Un run aplicado a un split, patch a patch. Las tres vistas leen{" "}
        <strong>una sola pasada</strong> sobre el split — una tabla por patch que se calcula al
        abrir y se invalida sola. Es un <strong>caché</strong>: no se nombra, no se guarda, y
        borrarlo no pierde nada.
      </p>

      {runs.loading && !runs.data && <Loading what="los runs" />}
      {runs.error && <ErrorNote problem={runs.error} />}
      {runs.data && usable.length === 0 && (
        <Empty>
          No hay ningún run con procedencia que diagnosticar. Entrena uno desde Entrenar — un run
          sin procedencia no puede decir de qué dataset salió, así que no hay contra qué medirlo.
        </Empty>
      )}

      {selected && (
        <>
          <div className="controls">
            <label className="field field--inline">
              <span className="field__label">Run (E)</span>
              <select value={run ?? ""} onChange={(e) => setRun(e.target.value)}>
                {usable.map((r) => (
                  <option key={r.name} value={r.name}>
                    {r.name}
                  </option>
                ))}
              </select>
            </label>

            <label className="field field--inline">
              <span className="field__label" title="el split de B: los patches que este run NO vio entrenando son los de val y test">
                Split de B
              </span>
              <select value={split} onChange={(e) => setSplit(e.target.value as Split)}>
                {SPLITS.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>

            <label className="field field--inline">
              <span className="field__label">Esquina</span>
              <select value={corner} onChange={(e) => setCorner(e.target.value)}>
                <option value="all">las 4</option>
                {CORNER_NAMES.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <Provenance run={selected} split={split} />

          {/* V8 first: it is the one that says what the threshold should be, and
              the other two read the threshold it sets. */}
          <ScoresPr
            run={selected.name}
            split={split}
            corner={corner}
            threshold={threshold}
            onThreshold={setThreshold}
          />
          <ErrorByPosition run={selected.name} split={split} corner={corner} />
          <Gallery
            run={selected.name}
            patchDataset={selected.provenance!.patch_dataset.name}
            split={split}
            corner={corner}
            threshold={threshold}
          />
        </>
      )}
    </section>
  );
}

/** What is being measured against what. **A measurement without its instrument
 *  named is an anecdote** (protocolo.md), so the B and its huella are on screen,
 *  not buried in a tooltip: the huella is what would tell you the dataset had
 *  been rebuilt under the same name — and the backend refuses outright if it has
 *  (contract ⑧). */
function Provenance({ run, split }: { run: RunRow; split: Split }) {
  const p = run.provenance!;
  return (
    <p className="card__foot diagnostics__provenance">
      Midiendo <code>{run.name}</code> (<code>best.pt</code>, el checkpoint que el run eligió por{" "}
      <code>{run.summary?.monitor ?? "su monitor"}</code>) sobre el split <strong>{split}</strong>{" "}
      de <code>{p.patch_dataset.name}</code>{" "}
      <span
        className="card__hint"
        title="la huella del contenido: si el dataset se reconstruye bajo el mismo nombre, deja de ser el mismo y el diagnóstico se niega (contrato ⑧)"
      >
        {p.patch_dataset.fingerprint.replace("sha256:", "").slice(0, 12)}…
      </span>{" "}
      · red <code>{p.network.name}</code> · receta <code>{p.recipe.name}</code>
    </p>
  );
}
