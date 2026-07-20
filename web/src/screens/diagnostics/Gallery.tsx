import { useEffect, useState } from "react";
import {
  BLIND_EVIDENCE as BLIND_CUT,
  getDiagnosticPatches,
  type Outcome,
  type PatchRow,
  type Split,
} from "../../api";
import { Empty, ErrorNote, Loading } from "../../components/Async";
import { Declares } from "../../components/Declares";
import { useAsync } from "../../useAsync";
import { BorderTest } from "./BorderTest";
import { FeatureMaps } from "./FeatureMaps";
import { Deconvolution } from "./Deconvolution";
import { Occlusion } from "./Occlusion";
import { PatchProvenance } from "./PatchProvenance";
import { Prediction } from "./Prediction";
import { Thumbnail } from "./Thumbnail";

const OUTCOMES: { value: Outcome; label: string; hint: string }[] = [
  { value: "all", label: "todos", hint: "sin filtrar" },
  { value: "fp", label: "falsos positivos", hint: "dijo que había esquina y no la hay" },
  { value: "fn", label: "falsos negativos", hint: "hay esquina y no la vio" },
  { value: "tp", label: "aciertos", hint: "dijo que había esquina y la hay" },
  { value: "tn", label: "rechazos", hint: "dijo que no había y no la hay" },
];

const PAGE = 12;

/** V18's cut, as a filter on the gallery. **This is what makes a false negative
 *  readable**: a miss on a blind corner and a miss on a visible one look identical
 *  as thumbnails and mean opposite things — one is the paragraph being outside the
 *  window, the other is the model failing at something it could see. */
const EVIDENCE: { value: string; label: string; hint: string }[] = [
  { value: "any", label: "toda", hint: "sin filtrar por evidencia" },
  {
    value: "blind",
    label: "solo ciegas",
    hint: "el párrafo de la esquina cae fuera del patch: los píxeles no lo enseñan",
  },
  {
    value: "seen",
    label: "solo visibles",
    hint: "había algo del párrafo dentro del patch que mirar",
  },
];

/** V6 — the worst-first gallery. The failures, in order of how bad they are.
 *
 * **The filter is the demonstration of the whole phase**: `outcome` and
 * `threshold` re-decide what counts as a false positive over stored scores, so
 * the model never runs again. On CPU that is the difference between free and
 * hours (ui.md §3).
 *
 * Clicking a patch opens V3 on it. The two are one gesture on purpose: a gallery
 * that could only show you thumbnails would make you leave to find out what the
 * model actually said about one.
 */
export function Gallery({
  run,
  patchDataset,
  split,
  corner,
  threshold,
}: {
  run: string;
  patchDataset: string;
  split: Split;
  corner: string;
  threshold: number;
}) {
  const [outcome, setOutcome] = useState<Outcome>("all");
  const [evidence, setEvidence] = useState("any");
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<PatchRow | null>(null);

  // Any change of question restarts the paging. Without this, narrowing to 3
  // false positives while sitting on page 4 shows an empty grid -- which reads
  // as "no failures", the most misleading answer possible.
  useEffect(() => setOffset(0), [run, split, corner, outcome, threshold, evidence]);

  const page = useAsync(
    () =>
      getDiagnosticPatches(run, {
        split,
        corner,
        outcome,
        order: "error",
        threshold,
        max_evidence: evidence === "blind" ? BLIND_CUT : undefined,
        min_evidence: evidence === "seen" ? BLIND_CUT : undefined,
        offset,
        limit: PAGE,
      }),
    [run, split, corner, outcome, threshold, evidence, offset]
  );
  const data = page.data;

  return (
    <section className="view">
      <Declares
        view="V6"
        title="Galería peor-primero"
        fixes={`el run y el split (${split})`}
        varies="el patch"
        measures="error por patch"
      >
        Filtrar por resultado y mover el umbral <strong>no recalcula nada</strong>: se filtra la
        tabla, no se vuelve a correr el modelo.
      </Declares>

      <div className="row-actions">
        <label className="field field--inline">
          <span className="field__label">Resultado</span>
          <select value={outcome} onChange={(e) => setOutcome(e.target.value as Outcome)}>
            {OUTCOMES.map((o) => (
              <option key={o.value} value={o.value} title={o.hint}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <label className="field field--inline">
          <span
            className="field__label"
            title="cuánto del párrafo de esa esquina cae dentro del patch (V18)"
          >
            Evidencia
          </span>
          <select value={evidence} onChange={(e) => setEvidence(e.target.value)}>
            {EVIDENCE.map((o) => (
              <option key={o.value} value={o.value} title={o.hint}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <span className="card__hint">
          {OUTCOMES.find((o) => o.value === outcome)?.hint}
          {outcome !== "all" && <> · con threshold {threshold.toFixed(2)}</>}
          {evidence !== "any" && (
            <> · {EVIDENCE.find((o) => o.value === evidence)?.hint}</>
          )}
        </span>
      </div>

      {page.loading && !data && <Loading what="la galería" />}
      {page.error && <ErrorNote problem={page.error} />}

      {data && data.total === 0 && (
        <Empty>
          Ningún patch cumple ese filtro con threshold {threshold.toFixed(2)}. Prueba a moverlo,
          o a cambiar de esquina.
        </Empty>
      )}

      {data && data.total > 0 && (
        <>
          <p className="card__hint">
            {data.total.toLocaleString("es")} patches · mostrando {offset + 1}–
            {Math.min(offset + PAGE, data.total)}, peor primero
          </p>
          <div className="gallery">
            {data.rows.map((row) => (
              <Thumbnail
                key={row.patch_idx}
                patchDataset={patchDataset}
                row={row}
                threshold={threshold}
                selected={selected?.patch_idx === row.patch_idx}
                onSelect={() => setSelected(row)}
              />
            ))}
          </div>
          <div className="row-actions">
            <button
              className="button button--quiet"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - PAGE))}
            >
              ← Anteriores
            </button>
            <button
              className="button button--quiet"
              disabled={offset + PAGE >= data.total}
              onClick={() => setOffset(offset + PAGE)}
            >
              Siguientes →
            </button>
          </div>
        </>
      )}

      {selected && (
        <>
          <Prediction
            patchDataset={patchDataset}
            row={selected}
            threshold={threshold}
            onClose={() => setSelected(null)}
          />
          {/* V2 fixes the same thing V3 does — this run, this patch — so it opens
              on the same click. V3 says what the model decided; V2 says what it
              saw on the way there. Splitting them across screens would mean
              picking the same patch twice to ask one question. */}
          <FeatureMaps run={run} patchDataset={patchDataset} patchIdx={selected.patch_idx} />
          {/* The probes (fase 8), all fixing this same run+patch (V4, V10) or the
              patch's B (V15) — so they open on the same click as V2/V3. */}
          <Occlusion run={run} patchDataset={patchDataset} patchIdx={selected.patch_idx} />
          {/* V16 fija lo mismo que V4 y contesta la misma pregunta con otro
              instrumento — V4 tapando y remidiendo, V16 con un backward — así que
              van seguidas a propósito: lo que informa es cuándo *discrepan*. */}
          <Deconvolution run={run} patchDataset={patchDataset} patchIdx={selected.patch_idx} />
          <BorderTest run={run} patchDataset={patchDataset} patchIdx={selected.patch_idx} />
          <PatchProvenance patchDataset={patchDataset} row={selected} />
        </>
      )}
    </section>
  );
}
