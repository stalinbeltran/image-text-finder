import { useState } from "react";
import { getFeatureMaps } from "../../api";
import { ErrorNote, Loading } from "../../components/Async";
import { Declares } from "../../components/Declares";
import { JobLegend, LayerMaps } from "../../components/LayerMaps";
import { useAsync } from "../../useAsync";

/** V2 — every layer's activations over **one patch** (contract ①).
 *
 * The patch is the real input of the CNN, which is why this view takes one and
 * not a whole image — the whole image is F's question and lives in Predecir
 * (organizacion.md §4, last note). It sits next to V3 because they fix the same
 * thing: *this run, this patch*. V3 says what the model decided; V2 says what it
 * saw on the way there.
 *
 * **This is where the deep layers actually speak.** V1 stops at layer 1 because a
 * filter over 32 input channels has no honest projection to a matrix — but its
 * *effect* on a real patch does, and this is it. So the reader V1 sends here is
 * not being fobbed off with a lesser view: it is the right one for that question.
 *
 * The colour job is per layer and comes from the server, which reads
 * `spec.activation`: `relu` gives magnitude → sequential; `tanh` has sign →
 * diverging (R3). Deducing it from the data instead would paint one `tanh` layer
 * two different ways depending on the patch.
 */
export function FeatureMaps({
  run,
  patchDataset,
  patchIdx,
}: {
  run: string;
  patchDataset: string;
  patchIdx: number;
}) {
  const maps = useAsync(
    () => getFeatureMaps(run, { patch_dataset: patchDataset, index: patchIdx }),
    [run, patchDataset, patchIdx]
  );
  const [layer, setLayer] = useState(0);
  // Kept ACROSS layer and patch changes, deliberately: the sibling's
  // `state.selectedMap`. An open table that closed itself every time the patch
  // changed would make the view unusable exactly when it is most useful —
  // watching one filter across several patches.
  const [selected, setSelected] = useState<number | null>(null);

  const data = maps.data;
  const current = data?.layers[Math.min(layer, data.layers.length - 1)];

  return (
    <section className="view">
      <Declares
        view="V2"
        title="Feature maps por capa"
        fixes={`el run y el patch #${patchIdx}`}
        varies="la capa"
        measures="la activación"
      >
        El efecto real de cada filtro sobre <strong>este</strong> patch. Es donde hablan las capas
        profundas, que en kernels no tienen proyección honesta.
      </Declares>

      {maps.loading && !data && <Loading what="los feature maps" />}
      {maps.error && <ErrorNote problem={maps.error} />}

      {data && current && (
        <>
          <div className="row-actions">
            <label className="field field--inline">
              <span className="field__label">Capa</span>
              <select value={layer} onChange={(e) => setLayer(Number(e.target.value))}>
                {data.layers.map((l, i) => (
                  <option key={l.layer} value={i}>
                    capa {l.layer} · {l.count} filtros · {l.height}×{l.width}
                  </option>
                ))}
              </select>
            </label>
            <span className="card__hint">
              {/* The spatial trace, observed rather than predicted: Redes shows
                  40 → 20 → 10 from the config; this is the same number measured
                  on a real tensor. */}
              activación {current.spec.activation ?? "relu"}
              {current.spec.pool ? ` · pool ${current.spec.pool}` : ""}
            </span>
          </div>

          <LayerMaps layer={current} size={72} selected={selected} onSelect={setSelected} />
          <JobLegend job={current.job} />
        </>
      )}
    </section>
  );
}
