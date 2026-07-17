import { getOcclusion } from "../../api";
import { ErrorNote, Loading } from "../../components/Async";
import { Declares } from "../../components/Declares";
import { MatrixCanvas } from "../../components/MatrixCanvas";
import { useAsync } from "../../useAsync";

/** V4 — occlusion sensitivity. **Fix E and the patch, vary the mask position.**
 *
 * Slide a small mask over the patch and, at each spot, ask what the four heads now
 * say. Where covering a region collapses `p(esquina)`, that region is what the
 * head was reading. It is the rigorous, in-distribution version of the sibling's
 * "paint over the digit and watch it change" (ui.md §5): every mask hides real
 * pixels, none invents new ones.
 *
 * **Each cell is `p(esquina | máscara aquí)`, so the ramp is sequential** (R3): a
 * probability is never signed, and dark = covering here killed the score = the
 * region that mattered. The *drop* against `baseline` is one subtraction the
 * reader can see — painting the drop directly would be signed, and a signed
 * quantity on a sequential ramp puts the neutral wherever the minimum falls, the
 * exact trap R2/R3 exist to stop.
 *
 * Four maps, one per corner, side by side: the mask is one, but its effect on each
 * head is its own picture, and a TL detector and a BR detector read different
 * parts of the same patch.
 */
export function Occlusion({
  run,
  patchDataset,
  patchIdx,
}: {
  run: string;
  patchDataset: string;
  patchIdx: number;
}) {
  const data = useAsync(
    () => getOcclusion(run, { patch_dataset: patchDataset, index: patchIdx }),
    [run, patchDataset, patchIdx]
  );
  const occ = data.data;

  return (
    <section className="view">
      <Declares
        view="V4"
        title="Occlusion sensitivity"
        fixes={`el run y el patch #${patchIdx}`}
        varies="la posición de la máscara"
        measures="cuánto cae p(esquina)"
      >
        Una máscara de {occ?.mask_size ?? 5}×{occ?.mask_size ?? 5} se desliza por el patch; cada
        celda es <strong>p con la máscara ahí</strong>. Lo <strong>oscuro</strong> es donde tapar
        mató el score: <strong>eso</strong> es lo que la cabeza estaba mirando.
      </Declares>

      {data.loading && !occ && <Loading what="la occlusion" />}
      {data.error && <ErrorNote problem={data.error} />}

      {occ && (
        <>
          <div className="occlusion">
            {occ.maps.map((m, c) => (
              <figure key={m.corner} className="occlusion__panel">
                <MatrixCanvas
                  matrix={m.matrix}
                  job={occ.job}
                  size={140}
                  label={`${m.corner}: p con la máscara en cada celda`}
                />
                <figcaption className="card__hint">
                  <span className="occlusion__swatch" data-corner={m.corner} aria-hidden="true" />{" "}
                  <strong>{m.corner}</strong> · base {occ.baseline[c].toFixed(2)} · mín{" "}
                  {m.min.toFixed(2)}{" "}
                  <span title="la mayor caída: base − el mínimo de este mapa">
                    (cae {Math.max(0, occ.baseline[c] - m.min).toFixed(2)})
                  </span>
                </figcaption>
              </figure>
            ))}
          </div>

          <p className="card__hint">
            {occ.bins}×{occ.bins} celdas, máscara {occ.mask_size}px, paso {occ.mask_stride}px, rellena
            con la media del patch (no negro: un rectángulo oscuro sería un borde que el modelo no
            vio). Lo que cambia es <strong>quitar señal</strong>, no <strong>añadir</strong> una.
          </p>
        </>
      )}
    </section>
  );
}
