import { getPatch, type PatchRow } from "../../api";
import { ErrorNote, Loading } from "../../components/Async";
import { Declares } from "../../components/Declares";
import { Meter } from "../../components/Meter";
import { PatchCanvas } from "../../components/PatchCanvas";
import { CORNER_NAMES, type CornerName } from "../../theme/palette";
import { useAsync } from "../../useAsync";

/** V3 — the prediction for one patch: 4 meters + the overlay.
 *
 * **Meters, not a 4-bar chart, and certainly not a pie** (ui.md §4.1): four
 * probabilities against a threshold are four ratios against a limit, and the
 * mark on the track is the component's whole point — `p = 0.6` means nothing
 * until you can see which side of the threshold it landed on. Move the slider and
 * the four marks move: that is the free sweep of V8, one patch at a time.
 *
 * It costs no forward pass. The scores and positions are already in the table, so
 * this view is a read of it joined to B's pixels — which is also why it works on
 * a run that finished weeks ago and is not loaded anywhere.
 */
export function Prediction({
  patchDataset,
  row,
  threshold,
  onClose,
}: {
  patchDataset: string;
  row: PatchRow;
  threshold: number;
  onClose: () => void;
}) {
  const patch = useAsync(() => getPatch(patchDataset, row.patch_idx), [patchDataset, row.patch_idx]);

  const overlay = CORNER_NAMES.map((corner, i) => ({
    corner: corner as CornerName,
    truth: row.exists[i] ? ([row.xy_true[i][0], row.xy_true[i][1]] as [number, number]) : null,
    pred: row.score[i] >= threshold ? ([row.xy_pred[i][0], row.xy_pred[i][1]] as [number, number]) : null,
  }));

  return (
    <article className="card prediction">
      <header className="card__head">
        <Declares
          view="V3"
          title={`Predicción del patch #${row.patch_idx}`}
          fixes="el run y este patch"
          varies="—"
          measures="4 × [p, x, y]"
        />
        <button className="button button--quiet" onClick={onClose}>
          Cerrar
        </button>
      </header>

      {patch.loading && !patch.data && <Loading what="el patch" />}
      {patch.error && <ErrorNote problem={patch.error} />}

      {patch.data && (
        <div className="prediction__body">
          <div>
            <PatchCanvas
              patch={patch.data.patch}
              size={240}
              overlay={overlay}
              label={`patch #${row.patch_idx} de la imagen ${row.sample_idx}`}
            />
            <p className="card__hint">
              El <strong>anillo</strong> es dónde está la esquina de verdad; el{" "}
              <strong>punto</strong>, dónde la puso el modelo. La línea entre los dos es el error.
              El color es <strong>qué esquina</strong> (nunca si acertó): un mismo TL es el mismo
              azul en toda la app.
            </p>
          </div>

          <div className="prediction__meters">
            {CORNER_NAMES.map((corner, i) => (
              <div key={corner} className="prediction__corner">
                <Meter corner={corner as CornerName} value={row.score[i]} threshold={threshold} />
                <p className="prediction__verdict">
                  <Verdict exists={row.exists[i]} score={row.score[i]} threshold={threshold} />
                  {/* n/a, never 0.0 px: no real corner means there was nothing to
                      localise, which is not the same as having localised it
                      perfectly (formatos.md §2). */}
                  {row.exists[i] && (
                    <span className="card__hint">
                      {" "}
                      · error {row.err_px[i] === null ? "n/a" : `${row.err_px[i]!.toFixed(1)} px`}
                    </span>
                  )}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </article>
  );
}

/** The four outcomes, named. The colour tokens here are STATUS, not series --
 *  reserved meaning, and always with a word next to them. */
function Verdict({
  exists,
  score,
  threshold,
}: {
  exists: boolean;
  score: number;
  threshold: number;
}) {
  const predicted = score >= threshold;
  if (predicted && exists) return <span data-verdict="tp">acierto</span>;
  if (predicted && !exists) return <span data-verdict="fp">falso positivo</span>;
  if (!predicted && exists) return <span data-verdict="fn">se la perdió</span>;
  return <span data-verdict="tn">no hay, y no la ve</span>;
}
