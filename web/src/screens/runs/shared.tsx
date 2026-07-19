import { useEffect, useRef, useState } from "react";
import { getRunMetrics, type EpochRecord, type Provenance, type RunState, type RunSummary } from "../../api";
import { TrainingCurves } from "../../components/TrainingCurves";

/** What the list and the detail of a run share. Split in two screens (list =
 *  qué runs hay + CRUD; detalle = qué pasó dentro de uno), so the pieces that
 *  both need live here and not duplicated: duplicating `Progress` would be the
 *  «definir un número dos veces» trap with a component instead of a metric. */

export const STATE_LABEL: Record<string, string> = {
  queued: "en cola",
  running: "corriendo",
  done: "terminado",
  error: "error",
  cancelled: "cancelado",
};

export const isLiveState = (s: RunState) => s === "running" || s === "queued";

/** Contract ③ on screen: the name to group, the value to reproduce.
 *
 * The fingerprint is here and not decoration: a B rebuilt under the same name is
 * a different B, and without the huella nothing would tell two runs apart that
 * "used the same dataset" (contract ⑧).
 */
export function ProvenanceFacts({ provenance: p }: { provenance: Provenance }) {
  return (
    <>
      <p className="card__section">
        Procedencia <span className="card__hint">de qué salió, por nombre — contrato ③</span>
      </p>
      <dl className="facts facts--inline">
        <div className="fact">
          <dt>Dataset de patches (B)</dt>
          <dd>
            <code>{p.patch_dataset.name}</code>{" "}
            <span
              className="card__hint"
              title="huella del contenido: distingue un dataset reconstruido bajo el mismo nombre"
            >
              {p.patch_dataset.fingerprint.replace("sha256:", "").slice(0, 12)}…
            </span>
          </dd>
        </div>
        <div className="fact">
          <dt>Red (C)</dt>
          <dd>
            <code>{p.network.name}</code>{" "}
            <span className="card__hint">input_size {p.network.value.input_size}</span>
          </dd>
        </div>
        <div className="fact">
          <dt>Receta (D)</dt>
          <dd>
            <code>{p.recipe.name}</code>{" "}
            <span className="card__hint">
              {p.recipe.value.optimizer}, lr {p.recipe.value.lr}
            </span>
          </dd>
        </div>
        <div className="fact">
          <dt>Barrido (H)</dt>
          <dd>{p.sweep ? <code>{p.sweep}</code> : <span className="card__hint">ninguno</span>}</dd>
        </div>
        <div className="fact">
          <dt title="el commit fija el código; no fija el intérprete — por eso va también el entorno">
            Commit
          </dt>
          <dd>
            <code>{p.git_commit.slice(0, 12)}</code>
          </dd>
        </div>
        <div className="fact">
          <dt title="dos runs solo son comparables con el mismo commit Y el mismo entorno: subir de torch 2.13 a 2.14 mueve los resultados sin mover el commit">
            Entorno
          </dt>
          <dd>
            <span className="card__hint">
              python {p.environment.python} · torch {p.environment.torch} · {p.environment.platform}
            </span>
          </dd>
        </div>
      </dl>
    </>
  );
}

/** The live readout, polled incrementally (R5).
 *
 * **The curves are small multiples** (V14, R4), which is what fase 4 deferred to
 * here rather than drawing a wrong chart in the meantime: `loss ≈ 0.28`,
 * `f1 ≈ 0.77` and `pos_err_px ≈ 11` are three scales, and one plot with two
 * y-axes would invent a correlation that is not in the data.
 *
 * The last epochs stay as numbers under them. That is not redundancy: a curve
 * answers "where is this going", a table answers "what exactly was epoch 17" —
 * and it is the same R5 twin the maps have.
 */
export function Progress({
  name,
  state,
  summary,
  secondsPerEpoch,
  epochsShown = 6,
}: {
  name: string;
  state: RunState;
  summary: RunSummary | null;
  secondsPerEpoch: number | null;
  /** The detail screen shows the whole history; the list does not use this at all. */
  epochsShown?: number;
}) {
  const [records, setRecords] = useState<EpochRecord[]>([]);
  const since = useRef(0);
  const isLive = isLiveState(state);

  useEffect(() => {
    let alive = true;

    async function poll() {
      try {
        // `since` is what the last call handed back, so only new epochs travel.
        // Re-sending the history each time is what made watching a run cost more
        // the longer it ran.
        const fresh = await getRunMetrics(name, since.current);
        if (!alive || fresh.records.length === 0) return;
        since.current = fresh.next;
        setRecords((prev) => [...prev, ...fresh.records]);
      } catch {
        // A run being deleted or renamed under a poll is not an error worth
        // shouting about: the list reload is what decides this card's fate.
      }
    }

    poll();
    // The cleanup is registered on BOTH paths. Returning early for a finished
    // run would leave `alive` true forever, so the one poll() above could still
    // land its setState after the card unmounted -- which is the exact thing the
    // flag is here to prevent.
    const timer = isLive ? setInterval(poll, 2000) : undefined;
    return () => {
      alive = false;
      if (timer !== undefined) clearInterval(timer);
    };
  }, [name, isLive]);

  if (records.length === 0) {
    return (
      <p className="card__foot">
        {isLive ? "Todavía no ha terminado ninguna época." : "Este run no dejó métricas."}
      </p>
    );
  }

  const last = records[records.length - 1];
  const total = summary?.epochs_requested;
  const shown = epochsShown <= 0 ? records : records.slice(-epochsShown);

  return (
    <>
      <p className="card__section">
        Progreso{" "}
        <span className="card__hint">
          época {last.epoch}
          {total ? ` de ${total}` : ""} · {secondsPerEpoch?.toFixed(1)} s/época de media
        </span>
      </p>
      {/* V14. Three panels with the epoch axis aligned — never one plot with two
          y-axes (R4). */}
      <TrainingCurves records={records} />
      <div className="table__scroll">
        <table className="table">
          <thead>
            <tr>
              <th>época</th>
              <th className="table__num">train loss</th>
              <th className="table__num">val loss</th>
              <th className="table__num">f1</th>
              <th className="table__num" title="error de posición en píxeles">
                pos_err_px
              </th>
              <th className="table__num">lr</th>
              <th className="table__num">s</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.epoch}>
                <td>{r.epoch}</td>
                <td className="table__num">{r.train_loss.toFixed(4)}</td>
                <td className="table__num">{r.val.loss.toFixed(4)}</td>
                <td className="table__num">{r.val.f1.toFixed(3)}</td>
                {/* n/a, never 0.0: no corners in val means NOT MEASURED, and a 0
                    would read as a perfect localisation (formatos.md §2). */}
                <td className="table__num">
                  {r.val.pos_err_px === null ? "n/a" : r.val.pos_err_px.toFixed(1)}
                </td>
                <td className="table__num">{r.lr.toExponential(1)}</td>
                <td className="table__num">{r.seconds.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {summary && (
        <p className="card__foot">
          Mejor <code>{summary.monitor}</code>:{" "}
          {/* "sin medir", never a number: a monitor that never fired is an
              absence, and printing something there would invent a result. */}
          <strong>{summary.best === null ? "sin medir" : summary.best.toFixed(4)}</strong> ·{" "}
          {summary.epochs_run} de {summary.epochs_requested} épocas
          {summary.stopped_early && " · cortado por patience"}
          {summary.cancelled && " · parado a mano"}
        </p>
      )}
    </>
  );
}
