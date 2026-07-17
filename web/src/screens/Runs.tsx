import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  deleteRun,
  getRunMetrics,
  listRuns,
  renameRun,
  stopRun,
  type EpochRecord,
  type Provenance,
  type RunRow,
} from "../api";
import { ErrorNote, Empty, Loading, type ApiProblem } from "../components/Async";
import { TrainingCurves } from "../components/TrainingCurves";
import { useAsync } from "../useAsync";
import { Link } from "react-router-dom";

/** Runs (E) — the trained model: weights + metrics + provenance.
 *
 * ui.md §2. The one thing this screen has that the old `RunsPanel` could not
 * have: **where each run came from, by name** (contract ③). Before, a run copied
 * the VALUE of C and D and lost their identity, so "which runs used network X?"
 * meant diffing dictionaries by hand — and that is the question a sweep asks all
 * the time.
 *
 * The `lockArchitecture` toggle is gone and has nothing to replace it: with C and
 * D separated, "re-train the same network" is *choose another recipe with the
 * same C*, which is what it always was.
 */
export function Runs() {
  const runs = useAsync(listRuns, []);
  const live = (runs.data?.runs ?? []).some((r) => r.state === "running" || r.state === "queued");

  // A queued run becomes running, and a running one finishes, without anyone
  // clicking. The per-epoch metrics have their own incremental poll (R5); this
  // one is just the state, and it stops as soon as nothing is moving.
  useEffect(() => {
    if (!live) return;
    const timer = setInterval(() => runs.reload(), 2000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live]);

  return (
    <section className="screen screen--wide">
      <h1 className="screen__title">Runs</h1>
      <p className="screen__lede">
        Un modelo entrenado: pesos, métricas y <strong>procedencia</strong>. De qué dataset, qué red
        y qué receta salió — <strong>por nombre</strong>, que es lo que permite agrupar y comparar.
      </p>

      {runs.loading && !runs.data && <Loading what="los runs" />}
      {runs.error && <ErrorNote problem={runs.error} />}
      {runs.data?.runs.length === 0 && (
        <Empty>Todavía no hay ningún run. Lanza uno desde Entrenar.</Empty>
      )}

      {runs.data?.runs.map((run) => <RunCard key={run.name} run={run} onChange={runs.reload} />)}
    </section>
  );
}

const STATE_LABEL: Record<string, string> = {
  queued: "en cola",
  running: "corriendo",
  done: "terminado",
  error: "error",
  cancelled: "cancelado",
};

function RunCard({ run, onChange }: { run: RunRow; onChange: () => void }) {
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [renaming, setRenaming] = useState(false);
  const [newName, setNewName] = useState(run.name);
  const isLive = run.state === "running" || run.state === "queued";

  /** Returns whether it worked, so a caller can tell "done" from "refused".
   *  Swallowing that made the rename form close on a 409 and throw away what the
   *  user had typed -- with the error showing, but the input gone. */
  async function act(fn: () => Promise<unknown>): Promise<boolean> {
    setProblem(null);
    try {
      await fn();
      onChange();
      return true;
    } catch (err) {
      setProblem(err instanceof ApiError ? err.problem : { code: "unknown", message: String(err) });
      return false;
    }
  }

  return (
    <article className="card">
      <header className="card__head">
        <h2 className="card__title">
          {run.name} <span className="run-state" data-state={run.state}>{STATE_LABEL[run.state] ?? run.state}</span>
        </h2>
        <div className="row-actions row-actions--tight">
          {/* The entrance to Diagnóstico (ui.md §2). Only for a run that has
              provenance: without it there is no B to measure against, and the
              screen would only be able to refuse. */}
          {run.provenance && !isLive && (
            <Link className="button button--quiet" to="/diagnostics">
              Diagnóstico
            </Link>
          )}
          {isLive && (
            <button className="button button--quiet" onClick={() => act(() => stopRun(run.name))}>
              Parar
            </button>
          )}
          <button className="button button--quiet" onClick={() => setRenaming((v) => !v)} disabled={isLive}>
            Renombrar
          </button>
          <button className="button button--quiet" onClick={() => act(() => deleteRun(run.name))} disabled={isLive}>
            Borrar
          </button>
        </div>
      </header>

      {problem && <ErrorNote problem={problem} />}

      {renaming && (
        <form
          className="row-actions"
          onSubmit={async (e) => {
            e.preventDefault();
            // Only close if it actually happened. A 409 leaves the form open
            // with the name still in it, which is what you need to fix it.
            if (await act(() => renameRun(run.name, newName))) setRenaming(false);
          }}
        >
          <input value={newName} onChange={(e) => setNewName(e.target.value)} />
          <button className="button" type="submit">
            Guardar
          </button>
        </form>
      )}

      {run.error && <ErrorNote problem={{ code: "run_corrupt", message: run.error }} />}
      {run.provenance ? <ProvenanceFacts provenance={run.provenance} /> : null}
      <Progress run={run} />
    </article>
  );
}

/** Contract ③ on screen: the name to group, the value to reproduce.
 *
 * The fingerprint is here and not decoration: a B rebuilt under the same name is
 * a different B, and without the huella nothing would tell two runs apart that
 * "used the same dataset" (contract ⑧).
 */
function ProvenanceFacts({ provenance: p }: { provenance: Provenance }) {
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
function Progress({ run }: { run: RunRow }) {
  const [records, setRecords] = useState<EpochRecord[]>([]);
  const since = useRef(0);
  const isLive = run.state === "running" || run.state === "queued";

  useEffect(() => {
    let alive = true;

    async function poll() {
      try {
        // `since` is what the last call handed back, so only new epochs travel.
        // Re-sending the history each time is what made watching a run cost more
        // the longer it ran.
        const fresh = await getRunMetrics(run.name, since.current);
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
  }, [run.name, isLive]);

  if (records.length === 0) {
    return (
      <p className="card__foot">
        {isLive ? "Todavía no ha terminado ninguna época." : "Este run no dejó métricas."}
      </p>
    );
  }

  const last = records[records.length - 1];
  const total = run.summary?.epochs_requested;

  return (
    <>
      <p className="card__section">
        Progreso{" "}
        <span className="card__hint">
          época {last.epoch}
          {total ? ` de ${total}` : ""} · {run.seconds_per_epoch?.toFixed(1)} s/época de media
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
            {records.slice(-6).map((r) => (
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
      {run.summary && (
        <p className="card__foot">
          Mejor <code>{run.summary.monitor}</code>:{" "}
          {/* "sin medir", never a number: a monitor that never fired is an
              absence, and printing something there would invent a result. */}
          <strong>{run.summary.best === null ? "sin medir" : run.summary.best.toFixed(4)}</strong> ·{" "}
          {run.summary.epochs_run} de {run.summary.epochs_requested} épocas
          {run.summary.stopped_early && " · cortado por patience"}
          {run.summary.cancelled && " · parado a mano"}
        </p>
      )}
    </>
  );
}
