import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, deleteRun, listRuns, renameRun, stopRun, type RunRow } from "../../api";
import { ErrorNote, Empty, Loading, type ApiProblem } from "../../components/Async";
import { useAsync } from "../../useAsync";
import { STATE_LABEL, isLiveState } from "./shared";

/** Runs (E) — la LISTA: qué runs hay y qué se puede hacer con ellos.
 *
 * ui.md §2. Antes cada run venía con sus curvas y su tabla de épocas desplegadas,
 * así que ver *qué runs existen* costaba una pantalla de scroll por run. Aquí una
 * fila por run con lo que distingue a uno de otro — estado, procedencia por
 * nombre (contrato ③) y el mejor valor del monitor — más el CRUD. Lo de dentro
 * (curvas, épocas, entorno, checkpoints) vive en `/runs/:name`.
 *
 * La procedencia sigue siendo **por nombre**: es lo que permite preguntar "¿qué
 * runs usaron la red X?" sin diffear diccionarios, que es lo que un barrido
 * pregunta todo el rato.
 */
export function Runs() {
  const runs = useAsync(listRuns, []);
  const live = (runs.data?.runs ?? []).some((r) => isLiveState(r.state));

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
        Abre uno para ver sus curvas y sus épocas.
      </p>

      {runs.loading && !runs.data && <Loading what="los runs" />}
      {runs.error && <ErrorNote problem={runs.error} />}
      {runs.data?.runs.length === 0 && (
        <Empty>Todavía no hay ningún run. Lanza uno desde Entrenar.</Empty>
      )}

      {runs.data && runs.data.runs.length > 0 && (
        <div className="table__scroll">
          {/* Densa: con nueve columnas sin partir, el padding normal empujaba
              «Borrar» fuera de la ventana a 1400 px — una acción que hay que
              desplazar para ver es una acción que no está. */}
          <table className="table table--dense">
            <thead>
              <tr>
                <th>run</th>
                <th>estado</th>
                <th>dataset (B)</th>
                <th>red (C)</th>
                <th>receta (D)</th>
                <th className="table__num">mejor</th>
                <th className="table__num">épocas</th>
                <th className="table__num" title="segundos por época de media">
                  s/época
                </th>
                <th />
              </tr>
            </thead>
            <tbody>
              {runs.data.runs.map((run) => (
                <RunRowView key={run.name} run={run} onChange={runs.reload} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function RunRowView({ run, onChange }: { run: RunRow; onChange: () => void }) {
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [renaming, setRenaming] = useState(false);
  const [newName, setNewName] = useState(run.name);
  const isLive = isLiveState(run.state);
  const p = run.provenance;

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
    <>
      <tr>
        <td className="table__nowrap">
          <Link to={`/runs/${encodeURIComponent(run.name)}`}>{run.name}</Link>
        </td>
        <td>
          <span className="run-state" data-state={run.state}>
            {STATE_LABEL[run.state] ?? run.state}
          </span>
        </td>
        {/* Un run sin procedencia (los de antes del contrato ③) no puede decir de
            qué salió, y la fila lo dice en voz alta en vez de dejar el hueco en
            blanco: ausente ≠ vacío. */}
        <td className="table__nowrap">
          {p ? <code>{p.patch_dataset.name}</code> : <span className="card__hint">sin procedencia</span>}
        </td>
        <td className="table__nowrap">{p ? <code>{p.network.name}</code> : null}</td>
        <td className="table__nowrap">{p ? <code>{p.recipe.name}</code> : null}</td>
        <td className="table__num" title={run.summary ? `monitor: ${run.summary.monitor}` : undefined}>
          {/* "sin medir", nunca un número: un monitor que no llegó a disparar es
              una ausencia, y poner algo ahí inventaría un resultado. */}
          {run.summary
            ? run.summary.best === null
              ? "sin medir"
              : run.summary.best.toFixed(4)
            : "—"}
        </td>
        <td className="table__num">
          {run.summary ? `${run.summary.epochs_run}/${run.summary.epochs_requested}` : "—"}
        </td>
        {/* `null` (no ha terminado ninguna época) se lee "—", no 0: un 0 diría
            "instantáneo". */}
        <td className="table__num">
          {run.seconds_per_epoch === null ? "—" : run.seconds_per_epoch.toFixed(1)}
        </td>
        <td className="table__nowrap">
          <div className="row-actions row-actions--tight">
            {isLive && (
              <button className="button button--quiet" onClick={() => act(() => stopRun(run.name))}>
                Parar
              </button>
            )}
            <button
              className="button button--quiet"
              onClick={() => setRenaming((v) => !v)}
              disabled={isLive}
            >
              Renombrar
            </button>
            <button
              className="button button--quiet"
              onClick={() => act(() => deleteRun(run.name))}
              disabled={isLive}
            >
              Borrar
            </button>
          </div>
        </td>
      </tr>

      {(problem || renaming || run.error) && (
        <tr>
          <td colSpan={9}>
            {problem && <ErrorNote problem={problem} />}
            {run.error && <ErrorNote problem={{ code: "run_corrupt", message: run.error }} />}
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
          </td>
        </tr>
      )}
    </>
  );
}
