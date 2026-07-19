import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ApiError, deleteRun, getRun, renameRun, stopRun } from "../../api";
import { ErrorNote, Loading, type ApiProblem } from "../../components/Async";
import { useAsync } from "../../useAsync";
import { Progress, ProvenanceFacts, STATE_LABEL, isLiveState } from "./shared";

/** Runs (E) — el DETALLE de uno: qué pasó dentro.
 *
 * Lo que la lista no puede enseñar sin dejar de ser una lista: la procedencia
 * entera (contrato ③), las curvas (V14) y **todas** las épocas, no las seis
 * últimas. El X (`device`, `num_workers`) sale aquí y no en la lista a propósito:
 * cuesta tiempo, no resultado, así que no distingue un run de otro.
 */
export function RunDetail() {
  const { name = "" } = useParams();
  const navigate = useNavigate();
  const run = useAsync(() => getRun(name), [name]);
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [renaming, setRenaming] = useState(false);
  const [newName, setNewName] = useState(name);

  const state = run.data?.state.state;
  const isLive = state !== undefined && isLiveState(state);

  // Igual que en la lista: un run en cola pasa a corriendo, y uno corriendo
  // termina, sin que nadie haga clic. Las épocas tienen su propio poll
  // incremental (R5); esto es solo el estado, y para en cuanto nada se mueve.
  useEffect(() => {
    if (!isLive) return;
    const timer = setInterval(() => run.reload(), 2000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLive]);

  async function act(fn: () => Promise<unknown>): Promise<boolean> {
    setProblem(null);
    try {
      await fn();
      return true;
    } catch (err) {
      setProblem(err instanceof ApiError ? err.problem : { code: "unknown", message: String(err) });
      return false;
    }
  }

  return (
    <section className="screen screen--wide">
      <p className="card__hint">
        <Link to="/runs">← Runs</Link>
      </p>
      <h1 className="screen__title">
        {name}{" "}
        {state && (
          <span className="run-state" data-state={state}>
            {STATE_LABEL[state] ?? state}
          </span>
        )}
      </h1>

      {run.loading && !run.data && <Loading what="el run" />}
      {run.error && <ErrorNote problem={run.error} />}
      {problem && <ErrorNote problem={problem} />}
      {run.data?.state.error && (
        <ErrorNote problem={{ code: "run_error", message: run.data.state.error }} />
      )}

      {run.data && (
        <>
          <div className="row-actions row-actions--tight">
            {/* La entrada a Diagnóstico (ui.md §2). Solo para un run con
                procedencia: sin ella no hay B contra el que medir, y la pantalla
                solo podría negarse. */}
            {run.data.provenance && !isLive && (
              <Link className="button button--quiet" to="/diagnostics">
                Diagnóstico
              </Link>
            )}
            {isLive && (
              <button className="button button--quiet" onClick={() => act(() => stopRun(name))}>
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
              disabled={isLive}
              onClick={async () => {
                // Borrado el run, esta pantalla ya no tiene sujeto: se vuelve a
                // la lista en vez de quedarse enseñando un 404.
                if (await act(() => deleteRun(name))) navigate("/runs");
              }}
            >
              Borrar
            </button>
          </div>

          {renaming && (
            <form
              className="row-actions"
              onSubmit={async (e) => {
                e.preventDefault();
                // La URL lleva el nombre, así que renombrar mueve la pantalla:
                // quedarse en la vieja daría un 404 al siguiente refresco.
                if (await act(() => renameRun(name, newName))) {
                  navigate(`/runs/${encodeURIComponent(newName)}`, { replace: true });
                  setRenaming(false);
                }
              }}
            >
              <input value={newName} onChange={(e) => setNewName(e.target.value)} />
              <button className="button" type="submit">
                Guardar
              </button>
            </form>
          )}

          <article className="card">
            <ProvenanceFacts provenance={run.data.provenance} />
            <p className="card__section">
              Ejecución <span className="card__hint">X — cuesta tiempo, no resultado</span>
            </p>
            <dl className="facts facts--inline">
              <div className="fact">
                <dt>device</dt>
                <dd>
                  <code>{run.data.config.execution.device}</code>
                </dd>
              </div>
              <div className="fact">
                <dt>num_workers</dt>
                <dd>
                  <code>{run.data.config.execution.num_workers}</code>
                </dd>
              </div>
              <div className="fact">
                <dt>checkpoints</dt>
                <dd>
                  {run.data.checkpoints.length === 0 ? (
                    <span className="card__hint">ninguno</span>
                  ) : (
                    run.data.checkpoints.map((c) => (
                      <code key={c} style={{ marginRight: 8 }}>
                        {c}
                      </code>
                    ))
                  )}
                </dd>
              </div>
            </dl>
          </article>

          <article className="card">
            {/* 0 = todas: aquí sí cabe la historia entera, que es justamente lo
                que la lista no podía enseñar. */}
            <Progress
              name={name}
              state={run.data.state.state}
              summary={run.data.summary}
              secondsPerEpoch={run.data.seconds_per_epoch}
              epochsShown={0}
            />
          </article>
        </>
      )}
    </section>
  );
}
