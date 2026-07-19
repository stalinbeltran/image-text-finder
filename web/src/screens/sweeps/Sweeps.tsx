import { useEffect, useMemo, useState } from "react";
import {
  ApiError,
  createSweep,
  getSweep,
  listNetworks,
  listPatchDatasets,
  listRecipes,
  listSweeps,
  resumeSweep,
  stopSweep,
  OBJECTIVE_DIRECTION,
  type Distribution,
  type Objective,
  type Strategy,
  type SweepDetail as SweepDetailT,
  type SweepRow,
} from "../../api";
import { Empty, ErrorNote, Loading, type ApiProblem } from "../../components/Async";
import { useAsync } from "../../useAsync";
import { Pareto } from "./Pareto";
import { Parallel } from "./Parallel";

/** The recipe fields worth sweeping, with the shape of their distribution. Not
 *  every field of D: `epochs` is set by the budget, `monitor`/`seed` are the
 *  replication and selection axes, not knobs to optimise (organizacion.md §1-D). */
const SWEEPABLE: Record<
  string,
  { kind: "float" | "int" | "categorical"; log?: boolean; choices?: string[]; hint: string }
> = {
  lr: { kind: "float", log: true, hint: "el más influyente; log 1e-4…3e-2" },
  weight_decay: { kind: "float", log: true, hint: "L2; log 0…1e-2" },
  lambda_pos: { kind: "float", hint: "detectar vs. localizar — el árbitro (⑨)" },
  pos_weight: { kind: "float", hint: "el desbalance; ~3.9 en clear-paragraphs" },
  smooth_l1_beta: { kind: "float", hint: "~0.05–0.1 para que Huber se active" },
  momentum: { kind: "float", hint: "solo sgd/rmsprop; ~0.9" },
  grad_clip: { kind: "float", hint: "estabiliza lr altos" },
  batch_size: { kind: "int", hint: "acoplado a lr; es D, no X (⑩)" },
  optimizer: { kind: "categorical", choices: ["adam", "adamw", "sgd", "rmsprop"], hint: "ojo momentum en sgd" },
  scheduler: { kind: "categorical", choices: ["none", "cosine", "step", "plateau"], hint: "la omisión más cara" },
};

/** Barridos (H) — a space of D explored with B and C fixed → many E.
 *
 * ui.md §2. The screen that gives D its name: a sweep is literally "a list of
 * recipes", so it only exists because C and D became nouns in fase 3. It fixes a
 * B and a C (contract ⑧: same ruler for every point), varies a space over D,
 * declares an objective, and blocks `loss` while `lambda_pos` varies (contract
 * ⑨) -- the same 400 the server enforces, surfaced before the trip.
 */
export function Sweeps() {
  const sweeps = useAsync(listSweeps, []);
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <section className="screen screen--wide">
      <h1 className="screen__title">Barridos</h1>
      <p className="screen__lede">
        Un espacio de recetas (D) con el dataset (B) y la red (C) <strong>fijos</strong>: la única
        forma de que los puntos sean comparables (contrato ⑧). optuna propone y poda; la
        organización sigue siendo nuestra —un <em>trial</em> no es un run, lanza uno—.{" "}
        <strong>En CPU el límite de workers es 1</strong>: torch ya usa todos los núcleos, así que
        los puntos corren de uno en uno.
      </p>

      {selected ? (
        <SweepDetail
          name={selected}
          onClose={() => {
            setSelected(null);
            sweeps.reload();
          }}
        />
      ) : (
        <>
          <SweepForm
            onCreated={(name) => {
              sweeps.reload();
              setSelected(name);
            }}
          />
          <SweepList
            rows={sweeps.data?.sweeps ?? []}
            loading={sweeps.loading}
            error={sweeps.error}
            onSelect={setSelected}
          />
        </>
      )}
    </section>
  );
}

// ── The creation form ────────────────────────────────────────────────────────

function SweepForm({ onCreated }: { onCreated: (name: string) => void }) {
  const datasets = useAsync(listPatchDatasets, []);
  const networks = useAsync(listNetworks, []);
  const recipes = useAsync(listRecipes, []);

  const [name, setName] = useState("");
  const [pickedDataset, setPickedDataset] = useState("");
  const [pickedNetwork, setPickedNetwork] = useState("");
  const [baseRecipe, setBaseRecipe] = useState("");
  const [objective, setObjective] = useState<Objective>("f1");
  const [strategy, setStrategy] = useState<Strategy>("tpe");
  const [points, setPoints] = useState(12);
  const [epochs, setEpochs] = useState(10);
  const [pruning, setPruning] = useState(true);
  const [seed, setSeed] = useState(0);
  const [space, setSpace] = useState<Record<string, Distribution>>({
    lr: { type: "float", low: 1e-4, high: 3e-2, log: true },
  });
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [busy, setBusy] = useState(false);

  const dataset = pickedDataset || datasets.data?.patch_datasets[0]?.name || "";
  const network = pickedNetwork || networks.data?.networks[0]?.name || "";

  const loading = datasets.loading || networks.loading || recipes.loading;
  const missing =
    !loading &&
    ((datasets.data?.patch_datasets.length ?? 0) === 0 || (networks.data?.networks.length ?? 0) === 0);

  // Contract ⑨, client-side: it is a 400 on the server too, but showing it here
  // stops you filling the whole form to be refused on submit.
  const lambdaTrap = objective === "loss" && "lambda_pos" in space;
  const emptySpace = Object.keys(space).length === 0;
  const gridWithFloat =
    strategy === "grid" && Object.values(space).some((d) => d.type === "float");

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setProblem(null);
    setBusy(true);
    try {
      const job = await createSweep({
        name,
        patch_dataset: dataset,
        network,
        recipe: baseRecipe || null,
        space,
        objective,
        strategy,
        budget: { points, epochs, pruning },
        seed,
      });
      onCreated((job.detail as { sweep: string }).sweep ?? name);
    } catch (err) {
      setProblem(err instanceof ApiError ? err.problem : { code: "unknown", message: String(err) });
    } finally {
      setBusy(false);
    }
  }

  if (loading) return <Loading what="datasets, redes y recetas" />;
  if (missing)
    return (
      <Empty>
        Un barrido fija un dataset de patches (Datos → Patches) y una red (Modelo → Redes). Crea al
        menos uno de cada primero.
      </Empty>
    );

  return (
    <form className="card card--form" onSubmit={submit}>
      <p className="card__section">Nuevo barrido</p>
      {problem && <ErrorNote problem={problem} />}

      <div className="form__grid">
        <label className="field">
          <span className="field__label">
            Nombre <span className="field__hint">no se sobrescribe; los runs salen &lt;nombre&gt;-0000</span>
          </span>
          <input value={name} onChange={(e) => setName(e.target.value)} required placeholder="lr-optim-01" />
        </label>

        <label className="field">
          <span className="field__label">
            Dataset (B) <span className="field__hint">fijo — el mismo para todos los puntos (⑧)</span>
          </span>
          <select value={dataset} onChange={(e) => setPickedDataset(e.target.value)}>
            {datasets.data?.patch_datasets.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} (n={d.manifest.config.patch_size})
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="field__label">
            Red (C) <span className="field__hint">fija</span>
          </span>
          <select value={network} onChange={(e) => setPickedNetwork(e.target.value)}>
            {networks.data?.networks.map((n) => (
              <option key={n.name} value={n.name}>
                {n.name} (input_size={n.config.input_size})
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="field__label">
            Receta base <span className="field__hint">opcional; los campos fuera del espacio la heredan</span>
          </span>
          <select value={baseRecipe} onChange={(e) => setBaseRecipe(e.target.value)}>
            <option value="">— defaults</option>
            {recipes.data?.recipes.map((r) => (
              <option key={r.name} value={r.name}>
                {r.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      <p className="card__section">
        El espacio <span className="card__hint">qué campos de D varían, y en qué rango</span>
      </p>
      <SpaceBuilder space={space} onChange={setSpace} />

      <p className="card__section">Objetivo y presupuesto</p>
      <div className="form__grid">
        <label className="field">
          <span className="field__label">
            Objetivo <span className="field__hint">lo que ordena los puntos</span>
          </span>
          <select value={objective} onChange={(e) => setObjective(e.target.value as Objective)}>
            <option value="f1">f1 (maximizar)</option>
            <option value="pos_err_px">pos_err_px (minimizar)</option>
            <option value="loss">loss (minimizar)</option>
          </select>
        </label>
        <label className="field">
          <span className="field__label">
            Estrategia <span className="field__hint">random gana a grid si un parámetro manda</span>
          </span>
          <select value={strategy} onChange={(e) => setStrategy(e.target.value as Strategy)}>
            <option value="tpe">tpe (bayesiana)</option>
            <option value="random">random</option>
            <option value="grid">grid</option>
          </select>
        </label>
        <label className="field">
          <span className="field__label">
            Puntos <span className="field__hint">cuántas recetas probar</span>
          </span>
          <input type="number" min={1} value={points} onChange={(e) => setPoints(+e.target.value)} />
        </label>
        <label className="field">
          <span className="field__label">
            Épocas / punto <span className="field__hint">corto: la poda corta antes</span>
          </span>
          <input type="number" min={1} value={epochs} onChange={(e) => setEpochs(+e.target.value)} />
        </label>
        <label className="field field--check">
          <input type="checkbox" checked={pruning} onChange={(e) => setPruning(e.target.checked)} />
          <span className="field__label">
            Poda <span className="field__hint">la palanca nº1 en CPU: mata lo malo en la época 3</span>
          </span>
        </label>
        <label className="field">
          <span className="field__label">
            seed del sampler <span className="field__hint">qué puntos se prueban; no el de D ni el de B</span>
          </span>
          <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} />
        </label>
      </div>

      {lambdaTrap && (
        <div className="async async--warning" role="alert">
          <p className="async__message">
            No puedes rankear por <code>loss</code> mientras barres <code>lambda_pos</code>:{" "}
            <code>loss = cls + λ·pos</code>, así que λ=0 gana por definición (no predecir posiciones).
            Es el contrato ⑨.
          </p>
          <p className="async__hint">
            Rankea por <code>f1</code> o <code>pos_err_px</code> (independientes de λ), o saca{" "}
            <code>lambda_pos</code> del espacio.
          </p>
        </div>
      )}
      {gridWithFloat && (
        <div className="async async--warning" role="alert">
          <p className="async__message">
            <code>grid</code> no admite campos continuos: un rango float no tiene rejilla sin paso.
          </p>
          <p className="async__hint">Usa random o tpe, o barre solo campos int/categóricos con grid.</p>
        </div>
      )}

      <div className="row-actions">
        <button
          className="button"
          type="submit"
          disabled={!name || busy || emptySpace || lambdaTrap || gridWithFloat}
        >
          {busy ? "Lanzando…" : "Lanzar barrido"}
        </button>
        {emptySpace && <span className="card__hint">añade al menos un campo al espacio</span>}
      </div>
    </form>
  );
}

// ── The space builder ────────────────────────────────────────────────────────

function SpaceBuilder({
  space,
  onChange,
}: {
  space: Record<string, Distribution>;
  onChange: (next: Record<string, Distribution>) => void;
}) {
  function toggle(field: string, on: boolean) {
    const next = { ...space };
    if (!on) {
      delete next[field];
    } else {
      const meta = SWEEPABLE[field];
      if (meta.kind === "categorical") next[field] = { type: "categorical", choices: [...(meta.choices ?? [])] };
      else if (meta.kind === "int") next[field] = { type: "int", low: 8, high: 128 };
      else next[field] = { type: "float", low: meta.log ? 1e-4 : 0, high: meta.log ? 1e-2 : 1, log: meta.log };
    }
    onChange(next);
  }

  function patch(field: string, dist: Distribution) {
    onChange({ ...space, [field]: dist });
  }

  return (
    <ul className="space-builder">
      {Object.entries(SWEEPABLE).map(([field, meta]) => {
        const dist = space[field];
        const on = dist !== undefined;
        return (
          <li key={field} className={`space-row ${on ? "is-on" : ""}`}>
            <label className="space-row__toggle">
              <input type="checkbox" checked={on} onChange={(e) => toggle(field, e.target.checked)} />
              <code>{field}</code>
              <span className="field__hint">{meta.hint}</span>
            </label>
            {on && dist.type !== "categorical" && (
              <div className="space-row__params">
                <label>
                  low
                  <input
                    type="number"
                    value={dist.low}
                    onChange={(e) => patch(field, { ...dist, low: +e.target.value })}
                  />
                </label>
                <label>
                  high
                  <input
                    type="number"
                    value={dist.high}
                    onChange={(e) => patch(field, { ...dist, high: +e.target.value })}
                  />
                </label>
                <label className="space-row__log">
                  <input
                    type="checkbox"
                    checked={!!dist.log}
                    onChange={(e) => patch(field, { ...dist, log: e.target.checked })}
                  />
                  log
                </label>
              </div>
            )}
            {on && dist.type === "categorical" && (
              <div className="space-row__params">
                {(meta.choices ?? []).map((choice) => {
                  const chosen = dist.choices.includes(choice);
                  return (
                    <label key={choice} className="space-row__choice">
                      <input
                        type="checkbox"
                        checked={chosen}
                        onChange={(e) => {
                          const choices = e.target.checked
                            ? [...dist.choices, choice]
                            : dist.choices.filter((c) => c !== choice);
                          patch(field, { ...dist, choices });
                        }}
                      />
                      {choice}
                    </label>
                  );
                })}
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}

// ── The list ─────────────────────────────────────────────────────────────────

function SweepList({
  rows,
  loading,
  error,
  onSelect,
}: {
  rows: SweepRow[];
  loading: boolean;
  error: ApiProblem | null;
  onSelect: (name: string) => void;
}) {
  return (
    <div className="card">
      <p className="card__section">Barridos</p>
      {loading && <Loading what="los barridos" />}
      {error && <ErrorNote problem={error} />}
      {!loading && rows.length === 0 && <Empty>Todavía no hay ningún barrido.</Empty>}
      {rows.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th>Nombre</th>
              <th>Estado</th>
              <th>B · C</th>
              <th>Objetivo</th>
              <th>Progreso</th>
              <th>Mejor</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((s) => (
              <tr key={s.name}>
                <td>
                  <code>{s.name}</code>
                </td>
                <td>
                  <SweepState state={s.state} />
                </td>
                <td className="table__muted">
                  {s.patch_dataset} · {s.network}
                </td>
                <td>{s.objective}</td>
                <td>
                  {s.completed}/{s.points}
                </td>
                <td>{s.best ? s.best.value.toFixed(3) : "—"}</td>
                <td>
                  <button className="button button--quiet" onClick={() => onSelect(s.name)}>
                    Ver
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function SweepState({ state }: { state: SweepRow["state"] }) {
  const label: Record<string, string> = {
    queued: "en cola",
    running: "corriendo",
    done: "hecho",
    error: "error",
    cancelled: "cancelado",
    interrupted: "interrumpido",
  };
  return (
    <span className="run-state" data-state={state}>
      {label[state] ?? state}
    </span>
  );
}

// ── The detail ───────────────────────────────────────────────────────────────

const LIVE = new Set(["queued", "running"]);

function SweepDetail({ name, onClose }: { name: string; onClose: () => void }) {
  const sweep = useAsync(() => getSweep(name), [name]);
  const [stopping, setStopping] = useState(false);
  const [resuming, setResuming] = useState(false);
  const [resumeError, setResumeError] = useState<ApiProblem | null>(null);
  const data = sweep.data;

  // Poll while the sweep is live: points land one at a time, and the snapshot the
  // API reads is rewritten after each trial.
  useEffect(() => {
    if (!data || !LIVE.has(data.state)) return;
    const id = setInterval(sweep.reload, 2000);
    return () => clearInterval(id);
  }, [data, sweep.reload]);

  async function stop() {
    setStopping(true);
    try {
      await stopSweep(name);
      sweep.reload();
    } finally {
      setStopping(false);
    }
  }

  async function resume() {
    setResuming(true);
    setResumeError(null);
    try {
      await resumeSweep(name);
      sweep.reload();
    } catch (e) {
      // Shown, not swallowed: the two refusals (already complete, already
      // running) are the answer to "why did nothing happen?" — and a button that
      // fails silently is what sent someone looking for a button that was there.
      setResumeError(e instanceof ApiError ? e.problem : null);
    } finally {
      setResuming(false);
    }
  }

  // Everything left to run, and nothing running it. `completed` and `points` are
  // the same numbers the table shows, so the button appears exactly when the
  // screen already says the sweep is unfinished.
  const canResume =
    data != null && !LIVE.has(data.state) && data.completed < data.budget.points;

  return (
    <div>
      <div className="row-actions">
        <button className="button button--quiet" onClick={onClose}>
          ← Barridos
        </button>
        {data && LIVE.has(data.state) && (
          <button className="button button--quiet" onClick={stop} disabled={stopping}>
            {stopping ? "Parando…" : "Parar (entre trials)"}
          </button>
        )}
        {canResume && (
          <button className="button" onClick={resume} disabled={resuming}>
            {resuming ? "Reanudando…" : `Continuar (faltan ${data.budget.points - data.completed})`}
          </button>
        )}
      </div>

      {canResume && (
        <p className="card__hint">
          Reanudar <strong>no repite</strong> los puntos ya hechos: el estudio vive en disco
          (`optuna.db`) y el barrido cuenta los trials terminados y corre el resto.
        </p>
      )}

      {sweep.loading && !data && <Loading what="el barrido" />}
      {sweep.error && <ErrorNote problem={sweep.error} />}
      {resumeError && <ErrorNote problem={resumeError} />}

      {data && <SweepBody data={data} />}
    </div>
  );
}

function SweepBody({ data }: { data: SweepDetailT }) {
  const dir = OBJECTIVE_DIRECTION[data.objective];
  const spaceFields = useMemo(() => Object.keys(data.space), [data.space]);

  return (
    <>
      <div className="card">
        <h2 className="screen__title" style={{ fontSize: "1.2rem" }}>
          <code>{data.name}</code> <SweepState state={data.state} />
        </h2>
        <div className="view__facts">
          <span>
            B <strong>{data.patch_dataset}</strong> · C <strong>{data.network}</strong>
          </span>
          <span>
            objetivo <strong>{data.objective}</strong> ({dir === "maximize" ? "↑ mejor" : "↓ mejor"})
          </span>
          <span>
            {data.completed}/{data.budget.points} puntos · {data.budget.epochs} épocas/punto ·{" "}
            {data.budget.pruning ? "con poda" : "sin poda"}
          </span>
          {data.best && (
            <span>
              mejor <strong>{data.best.value.toFixed(3)}</strong> ({data.best.run})
            </span>
          )}
        </div>
        <p className="card__hint">espacio: {spaceFields.join(", ") || "—"}</p>
      </div>

      <Pareto progress={data} />
      <Parallel progress={data} />

      <div className="card">
        <p className="card__section">Puntos</p>
        <table className="table">
          <thead>
            <tr>
              <th>#</th>
              <th>Estado</th>
              <th>Run</th>
              {spaceFields.map((f) => (
                <th key={f}>{f}</th>
              ))}
              <th>{data.objective}</th>
              <th>f1</th>
              <th>pos_err_px</th>
            </tr>
          </thead>
          <tbody>
            {data.trials.map((t) => (
              <tr key={t.number} className={data.best?.number === t.number ? "is-best" : ""}>
                <td>{t.number}</td>
                <td>
                  <span className="run-state" data-state={t.state}>
                    {t.state}
                  </span>
                </td>
                <td className="table__muted">{t.run ?? "—"}</td>
                {spaceFields.map((f) => (
                  <td key={f}>{formatParam(t.params[f])}</td>
                ))}
                <td>{t.value !== null ? t.value.toFixed(3) : "—"}</td>
                <td>{t.f1 !== null ? t.f1.toFixed(3) : "—"}</td>
                <td>{t.pos_err_px !== null ? t.pos_err_px.toFixed(1) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function formatParam(v: number | string | undefined): string {
  if (v === undefined) return "—";
  if (typeof v === "string") return v;
  if (v !== 0 && (Math.abs(v) < 0.01 || Math.abs(v) >= 1000)) return v.toExponential(2);
  return String(Math.round(v * 1e6) / 1e6);
}
