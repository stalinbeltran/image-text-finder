import { useEffect, useState } from "react";
import {
  ApiError,
  createNetwork,
  deleteNetwork,
  listNetworks,
  validateNetwork,
  type BlockSpec,
  type NetworkConfig,
  type NetworkDescription,
} from "../api";
import { ErrorNote, Empty, Loading, type ApiProblem } from "../components/Async";
import { useAsync } from "../useAsync";

/** Redes (C) — the architecture, and nothing else.
 *
 * ui.md §2. It does not touch hyperparameters, datasets or weights. This is the
 * screen that did not exist: the endpoints were there and the front never called
 * them, so an architecture only ever existed embedded in a training form and
 * frozen inside a run — you could not name one, list one, or reuse one without
 * training it.
 *
 * It deliberately does NOT reuse the old `ModelConfigForm`: that component was
 * C + D + X in one form (its own comment admitted it), and `lockArchitecture` was
 * a boolean papering over the C/D border instead of resolving it.
 */
export function Networks() {
  const networks = useAsync(listNetworks, []);
  const [creating, setCreating] = useState(false);

  return (
    <section className="screen screen--wide">
      <h1 className="screen__title">Redes</h1>
      <p className="screen__lede">
        La arquitectura: <strong>config puro, cero datos</strong>. Una red se puede listar,
        comparar y versionar sin tocar un dataset ni entrenar nada — y esa propiedad es la que hace
        que tenga nombre y pantalla propia. Aquí no hay <code>lr</code>, ni <code>epochs</code>, ni{" "}
        <code>device</code>: eso es de la Receta y de Entrenar.
      </p>

      <div className="row-actions">
        <button className="button" onClick={() => setCreating((v) => !v)}>
          {creating ? "Cancelar" : "Crear una red"}
        </button>
      </div>

      {creating && (
        <NetworkForm
          onDone={() => {
            setCreating(false);
            networks.reload();
          }}
        />
      )}

      {networks.loading && <Loading what="las redes" />}
      {networks.error && <ErrorNote problem={networks.error} />}
      {networks.data?.networks.length === 0 && !creating && (
        <Empty>Todavía no hay ninguna red.</Empty>
      )}

      {networks.data?.networks.map((n) => (
        <NetworkCard key={n.name} name={n.name} config={n.config} onChange={networks.reload} />
      ))}
    </section>
  );
}

function NetworkCard({
  name,
  config,
  onChange,
}: {
  name: string;
  config: NetworkConfig;
  onChange: () => void;
}) {
  const described = useAsync<NetworkDescription>(() => validateNetwork(config), [name]);
  const [problem, setProblem] = useState<ApiProblem | null>(null);

  async function remove() {
    try {
      await deleteNetwork(name);
      onChange();
    } catch (err) {
      setProblem(err instanceof ApiError ? err.problem : { code: "unknown", message: String(err) });
    }
  }

  return (
    <article className="card">
      <header className="card__head">
        <h2 className="card__title">{name}</h2>
        <button className="button button--quiet" onClick={remove}>
          Borrar
        </button>
      </header>
      {problem && <ErrorNote problem={problem} />}

      <dl className="facts facts--inline">
        <dt title="tiene que ser igual al patch_size del dataset — es el contrato ①">input_size</dt>
        <dd>
          <strong>{config.input_size}</strong>
        </dd>
        <dt>Canales</dt>
        <dd>{config.in_channels}</dd>
        <dt title="mete los 4 flags de borde en la cabeza; el dataset ofrece y la red decide">
          border_features
        </dt>
        <dd>{config.border_features ? "sí" : "no"}</dd>
        <dt>Parámetros</dt>
        <dd>{described.data ? described.data.num_params.toLocaleString("es") : "…"}</dd>
      </dl>

      {described.data && <Trace trace={described.data.trace} input={config.input_size} />}
      {described.error && <ErrorNote problem={described.error} />}
    </article>
  );
}

/** The spatial trace: `40 → 20 → 10 → 5`.
 *
 * Free — no weights needed — and the only thing an untrained network can show
 * about itself. Which is exactly what makes this screen worth having before
 * anything has been trained.
 */
function Trace({ trace, input }: { trace: NetworkDescription["trace"]; input: number }) {
  return (
    <>
      <p className="card__section">
        Traza espacial <span className="card__hint">cómo encoge el patch, capa a capa</span>
      </p>
      <div className="trace">
        <span className="trace__size">{input}</span>
        {trace.map((t) => (
          <span className="trace__step" key={t.layer}>
            <span className="trace__arrow">→</span>
            <span className="trace__size">{t.out}</span>
            <span className="trace__channels">{t.channels}ch</span>
          </span>
        ))}
      </div>
    </>
  );
}

const EMPTY_BLOCK: BlockSpec = {
  filters: 32,
  kernel: 3,
  stride: 1,
  batchnorm: true,
  activation: "relu",
  pool: 2,
  dropout: 0,
};

function NetworkForm({ onDone }: { onDone: () => void }) {
  const [name, setName] = useState("");
  const [config, setConfig] = useState<NetworkConfig>({
    input_size: 40,
    in_channels: 1,
    border_features: false,
    backbone: [{ ...EMPTY_BLOCK }, { ...EMPTY_BLOCK, filters: 64 }],
    head: { hidden: [128], dropout: 0.3 },
  });
  const [described, setDescribed] = useState<NetworkDescription | null>(null);
  const [traceProblem, setTraceProblem] = useState<ApiProblem | null>(null);
  const [problem, setProblem] = useState<ApiProblem | null>(null);

  // Live: every edit re-validates. `POST /networks/validate` is pure and takes
  // milliseconds, so "does this even fit?" is answered while you type rather
  // than by a job exploding half an hour later.
  useEffect(() => {
    let live = true;
    validateNetwork(config)
      .then((d) => live && (setDescribed(d), setTraceProblem(null)))
      .catch((err) => {
        if (!live) return;
        setDescribed(null);
        setTraceProblem(err instanceof ApiError ? err.problem : { code: "unknown", message: String(err) });
      });
    return () => {
      live = false;
    };
  }, [config]);

  const setBlock = (i: number, patch: Partial<BlockSpec>) =>
    setConfig((c) => ({
      ...c,
      backbone: c.backbone.map((b, j) => (i === j ? { ...b, ...patch } : b)),
    }));

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setProblem(null);
    try {
      await createNetwork(name, config);
      onDone();
    } catch (err) {
      setProblem(err instanceof ApiError ? err.problem : { code: "unknown", message: String(err) });
    }
  }

  return (
    <form className="card card--form" onSubmit={submit}>
      <h2 className="card__title">Crear una red</h2>
      {problem && <ErrorNote problem={problem} />}

      <div className="form__grid">
        <label className="field">
          <span className="field__label">Nombre</span>
          <input value={name} onChange={(e) => setName(e.target.value)} required placeholder="cnn-a" />
        </label>
        <label className="field">
          <span className="field__label">
            input_size <span className="field__hint">tiene que casar con el n del dataset (contrato ①)</span>
          </span>
          <input
            type="number"
            min={1}
            value={config.input_size}
            onChange={(e) => setConfig({ ...config, input_size: +e.target.value })}
          />
        </label>
        <label className="field field--check">
          <input
            type="checkbox"
            checked={config.border_features}
            onChange={(e) => setConfig({ ...config, border_features: e.target.checked })}
          />
          <span className="field__label">
            border_features <span className="field__hint">exige un dataset que traiga los flags</span>
          </span>
        </label>
      </div>

      <p className="card__section">
        Backbone <span className="card__hint">cada bloque: conv → (batchnorm) → activación → (pool)</span>
      </p>
      {config.backbone.map((b, i) => (
        <div className="form__grid form__grid--block" key={i}>
          <label className="field">
            <span className="field__label">filtros</span>
            <input type="number" min={1} value={b.filters} onChange={(e) => setBlock(i, { filters: +e.target.value })} />
          </label>
          <label className="field">
            <span className="field__label">kernel</span>
            <input type="number" min={1} value={b.kernel} onChange={(e) => setBlock(i, { kernel: +e.target.value })} />
          </label>
          <label className="field">
            <span className="field__label">stride</span>
            <input type="number" min={1} value={b.stride} onChange={(e) => setBlock(i, { stride: +e.target.value })} />
          </label>
          <label className="field">
            <span className="field__label">pool</span>
            <input type="number" min={0} value={b.pool ?? 0} onChange={(e) => setBlock(i, { pool: +e.target.value })} />
          </label>
          <label className="field field--check">
            <input
              type="checkbox"
              checked={b.batchnorm ?? false}
              onChange={(e) => setBlock(i, { batchnorm: e.target.checked })}
            />
            <span className="field__label">batchnorm</span>
          </label>
          <button
            type="button"
            className="button button--quiet"
            onClick={() => setConfig({ ...config, backbone: config.backbone.filter((_, j) => j !== i) })}
          >
            Quitar
          </button>
        </div>
      ))}
      <div className="row-actions">
        <button
          type="button"
          className="button button--quiet"
          onClick={() => setConfig({ ...config, backbone: [...config.backbone, { ...EMPTY_BLOCK }] })}
        >
          + Bloque
        </button>
      </div>

      <p className="card__section">Traza espacial y tamaño</p>
      {traceProblem ? (
        <ErrorNote problem={traceProblem} />
      ) : described ? (
        <>
          <Trace trace={described.trace} input={config.input_size} />
          <p className="card__foot">
            {described.num_params.toLocaleString("es")} parámetros · {described.flat_features} features
            aplanadas
          </p>
        </>
      ) : (
        <Loading what="la traza" />
      )}

      <div className="row-actions">
        <button className="button" type="submit" disabled={!name || !described}>
          Guardar
        </button>
      </div>
    </form>
  );
}
