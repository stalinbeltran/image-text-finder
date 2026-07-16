import { useEffect, useState } from "react";
import {
  buildPatchDataset,
  deletePatchDataset,
  getJob,
  listPatchDatasets,
  listSources,
  type Job,
  type Manifest,
  type PatchDataset,
} from "../api";
import { ApiError } from "../api";
import { ErrorNote, Empty, Loading, type ApiProblem } from "../components/Async";
import { useAsync } from "../useAsync";

/** Patches (B) — the data the CNN actually consumes.
 *
 * ui.md §2: this is where `n` is decided, and therefore where contract ① is
 * born. It shows the imbalance, which has been sitting in the manifest with
 * nobody looking at it and is the number that governs `pos_weight`.
 */
export function Patches() {
  const datasets = useAsync(listPatchDatasets, []);
  const [building, setBuilding] = useState(false);

  return (
    <section className="screen screen--wide">
      <h1 className="screen__title">Patches</h1>
      <p className="screen__lede">
        El dato que la CNN consume de verdad. Un dataset de patches se define por sus parámetros de
        extracción; una vez construido es <strong>autocontenido</strong> — lleva los píxeles, y
        entrenar ya no necesita la fuente. <strong>Aquí se decide <code>n</code></strong>, que es el
        número que tiene que casar con el <code>input_size</code> de la red.
      </p>

      <div className="row-actions">
        <button className="button" onClick={() => setBuilding((b) => !b)}>
          {building ? "Cancelar" : "Construir uno nuevo"}
        </button>
      </div>

      {building && (
        <BuildForm
          onDone={() => {
            setBuilding(false);
            datasets.reload();
          }}
        />
      )}

      {datasets.loading && <Loading what="los datasets de patches" />}
      {datasets.error && <ErrorNote problem={datasets.error} />}
      {datasets.data?.patch_datasets.length === 0 && !building && (
        <Empty>Todavía no hay ninguno. Construye uno desde una fuente.</Empty>
      )}

      {datasets.data?.patch_datasets.map((d) => (
        <PatchDatasetCard key={d.name} dataset={d} onChange={datasets.reload} />
      ))}
    </section>
  );
}

function PatchDatasetCard({ dataset, onChange }: { dataset: PatchDataset; onChange: () => void }) {
  const m = dataset.manifest;
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [busy, setBusy] = useState(false);

  async function remove() {
    setBusy(true);
    setProblem(null);
    try {
      await deletePatchDataset(dataset.name);
      onChange();
    } catch (err) {
      // The 409 of contract ③. Not a failure to handle away: it is the endpoint
      // telling you which runs would lose their provenance, and the whole reason
      // the check exists is that deleting anyway fails SILENTLY.
      setProblem(err instanceof ApiError ? err.problem : { code: "unknown", message: String(err) });
    } finally {
      setBusy(false);
    }
  }

  return (
    <article className="card">
      <header className="card__head">
        <h2 className="card__title">{dataset.name}</h2>
        <button className="button button--quiet" onClick={remove} disabled={busy}>
          Borrar
        </button>
      </header>

      {m.warnings?.map((w) => (
        <div className="async async--warning" key={w.code} role="alert">
          <p className="async__message">{w.message}</p>
          {w.hint && <p className="async__hint">{w.hint}</p>}
        </div>
      ))}
      {problem && <ErrorNote problem={problem} />}

      <dl className="facts facts--inline">
        <dt>n</dt>
        <dd>
          <strong>{m.config.patch_size}</strong>
        </dd>
        <dt title="stride de EXTRACCIÓN: parte de la identidad de B, fijo una vez construido">
          stride de extracción
        </dt>
        <dd>{m.config.stride}</dd>
        <dt>Fuente</dt>
        <dd>
          <code>{m.source_id}</code>
        </dd>
        <dt>Imágenes</dt>
        <dd>{m.num_samples}</dd>
        <dt>Patches</dt>
        <dd>{m.num_patches.toLocaleString("es")}</dd>
        <dt title="la semilla de B: fija QUÉ IMÁGENES caen en cada split. No es la semilla de la receta">
          semilla de split
        </dt>
        <dd>{m.config.seed}</dd>
      </dl>

      <Splits manifest={m} />
      <Imbalance manifest={m} />

      <p className="card__foot">
        <span title="huella del contenido: distingue un dataset reconstruido bajo el mismo nombre (contrato ⑧)">
          huella
        </span>{" "}
        <code>{m.fingerprint.replace("sha256:", "").slice(0, 12)}…</code>
      </p>
    </article>
  );
}

function Splits({ manifest: m }: { manifest: Manifest }) {
  const total = m.num_patches || 1;
  return (
    <div className="splits">
      {(["train", "val", "test"] as const).map((s) => {
        const count = m.patches_per_split[s] ?? 0;
        return (
          <div className="splits__row" key={s}>
            <span className="splits__label">{s}</span>
            <div className="splits__track">
              <div className="splits__fill" data-split={s} style={{ width: `${(count / total) * 100}%` }} />
            </div>
            <span className="splits__value">{count.toLocaleString("es")}</span>
          </div>
        );
      })}
    </div>
  );
}

/** The imbalance: positives per corner over total patches.
 *
 * ui.md §2 asks for it by name because it is the number that governs
 * `pos_weight` and it has been in the manifest all along with nobody reading it.
 * Shown as a ratio too, because that is the form `pos_weight` takes -- and
 * because the real one turned out to be 3.9:1, "modest", after the docs had
 * called the imbalance brutal for months. Seeing it is how that got corrected.
 */
function Imbalance({ manifest: m }: { manifest: Manifest }) {
  const total = m.num_patches || 1;
  return (
    <>
      <p className="card__section">
        Desbalance <span className="card__hint">positivos por esquina ÷ patches — gobierna <code>pos_weight</code></span>
      </p>
      <div className="imbalance">
        {m.corner_order.map((corner) => {
          const positives = m.positives_per_corner[corner] ?? 0;
          const frac = positives / total;
          const ratio = positives > 0 ? (total - positives) / positives : Infinity;
          return (
            <div className="imbalance__item" key={corner}>
              <div className="imbalance__head">
                <span className="meter__swatch" data-corner={corner} aria-hidden="true" />
                <span className="meter__label">{corner}</span>
              </div>
              <div className="meter__track">
                <div className="meter__fill" data-corner={corner} style={{ width: `${frac * 100}%` }} />
              </div>
              <p className="imbalance__value">
                {(frac * 100).toFixed(1)} %{" "}
                <span className="imbalance__ratio">
                  {Number.isFinite(ratio) ? `${ratio.toFixed(1)}:1` : "sin positivos"}
                </span>
              </p>
            </div>
          );
        })}
      </div>
    </>
  );
}

function BuildForm({ onDone }: { onDone: () => void }) {
  const sources = useAsync(listSources, []);
  const [name, setName] = useState("");
  // Derived, not defaulted in an effect. An effect runs AFTER the render that
  // already drew the options, so there is a window where the select shows its
  // list with nothing chosen and `source` is still "" -- and a submit in that
  // window posts an empty source. Deriving it is correct on the first render
  // that has data, so the window does not exist.
  const [picked, setPicked] = useState("");
  const source = picked || sources.data?.sources[0]?.id || "";
  const setSource = setPicked;
  const [patchSize, setPatchSize] = useState(40);
  const [stride, setStride] = useState(20);
  const [seed, setSeed] = useState(1);
  const [split, setSplit] = useState({ train: 0.8, val: 0.1, test: 0.1 });
  const [dropOverlap, setDropOverlap] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [problem, setProblem] = useState<ApiProblem | null>(null);

  // Building is a job (R3), so the result arrives by polling, not by the POST.
  useEffect(() => {
    if (!job || job.state === "done" || job.state === "error") return;
    const timer = setInterval(async () => {
      const fresh = await getJob(job.id);
      setJob(fresh);
      if (fresh.state === "done") {
        clearInterval(timer);
        onDone();
      }
    }, 500);
    return () => clearInterval(timer);
  }, [job, onDone]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setProblem(null);
    try {
      setJob(await buildPatchDataset({ name, source, patch_size: patchSize, stride, drop_overlap: dropOverlap, split, seed }));
    } catch (err) {
      setProblem(err instanceof ApiError ? err.problem : { code: "unknown", message: String(err) });
    }
  }

  return (
    <form className="card card--form" onSubmit={submit}>
      <h2 className="card__title">Construir un dataset de patches</h2>

      {sources.error && <ErrorNote problem={sources.error} />}
      {problem && <ErrorNote problem={problem} />}

      <div className="form__grid">
        <label className="field">
          <span className="field__label">Nombre</span>
          <input value={name} onChange={(e) => setName(e.target.value)} required placeholder="clear-40" />
        </label>

        <label className="field">
          <span className="field__label">Fuente</span>
          <select value={source} onChange={(e) => setSource(e.target.value)} disabled={!sources.data}>
            {sources.loading && <option value="">cargando…</option>}
            {sources.data?.sources.length === 0 && <option value="">no hay ninguna fuente</option>}
            {sources.data?.sources.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id} ({s.num_samples} imágenes)
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="field__label">
            n <span className="field__hint">el lado del patch; tendrá que casar con input_size</span>
          </span>
          <input type="number" min={1} value={patchSize} onChange={(e) => setPatchSize(+e.target.value)} />
        </label>

        <label className="field">
          <span className="field__label">
            stride de extracción <span className="field__hint">fijo una vez construido</span>
          </span>
          <input type="number" min={1} value={stride} onChange={(e) => setStride(+e.target.value)} />
        </label>

        <label className="field">
          <span className="field__label">
            semilla de split <span className="field__hint">qué imágenes caen en cada split</span>
          </span>
          <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} />
        </label>

        <label className="field field--check">
          <input type="checkbox" checked={dropOverlap} onChange={(e) => setDropOverlap(e.target.checked)} />
          <span className="field__label">Descartar imágenes con bloques solapados</span>
        </label>
      </div>

      <p className="card__section">
        Split <span className="card__hint">por imagen, nunca por patch: ventanas solapadas de una
        misma imagen en dos splits contaminarían el val</span>
      </p>
      <div className="form__grid form__grid--three">
        {(["train", "val", "test"] as const).map((k) => (
          <label className="field" key={k}>
            <span className="field__label">{k}</span>
            <input
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={split[k]}
              onChange={(e) => setSplit({ ...split, [k]: +e.target.value })}
            />
          </label>
        ))}
      </div>
      {split.val === 0 && (
        <div className="async async--warning" role="alert">
          <p className="async__message">Con val a 0 el dataset no sirve para medir.</p>
          <p className="async__hint">
            Sin val, elegir el mejor checkpoint cae en la pérdida de entrenamiento — o sea, en el más
            sobreajustado, y sin avisar.
          </p>
        </div>
      )}

      <div className="row-actions">
        {/* Not submittable until there IS a source. Without this the form posts
            an empty `source` while the list is still loading, and the server
            answers with FastAPI's own 422 -- which is not R4-shaped, so the user
            gets "422 Unprocessable Entity" instead of a reason and a fix. An
            empty required field should never reach the server. */}
        <button
          className="button"
          type="submit"
          disabled={!name || !source || Boolean(job && job.state !== "error")}
        >
          Construir
        </button>
        {job && (
          <span className="job-state" data-state={job.state}>
            job {job.id} · {job.state}
          </span>
        )}
      </div>
      {job?.state === "error" && (
        <ErrorNote problem={{ code: "job_failed", message: job.error ?? "el job falló" }} />
      )}
    </form>
  );
}
