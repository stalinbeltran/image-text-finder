import { useEffect, useState } from "react";
import {
  ApiError,
  getJob,
  listSamples,
  listSources,
  resizeSource,
  sampleGeometry,
  sampleImageUrl,
  type Derived,
  type Geometry,
  type Job,
  type SampleInfo,
  type Source,
} from "../api";
import { ErrorNote, Empty, Loading, type ApiProblem } from "../components/Async";
import { useAsync } from "../useAsync";

/** Fuentes (A) — read only.
 *
 * ui.md §2: it does not touch patches. `n` is not decided here, and there is no
 * form to build anything. That is the whole reason it is its own screen: the
 * source used to be chosen inside the extraction form, which meant you could
 * never look at a dataset on its own -- and looking at it on its own is exactly
 * what you need in order to judge whether it is any good.
 */
export function Sources() {
  const sources = useAsync(listSources, []);
  const [selected, setSelected] = useState<Source | null>(null);

  useEffect(() => {
    if (!selected && sources.data?.sources.length) setSelected(sources.data.sources[0]);
  }, [sources.data, selected]);

  return (
    <section className="screen screen--wide">
      <h1 className="screen__title">Fuentes</h1>
      <p className="screen__lede">
        Las imágenes y la geometría de sus párrafos: la verdad de campo. Las produce{" "}
        <code>image-text-sample-generator</code>, viven fuera del repo y aquí <strong>solo se
        leen</strong>. Las <strong>derivadas</strong> (un resize) son fuentes de pleno derecho y
        salen en esta misma lista, con su procedencia. Aquí no se decide <code>n</code> — eso es
        de Patches.
      </p>

      {sources.loading && <Loading what="las fuentes" />}
      {sources.error && <ErrorNote problem={sources.error} />}
      {sources.data && sources.data.sources.length === 0 && (
        <Empty>
          No hay ninguna fuente bajo <code>{sources.data.root}</code>. Genera uno con{" "}
          <code>image-text-sample-generator</code>, o apunta <code>ITF_DATASETS_ROOT</code> a otra
          carpeta.
        </Empty>
      )}

      {sources.data && sources.data.sources.length > 0 && (
        <>
          {/* Two roots since D19, and they are not interchangeable: the first is
              external and read-only, the second is ours and is where a resize
              writes. Saying only one of them would make a derived source look
              like it came from the generator. */}
          <p className="screen__note">
            Raíz: <code>{sources.data.root}</code>
            <br />
            Derivadas: <code>{sources.data.derived_root}</code>
          </p>
          <div className="table__scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Fuente</th>
                  <th>Procedencia</th>
                  <th className="table__num">Imágenes</th>
                  <th className="table__num">Con solape</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {sources.data.sources.map((s) => (
                  <tr key={s.id} className={selected?.id === s.id ? "is-selected" : ""}>
                    <td>
                      <code>{s.id}</code>
                    </td>
                    <td>
                      <Provenance derived={s.derived} />
                    </td>
                    <td className="table__num">{s.num_samples}</td>
                    <td className="table__num">
                      {s.num_overlapping > 0 ? (
                        <span title="bloques que se solapan de verdad; drop_overlap los descarta">
                          {s.num_overlapping}
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td className="table__num">
                      <button className="button button--quiet" onClick={() => setSelected(s)}>
                        Ver
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {selected && <SampleBrowser source={selected} onResized={sources.reload} />}
        </>
      )}
    </section>
  );
}

/** Where a source came from: its parent and the scale, or nothing if it is an original.
 *
 * The column exists because **the directory name is not a datum** (organizacion.md
 * ⑧). A derived source has exactly the same number of images as its parent, so
 * "Imágenes" does not separate them either -- what changed is the size, and
 * without this the two rows read identically.
 *
 * `null` is rendered as a dash, not as "unknown": absent means ORIGINAL, and that
 * is why the absence is legal (formatos.md §2).
 *
 * Shows `from`, never `from_declared_id`: the addressable id is the one that
 * takes you back to the parent, and the declared one is **not unique** -- the two
 * `clear-paragraphs-02` share theirs, which is the exact pair behind the 14.5x
 * mistake this project already made once.
 */
function Provenance({ derived }: { derived: Derived | null }) {
  if (!derived) return <span className="muted">—</span>;

  const scale = derived.scale ? `×${derived.scale[0].toFixed(2).replace(/\.?0+$/, "")}` : "";
  const size = derived.size ? `${derived.size[0]}×${derived.size[1]}` : "tamaños mixtos";
  return (
    <span
      className="provenance"
      title={`${derived.op} de ${derived.from} → ${size}. El padre se declara a sí mismo como '${derived.from_declared_id}', que no es único.`}
    >
      ← <code>{derived.from}</code> {scale}
    </span>
  );
}

function SampleBrowser({ source, onResized }: { source: Source; onResized: () => void }) {
  const samples = useAsync<{ samples: SampleInfo[] }>(() => listSamples(source.id), [source.id]);
  const [index, setIndex] = useState(0);

  useEffect(() => setIndex(0), [source.id]);

  return (
    <>
      <h2 className="screen__section">Muestras de {source.id}</h2>
      {samples.loading && <Loading what="las muestras" />}
      {samples.error && <ErrorNote problem={samples.error} />}
      {samples.data && (
        <>
          <p className="screen__note">
            Una <strong>muestra es una imagen</strong>, no un ejemplo de entrenamiento — el ejemplo
            es el patch. {samples.data.samples.length} imágenes.
          </p>
          <div className="thumbs">
            {samples.data.samples.slice(0, 24).map((s) => (
              <button
                key={s.index}
                className={`thumbs__item ${index === s.index ? "is-selected" : ""}`}
                onClick={() => setIndex(s.index)}
                title={`imagen ${s.index} · ${s.width}×${s.height} · ${s.num_blocks} bloques`}
              >
                <img src={sampleImageUrl(source.id, s.index, 96)} alt={`imagen ${s.index}`} />
                <span className="thumbs__label">{s.index}</span>
              </button>
            ))}
          </div>
          <SampleView source={source} index={index} />
          <ResizeForm source={source} samples={samples.data.samples} onDone={onResized} />
        </>
      )}
    </>
  );
}

/** Resize the SELECTED source into a derived one (D19, ui.md §2).
 *
 * It lives here, under the source you are looking at, and takes no source
 * `select` of its own **on purpose**: a second picker would let you resize A
 * while inspecting B, and picking the wrong source does not fail -- it builds a
 * perfectly valid dataset that measures something else (organizacion.md §3).
 */
function ResizeForm({
  source,
  samples,
  onDone,
}: {
  source: Source;
  samples: SampleInfo[];
  onDone: () => void;
}) {
  const [name, setName] = useState("");
  const [axis, setAxis] = useState<"width" | "height">("width");
  const [value, setValue] = useState(80);
  const [job, setJob] = useState<Job | null>(null);
  const [problem, setProblem] = useState<ApiProblem | null>(null);

  useEffect(() => {
    setName("");
    setJob(null);
    setProblem(null);
  }, [source.id]);

  // A job (R3): the result arrives by polling, not from the POST.
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

  // Shown live, from a real sample: it turns "80" into a decision instead of a
  // bet, and it is where you SEE that this only ever shrinks -- before sending.
  const first = samples[0];
  const preview = previewSize(first, axis, value);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setProblem(null);
    try {
      setJob(
        await resizeSource(source.id, {
          name,
          [axis]: value,
        } as { name: string; width?: number; height?: number })
      );
    } catch (err) {
      setProblem(err instanceof ApiError ? err.problem : { code: "unknown", message: String(err) });
    }
  }

  return (
    <form className="card card--form" onSubmit={submit}>
      <h2 className="card__title">Redimensionar esta fuente</h2>
      <p className="card__hint">
        Crea una <strong>fuente derivada</strong> reduciendo las imágenes y reescalando su
        geometría. Se mantiene la proporción: das <strong>el ancho o el alto</strong>, y el otro
        sale solo. <strong>Solo reduce</strong> — ampliar interpola, y un dataset de patches
        extraído de ahí mediría el interpolador. El original no se toca.
      </p>

      {problem && <ErrorNote problem={problem} />}

      <div className="form__grid">
        <label className="field">
          <span className="field__label">Nombre de la derivada</span>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            placeholder={`${source.id.split("/").pop()}-${axis === "width" ? "w" : "h"}${value}`}
          />
        </label>

        <label className="field">
          <span className="field__label">
            Dimensión <span className="field__hint">una de las dos; la otra se deriva</span>
          </span>
          <div className="resize-dim">
            <select value={axis} onChange={(e) => setAxis(e.target.value as "width" | "height")}>
              <option value="width">ancho</option>
              <option value="height">alto</option>
            </select>
            <input
              type="number"
              min={1}
              value={value}
              onChange={(e) => setValue(+e.target.value)}
              required
            />
            <span className="resize-dim__unit">px</span>
          </div>
        </label>
      </div>

      {preview && (
        <p className={`resize-preview ${preview.grows ? "resize-preview--grows" : ""}`}>
          {preview.from} → <strong>{preview.to}</strong>{" "}
          {preview.grows ? (
            <span className="field__hint">
              — esto <strong>ampliaría</strong>, y se rechazará: pide una dimensión menor que{" "}
              {axis === "width" ? first.width : first.height}
            </span>
          ) : (
            <span className="field__hint">
              — {source.num_samples} imágenes, ×{preview.scale}
            </span>
          )}
        </p>
      )}

      {/* Same shape as the extraction form: `row-actions` + `job-state`, and the
          job's failure rendered as an ErrorNote. Reused rather than restyled --
          two forms that do the same thing (POST -> job -> poll) should not look
          like two different mechanisms. */}
      <div className="row-actions">
        <button
          className="button"
          type="submit"
          disabled={!name || !value || Boolean(job && job.state !== "error")}
        >
          Redimensionar
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

/** Python's `round`: half to EVEN, not half up. `Math.round(2.5)` is 3; this is 2.
 *
 * Measured, not theorised: a 100×50 source asked for width 5 gives height **2**
 * on the server and **3** from `Math.round`. One pixel, in a preview, in a case
 * that needs an exact .5 -- which is precisely the profile of every trap in
 * organizacion.md §3: small, silent, and nobody's decision.
 */
function roundHalfToEven(x: number): number {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff !== 0.5) return Math.round(x);
  return floor % 2 === 0 ? floor : floor + 1;
}

/** The output size for one sample, mirroring `itf.imageops.target_size`.
 *
 * A COPY of a rule that lives in Python, and worth being uneasy about -- it is
 * the shape of contract ⑤. Tolerable only because it is a *preview*: the
 * authority is `check_resize`, which refuses with a 400, and nothing here can let
 * a bad request through. If it ever starts deciding instead of previewing, it
 * moves to the server.
 *
 * But a preview that lies is worse than no preview, so the mirror is exact,
 * rounding included.
 */
function previewSize(sample: SampleInfo | undefined, axis: "width" | "height", value: number) {
  if (!sample || !value || value < 1) return null;
  const [w, h] =
    axis === "width"
      ? [value, Math.max(1, roundHalfToEven((sample.height * value) / sample.width))]
      : [Math.max(1, roundHalfToEven((sample.width * value) / sample.height)), value];
  return {
    from: `${sample.width}×${sample.height}`,
    to: `${w}×${h}`,
    grows: w > sample.width || h > sample.height,
    scale: (w / sample.width).toFixed(2).replace(/\.?0+$/, ""),
  };
}

function SampleView({ source, index }: { source: Source; index: number }) {
  const geom = useAsync<Geometry>(() => sampleGeometry(source.id, index), [source.id, index]);

  if (geom.loading) return <Loading what="la geometría" />;
  if (geom.error) return <ErrorNote problem={geom.error as ApiProblem} />;
  if (!geom.data) return null;

  const g = geom.data;
  const paragraphs = g.blocks.filter((b) => b.kind === "paragraph");

  return (
    <div className="sample-view">
      {/* The server sends the quads as NUMBERS and the browser draws them --
          same reasoning as map_payload. An SVG overlaid on the image scales with
          it and needs no canvas. */}
      <div className="sample-view__stage" style={{ aspectRatio: `${g.width} / ${g.height}` }}>
        <img src={sampleImageUrl(source.id, index)} alt={`imagen ${index}`} />
        <svg viewBox={`0 0 ${g.width} ${g.height}`} className="sample-view__overlay">
          {paragraphs.map((b) => (
            <g key={b.block_id}>
              <polygon
                points={b.quad.map(([x, y]) => `${x},${y}`).join(" ")}
                className="sample-view__quad"
              />
              {/* The 4 corners in their fixed slots (R1). Same TL colour here as
                  in every other view: the colour follows the entity. */}
              {b.quad.map(([x, y], c) => (
                <circle key={c} cx={x} cy={y} r={3} data-corner-fill={["TL", "TR", "BR", "BL"][c]} />
              ))}
            </g>
          ))}
        </svg>
      </div>
      <dl className="facts">
        <dt>Tamaño</dt>
        <dd>
          {g.width} × {g.height}
        </dd>
        <dt>Párrafos</dt>
        <dd>{paragraphs.length}</dd>
        <dt>Solape</dt>
        <dd>{g.has_overlap ? "sí" : "no"}</dd>
        <dt>Ángulo</dt>
        <dd>
          {paragraphs.every((b) => Math.abs(b.angle) < 0.01)
            ? "0° (todos)"
            : `hasta ${Math.max(...paragraphs.map((b) => Math.abs(b.angle))).toFixed(1)}°`}
        </dd>
      </dl>
    </div>
  );
}
