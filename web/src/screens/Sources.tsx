import { useEffect, useState } from "react";
import {
  listSamples,
  listSources,
  sampleGeometry,
  sampleImageUrl,
  type Geometry,
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
        leen</strong>. Aquí no se decide <code>n</code> — eso es de Patches.
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
          <p className="screen__note">
            Raíz: <code>{sources.data.root}</code>
          </p>
          <div className="table__scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Fuente</th>
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

          {selected && <SampleBrowser source={selected} />}
        </>
      )}
    </section>
  );
}

function SampleBrowser({ source }: { source: Source }) {
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
        </>
      )}
    </>
  );
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
