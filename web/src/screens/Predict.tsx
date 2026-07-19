import { useEffect, useMemo, useState } from "react";
import {
  listRuns,
  listSamples,
  listSources,
  predict,
  sampleImageUrl,
  type Prediction,
  type SampleInfo,
} from "../api";
import { Empty, ErrorNote, Loading, Working } from "../components/Async";
import { Declares } from "../components/Declares";
import { cornerColors } from "../theme/palette";
import { useAsync } from "../useAsync";
import { Scrubber } from "./Scrubber";

/** Predecir (F) — a run + a whole image → corners and paragraphs (ui.md §2).
 *
 * **The one screen whose input is a whole image.** Everything in Diagnóstico
 * takes a patch, because the patch is the real input of the CNN (contract ①); F
 * is the question of what happens when you slide that network across an image and
 * reassemble the answers.
 *
 * Two things carry the screen, both from the spec:
 *
 *  - **V11: the three stages, toggleable.** The failure of "the paragraph came
 *    out wrong" is born in one of three places — the per-patch prediction, NMS, or
 *    the greedy TL→BR pairing — and they are fixed in different domains. Overlaying
 *    the raw pre-NMS detections, the merged corners and the boxes, switchable,
 *    says which one lost the paragraph. Without `raw` that is not diagnosable.
 *
 *  - **The knobs are sliders with live repaint** (ui.md §2), not a form you
 *    submit. `threshold`, `stride`, the NMS radius and `min_size` are F, not D
 *    (organizacion.md §1-D): tuned post-hoc over a model already trained, so
 *    sweeping them costs a forward pass, not an afternoon. Making them a form
 *    would hide that they are free.
 */
type Stage = "raw" | "corners" | "paragraphs";

export function Predict() {
  const runs = useAsync(listRuns, []);
  const [run, setRun] = useState<string | null>(null);

  // Only runs that have a checkpoint can predict. A run with no provenance is
  // fine here — F never needs B, the checkpoint describes itself (contract ④) —
  // but one that never finished an epoch has no `best.pt` to load.
  const usable = (runs.data?.runs ?? []).filter(
    (r) => r.state === "done" || r.state === "cancelled" || r.state === "running"
  );

  useEffect(() => {
    if (!run && usable.length) setRun(usable[0].name);
  }, [run, usable]);

  const selected = usable.find((r) => r.name === run) ?? null;

  return (
    <section className="screen screen--wide">
      <h1 className="screen__title">Predecir</h1>
      <p className="screen__lede">
        Un run sobre una imagen entera: esquinas y párrafos. Es la única pantalla cuya entrada es
        una <strong>imagen</strong> y no un patch — deslizar la ventana y recomponer es justo lo
        que es F. Los <strong>knobs</strong> (umbral, stride, NMS) se ajustan aquí,{" "}
        <strong>sin reentrenar</strong>.
      </p>

      {runs.loading && !runs.data && <Loading what="los runs" />}
      {runs.error && <ErrorNote problem={runs.error} />}
      {runs.data && usable.length === 0 && (
        <Empty>
          No hay ningún run con checkpoint. Entrena uno desde Entrenar: sin <code>best.pt</code> no
          hay modelo que aplicar.
        </Empty>
      )}

      {selected && (
        <>
          <label className="field field--inline">
            <span className="field__label">Run (E)</span>
            <select value={run ?? ""} onChange={(e) => setRun(e.target.value)}>
              {usable.map((r) => (
                <option key={r.name} value={r.name}>
                  {r.name}
                </option>
              ))}
            </select>
          </label>
          <SourcePicker run={selected.name} />
        </>
      )}
    </section>
  );
}

const STAGES: { value: Stage; label: string; hint: string }[] = [
  { value: "raw", label: "crudas (pre-NMS)", hint: "cada ventana que superó el umbral, sin fusionar" },
  { value: "corners", label: "esquinas (post-NMS)", hint: "tras fusionar los duplicados" },
  { value: "paragraphs", label: "párrafos", hint: "tras emparejar TL→BR" },
];

function SourcePicker({ run }: { run: string }) {
  const sources = useAsync(listSources, []);
  const [source, setSource] = useState<string | null>(null);

  useEffect(() => {
    if (!source && sources.data?.sources.length) setSource(sources.data.sources[0].id);
  }, [source, sources.data]);

  return (
    <>
      {sources.loading && !sources.data && <Loading what="las fuentes" />}
      {sources.error && <ErrorNote problem={sources.error} />}
      {sources.data && (
        <label className="field field--inline">
          <span className="field__label">Fuente (A)</span>
          <select value={source ?? ""} onChange={(e) => setSource(e.target.value)}>
            {sources.data.sources.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id} ({s.num_samples})
              </option>
            ))}
          </select>
        </label>
      )}
      {source && <ImagePicker run={run} source={source} />}
    </>
  );
}

function ImagePicker({ run, source }: { run: string; source: string }) {
  const samples = useAsync<{ samples: SampleInfo[] }>(() => listSamples(source), [source]);
  const [index, setIndex] = useState(0);

  useEffect(() => setIndex(0), [source]);

  return (
    <>
      {samples.loading && !samples.data && <Loading what="las imágenes" />}
      {samples.error && <ErrorNote problem={samples.error} />}
      {samples.data && (
        <>
          <div className="thumbs">
            {samples.data.samples.slice(0, 24).map((s) => (
              <button
                key={s.index}
                className={`thumbs__item ${index === s.index ? "is-selected" : ""}`}
                onClick={() => setIndex(s.index)}
                title={`imagen ${s.index} · ${s.width}×${s.height}`}
              >
                <img src={sampleImageUrl(source, s.index, 96)} alt={`imagen ${s.index}`} />
                <span className="thumbs__label">{s.index}</span>
              </button>
            ))}
          </div>
          <PredictStage run={run} source={source} index={index} />
          {/* V5 fixes the same run+image and is a probe, not the pipeline: dragging
              a single window over the image and reading the 4 heads live, plus the
              1-px stability that sets the inference stride. */}
          <Scrubber run={run} source={source} index={index} />
        </>
      )}
    </>
  );
}

function PredictStage({ run, source, index }: { run: string; source: string; index: number }) {
  const [threshold, setThreshold] = useState(0.5);
  const [stride, setStride] = useState(20);
  const [nmsRadius, setNmsRadius] = useState(10);
  const [minSize, setMinSize] = useState(4);
  const [visible, setVisible] = useState<Record<Stage, boolean>>({
    raw: false,
    corners: true,
    paragraphs: true,
  });

  const result = useAsync(
    () =>
      predict(run, source, index, {
        threshold,
        stride,
        nms_radius: nmsRadius,
        min_size: minSize,
      }),
    [run, source, index, threshold, stride, nmsRadius, minSize]
  );

  return (
    <section className="view">
      <Declares
        view="V11"
        title="Etapas del pipeline"
        fixes="el run y la imagen"
        varies="la etapa (y los knobs de F)"
        measures="qué se pierde y dónde"
      >
        El fallo nace en una de tres etapas —predicción por patch, NMS, o reconstrucción voraz
        TL→BR— y cada una se arregla en un sitio distinto. Superponerlas dice cuál perdió el
        párrafo. <strong>Sin las crudas, "salió mal" no es diagnosticable.</strong>
      </Declares>

      <div className="controls controls--knobs">
        <Slider label="Umbral" title="p(esquina) por encima del cual cuenta una detección" value={threshold} min={0} max={1} step={0.01} onChange={setThreshold} format={(v) => v.toFixed(2)} />
        <Slider label="Stride" title="el stride de INFERENCIA, no el de B: cada cuántos px se coloca la ventana" value={stride} min={4} max={40} step={2} onChange={setStride} format={(v) => `${v} px`} />
        <Slider label="Radio NMS" title="dos detecciones más cerca que esto, de la misma esquina, son la misma vista dos veces" value={nmsRadius} min={0} max={30} step={1} onChange={setNmsRadius} format={(v) => `${v} px`} />
        <Slider label="Mínimo" title="cajas más finas que esto son dos esquinas que se emparejaron por azar, no un párrafo" value={minSize} min={0} max={40} step={1} onChange={setMinSize} format={(v) => `${v} px`} />
      </div>
      <p className="card__hint">
        Son knobs de <strong>F</strong>: se ajustan post-hoc, sobre el modelo ya entrenado. Mover
        cualquiera <strong>no reentrena nada</strong> — es un forward, no una tarde.
      </p>

      <div className="row-actions">
        {STAGES.map((s) => (
          <label key={s.value} className="check" title={s.hint}>
            <input
              type="checkbox"
              checked={visible[s.value]}
              onChange={(e) => setVisible((v) => ({ ...v, [s.value]: e.target.checked }))}
            />
            {s.label}
          </label>
        ))}
      </div>

      {result.loading && !result.data && <Loading what="la predicción" />}
      {result.loading && result.data && <Working what="la predicción" />}
      {result.error && <ErrorNote problem={result.error} />}
      {result.data && (
        <PipelineOverlay
          source={source}
          index={index}
          prediction={result.data}
          visible={visible}
          stale={result.loading}
        />
      )}
    </section>
  );
}

function Slider({
  label,
  title,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  title: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
}) {
  return (
    <label className="knob" title={title}>
      <span className="knob__label">
        {label} <strong>{format(value)}</strong>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

function PipelineOverlay({
  source,
  index,
  prediction,
  visible,
  stale,
}: {
  source: string;
  index: number;
  prediction: Prediction;
  visible: Record<Stage, boolean>;
  stale: boolean;
}) {
  const [w, h] = prediction.image_size;
  const colors = useMemo(() => cornerColors(), []);
  // Marks scale with the image, so a radius in image px must be scaled to the
  // SVG's own units — the viewBox IS image px, so 1:1, and a fixed r keeps the
  // dots readable regardless of image size via non-scaling stroke below.
  const r = Math.max(2, w / 120);

  return (
    <div className={`predict-stage ${stale ? "is-stale" : ""}`}>
      <div className="sample-view__stage" style={{ aspectRatio: `${w} / ${h}` }}>
        <img src={sampleImageUrl(source, index)} alt={`imagen ${index}`} />
        <svg viewBox={`0 0 ${w} ${h}`} className="sample-view__overlay">
          {/* Paragraphs first, so the corners sit on top of their boxes. */}
          {visible.paragraphs &&
            prediction.paragraphs.map((p, i) => (
              <rect
                key={`box-${i}`}
                x={p.box[0]}
                y={p.box[1]}
                width={p.box[2]}
                height={p.box[3]}
                className="predict-stage__box"
              />
            ))}
          {visible.raw &&
            prediction.raw.map((d, i) => (
              <circle
                key={`raw-${i}`}
                cx={d.x}
                cy={d.y}
                r={r * 0.6}
                fill={colors[d.corner as keyof typeof colors]}
                opacity={0.25}
              />
            ))}
          {visible.corners &&
            prediction.corners.map((d, i) => (
              <circle
                key={`corner-${i}`}
                cx={d.x}
                cy={d.y}
                r={r}
                fill={colors[d.corner as keyof typeof colors]}
              />
            ))}
        </svg>
      </div>

      <div className="predict-stage__facts">
        <span title="cada ventana × esquina que superó el umbral, antes de fusionar">
          <strong>{prediction.raw.length}</strong> crudas
        </span>
        <span title="tras fusionar los duplicados por NMS">
          <strong>{prediction.corners.length}</strong> esquinas
        </span>
        <span title="tras emparejar TL→BR">
          <strong>{prediction.paragraphs.length}</strong> párrafos
        </span>
        {/* The knobs the payload came from. The sliders are live and answers
            arrive out of order, so this is what a payload is FOR, not decoration:
            it says which parameters produced what is on screen. */}
        <span className="card__hint">
          umbral {prediction.knobs.threshold.toFixed(2)} · stride {prediction.knobs.stride} · NMS{" "}
          {prediction.knobs.nms_radius}
        </span>
      </div>
    </div>
  );
}
