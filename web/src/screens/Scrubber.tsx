import { useEffect, useRef, useState } from "react";
import { getWindow, sampleImageUrl, type WindowPrediction } from "../api";
import { ErrorNote, Loading, Working } from "../components/Async";
import { Declares } from "../components/Declares";
import { Meter } from "../components/Meter";
import { cornerColors, CORNER_NAMES, type CornerName } from "../theme/palette";

/** V5 — the scrubber. **Fix E and the image, drag one n×n window over it.**
 *
 * `n` is the run's `patch_size` and comes from the payload, never from a
 * constant: the runs in this project are not all 40 px, and a screen that says
 * "40×40" against a 10-px model is confidently wrong.
 *
 * Two things make it worth a view of its own (ui.md §4.1):
 *
 *  - **The input stays in distribution.** Unlike a pixel editor, every crop is
 *    real pixels the extractor could have produced, so the prediction says
 *    something about the model rather than about an artefact it never saw
 *    (ui.md §5). The crop's border flags come from `window_at` — the same formula
 *    B extracted with (contract ⑤) — so a crop flush against an edge is flagged
 *    exactly as its trained-on twin.
 *
 *  - **It measures the stability the stride depends on.** Move the window one
 *    pixel and the prediction should barely move; if it jumps, that jitter is what
 *    sets how fine an inference `stride` can be and how big the NMS radius must be.
 *    The payload carries, per corner, the largest change in score over the four
 *    1-px neighbours — a number a single prediction cannot show.
 *
 * The response is **authoritative**: the backend clamps the crop inside the image
 * and hands back the actual `(x0, y0)` and `patch_size`, so the box always sits
 * where the model really looked, not merely where the pointer was. And the request
 * is **debounced** (ui.md §5): a drag fires one forward per settle, not per pixel.
 */
const DEBOUNCE_MS = 60;

export function Scrubber({ run, source, index }: { run: string; source: string; index: number }) {
  const stageRef = useRef<HTMLDivElement>(null);
  // Where the pointer asked for the crop. The response carries where the model
  // actually looked (clamped), which is what we draw.
  const [req, setReq] = useState<{ x0: number; y0: number }>({ x0: 0, y0: 0 });
  const [debounced, setDebounced] = useState(req);
  const [dragging, setDragging] = useState(false);
  const [data, setData] = useState<WindowPrediction | null>(null);
  const [error, setError] = useState<Parameters<typeof ErrorNote>[0]["problem"] | null>(null);
  const [loading, setLoading] = useState(true);

  // A new image resets the window to the top-left; the first fetch learns the
  // image size and patch size from the response.
  useEffect(() => {
    setReq({ x0: 0, y0: 0 });
    setDebounced({ x0: 0, y0: 0 });
  }, [run, source, index]);

  useEffect(() => {
    const t = setTimeout(() => setDebounced(req), DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [req]);

  useEffect(() => {
    let live = true;
    setLoading(true);
    getWindow(run, source, index, debounced.x0, debounced.y0)
      .then((d) => {
        if (!live) return;
        setData(d);
        setError(null);
      })
      .catch((e) => {
        if (!live) return;
        setError(e?.problem ?? { code: "network_error", message: String(e) });
      })
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [run, source, index, debounced.x0, debounced.y0]);

  // Pointer → image px. The stage's box maps to the image's [0, w]×[0, h], and the
  // requested top-left centres the window on the pointer.
  const moveTo = (clientX: number, clientY: number) => {
    const stage = stageRef.current;
    if (!stage || !data) return;
    const rect = stage.getBoundingClientRect();
    const [w, h] = data.image_size;
    const n = data.patch_size;
    const ix = ((clientX - rect.left) / rect.width) * w;
    const iy = ((clientY - rect.top) / rect.height) * h;
    setReq({
      x0: Math.round(Math.max(0, Math.min(ix - n / 2, w - n))),
      y0: Math.round(Math.max(0, Math.min(iy - n / 2, h - n))),
    });
  };

  const colors = cornerColors();

  return (
    <section className="view">
      <Declares
        view="V5"
        title="Scrubber de la ventana"
        fixes="el run y la imagen"
        varies="el recorte (x0, y0)"
        measures="la predicción y su estabilidad"
      >
        {/* El tamaño lo dice el modelo, no la prosa: este texto decía «40×40»
            fijo, y contra un run de patches de 10 px era simplemente falso — la
            misma constante heredada que hacía que V11 mandara stride 20. */}
        Arrastra la ventana de {data ? `${data.patch_size}×${data.patch_size}` : "n×n"} por la
        imagen y lee las 4 cabezas en vivo. La entrada está{" "}
        <strong>en distribución</strong> (recorta píxeles reales, no los inventa), y mide{" "}
        <strong>cuánto tiembla</strong> la predicción al mover la ventana 1 px — que es lo que fija
        el <code>stride</code> de inferencia y el radio de NMS. El <strong>área</strong> de cada
        punto es proporcional a su score (relleno si pasa de 0,5; aro vacío si no), así que se ve
        qué esquinas detecta sin mirar los medidores.
      </Declares>

      {loading && !data && <Loading what="la ventana" />}
      {loading && data && <Working what="la ventana" />}
      {error && <ErrorNote problem={error} />}

      {data && (
        <div className={`scrubber ${loading ? "is-stale" : ""}`}>
          <div
            ref={stageRef}
            className="scrubber__stage sample-view__stage"
            style={{ aspectRatio: `${data.image_size[0]} / ${data.image_size[1]}`, touchAction: "none" }}
            onPointerDown={(e) => {
              (e.target as Element).setPointerCapture?.(e.pointerId);
              setDragging(true);
              moveTo(e.clientX, e.clientY);
            }}
            onPointerMove={(e) => dragging && moveTo(e.clientX, e.clientY)}
            onPointerUp={() => setDragging(false)}
            onPointerCancel={() => setDragging(false)}
          >
            <img src={sampleImageUrl(source, index)} alt={`imagen ${index}`} draggable={false} />
            <svg viewBox={`0 0 ${data.image_size[0]} ${data.image_size[1]}`} className="sample-view__overlay">
              {/* The window, where the model actually looked (authoritative). */}
              <rect
                x={data.x0}
                y={data.y0}
                width={data.patch_size}
                height={data.patch_size}
                className="scrubber__window"
              />
              {/* The predicted corners, in image px. **El radio lleva el score**:
                  el área es proporcional a la puntuación (r ∝ √score), que es la
                  codificación honesta para un punto — el ojo lee área, no radio.
                  Un punto grande = detectada; uno diminuto = la cabeza dice que no,
                  pero sigue dibujado (con el aro vacío) porque *dónde* la sitúa
                  cuando no la ve también es información. Sin esto los cuatro puntos
                  se veían idénticos y una esquina inventada tenía la misma cara
                  que una detectada. */}
              {data.corners.map((c) => {
                // El punto más grande cabe en la ventana: el radio se mide en
                // unidades del patch (que es de 10 px en unos runs y de 40 en
                // otros), con un tope en unidades de imagen para que un patch
                // grande no tape el recorte. El radio fijo de antes —imagen/140—
                // dibujaba sobre un patch de 10 px puntos más anchos que la
                // propia ventana.
                const rMax = Math.max(3, Math.min(data.patch_size * 0.5, data.image_size[0] / 40));
                // El suelo se mide contra `rMax`, no en píxeles absolutos: un
                // score de 0,000 tenía que seguir siendo un punto localizable.
                const rMin = rMax * 0.18;
                const r = rMin + (rMax - rMin) * Math.sqrt(Math.max(0, Math.min(1, c.score)));
                const hit = c.score >= 0.5;
                return (
                  <circle
                    key={c.corner}
                    cx={c.image_x}
                    cy={c.image_y}
                    r={r}
                    fill={hit ? colors[c.corner as CornerName] : "none"}
                    stroke={colors[c.corner as CornerName]}
                    // El trazo también en px de imagen: con `rMax * 0.14` sobre un
                    // patch de 10 px salía un pelo de medio píxel — un aro correcto
                    // e invisible en pantalla.
                    strokeWidth={Math.max(rMax * 0.25, data.image_size[0] / 400)}
                    opacity={hit ? 0.9 : 0.65}
                  >
                    <title>
                      {c.corner}: {c.score.toFixed(3)}
                    </title>
                  </circle>
                );
              })}
            </svg>
          </div>

          <div className="scrubber__panel">
            <div className="scrubber__meters">
              {CORNER_NAMES.map((corner, c) => (
                <div key={corner} className="scrubber__corner">
                  <Meter
                    corner={corner as CornerName}
                    value={data.corners[c].score}
                    threshold={0.5}
                  />
                  {/* The 1-px jitter, per head: near 0 = stable, so a fine stride is
                      safe; large = the prediction trembles and the stride/NMS must
                      allow for it. */}
                  <p className="scrubber__stability card__hint">
                    tiembla{" "}
                    <strong>±{data.stability.per_corner[c].toFixed(3)}</strong> al mover 1 px
                  </p>
                </div>
              ))}
            </div>
            <p className="card__hint">
              Ventana en ({data.x0}, {data.y0}) · bordes{" "}
              {data.border
                .map((f, b) => (f ? ["top", "right", "bottom", "left"][b] : null))
                .filter(Boolean)
                .join(", ") || "ninguno"}{" "}
              · máxima inestabilidad <strong>±{data.stability.max.toFixed(3)}</strong>
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
