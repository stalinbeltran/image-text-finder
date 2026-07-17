import * as Plot from "@observablehq/plot";
import { useMemo } from "react";
import { getPrCurve, type Split } from "../../api";
import { ErrorNote, Loading } from "../../components/Async";
import { Declares } from "../../components/Declares";
import { PlotFigure } from "../../components/PlotFigure";
import { polarityColors } from "../../theme/palette";
import { useThemeVersion } from "../../theme/useTheme";
import { useAsync } from "../../useAsync";

/** V8 — score histogram + PR curve. **The free sweep** (ui.md §4.1).
 *
 * Every point on this curve is a threshold, and not one of them runs the model:
 * the scores were computed once into the table and re-thresholding filters a
 * column. That is why the plan puts this view before H — entering the sweep
 * without it means spending CPU hours searching in D for what was sitting in F
 * (ui.md §6).
 *
 * **Two charts, not one, and R4 is not a style rule here.** Precision and recall
 * share the 0–1 scale; the histogram's bucket counts are in the thousands.
 * Putting them on one plot with two y-axes would invent a correlation that is not
 * in the data.
 */
export function ScoresPr({
  run,
  split,
  corner,
  threshold,
  onThreshold,
}: {
  run: string;
  split: Split;
  corner: string;
  threshold: number;
  onThreshold: (value: number) => void;
}) {
  const pr = useAsync(() => getPrCurve(run, split, corner), [run, split, corner]);
  const version = useThemeVersion();
  const data = pr.data;

  const histogram = useMemo(() => {
    if (!data) return null;
    const { edges, positive, negative } = data.histogram;
    const polarity = polarityColors();
    // Two classes, and they may NOT borrow a corner slot: colour there belongs to
    // the corner's identity (R1), and the corner selector is on this very screen.
    // The diverging ramp's two ends are already in the fixed palette and were
    // validated against each other (see theme/palette.ts).
    // **Side by side inside each bin, not on top of each other.** Two series of
    // rects sharing one x-range overlap exactly, so whichever draws last hides
    // the other -- and the chart still looks like a chart. The question here is
    // "are the two distributions separable?", which cannot be read off a bar you
    // cannot see. So each class gets half the bin.
    const bars = edges.slice(0, -1).flatMap((low, i) => {
      const mid = (low + edges[i + 1]) / 2;
      return [
        { low, high: mid, count: negative[i], clase: "sin esquina" },
        { low: mid, high: edges[i + 1], count: positive[i], clase: "con esquina" },
      ];
    });
    return {
      bars,
      options: {
        height: 200,
        marginLeft: 56,
        x: { label: "score p(exists) →", domain: [0, 1], grid: true },
        // **Linear, and a log scale here is a bug twice over.** A `rectY` spans
        // from an implicit y=0, and log(0) is undefined -- so Plot silently drops
        // every bar and leaves a chart with axes and no data, which looks like a
        // chart. And it was not needed anyway: the imbalance is 3.9:1, "modesto,
        // no brutal" (organizacion.md §1-D), so both classes read fine linear.
        y: { label: "↑ patches", grid: true },
        color: {
          legend: true,
          domain: ["sin esquina", "con esquina"],
          range: [polarity.negative, polarity.positive],
        },
        marks: [
          Plot.rectY(bars, { x1: "low", x2: "high", y: "count", fill: "clase", inset: 0.25 }),
          Plot.ruleY([0]),
          Plot.ruleX([threshold], { strokeWidth: 2, strokeDasharray: "4 3" }),
        ],
      } satisfies Plot.PlotOptions,
    };
    // `version` is in here so the colours are re-read when the mode flips.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, threshold, version]);

  const curve = useMemo(() => {
    if (!data) return null;
    const polarity = polarityColors();
    const points = data.curve.flatMap((p) => [
      { threshold: p.threshold, value: p.precision, metrica: "precision" },
      { threshold: p.threshold, value: p.recall, metrica: "recall" },
    ]);
    return {
      options: {
        height: 200,
        marginLeft: 56,
        x: { label: "threshold →", domain: [0, 1], grid: true },
        // One shared 0–1 scale, so these two DO belong on one chart. That is the
        // same rule that keeps the histogram off it.
        y: { label: "↑ precision / recall", domain: [0, 1], grid: true },
        color: {
          legend: true,
          domain: ["precision", "recall"],
          range: [polarity.negative, polarity.positive],
        },
        marks: [
          Plot.lineY(points, { x: "threshold", y: "value", stroke: "metrica", strokeWidth: 2 }),
          Plot.ruleX([threshold], { strokeWidth: 2, strokeDasharray: "4 3" }),
        ],
      } satisfies Plot.PlotOptions,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, threshold, version]);

  const at = data?.curve.reduce((best, p) =>
    Math.abs(p.threshold - threshold) < Math.abs(best.threshold - threshold) ? p : best
  );

  return (
    <section className="view">
      <Declares
        view="V8"
        title="Scores y curva PR"
        fixes={`el run y el split (${split})`}
        varies="el threshold"
        measures="precision y recall"
      >
        Es el <strong>barrido gratis</strong>: los scores están guardados, así que mover el
        umbral no vuelve a correr el modelo. Ajustarlo aquí cuesta segundos; buscarlo
        reentrenando cuesta horas de CPU — y <code>threshold</code> es F, no D.
      </Declares>

      {pr.loading && !data && <Loading what="la curva PR" />}
      {pr.error && <ErrorNote problem={pr.error} />}

      {data && (
        <>
          <div className="view__facts">
            <span>
              <strong>{data.positives.toLocaleString("es")}</strong> con esquina ·{" "}
              <strong>{data.negatives.toLocaleString("es")}</strong> sin
            </span>
            {data.positive_rate !== null && (
              <span
                title="con este desbalance, acertar siempre «no hay esquina» ya da una accuracy alta: por eso se mira la PR y no la accuracy"
              >
                desbalance <strong>{(data.positive_rate * 100).toFixed(1)} %</strong> positivos (
                {(((1 - data.positive_rate) / data.positive_rate) || 0).toFixed(1)}:1)
              </span>
            )}
          </div>

          <div className="view__charts">
            <PlotFigure
              options={histogram!.options}
              ariaLabel="histograma de scores, positivos contra negativos"
            />
            <PlotFigure options={curve!.options} ariaLabel="curva de precision y recall" />
          </div>

          <div className="threshold">
            <label className="threshold__control">
              <span>
                threshold <code>{threshold.toFixed(2)}</code>
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={threshold}
                onChange={(e) => onThreshold(Number(e.target.value))}
              />
            </label>
            {at && (
              <p className="threshold__readout">
                precision <strong>{at.precision.toFixed(3)}</strong> · recall{" "}
                <strong>{at.recall.toFixed(3)}</strong> · f1 <strong>{at.f1.toFixed(3)}</strong>
              </p>
            )}
            {/* Reported, never applied on its own: choosing the threshold is a
                decision, and a view that moved it for you would be making it. */}
            {data.best && (
              <button
                className="button button--quiet"
                onClick={() => onThreshold(data.best!.threshold)}
                title="el umbral que maximiza la f1 sobre este split, elegido post-hoc y gratis"
              >
                Mejor f1: {data.best.f1.toFixed(3)} en {data.best.threshold.toFixed(2)}
              </button>
            )}
          </div>
        </>
      )}
    </section>
  );
}
