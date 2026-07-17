import * as Plot from "@observablehq/plot";
import { useMemo } from "react";
import type { SweepProgress, Trial } from "../../api";
import { Declares } from "../../components/Declares";
import { PlotFigure } from "../../components/PlotFigure";
import { sequentialRamp } from "../../theme/palette";
import { useThemeVersion } from "../../theme/useTheme";

/** V13 — parallel coordinates: every swept knob and the objective, one axis each.
 *
 * A trial is a line crossing all the axes. It answers the question a scatter
 * cannot when there are three or more swept fields: **which combinations of knobs
 * land where on the objective?** Each axis is normalised to its own [0,1], so
 * axes with different units share the frame; the raw range sits under each one.
 *
 * Colour is the objective, sequential, darker = better (whichever way "better"
 * runs -- `f1` up, `pos_err_px` down). A band of dark lines converging on one end
 * of a knob's axis is that knob mattering.
 */
export function Parallel({ progress }: { progress: SweepProgress }) {
  const version = useThemeVersion();

  const chart = useMemo(() => {
    const scored = progress.trials.filter(
      (t): t is Trial & { value: number } => t.value !== null
    );
    if (scored.length < 2) return null;

    const dims = [...Object.keys(progress.space), progress.objective];

    // Per-dim scale: numeric -> [min,max]; categorical -> ordered unique list.
    type Scale =
      | { kind: "num"; min: number; max: number }
      | { kind: "cat"; order: string[] };
    const rawOf = (t: Trial, dim: string): number | string =>
      dim === progress.objective ? (t.value as number) : t.params[dim];
    const scales: Record<string, Scale> = {};
    for (const dim of dims) {
      const raws = scored.map((t) => rawOf(t, dim));
      if (raws.every((v) => typeof v === "number")) {
        const nums = raws as number[];
        scales[dim] = { kind: "num", min: Math.min(...nums), max: Math.max(...nums) };
      } else {
        scales[dim] = { kind: "cat", order: [...new Set(raws.map(String))] };
      }
    }
    const norm = (dim: string, raw: number | string): number => {
      const s = scales[dim];
      if (s.kind === "num") return s.max > s.min ? ((raw as number) - s.min) / (s.max - s.min) : 0.5;
      return s.order.length > 1 ? s.order.indexOf(String(raw)) / (s.order.length - 1) : 0.5;
    };

    // Colour by the objective, darker = better.
    const values = scored.map((t) => t.value);
    const vmin = Math.min(...values);
    const vmax = Math.max(...values);
    const ramp = sequentialRamp();
    const colorOf = (t: Trial): string => {
      if (vmax <= vmin) return ramp.at(0.6);
      const t01 = ((t.value as number) - vmin) / (vmax - vmin);
      return ramp.at(progress.direction === "maximize" ? t01 : 1 - t01);
    };
    const trialColor: Record<number, string> = {};
    for (const t of scored) trialColor[t.number] = colorOf(t);

    const lines = scored.flatMap((t) =>
      dims.map((dim) => ({
        trial: t.number,
        dim,
        y: norm(dim, rawOf(t, dim)),
        color: trialColor[t.number],
      }))
    );

    // Raw min/max labels under each axis.
    const labels = dims.flatMap((dim) => {
      const s = scales[dim];
      const fmt = (v: number) => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01) ? v.toExponential(1) : v.toFixed(2));
      if (s.kind === "num") {
        return [
          { dim, y: 0, text: fmt(s.min) },
          { dim, y: 1, text: fmt(s.max) },
        ];
      }
      return [
        { dim, y: 0, text: s.order[0] },
        { dim, y: 1, text: s.order[s.order.length - 1] },
      ];
    });

    return {
      options: {
        height: 340,
        marginTop: 28,
        marginBottom: 56,
        marginLeft: 30,
        marginRight: 30,
        x: { domain: dims, label: null, tickRotate: -20 },
        y: { domain: [-0.08, 1.08], axis: null },
        marks: [
          Plot.ruleX(dims, { stroke: "currentColor", strokeOpacity: 0.25 }),
          Plot.line(lines, {
            x: "dim",
            y: "y",
            z: "trial",
            stroke: (d: { color: string }) => d.color,
            strokeWidth: 1.75,
            strokeOpacity: 0.75,
          }),
          // `dy` must be a constant, so the two ends are two marks.
          Plot.text(labels.filter((l) => l.y === 0), { x: "dim", y: "y", text: "text", dy: 12, fontSize: 10, fill: "currentColor", fillOpacity: 0.7 }),
          Plot.text(labels.filter((l) => l.y === 1), { x: "dim", y: "y", text: "text", dy: -12, fontSize: 10, fill: "currentColor", fillOpacity: 0.7 }),
        ],
      } satisfies Plot.PlotOptions,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [progress, version]);

  return (
    <section className="view">
      <Declares
        view="V13"
        title="Coordenadas paralelas"
        fixes="el dataset (B) y la red (C)"
        varies="cada campo del espacio (D)"
        measures={`el objetivo (${progress.objective})`}
      >
        Cada línea es un punto del barrido cruzando todos los ejes. Un haz de líneas oscuras
        convergiendo en un extremo de un eje es <strong>ese knob importando</strong>. Cada eje va
        normalizado a su propio rango (abajo y arriba, los valores reales).
      </Declares>

      {!chart && (
        <p className="async async--empty">
          Hacen falta al menos dos puntos con el objetivo medido para cruzar los ejes.
        </p>
      )}
      {chart && (
        <PlotFigure options={chart.options} ariaLabel="coordenadas paralelas de los puntos del barrido" />
      )}
    </section>
  );
}
