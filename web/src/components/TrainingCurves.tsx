import * as Plot from "@observablehq/plot";
import { useMemo } from "react";
import type { EpochRecord } from "../api";
import { PlotFigure } from "./PlotFigure";
import { inkColors, polarityColors } from "../theme/palette";
import { useThemeVersion } from "../theme/useTheme";

/** V14 — the training curves, as **small multiples** (R4).
 *
 * **This is the rule's concrete case, not an illustration of it.**
 * `metrics.jsonl` puts `loss ≈ 0.28`, `f1 ≈ 0.77` and `pos_err_px ≈ 11` on the
 * same line: three different scales. Drawing them on one plot with two y-axes
 * would invent a correlation that is not in the data — the eye reads "these two
 * lines move together" straight off the pixels, and the pixels were put there by
 * the axis choice, not by the model.
 *
 * So: three panels, stacked, with the epoch axis aligned. Comparing across them
 * is then something the reader does on purpose, over a shared x, instead of
 * something the chart asserts for them.
 *
 * Colour is train vs val — two classes, so the diverging ends, never a corner
 * slot (see `polarityColors`). `pos_err_px` has one series because it is only
 * ever measured on val.
 */
export function TrainingCurves({ records }: { records: EpochRecord[] }) {
  const version = useThemeVersion();

  const panels = useMemo(() => {
    if (records.length === 0) return [];
    const polarity = polarityColors();
    const ink = inkColors();

    const loss = records.flatMap((r) => [
      { epoch: r.epoch, value: r.train_loss, serie: "train" },
      { epoch: r.epoch, value: r.val.loss, serie: "val" },
    ]);
    const f1 = records.map((r) => ({ epoch: r.epoch, value: r.val.f1 }));
    // `null` is the honest gap: a val split with no corners measures nothing, and
    // Plot leaves a break in the line rather than drawing through it. Coercing to
    // 0 would draw "perfect localisation" (formatos.md §2, on a chart).
    const err = records.map((r) => ({ epoch: r.epoch, value: r.val.pos_err_px }));

    // **Epochs are integers, so the ticks are the epochs themselves.** Left to
    // pick its own, Plot spaces ticks evenly over a continuous domain and
    // `tickFormat: "d"` rounds them for display -- so a 2-epoch run draws
    // `1 1 1 1 1 2 2 2 2 2`, ten ticks pretending to be two. Handing it the
    // actual values is what makes the axis mean epochs.
    const epochs = records.map((r) => r.epoch);
    const ticks = epochs.length <= 12 ? epochs : undefined;

    // One `base` for the three, and the shared `marginLeft` is what actually
    // aligns the epoch axis down the column (R4). Per-panel margins would drift
    // apart the moment one y label got longer than another.
    const base = {
      height: 130,
      marginLeft: 56,
      marginRight: 12,
      marginBottom: 26,
      x: { label: null, grid: true, tickFormat: "d", ticks, domain: [1, Math.max(2, records.length)] },
    };
    // The epoch label goes on the bottom panel alone: repeated three times down
    // an aligned column it is noise, and R6 asks for selective labelling.
    const bottom = { ...base, marginBottom: 32, x: { ...base.x, label: "época →" } };

    return [
      {
        key: "loss",
        aria: "pérdida por época, train contra val",
        options: {
          ...base,
          y: { label: "↑ loss", grid: true, zero: true },
          color: {
            legend: true,
            domain: ["train", "val"],
            range: [polarity.negative, polarity.positive],
          },
          marks: [
            Plot.lineY(loss, { x: "epoch", y: "value", stroke: "serie", strokeWidth: 2 }),
            Plot.dot(loss, { x: "epoch", y: "value", fill: "serie", r: 2 }),
          ],
        } satisfies Plot.PlotOptions,
      },
      {
        key: "f1",
        aria: "f1 de val por época",
        options: {
          ...base,
          y: { label: "↑ f1 (val)", domain: [0, 1], grid: true },
          marks: [
            Plot.lineY(f1, { x: "epoch", y: "value", stroke: polarity.positive, strokeWidth: 2 }),
            Plot.dot(f1, { x: "epoch", y: "value", fill: polarity.positive, r: 2 }),
          ],
        } satisfies Plot.PlotOptions,
      },
      {
        key: "pos_err_px",
        aria: "error de posición en píxeles, val, por época",
        options: {
          ...bottom,
          y: { label: "↑ pos_err_px (val)", grid: true, zero: true },
          marks: [
            // Plot breaks the line at a null y of its own accord, which is the
            // behaviour we want: the gap says "not measured", where a line drawn
            // through it would say "it kept going".
            Plot.lineY(err, { x: "epoch", y: "value", stroke: polarity.positive, strokeWidth: 2 }),
            Plot.dot(err, { x: "epoch", y: "value", fill: polarity.positive, r: 2 }),
            Plot.ruleY([0], { stroke: ink.grid }),
          ],
        } satisfies Plot.PlotOptions,
      },
    ];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [records, version]);

  if (panels.length === 0) return null;

  // No caption over each panel: the y axis already names the metric, and R6 asks
  // for selective labelling -- `loss` written twice, once above and once beside,
  // is the kind of decoration that makes three small charts look like six things.
  return (
    <div className="small-multiples">
      {panels.map((panel) => (
        <div className="small-multiples__panel" key={panel.key}>
          <PlotFigure options={panel.options} ariaLabel={panel.aria} />
        </div>
      ))}
    </div>
  );
}
