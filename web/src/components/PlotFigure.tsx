import * as Plot from "@observablehq/plot";
import { useEffect, useMemo, useRef } from "react";
import { inkColors } from "../theme/palette";
import { useThemeVersion } from "../theme/useTheme";

/** Observable Plot, mounted the standard way: `useEffect` + `ref` + `replaceChildren`.
 *
 * Plot enters in fase 5 and pays its weight where drawing by hand would not
 * (ui.md §0): a histogram or a PR curve with axes, scales and a legend is 3–5
 * lines. What it does NOT get is the matrices -- V1, V2, V4 and V7 stay on the
 * hand-rolled canvas, because `drawMap` is ~15 lines and gives finer control over
 * per-map normalisation than a charting library would.
 *
 * **The deuda it creates is real and named**: `LineChart.tsx` in the tag drew
 * charts by hand, so entering Plot means two ways of drawing. The direction is to
 * migrate toward Plot when a chart is touched, not to keep both -- V14 is built
 * with Plot here for exactly that reason.
 *
 * Two things this wrapper adds over calling `Plot.plot` directly, both of which
 * would otherwise be re-solved in every view:
 *
 *  1. **The ink comes from tokens.css.** Plot ships its own greys, and they go
 *     invisible on the dark surface. The palette lives in one file (D12) and this
 *     is how an SVG built in JS obeys it.
 *  2. **It repaints when the mode flips.** An SVG already in the DOM does not
 *     restyle itself the way CSS does, so without `useThemeVersion` a chart keeps
 *     the palette of the mode it was born in.
 */
export interface PlotFigureProps {
  /** Everything but `style` and `marks` defaults; `options.marks` is required. */
  options: Plot.PlotOptions;
  /** The accessible name. A chart is an image: it needs one. */
  ariaLabel: string;
}

export function PlotFigure({ options, ariaLabel }: PlotFigureProps) {
  const ref = useRef<HTMLDivElement>(null);
  const version = useThemeVersion();

  // The options object is rebuilt on every render by the caller, so depending on
  // it directly would rebuild the chart on every render. The caller memoises what
  // matters; here we key on the theme and the options identity together.
  const memo = useMemo(() => options, [options]);

  useEffect(() => {
    const host = ref.current;
    if (!host) return;
    const ink = inkColors(host);
    const chart = Plot.plot({
      ...memo,
      style: {
        background: "transparent",
        color: ink.secondary,
        fontSize: "12px",
        // Plot's own font stack does not match the app's; inheriting keeps the
        // numbers on a chart looking like the numbers in the table next to it.
        fontFamily: "inherit",
        overflow: "visible",
        ...(memo.style as object),
      },
    });
    chart.setAttribute("aria-label", ariaLabel);
    host.replaceChildren(chart);
    return () => chart.remove();
  }, [memo, version, ariaLabel]);

  return <div className="plot-figure" ref={ref} />;
}
