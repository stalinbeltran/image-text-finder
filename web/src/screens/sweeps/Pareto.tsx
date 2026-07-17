import * as Plot from "@observablehq/plot";
import { useMemo } from "react";
import type { SweepProgress, Trial } from "../../api";
import { Declares } from "../../components/Declares";
import { PlotFigure } from "../../components/PlotFigure";
import { sequentialRamp } from "../../theme/palette";
import { useThemeVersion } from "../../theme/useTheme";

/** V12 — the Pareto view: the two objectives in tension, coloured by λ.
 *
 * `f1` (detect) and `pos_err_px` (localise) pull in opposite directions, and
 * `lambda_pos` is what arbitrates between them (contract ⑨). Plotting them against
 * each other is what makes visible *what λ buys* -- which is why the view exists
 * even though the sweep now ranks by a single objective and no longer needs a
 * Pareto front to pick the winner (organizacion.md §2-⑨).
 *
 * **The good corner is up-and-left**: high f1, low pos_err_px. The non-dominated
 * points -- the ones nothing beats on both axes at once -- are joined into the
 * frontier. Colour is `lambda_pos`, sequential, so you can read the trade the
 * knob makes as you move along the front (ui.md §4.1: "secuencial por λ").
 */
export function Pareto({ progress }: { progress: SweepProgress }) {
  const version = useThemeVersion();

  const chart = useMemo(() => {
    // Only trials that produced both numbers can sit on this plane. A pruned
    // point that never measured pos_err_px is not plotted -- an invented 0 would
    // read as "localised perfectly", which is formatos.md §2 in a chart.
    const points = progress.trials.filter(
      (t): t is Trial & { f1: number; pos_err_px: number } =>
        t.f1 !== null && t.pos_err_px !== null
    );
    if (points.length === 0) return null;

    const swpLambda = "lambda_pos" in progress.space;
    const lambdas = points.map((t) => Number(t.params.lambda_pos ?? 0));
    const lo = Math.min(...lambdas);
    const hi = Math.max(...lambdas);
    const ramp = sequentialRamp();
    const fillOf = (t: Trial) =>
      swpLambda && hi > lo ? ramp.at((Number(t.params.lambda_pos ?? 0) - lo) / (hi - lo)) : ramp.at(0.6);

    // The frontier: sort by pos_err_px ascending, keep a point only if its f1
    // beats every kept point so far. That is the non-dominated set for
    // (min pos_err_px, max f1).
    const byErr = [...points].sort((a, b) => a.pos_err_px - b.pos_err_px);
    const front: typeof points = [];
    let bestF1 = -Infinity;
    for (const p of byErr) {
      if (p.f1 > bestF1) {
        front.push(p);
        bestF1 = p.f1;
      }
    }

    return {
      options: {
        height: 320,
        marginLeft: 56,
        marginBottom: 44,
        x: { label: "pos_err_px →  (menos es mejor)", grid: true, nice: true },
        y: { label: "↑ f1  (más es mejor)", grid: true, nice: true, domain: [0, 1] },
        marks: [
          Plot.line(front, {
            x: "pos_err_px",
            y: "f1",
            stroke: "currentColor",
            strokeOpacity: 0.35,
            strokeWidth: 1.5,
          }),
          Plot.dot(points, {
            x: "pos_err_px",
            y: "f1",
            r: 6,
            fill: fillOf,
            stroke: "currentColor",
            strokeOpacity: 0.5,
            title: (t: Trial) =>
              `${t.run ?? `#${t.number}`}\nf1 ${t.f1?.toFixed(3)} · pos_err ${t.pos_err_px?.toFixed(
                1
              )} px${swpLambda ? `\nλ ${Number(t.params.lambda_pos ?? 0).toFixed(2)}` : ""}`,
          }),
        ],
      } satisfies Plot.PlotOptions,
      swpLambda,
      lo,
      hi,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [progress, version]);

  return (
    <section className="view">
      <Declares
        view="V12"
        title="Frente de Pareto"
        fixes="el dataset (B) y la red (C)"
        varies="la receta (D) — el espacio del barrido"
        measures="f1 contra pos_err_px"
      >
        Las dos métricas <strong>tiran en direcciones opuestas</strong> —detectar vs. localizar— y{" "}
        <code>lambda_pos</code> es quien las arbitra. Esta vista no elige el ganador (eso lo hace el
        objetivo escalar): enseña <strong>qué compra λ</strong>. El buen rincón es{" "}
        <strong>arriba a la izquierda</strong>.
      </Declares>

      {!chart && (
        <p className="async async--empty">
          Todavía no hay ningún punto con las dos métricas medidas. Aparecerán según terminen los
          trials.
        </p>
      )}
      {chart && (
        <>
          <PlotFigure options={chart.options} ariaLabel="frente de Pareto: f1 contra error de posición" />
          {chart.swpLambda && (
            <p className="view__facts">
              <span className="card__hint">
                color = <code>lambda_pos</code>, de {chart.lo.toFixed(2)} (claro) a{" "}
                {chart.hi.toFixed(2)} (oscuro)
              </span>
            </p>
          )}
        </>
      )}
    </section>
  );
}
