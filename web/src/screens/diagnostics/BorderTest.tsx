import * as Plot from "@observablehq/plot";
import { useMemo } from "react";
import { getBorderTest } from "../../api";
import { ErrorNote, Loading } from "../../components/Async";
import { Declares } from "../../components/Declares";
import { PlotFigure } from "../../components/PlotFigure";
import { rampFor } from "../../theme/palette";
import { useThemeVersion } from "../../theme/useTheme";
import { useAsync } from "../../useAsync";

/** V10 — the border-flag test. **Fix E and the patch, flip each of the 4 flags.**
 *
 * The 4 border flags say whether the patch sits flush against the source image's
 * edge — real information, because a paragraph corner cannot sit just outside the
 * image, so a patch against the top carries a different prior than one in the
 * middle (formatos.md §2). This probe measures how much the network actually
 * *uses* it: baseline with the real flags, then one forward per flag flipped, and
 * a dumbbell draws each head's score moving before→después.
 *
 * **Only meaningful for a network with `border_features`.** A network without it
 * ignores the flags, so the backend refuses (409 `border_not_used`) rather than
 * drawing four flat dumbbells that would read as "the border does not matter to
 * this patch" — a claim about the data when the truth is the architecture never
 * looks at it. That refusal surfaces here as the error's own hint.
 *
 * **One tint, two tones** (ui.md §4.1): the colour encodes *phase* (real flag vs
 * flipped), not the corner — the corner is the row label. So it uses two tones of
 * the sequential ramp, never a corner slot (R1).
 */
export function BorderTest({
  run,
  patchDataset,
  patchIdx,
}: {
  run: string;
  patchDataset: string;
  patchIdx: number;
}) {
  const data = useAsync(
    () => getBorderTest(run, patchDataset, patchIdx),
    [run, patchDataset, patchIdx]
  );
  const version = useThemeVersion();
  const bt = data.data;

  const chart = useMemo(() => {
    if (!bt) return null;
    const ramp = rampFor("sequential");
    // Two tones of one tint: light for the real flag, dark for the flipped one.
    const toneReal = ramp.at(0.32);
    const toneFlipped = ramp.at(0.82);
    const PHASE_REAL = "flag real";
    const PHASE_FLIP = "volteado";

    // Wide rows carry the link (before→after); long rows carry the two dots so a
    // colour legend can name the phases.
    const links = bt.flips.flatMap((flip) =>
      bt.corner_order.map((corner, c) => ({
        flip: `${flip.border}: ${flip.flag_from}→${flip.flag_to}`,
        corner,
        real: bt.baseline[c],
        flipped: flip.scores[c],
      }))
    );
    const dots = links.flatMap((r) => [
      { flip: r.flip, corner: r.corner, score: r.real, phase: PHASE_REAL },
      { flip: r.flip, corner: r.corner, score: r.flipped, phase: PHASE_FLIP },
    ]);

    return {
      options: {
        // Four facets stacked (one per flag), corners aligned across them.
        height: 60 * bt.flips.length + 60,
        marginLeft: 40,
        marginRight: 12,
        x: { label: "p(esquina) →", domain: [0, 1], grid: true },
        y: { label: null, domain: bt.corner_order },
        fy: { label: null },
        color: {
          legend: true,
          domain: [PHASE_REAL, PHASE_FLIP],
          range: [toneReal, toneFlipped],
        },
        marks: [
          Plot.link(links, {
            x1: "real",
            x2: "flipped",
            y: "corner",
            fy: "flip",
            stroke: "currentColor",
            strokeOpacity: 0.35,
            strokeWidth: 2,
          }),
          Plot.dot(dots, {
            x: "score",
            y: "corner",
            fy: "flip",
            fill: "phase",
            r: 4.5,
          }),
        ],
      } satisfies Plot.PlotOptions,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bt, version]);

  // The headline: the biggest move any flag caused. Near 0 = this patch does not
  // lean on the border at all; large = flipping a flag changed the answer.
  const biggest = bt
    ? Math.max(
        ...bt.flips.flatMap((f) => f.scores.map((s, c) => Math.abs(s - bt.baseline[c])))
      )
    : 0;

  return (
    <section className="view">
      <Declares
        view="V10"
        title="Test del flag de borde"
        fixes={`el run y el patch #${patchIdx}`}
        varies="los 4 flags de borde"
        measures="cuánto se mueve la predicción"
      >
        Los 4 flags dicen si el patch está pegado a un borde de la imagen. Esta sonda voltea cada
        uno y mide cuánto lo usa la red: el <strong>dumbbell</strong> va del flag real al volteado.
      </Declares>

      {data.loading && !bt && <Loading what="el test de borde" />}
      {data.error && <ErrorNote problem={data.error} />}

      {bt && chart && (
        <>
          <p className="card__hint">
            Flags reales de este patch:{" "}
            {bt.border_order.map((name, b) => (
              <span key={name}>
                {b > 0 && " · "}
                <strong>{name}</strong> {bt.border[b]}
              </span>
            ))}{" "}
            · mayor cambio al voltear un flag: <strong>{biggest.toFixed(3)}</strong>{" "}
            {biggest < 0.02 ? "(este patch casi no se apoya en el borde)" : ""}
          </p>
          <PlotFigure
            options={chart.options}
            ariaLabel="dumbbell del cambio en la predicción al voltear cada flag de borde"
          />
        </>
      )}
    </section>
  );
}
