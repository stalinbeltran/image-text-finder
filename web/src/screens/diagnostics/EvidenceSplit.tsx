import { getEvidenceSplit, type EvidenceBand, type EvidencePopulation, type Split } from "../../api";
import { ErrorNote, Loading } from "../../components/Async";
import { Declares } from "../../components/Declares";
import { useAsync } from "../../useAsync";

/** V18 — the same numbers, split by how much there was to look at.
 *
 * **The question**: when the label says "there is a TL here", is that corner's
 * paragraph actually inside the patch? A TL whose point lands near the patch's
 * bottom-right has its whole paragraph *outside* the window — the label asks for
 * something the pixels do not show. It is not a rare case: it is the structural
 * price of a sliding window, and measured on `dirty-20` it is 14 % of all corners.
 *
 * **What it found, and what it did not.** Those 14 % carry **31 % of the position
 * error** (5,3 px vs 2,0 px on a 20 px patch). But detection on them *generalises*
 * — mean score 0,623 / 0,621 / 0,619 on train / val / test, a gap of 0,002 — so
 * the model is reading real context, not memorising an impossible label. The
 * deficit is in *position*, not in detection, which is why this view reports the
 * two side by side instead of folding them into one number. Reading it as "the
 * model fails here" is the misreading it exists to prevent.
 *
 * **No chart, on purpose.** The finding is two shares over the same six bands, and
 * two `rect` series over one x range do not group in Plot — they cover each other
 * (the fase 5 lesson). Two bars per row in a table says it without the trap, and
 * the numbers are right there rather than in a tooltip.
 */
export function EvidenceSplit({
  run,
  split,
  corner,
  threshold,
}: {
  run: string;
  split: Split;
  corner: string;
  threshold: number;
}) {
  const data = useAsync(
    () => getEvidenceSplit(run, split, corner === "all" ? undefined : corner, threshold),
    [run, split, corner, threshold]
  );
  const ev = data.data;

  return (
    <section className="view">
      <Declares
        view="V18"
        title="Detección y error por evidencia disponible"
        fixes={`el run, el split (${split}) y el umbral`}
        varies="cuánto del párrafo cabe en el patch"
        measures="recall y error de posición"
      >
        Una esquina cuyo punto cae en el borde <em>lejano</em> del patch tiene su párrafo{" "}
        <strong>fuera</strong> de la ventana: la etiqueta pide algo que los píxeles no enseñan. La
        medida es geométrica — no mira el modelo — y es una <strong>cota superior</strong>: lo que
        marca como ciego lo está.
      </Declares>

      {data.loading && !ev && <Loading what="el desglose por evidencia" />}
      {data.error && <ErrorNote problem={data.error} />}

      {ev && (
        <>
          <div className="evidence__headline">
            <Population
              label="Ciegas"
              hint={`evidencia < ${ev.blind_cut}`}
              population={ev.blind}
              patchSize={ev.patch_size}
              accent
            />
            <Population
              label="Visibles"
              hint={`evidencia ≥ ${ev.blind_cut}`}
              population={ev.seen}
              patchSize={ev.patch_size}
            />
          </div>

          <p className="card__hint">
            Lo que hay que leer es <strong>la población contra el error</strong>: si una banda se
            lleva la misma fracción de las dos, no tiene nada de particular.
          </p>

          <div className="table__scroll">
            <table className="table table--dense">
              <caption className="card__hint">
                {ev.corners.toLocaleString("es")} esquinas reales
                {ev.corner !== "all" && <> de tipo <strong>{ev.corner}</strong></>}, patch de{" "}
                {ev.patch_size} px. <strong>err</strong> es error de posición en píxeles;{" "}
                <strong>recall</strong> es a umbral {ev.threshold}.
              </caption>
              <thead>
                <tr>
                  <th>evidencia</th>
                  <th className="table__num">esquinas</th>
                  <th>población vs. error</th>
                  <th className="table__num">err px</th>
                  <th className="table__num">recall</th>
                </tr>
              </thead>
              <tbody>
                {ev.bands.map((band) => (
                  <Band key={band.low} band={band} blindCut={ev.blind_cut} />
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

/** One band. The two bars share a scale so "carries more error than population"
 *  is a length you can see rather than two percentages you have to subtract. */
function Band({ band, blindCut }: { band: EvidenceBand; blindCut: number }) {
  const blind = band.high <= blindCut;
  return (
    <tr className={blind ? "evidence__row--blind" : undefined}>
      <th scope="row">
        [{band.low.toFixed(2)}, {band.high.toFixed(2)})
        {blind && <span className="card__hint"> ciega</span>}
      </th>
      <td className="table__num">{band.corners.toLocaleString("es")}</td>
      <td>
        <Bar value={band.corner_share} kind="population" title="fracción de las esquinas" />
        <Bar value={band.error_share} kind="error" title="fracción del error total" />
      </td>
      <td className="table__num">{band.err_px?.toFixed(2) ?? <Absent />}</td>
      <td className="table__num">
        {band.recall === null ? <Absent /> : `${(band.recall * 100).toFixed(1)} %`}
      </td>
    </tr>
  );
}

function Bar({
  value,
  kind,
  title,
}: {
  value: number | null;
  kind: "population" | "error";
  title: string;
}) {
  if (value === null) return <div className="evidence__bar-row"><Absent /></div>;
  return (
    <div className="evidence__bar-row" title={title}>
      <span className="evidence__bar-label">{kind === "population" ? "pob" : "err"}</span>
      <span className="evidence__track">
        <span
          className={`evidence__fill evidence__fill--${kind}`}
          style={{ width: `${Math.min(value, 1) * 100}%` }}
        />
      </span>
      <span className="evidence__bar-value">{(value * 100).toFixed(1)} %</span>
    </div>
  );
}

/** The headline pair. `error_share` next to `corner_share` IS the finding. */
function Population({
  label,
  hint,
  population,
  patchSize,
  accent,
}: {
  label: string;
  hint: string;
  population: EvidencePopulation;
  patchSize: number;
  accent?: boolean;
}) {
  return (
    <div className={`evidence__card${accent ? " evidence__card--accent" : ""}`}>
      <h3 className="evidence__card-title">
        {label} <span className="card__hint">{hint}</span>
      </h3>
      <dl className="evidence__stats">
        <div>
          <dt>de las esquinas</dt>
          <dd>{(population.corner_share * 100).toFixed(1)} %</dd>
        </div>
        <div>
          <dt>del error total</dt>
          <dd>
            {population.error_share === null ? (
              <Absent />
            ) : (
              `${(population.error_share * 100).toFixed(1)} %`
            )}
          </dd>
        </div>
        <div>
          <dt>error de posición</dt>
          <dd>
            {population.err_px === null ? (
              <Absent />
            ) : (
              <>
                {population.err_px.toFixed(2)} px{" "}
                <span className="card__hint">
                  ({((population.err_px / patchSize) * 100).toFixed(0)} % del patch)
                </span>
              </>
            )}
          </dd>
        </div>
        <div>
          <dt>recall</dt>
          <dd>
            {population.recall === null ? <Absent /> : `${(population.recall * 100).toFixed(1)} %`}
          </dd>
        </div>
      </dl>
    </div>
  );
}

/** Never a 0: an empty population was not measured (formatos.md §2). */
const Absent = () => <span className="table__absent">—</span>;
