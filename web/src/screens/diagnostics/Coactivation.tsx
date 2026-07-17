import { useState } from "react";
import { getCoactivation, type Coactivation as Data, type Split } from "../../api";
import { ErrorNote, Loading } from "../../components/Async";
import { Declares } from "../../components/Declares";
import { MatrixCanvas } from "../../components/MatrixCanvas";
import { useAsync } from "../../useAsync";

/** V9 — given the truth was TL, which heads fired?
 *
 * **Not a confusion matrix, and the difference is the view** (ui.md §4.1). The
 * four heads are independent binaries, not a softmax: a patch can fire two at
 * once, or none, so the rows sum to nothing in particular. What is tabulated is
 * `P(head j fires | corner i really there)` — the diagonal is recall, and the
 * off-diagonal is what no other view in the catalogue shows. **TL↔TR is the
 * textbook failure here** (they are mirror images), and this is the only place it
 * would be visible.
 *
 * **The baseline is what makes it honest.** A high `matrix[TL][TR]` has two
 * opposite explanations — the TR head is confused by a TL, *or* these patches
 * genuinely contain a real TR too (two paragraphs, or one narrow enough to fit
 * both corners in 40 px). The matrix alone cannot tell them apart, and reading a
 * co-occurrence as confusion would send the fix into the wrong domain. So the
 * truth's own co-occurrence rate is shown next to it: **where the two agree the
 * head is right; where the fired rate runs above the truth rate, that is the
 * confusion.**
 *
 * It is the same family as V7's speckle — something true, presented so that the
 * obvious reading of it is false.
 */
export function Coactivation({
  run,
  split,
  threshold,
}: {
  run: string;
  split: Split;
  threshold: number;
}) {
  const data = useAsync(() => getCoactivation(run, split, threshold), [run, split, threshold]);
  const [showNumbers, setShowNumbers] = useState(false);
  const co = data.data;

  return (
    <section className="view">
      <Declares
        view="V9"
        title="Co-activación de tipos"
        fixes={`el run y el split (${split})`}
        varies="el tipo de esquina que hay de verdad"
        measures="qué cabezas disparan"
      >
        <strong>No es una matriz de confusión</strong>: las 4 cabezas son binarias e
        independientes, así que un patch puede disparar dos a la vez o ninguna. La confusión{" "}
        <strong>TL↔TR</strong> es el fallo de manual aquí, y esta es la única vista que lo enseña.
      </Declares>

      {data.loading && !co && <Loading what="la co-activación" />}
      {data.error && <ErrorNote problem={data.error} />}

      {co && (
        <>
          <div className="coactivation">
            <figure className="coactivation__panel">
              <MatrixCanvas
                matrix={co.matrix}
                job={co.job}
                size={200}
                label="disparó la cabeza (fila = la esquina real)"
              />
              <figcaption className="card__hint">
                <strong>Lo que dispara</strong>: P(cabeza j | esquina i de verdad)
              </figcaption>
            </figure>

            {/* The control, side by side and at the same scale. Buried in a
                tooltip it would not be read, and unread it is not a control. */}
            <figure className="coactivation__panel">
              <MatrixCanvas
                matrix={co.truth_rate}
                job={co.job}
                size={200}
                label="co-ocurrencia real (fila = la esquina real)"
              />
              <figcaption className="card__hint">
                <strong>La verdad</strong>: P(esquina j de verdad | esquina i de verdad)
              </figcaption>
            </figure>
          </div>

          <p className="card__hint">
            Se leen <strong>una contra otra</strong>. Donde coinciden, la cabeza acierta y las
            esquinas simplemente conviven en el patch. Donde la izquierda va{" "}
            <strong>por encima</strong> de la derecha, eso es confusión.{" "}
            <button className="button button--quiet" onClick={() => setShowNumbers((v) => !v)}>
              {showNumbers ? "Ocultar" : "Ver"} los números
            </button>
          </p>

          {/* R5: two heat maps encoding with colour alone owe a number table. */}
          {showNumbers && <Numbers co={co} />}
        </>
      )}
    </section>
  );
}

/** The difference, computed. The two maps show it; this says it.
 *
 * `counts` rides along because a row built on 3 patches paints exactly like one
 * built on 300 — the lesson V7's speckle taught, applied before it bites.
 */
function Numbers({ co }: { co: Data }) {
  return (
    <div className="table__scroll">
      <table className="table table--dense">
        <caption className="card__hint">
          Disparó · (de verdad) · <strong>exceso</strong>. El exceso es lo que la cabeza dispara de
          más sobre lo que hay: es la confusión. “—” = ningún patch tiene esa esquina.
        </caption>
        <thead>
          <tr>
            <th>real \ disparó</th>
            {co.corner_order.map((c) => (
              <th key={c} className="table__num">
                {c}
              </th>
            ))}
            <th className="table__num" title="patches con esa esquina de verdad: una fila de 3 se pinta igual que una de 300">
              n
            </th>
          </tr>
        </thead>
        <tbody>
          {co.corner_order.map((row, i) => (
            <tr key={row}>
              <th scope="row">{row}</th>
              {co.corner_order.map((_, j) => {
                const fired = co.matrix[i][j];
                const truth = co.truth_rate[i][j];
                if (fired === null || truth === null) {
                  return (
                    <td key={j} className="table__num">
                      <span className="table__absent">—</span>
                    </td>
                  );
                }
                const excess = fired - truth;
                return (
                  <td key={j} className="table__num">
                    {fired.toFixed(2)}
                    <span className="card__hint"> ({truth.toFixed(2)})</span>
                    {i !== j && excess > 0.05 && (
                      <>
                        {" "}
                        <strong title="la cabeza dispara más de lo que la verdad justifica: eso es confusión">
                          +{excess.toFixed(2)}
                        </strong>
                      </>
                    )}
                  </td>
                );
              })}
              <td className="table__num">{co.counts[i].toLocaleString("es")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
