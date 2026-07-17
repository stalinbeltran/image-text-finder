import { useState } from "react";
import { getErrorMap, type ErrorMap, type Split } from "../../api";
import { ErrorNote, Loading } from "../../components/Async";
import { Declares } from "../../components/Declares";
import { MatrixCanvas } from "../../components/MatrixCanvas";
import { useAsync } from "../../useAsync";

/** V7 — where inside the patch the corner really was, and how far off we were.
 *
 * **The view that says which domain to fix**, and the most valuable in the
 * catalogue for this project's actual question (ui.md §4.1): if the error piles
 * up at the patch's edges — corners only half visible — the answer is to lower
 * B's `stride`, not to add filters to C. Without this map that diagnosis is
 * systematically misread as "the network is too small", and the fix goes into the
 * wrong domain.
 *
 * **The resolution is a control, and that is what made the view work.** ui.md
 * says 40×40; measured on real data, ~200 corners over 1600 cells is 0.1 samples
 * per cell and the map renders as speckle — true, and unreadable. At 10×10 the
 * edge-vs-centre structure is visible at a glance. The finer grids stay
 * reachable because the right resolution grows with the dataset.
 *
 * Empty cells are empty, never 0 (formatos.md §2 drawn in colour), and `counts`
 * is shown alongside because a cell built on 2 samples must not read like one
 * built on 200.
 */
const RESOLUTIONS = [5, 10, 20, 40];

export function ErrorByPosition({
  run,
  split,
  corner,
}: {
  run: string;
  split: Split;
  corner: string;
}) {
  const [bins, setBins] = useState(10);
  const map = useAsync(() => getErrorMap(run, split, corner, bins), [run, split, corner, bins]);
  const [showNumbers, setShowNumbers] = useState(false);
  const data = map.data;

  const values = data?.matrix.flat().filter((v): v is number => v !== null) ?? [];
  const worst = values.length ? Math.max(...values) : null;
  const best = values.length ? Math.min(...values) : null;
  const covered = data ? values.length / (data.bins * data.bins) : 0;
  const perCell = data && values.length ? data.samples / values.length : 0;

  return (
    <section className="view">
      <Declares
        view="V7"
        title="Error por posición dentro del patch"
        fixes={`el run y el split (${split})`}
        varies="la posición real de la esquina"
        measures="error en px"
      >
        Es la vista que dice <strong>qué dominio arreglar</strong>. Si el error se concentra en
        los bordes del patch —esquinas medio visibles— la respuesta es bajar el{" "}
        <code>stride</code> de B, no meter filtros en C.
      </Declares>

      <div className="row-actions">
        <label className="field field--inline">
          <span className="field__label">Resolución</span>
          <select value={bins} onChange={(e) => setBins(Number(e.target.value))}>
            {RESOLUTIONS.map((b) => (
              <option key={b} value={b}>
                {b}×{b}
              </option>
            ))}
          </select>
        </label>
        {/* The honest caveat, on screen and not in a doc: at 40×40 this data
            gives ~0.1 corners per cell, and speckle is not structure. */}
        <span className="card__hint">
          {data && (
            <>
              celdas de {data.cell_px.toFixed(0)}×{data.cell_px.toFixed(0)} px ·{" "}
              <strong>{perCell.toFixed(1)}</strong> esquinas por celda
              {perCell < 3 && " — demasiado pocas para leer nada: baja la resolución"}
            </>
          )}
        </span>
      </div>

      {map.loading && !data && <Loading what="el mapa de error" />}
      {map.error && <ErrorNote problem={map.error} />}

      {data && (
        <>
          <div className="view__facts">
            <span>
              <strong>{data.samples.toLocaleString("es")}</strong> esquinas reales
            </span>
            <span>
              {best !== null && worst !== null && (
                <>
                  error <strong>{best.toFixed(1)}</strong>–<strong>{worst.toFixed(1)}</strong> px
                </>
              )}
            </span>
            <span title="celdas donde cayó al menos una esquina real; el resto están vacías, no en cero">
              cobertura <strong>{(covered * 100).toFixed(0)} %</strong> de las celdas
            </span>
            <EdgeVsCentre map={data} />
          </div>

          <div className="error-map">
            <MatrixCanvas
              matrix={data.matrix}
              job={data.job}
              size={320}
              label={`error medio en px · ${data.corner} · ${data.bins}×${data.bins}`}
              onSelect={() => setShowNumbers((v) => !v)}
              selected={showNumbers}
            />
            <div className="error-map__legend">
              <p className="card__hint">
                Claro = poco error, oscuro = mucho. Las celdas <strong>vacías</strong> son sitios
                donde nunca cayó una esquina: no se sabe, que no es lo mismo que cero.
              </p>
              <button className="button button--quiet" onClick={() => setShowNumbers((v) => !v)}>
                {showNumbers ? "Ocultar" : "Ver"} los números
              </button>
            </div>
          </div>

          {/* R5: the map encodes with COLOUR ALONE, so the number table is the
              accessible twin, not a nicety — and it happens to be the best way to
              debug it. Both matrices, because the error means little without the
              count behind it. */}
          {showNumbers && <NumberGrid map={data} />}
        </>
      )}
    </section>
  );
}

/** The diagnosis V7 exists for, computed rather than left to the eye.
 *
 * The map shows it; this says it. If the outer ring is clearly worse than the
 * middle, the corners that hurt are the half-visible ones — and that is B's
 * `stride`, not C's capacity. Reported as two numbers with their n, never as a
 * verdict: it is evidence for a decision, not the decision.
 */
function EdgeVsCentre({ map }: { map: ErrorMap }) {
  const n = map.bins;
  if (n < 3) return null;
  const ring: number[] = [];
  const middle: number[] = [];
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const value = map.matrix[y][x];
      if (value === null) continue;
      const isEdge = x === 0 || y === 0 || x === n - 1 || y === n - 1;
      (isEdge ? ring : middle).push(value);
    }
  }
  if (!ring.length || !middle.length) return null;
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  return (
    <span title="si el borde es claramente peor que el centro, lo que falla son las esquinas medio visibles: eso se arregla en B (bajando el stride), no en C">
      borde <strong>{mean(ring).toFixed(1)}</strong> px vs centro{" "}
      <strong>{mean(middle).toFixed(1)}</strong> px
    </span>
  );
}

function NumberGrid({ map }: { map: ErrorMap }) {
  const axis = Array.from({ length: map.bins }, (_, i) => i);
  const px = (i: number) => Math.round(i * map.cell_px);
  return (
    <div className="table__scroll">
      <table className="table table--dense">
        <caption className="card__hint">
          Error medio en px (y nº de esquinas detrás de cada celda). Los ejes van en píxeles del
          patch. “—” = ninguna esquina cayó ahí.
        </caption>
        <thead>
          <tr>
            <th>y \ x</th>
            {axis.map((x) => (
              <th key={x} className="table__num">
                {px(x)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {axis.map((y) => (
            <tr key={y}>
              <th scope="row">{px(y)}</th>
              {axis.map((x) => {
                const value = map.matrix[y][x];
                const count = map.counts[y][x];
                return (
                  <td key={x} className="table__num">
                    {value === null ? (
                      <span className="table__absent" title="ninguna esquina cayó aquí">
                        —
                      </span>
                    ) : (
                      <>
                        {value.toFixed(1)}
                        <span className="card__hint"> ({count})</span>
                      </>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
