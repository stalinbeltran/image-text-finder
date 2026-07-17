import { rampFor, type ColorJob } from "../theme/palette";

/** The matrix as numbers. R5: this is the "table-view twin".
 *
 * Not an extra: a heat map encodes with COLOUR ALONE, so every one of them owes
 * an accessible equivalent, and this is it (V1, V2, V7, V9). It comes from the
 * sibling project's click-a-map-see-the-numbers, where it turned out to also be
 * the most useful debugging view in the app -- but accessibility is why it is
 * mandatory rather than nice.
 *
 * Same shading scheme as MatrixCanvas, deliberately muted: the cell tint is
 * orientation, the number is the content, and the number has to stay legible.
 */
export interface NumberTableProps {
  matrix: number[][];
  job: ColorJob;
  digits?: number;
  caption?: string;
}

/** Matches MatrixCanvas: symmetric about 0 when diverging, own min/max when not. */
function normaliser(matrix: number[][], job: ColorJob): (v: number) => number {
  const flat = matrix.flat();
  if (job === "diverging") {
    const extent = Math.max(...flat.map(Math.abs)) || 1;
    return (v) => v / extent;
  }
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const range = max - min || 1;
  return (v) => (v - min) / range;
}

export function NumberTable({ matrix, job, digits = 3, caption }: NumberTableProps) {
  if (matrix.length === 0) return null;
  const ramp = rampFor(job);
  const norm = normaliser(matrix, job);

  return (
    <div className="number-table__scroll">
      <table className="number-table">
        {caption && <caption className="number-table__caption">{caption}</caption>}
        <tbody>
          {matrix.map((row, y) => (
            <tr key={y}>
              {row.map((v, x) => (
                <td
                  key={x}
                  className="number-table__cell"
                  // Tint the BACKGROUND only. `opacity` here would fade the digits
                  // too, which is exactly what this table exists to keep legible.
                  style={{
                    backgroundColor: `color-mix(in srgb, ${ramp.at(norm(v))} 35%, transparent)`,
                  }}
                >
                  <span className="number-table__value">{v.toFixed(digits)}</span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
