import { useEffect, useRef } from "react";
import { rampFor, type ColorJob } from "../theme/palette";

/** A matrix as a heat map. `drawMap` from the sibling project, with two changes.
 *
 * Ported as-is: a canvas with one fillRect per cell, `image-rendering: pixelated`
 * so a 3x3 kernel reads as 3 crisp cells and not a blurred smear, and
 * NORMALISATION PER MAP -- each map against its own min/max. Without that, the
 * quiet feature maps all render black (ui.md §5).
 *
 * Changed, and neither is cosmetic:
 *
 *  1. `job` is required and comes from the payload, never guessed. The client
 *     cannot know whether it is looking at a signed weight or a non-negative
 *     activation, so the server declares it (api.md §3). Guessing is how kernels
 *     end up on a sequential ramp.
 *
 *  2. For `diverging`, the range is SYMMETRIC (±max|v|) and 0 is pinned to the
 *     neutral (R2). The sibling normalised min->max, which puts zero wherever it
 *     falls and hides the sign structure. Passing min/max in would reintroduce
 *     exactly that bug, so this component derives the range itself.
 */
export interface MatrixCanvasProps {
  matrix: number[][];
  job: ColorJob;
  /** Target size in px of the longest side. The sibling used 72. */
  size?: number;
  label?: string;
  onSelect?: () => void;
  selected?: boolean;
}

export function MatrixCanvas({
  matrix,
  job,
  size = 72,
  label,
  onSelect,
  selected = false,
}: MatrixCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || matrix.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const height = matrix.length;
    const width = matrix[0].length;
    const cell = Math.max(2, Math.round(size / Math.max(height, width)));
    canvas.width = width * cell;
    canvas.height = height * cell;

    // Resolved once per repaint, not once per cell: a 40x40 map is 1600 cells,
    // and V5 repaints on every mouse move.
    const ramp = rampFor(job, canvas);

    const flat = matrix.flat();
    let norm: (v: number) => number;
    if (job === "diverging") {
      // Symmetric about 0, so the neutral always lands on 0. `|| 1` keeps an
      // all-zero kernel from dividing by zero.
      const extent = Math.max(...flat.map(Math.abs)) || 1;
      norm = (v) => v / extent;
    } else {
      const min = Math.min(...flat);
      const max = Math.max(...flat);
      const range = max - min || 1; // a flat map is legal, and must not divide by zero
      norm = (v) => (v - min) / range;
    }

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        ctx.fillStyle = ramp.at(norm(matrix[y][x]));
        ctx.fillRect(x * cell, y * cell, cell, cell);
      }
    }
  }, [matrix, job, size]);

  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  const description = label
    ? `${label}, matriz ${rows}×${cols}`
    : `matriz ${rows}×${cols}`;

  // R5: the map encodes with colour ALONE, so it is never the only way to read
  // the numbers. Clicking opens the number table, which is the accessible twin
  // and not a nicety. onSelect is what wires it up.
  const interactive = Boolean(onSelect);

  return (
    <figure className={`matrix-canvas ${selected ? "is-selected" : ""}`}>
      <canvas
        ref={ref}
        className="matrix-canvas__canvas"
        role="img"
        aria-label={description}
        onClick={onSelect}
        tabIndex={interactive ? 0 : undefined}
        onKeyDown={(e) => {
          if (interactive && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            onSelect?.();
          }
        }}
        style={{ cursor: interactive ? "pointer" : undefined }}
      />
      {label && <figcaption className="matrix-canvas__label">{label}</figcaption>}
    </figure>
  );
}
