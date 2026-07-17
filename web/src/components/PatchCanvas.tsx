import { useEffect, useRef } from "react";
import { cornerColors, type CornerName } from "../theme/palette";
import { useThemeVersion } from "../theme/useTheme";

/** A patch, with where the corners really were and where the model said they were.
 *
 * **The patch is the real input of the CNN** (contract ①), which is why this --
 * and never a whole image -- is what the diagnóstico looks at. The whole image
 * belongs to F.
 *
 * The overlay carries two facts per corner and encodes them differently on
 * purpose, because colour is already spoken for: **colour is the corner's
 * identity** (R1 -- TL is that blue in the meters, here, and in V11), so it
 * cannot also mean truth-vs-prediction. So:
 *
 *  - the truth is a ring, the prediction is a filled dot, and a line joins them.
 *    That line IS the error, at the scale the error is actually made.
 *  - the corner is named next to its mark. R1 makes direct labelling mandatory
 *    for these four rather than a courtesy: in dark mode the worst pair sits at
 *    ΔE 6.9, which is only legal with a secondary encoding.
 *
 * `image-rendering: pixelated` and `imageSmoothingEnabled = false` matter here
 * beyond taste: the extractor reads the source images ALREADY pixelated and cuts
 * them as they are (ui.md §5). Smoothing a 40×40 patch up to 240 px would show
 * a blur the model never saw.
 */
export interface PatchOverlay {
  corner: CornerName;
  /** Normalised in [0,1] within the patch. Absent when there is no real corner. */
  truth?: [number, number] | null;
  pred?: [number, number] | null;
}

export interface PatchCanvasProps {
  /** (n, n), 0–255 greyscale, exactly as it sits in B's `.npz`. */
  patch: number[][];
  size?: number;
  overlay?: PatchOverlay[];
  label?: string;
}

export function PatchCanvas({ patch, size = 240, overlay = [], label }: PatchCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const version = useThemeVersion();

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || patch.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const n = patch.length;
    const scale = Math.max(1, Math.floor(size / n));
    const side = n * scale;
    canvas.width = side;
    canvas.height = side;

    // Painted at 1:1 into an offscreen buffer and blown up with smoothing OFF:
    // one source pixel becomes one crisp block, which is what the model sees.
    const buffer = document.createElement("canvas");
    buffer.width = n;
    buffer.height = n;
    const bufferCtx = buffer.getContext("2d");
    if (!bufferCtx) return;
    const image = bufferCtx.createImageData(n, n);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const v = patch[y][x];
        const i = (y * n + x) * 4;
        image.data[i] = image.data[i + 1] = image.data[i + 2] = v;
        image.data[i + 3] = 255;
      }
    }
    bufferCtx.putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(buffer, 0, 0, side, side);

    const colors = cornerColors(canvas);
    for (const mark of overlay) {
      const color = colors[mark.corner];
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;

      const at = (p: [number, number]): [number, number] => [p[0] * side, p[1] * side];

      if (mark.truth && mark.pred) {
        // The error, drawn at the scale it is made at.
        const [tx, ty] = at(mark.truth);
        const [px, py] = at(mark.pred);
        ctx.beginPath();
        ctx.setLineDash([3, 3]);
        ctx.moveTo(tx, ty);
        ctx.lineTo(px, py);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      if (mark.truth) {
        const [x, y] = at(mark.truth);
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (mark.pred) {
        const [x, y] = at(mark.pred);
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fill();
        // Direct labelling (R1). Outlined so it stays readable over both black
        // text pixels and white background -- a patch has no reliable contrast.
        ctx.font = "600 11px system-ui, sans-serif";
        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
        ctx.strokeText(mark.corner, x + 7, y - 6);
        ctx.fillText(mark.corner, x + 7, y - 6);
      }
    }
  }, [patch, overlay, size, version]);

  return (
    <figure className="patch-canvas">
      <canvas
        ref={ref}
        className="patch-canvas__canvas"
        role="img"
        aria-label={label ?? `patch de ${patch.length}×${patch.length}`}
      />
      {label && <figcaption className="patch-canvas__label">{label}</figcaption>}
    </figure>
  );
}
