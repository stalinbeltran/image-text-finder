/** Reading the palette from CSS, which is where it lives.
 *
 * tokens.css is the single source of truth (and what `npm run validate:palette`
 * checks). Canvas drawing needs the values in JS, so it reads them back out of
 * the cascade instead of re-declaring them.
 *
 * That indirection buys the thing that matters: `getComputedStyle` resolves the
 * ACTIVE mode, so a canvas painted through here is correct in light and in dark
 * with no mode-detection logic of its own -- and cannot drift from the CSS the
 * validator approved.
 */

export const CORNER_NAMES = ["TL", "TR", "BR", "BL"] as const;
export type CornerName = (typeof CORNER_NAMES)[number];

/** Colour work a payload declares (api.md §3: the client cannot know whether it
 *  is looking at a signed weight or a non-negative activation). */
export type ColorJob = "sequential" | "diverging";

const read = (name: string, root: Element): string =>
  getComputedStyle(root).getPropertyValue(name).trim();

/** The 4 corner slots, in CORNER_NAMES order. R1: colour follows the entity. */
export function cornerColors(root: Element = document.documentElement): Record<CornerName, string> {
  return {
    TL: read("--itf-corner-tl", root),
    TR: read("--itf-corner-tr", root),
    BR: read("--itf-corner-br", root),
    BL: read("--itf-corner-bl", root),
  };
}

export type Rgb = [number, number, number];

function parseColor(css: string): Rgb {
  const hex = css.replace("#", "");
  if (hex.length === 6) {
    return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)) as Rgb;
  }
  const nums = css.match(/[\d.]+/g);
  if (!nums || nums.length < 3) throw new Error(`no sé leer el color "${css}"`);
  return [Number(nums[0]), Number(nums[1]), Number(nums[2])];
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

function lerpRgb(a: Rgb, b: Rgb, t: number): Rgb {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

export const rgbCss = ([r, g, b]: Rgb) => `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;

/** A colour ramp resolved from the cascade once, then sampled per cell.
 *
 *  Resolving per cell would mean a getComputedStyle call per pixel of a 40x40
 *  map -- 1600 of them per repaint, and V5 repaints on every mouse move. */
export interface Ramp {
  /** @param t 0..1 for sequential; -1..1 for diverging, 0 being the neutral. */
  at(t: number): string;
}

export function sequentialRamp(root: Element = document.documentElement): Ramp {
  const steps = Array.from({ length: 7 }, (_, i) => parseColor(read(`--itf-seq-${i}`, root)));
  return {
    at(t: number) {
      const clamped = Math.min(1, Math.max(0, t));
      const pos = clamped * (steps.length - 1);
      const i = Math.min(steps.length - 2, Math.floor(pos));
      return rgbCss(lerpRgb(steps[i], steps[i + 1], pos - i));
    },
  };
}

/** Diverging, with the neutral pinned at t = 0.
 *
 *  R2, and it is the whole point: the sibling project normalised min->max on a
 *  continuous ramp, so zero landed wherever it landed and the sign structure --
 *  what excites vs what inhibits, which is what a kernel IS -- was invisible.
 *  Here 0 is always the neutral grey, so the sign is readable at a glance. */
export function divergingRamp(root: Element = document.documentElement): Ramp {
  const neg = parseColor(read("--itf-div-neg", root));
  const mid = parseColor(read("--itf-div-mid", root));
  const pos = parseColor(read("--itf-div-pos", root));
  return {
    at(t: number) {
      const clamped = Math.min(1, Math.max(-1, t));
      return clamped < 0
        ? rgbCss(lerpRgb(mid, neg, -clamped))
        : rgbCss(lerpRgb(mid, pos, clamped));
    },
  };
}

export const rampFor = (job: ColorJob, root?: Element): Ramp =>
  job === "diverging" ? divergingRamp(root) : sequentialRamp(root);
