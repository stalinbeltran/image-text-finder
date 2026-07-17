import type { CornerName } from "../theme/palette";

/** A probability against a threshold.
 *
 * ui.md §4.1: four probabilities against a threshold are METERS -- a ratio
 * against a limit, with the threshold marked on the track -- not a 4-bar chart,
 * and certainly not a pie. The mark on the track is the point of the component:
 * `p=0.6` means nothing until you can see which side of the threshold it fell on.
 *
 * The value is always written out. R1 makes direct labelling mandatory for the
 * corner slots ("no cortesía"), and it is also what discharges the light-mode
 * contrast WARN that `npm run validate:palette` reports for BR and BL.
 */
export interface MeterProps {
  corner: CornerName;
  value: number;
  /** Marked on the track. A knob of F, tuned post-hoc and free (V8). */
  threshold?: number;
  label?: string;
}

export function Meter({ corner, value, threshold, label }: MeterProps) {
  const pct = Math.min(100, Math.max(0, value * 100));
  const caption = label ?? corner;
  return (
    <div className="meter">
      <div className="meter__head">
        {/* The swatch carries identity; the text stays in ink tokens. Text never
            wears the series colour. */}
        <span className="meter__swatch" data-corner={corner} aria-hidden="true" />
        <span className="meter__label">{caption}</span>
        <span className="meter__value">{value.toFixed(3)}</span>
      </div>
      <div
        className="meter__track"
        role="meter"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-label={caption}
      >
        <div className="meter__fill" data-corner={corner} style={{ width: `${pct}%` }} />
        {threshold !== undefined && (
          <div
            className="meter__threshold"
            style={{ left: `${Math.min(100, Math.max(0, threshold * 100))}%` }}
            title={`umbral ${threshold.toFixed(2)}`}
          />
        )}
      </div>
    </div>
  );
}
