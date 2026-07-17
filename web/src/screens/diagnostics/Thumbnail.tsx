import { getPatch, type PatchRow } from "../../api";
import { PatchCanvas } from "../../components/PatchCanvas";
import { CORNER_NAMES, type CornerName } from "../../theme/palette";
import { useAsync } from "../../useAsync";

/** One cell of V6's gallery: the patch, plus how badly it went.
 *
 * **The pixels come from B's own endpoint**, not from the diagnostics payload.
 * The table says *which* patch (`patch_idx`); B says what a patch looks like.
 * Inlining them would have put a copy of B's data inside E×B's rows and made a
 * page ~300 KB of JSON — and left two endpoints able to disagree about what
 * patch 37 is.
 */
export function Thumbnail({
  patchDataset,
  row,
  threshold,
  selected,
  onSelect,
}: {
  patchDataset: string;
  row: PatchRow;
  threshold: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const patch = useAsync(() => getPatch(patchDataset, row.patch_idx), [patchDataset, row.patch_idx]);

  const worst = Math.max(...row.err_px.map((e) => e ?? -1));
  const overlay = CORNER_NAMES.flatMap((corner, i) =>
    row.exists[i] || row.score[i] >= threshold
      ? [
          {
            corner: corner as CornerName,
            truth: row.exists[i] ? ([row.xy_true[i][0], row.xy_true[i][1]] as [number, number]) : null,
            pred:
              row.score[i] >= threshold
                ? ([row.xy_pred[i][0], row.xy_pred[i][1]] as [number, number])
                : null,
          },
        ]
      : []
  );

  return (
    <button
      className={`gallery__item ${selected ? "is-selected" : ""}`}
      onClick={onSelect}
      aria-pressed={selected}
    >
      {patch.data ? (
        <PatchCanvas patch={patch.data.patch} size={112} overlay={overlay} />
      ) : (
        <div className="gallery__placeholder" aria-hidden="true" />
      )}
      <span className="gallery__caption">
        <code>#{row.patch_idx}</code>
        {/* "sin esquina", never "0.0 px": a patch with nothing to find has no
            error to report, and a 0 there would read as a perfect localisation
            (formatos.md §2). */}
        <span className="gallery__error">
          {worst < 0 ? "sin esquina" : `${worst.toFixed(1)} px`}
        </span>
      </span>
      <span className="gallery__origin" title="de qué imagen de la fuente salió, y dónde (V15)">
        img {row.sample_idx} · ({row.patch_xy[0]}, {row.patch_xy[1]})
      </span>
    </button>
  );
}
