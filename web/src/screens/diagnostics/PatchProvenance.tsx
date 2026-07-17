import { getPatchDataset, listSources, sampleGeometry, sampleImageUrl, type PatchRow } from "../../api";
import { ErrorNote, Loading } from "../../components/Async";
import { Declares } from "../../components/Declares";
import { useAsync } from "../../useAsync";

/** V15 — patch provenance. **Fix B, show which image the patch was cut from.**
 *
 * The one probe that fixes B and not E: it says nothing about the model, only
 * where in its source image a patch sits. It is **free** — `sample_idx` and
 * `patch_xy` have been in the `.npz` all along with nobody reading them (api.md
 * §3) — and it is what turns a 40×40 crop back into a place: a false positive at
 * the top of a page is a different story from one in a margin.
 *
 * All front: the backend already serves the numbers. The source image's relative
 * id (what `sampleImageUrl` needs) is not in the patch payload, so it is recovered
 * by matching the dataset's `source_id` against the sources list — A knows its own
 * id, and B recorded which one it drew from.
 */
export function PatchProvenance({ patchDataset, row }: { patchDataset: string; row: PatchRow }) {
  // The manifest (for the source id and the patch size) and the sources (to turn
  // that source id into the relative id the image URL takes). Both are cheap and
  // cached upstream; fetched together so the view has one loading state.
  const meta = useAsync(
    () => Promise.all([getPatchDataset(patchDataset), listSources()]),
    [patchDataset]
  );

  const manifest = meta.data?.[0].manifest;
  const sources = meta.data?.[1].sources;
  const source = sources?.find((s) => s.source_id === manifest?.source_id) ?? null;
  const patchSize = manifest?.config.patch_size ?? 40;

  const geom = useAsync(
    () => (source ? sampleGeometry(source.id, row.sample_idx) : Promise.resolve(null)),
    [source?.id, row.sample_idx]
  );

  return (
    <section className="view">
      <Declares
        view="V15"
        title="Procedencia del patch"
        fixes="el dataset de patches (B)"
        varies="—"
        measures="de qué imagen salió, y dónde"
      >
        Devuelve el recorte a su sitio: <code>sample_idx</code> y <code>patch_xy</code> ya estaban en
        el <code>.npz</code>. Un falso positivo arriba de la página cuenta otra historia que uno en
        un margen.
      </Declares>

      {meta.loading && !meta.data && <Loading what="la procedencia" />}
      {meta.error && <ErrorNote problem={meta.error} />}

      {meta.data && !source && (
        <p className="card__hint">
          No encuentro la fuente <code>{manifest?.source_id}</code> de la que salió{" "}
          <code>{patchDataset}</code>: puede que se haya movido o borrado.
        </p>
      )}

      {source && (
        <>
          {geom.error && <ErrorNote problem={geom.error} />}
          {geom.data && (
            <figure className="patch-provenance">
              <div
                className="sample-view__stage"
                style={{ aspectRatio: `${geom.data.width} / ${geom.data.height}` }}
              >
                <img
                  src={sampleImageUrl(source.id, row.sample_idx)}
                  alt={`imagen ${row.sample_idx} de ${source.id}`}
                />
                <svg
                  viewBox={`0 0 ${geom.data.width} ${geom.data.height}`}
                  className="sample-view__overlay"
                >
                  <rect
                    x={row.patch_xy[0]}
                    y={row.patch_xy[1]}
                    width={patchSize}
                    height={patchSize}
                    className="patch-provenance__box"
                  />
                </svg>
              </div>
              <figcaption className="card__hint">
                Patch #{row.patch_idx} · imagen <strong>{row.sample_idx}</strong> de{" "}
                <code>{source.id}</code> · esquina en ({row.patch_xy[0]}, {row.patch_xy[1]}),{" "}
                {patchSize}×{patchSize} px
              </figcaption>
            </figure>
          )}
        </>
      )}
    </section>
  );
}
