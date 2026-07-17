import { useState } from "react";
import { getKernels } from "../../api";
import { ErrorNote, Loading } from "../../components/Async";
import { Declares } from "../../components/Declares";
import { JobLegend, LayerMaps } from "../../components/LayerMaps";
import { useAsync } from "../../useAsync";

/** V1 — the learned kernels of layer 1. **Layer 1 only** (D13, 2026-07-17).
 *
 * **The cheapest real check in the app**: no input, no forward pass, just the
 * weights. And it answers a question nothing else does — *did this network learn
 * anything at all?* With `in_channels: 1` the layer-1 filters apply to the patch
 * itself, so they should come out looking like **oriented edge detectors**. If
 * they look like noise, the network did not learn, and that is information rather
 * than a bug in the view (plan-ui.md fase 6).
 *
 * **Why only layer 1, and why that is not a gap to fill later.** The rule is
 * `in_channels === 1`, not "the first layer" — layer 1 is simply the only one
 * that satisfies it. From layer 2 on a filter is 32 or 64 matrices operating on
 * channels that are not the image, and there is no honest projection to one
 * matrix. The sibling painted `weight[k, 0]` — one thirty-second of the kernel,
 * picked arbitrarily — which looks like a view while telling you nothing. What
 * the deep layers have to say is in their feature maps, so that is where the
 * reader is sent.
 *
 * **Diverging, centred on 0** (R2). Not cosmetic: a weight has sign, and what a
 * kernel *is* is its structure of excitation and inhibition.
 */
export function Kernels({ run }: { run: string }) {
  const kernels = useAsync(() => getKernels(run), [run]);
  const [selected, setSelected] = useState<number | null>(null);
  const data = kernels.data;

  return (
    <section className="view">
      <Declares
        view="V1"
        title="Kernels de la capa 1"
        fixes="el run (E)"
        varies="—"
        measures="los pesos aprendidos"
      >
        Deberían parecer <strong>detectores de borde orientados</strong>. Si parecen ruido, la red
        no aprendió — y eso es información, no un fallo de la vista.
      </Declares>

      {kernels.loading && !data && <Loading what="los kernels" />}
      {kernels.error && <ErrorNote problem={kernels.error} />}

      {data && (
        <>
          <div className="view__facts">
            <span>
              <strong>{data.count}</strong> filtros de {data.kernel_size}×{data.kernel_size}
            </span>
            <span title="con un solo canal de entrada, un filtro ES una matriz: lo que ves es exacto, no una proyección">
              <strong>{data.in_channels}</strong> canal de entrada
            </span>
            <span>
              capa <strong>1</strong> de {data.layers_in_backbone}
            </span>
          </div>

          <LayerMaps layer={data} size={64} selected={selected} onSelect={setSelected} />
          <JobLegend job={data.job} />

          {/* The absence, explained where it is noticed. An unexplained "only one
              layer" reads as a bug and invites someone to "fix" it back into the
              dishonest channel-0 slice. */}
          {data.layers_in_backbone > 1 && (
            <p className="card__foot">{data.deep_layers_note}</p>
          )}
        </>
      )}
    </section>
  );
}
