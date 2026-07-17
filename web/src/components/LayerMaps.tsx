import { useState } from "react";
import type { LayerPayload } from "../api";
import { MatrixCanvas } from "./MatrixCanvas";
import { NumberTable } from "./NumberTable";

/** A layer of matrices as a grid of heat maps. **The same renderer for V1 and V2.**
 *
 * Ported from the sibling project, where `renderLayers(layers, kind)` also served
 * both, and for the same reason: a kernel and a feature map differ in what the
 * numbers *mean*, not in what they *are*. Both are `(K, H, W)` of floats with
 * per-map stats.
 *
 * Three things it does not decide, and each one is a rule:
 *
 *  1. **The colour job comes from the payload** (`layer.job`), never from the
 *     data. The client cannot know whether it holds a signed weight or a
 *     non-negative activation (api.md §3), and guessing is the one decision the
 *     sibling got wrong: it painted kernels on a `min→max` ramp, so zero landed
 *     wherever it fell and the sign structure — what excites, what inhibits, which
 *     is *what a kernel is* — was invisible.
 *
 *  2. **Normalisation is per map**, which `MatrixCanvas` does against each map's
 *     own range. Global, and every quiet feature map renders black — and "the
 *     deep layers are dead" is a conclusion someone would then draw about the
 *     network rather than about the view.
 *
 *  3. **Click → the numbers** (R5). A heat map encodes with *colour alone*, so the
 *     number table is its accessible twin and not an extra. It also happens to be
 *     the best way to debug the thing.
 *
 * The open map stays open across reloads (`selected` lives above the payload):
 * the sibling's `state.selectedMap`, and it is what makes any live view usable —
 * V2 re-fetches on every patch change and a table that closed itself each time
 * would be unusable.
 */
export interface LayerMapsProps {
  layer: LayerPayload;
  /** Target px of a map's longest side. Kernels are 3×3 and need the help. */
  size?: number;
  selected: number | null;
  onSelect: (index: number | null) => void;
}

export function LayerMaps({ layer, size = 72, selected, onSelect }: LayerMapsProps) {
  const open = selected !== null ? layer.maps[selected] : undefined;

  return (
    <div className="layer">
      <div className="layer__maps">
        {layer.maps.map((map) => (
          <MatrixCanvas
            key={map.index}
            matrix={map.matrix}
            job={layer.job}
            size={size}
            label={`#${map.index}`}
            selected={selected === map.index}
            onSelect={() => onSelect(selected === map.index ? null : map.index)}
          />
        ))}
      </div>

      {/* Truncation, said out loud. Dropping maps silently would answer "how many
          filters fired?" with a lie — and a layer of 128 in JSON hangs the tab,
          so the truncation itself is not optional. */}
      {layer.truncated && (
        <p className="card__hint">
          Enseñando {layer.shown} de {layer.count}: el resto se truncan para no reventar el
          navegador.
        </p>
      )}

      {open && (
        <NumberTable
          matrix={open.matrix}
          job={layer.job}
          caption={`Mapa #${open.index} · min ${open.min.toFixed(3)} · max ${open.max.toFixed(
            3
          )} · media ${open.mean.toFixed(3)}`}
        />
      )}
    </div>
  );
}

/** What the colour is doing, in words. A legend for a ramp with no axis.
 *
 * Diverging says *centrado en 0* explicitly because that is the property doing
 * the work (R2): without it a reader has no way to know the neutral is zero
 * rather than the midpoint of the range, which is exactly the ambiguity the ramp
 * exists to remove.
 */
export function JobLegend({ job }: { job: LayerPayload["job"] }) {
  const [help, setHelp] = useState(false);
  return (
    <p className="card__hint">
      {job === "diverging" ? (
        <>
          Paleta <strong>divergente centrada en 0</strong>: una tinta para los positivos, la
          opuesta para los negativos, <strong>gris en el cero</strong>, rango simétrico.
        </>
      ) : (
        <>
          Paleta <strong>secuencial</strong>: claro = poco, oscuro = mucho. Cada mapa se normaliza
          contra <strong>su propio</strong> rango.
        </>
      )}{" "}
      <button className="button button--quiet" onClick={() => setHelp((v) => !v)}>
        ¿por qué?
      </button>
      {help &&
        (job === "diverging" ? (
          <span className="card__hint">
            {" "}
            Un peso tiene signo, y lo que un kernel <em>es</em> son sus zonas de excitación y de
            inhibición. Con una rampa <code>min→max</code> el cero cae en cualquier sitio y esa
            estructura se vuelve invisible.
          </span>
        ) : (
          <span className="card__hint">
            {" "}
            Tras una activación no negativa (relu, sigmoid) el valor es magnitud, no signo. El
            trabajo de color lo declara el servidor mirando <code>spec.activation</code>: deducirlo
            del dato haría que una capa <code>tanh</code> se pintara de dos formas distintas según
            el patch.
          </span>
        ))}
    </p>
  );
}
