import { useState } from "react";
import { MatrixCanvas } from "../components/MatrixCanvas";
import { NumberTable } from "../components/NumberTable";
import { Meter } from "../components/Meter";
import { CORNER_NAMES, type CornerName } from "../theme/palette";

/** The palette and the base components, on synthetic data.
 *
 * Not a demo page: `npm run validate:palette` computes the colour, but it cannot
 * see a layout. This is where you LOOK at the thing -- in both modes, via the
 * toggle in the bar -- which is the step the validator does not cover.
 *
 * It also keeps the fase-1 components honest: they exist here with real data
 * shapes, so a phase that reaches for MatrixCanvas can see what it does before
 * there is a backend.
 */

/** A 3x3 oriented edge detector, which is what layer-1 kernels SHOULD look like
 *  once a run has trained (V1). Signed on purpose: it is what the diverging ramp
 *  is for, and negative means "inhibits". */
const SOBEL_X = [
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1],
];

/** A quiet activation map. Sequential, non-negative (post-ReLU).
 *  Deliberately low-amplitude: normalising per map is what keeps it readable,
 *  and a global normalisation would render it black (ui.md §5). */
const FEATURE_MAP = Array.from({ length: 12 }, (_, y) =>
  Array.from({ length: 12 }, (_, x) => {
    const d = Math.hypot(x - 3.5, y - 3.5);
    return Math.max(0, 0.08 * Math.exp(-d / 3));
  })
);

const SCORES: Record<CornerName, number> = { TL: 0.94, TR: 0.61, BR: 0.12, BL: 0.78 };

export function Kitchen() {
  const [selected, setSelected] = useState<"kernel" | "map" | null>("kernel");

  return (
    <section className="screen">
      <h1 className="screen__title">Paleta y componentes</h1>
      <p className="screen__lede">
        La paleta la valida <code>npm run validate:palette</code>, que computa el color y pasa en
        claro y en oscuro. Lo que un script no puede ver es la maquetación: esta pantalla es para
        mirarla. Cambia el tema en la barra de arriba.
      </p>

      <h2 className="screen__section">Los 4 tipos de esquina</h2>
      <p className="screen__note">
        Cuatro slots categóricos fijos, en el orden de <code>CORNER_NAMES</code>. El color sigue a
        la <strong>entidad</strong>, nunca a su rango: filtrar o reordenar no repinta a los
        supervivientes. Van siempre con etiqueta directa — es obligatorio, no cortesía, y es lo
        que descarga los WARN del validador.
      </p>
      <div className="kitchen__meters">
        {CORNER_NAMES.map((c) => (
          <Meter key={c} corner={c} value={SCORES[c]} threshold={0.5} />
        ))}
      </div>

      <h2 className="screen__section">Divergente — los kernels tienen signo</h2>
      <p className="screen__note">
        Rango simétrico <code>±max|w|</code> y <strong>gris neutro en el 0</strong>. El proyecto
        hermano normalizaba <code>min→max</code>, así que el cero caía en cualquier sitio y la
        estructura de signo —qué excita y qué inhibe, que es lo que un kernel <em>es</em>— quedaba
        invisible.
      </p>
      <div className="kitchen__row">
        <MatrixCanvas
          matrix={SOBEL_X}
          job="diverging"
          size={96}
          label="detector de borde vertical"
          selected={selected === "kernel"}
          onSelect={() => setSelected("kernel")}
        />
        <MatrixCanvas matrix={SOBEL_X.map((r) => r.map((v) => -v))} job="diverging" size={96} label="el mismo, invertido" />
      </div>

      <h2 className="screen__section">Secuencial — las activaciones son magnitud</h2>
      <p className="screen__note">
        Una tinta, clara→oscura, y <strong>normalizada por mapa</strong> (contra su propio
        min/max). Este mapa tiene amplitud 0,08: con una normalización global se vería negro. En
        oscuro la rampa invierte el ancla, porque «cerca de cero» es lo que se funde con la
        superficie.
      </p>
      <div className="kitchen__row">
        <MatrixCanvas
          matrix={FEATURE_MAP}
          job="sequential"
          size={96}
          label="activación tras ReLU"
          selected={selected === "map"}
          onSelect={() => setSelected("map")}
        />
      </div>

      <h2 className="screen__section">La tabla de números</h2>
      <p className="screen__note">
        Un mapa de calor codifica <strong>solo con color</strong>, así que todos deben tener su
        equivalente accesible. Click en cualquier mapa de arriba la cambia.
      </p>
      {selected === "kernel" ? (
        <NumberTable matrix={SOBEL_X} job="diverging" caption="detector de borde vertical (divergente)" />
      ) : selected === "map" ? (
        <NumberTable matrix={FEATURE_MAP} job="sequential" digits={3} caption="activación tras ReLU (secuencial)" />
      ) : null}
    </section>
  );
}
