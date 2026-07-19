import { useState } from "react";
import { getDeconvolution } from "../../api";
import { ErrorNote, Loading } from "../../components/Async";
import { Declares } from "../../components/Declares";
import { LayerMaps } from "../../components/LayerMaps";
import { useAsync } from "../../useAsync";

/** V16 — deconvolución. **Fija E y el patch, varía el filtro.**
 *
 * Para cada filtro, el gradiente de su activación respecto de la **entrada**: de
 * los píxeles que había, cuáles lo encendieron. Es la pregunta que V1 no puede
 * contestar de la capa 2 en adelante (D13: 32 o 64 canales no proyectan a una
 * matriz) y que V2 contesta a medias — V2 enseña *qué hizo* el filtro, esto enseña
 * *qué miraba*.
 *
 * **Divergente ±0, y no es estética** (R2). Un gradiente negativo es un píxel que
 * empuja al filtro hacia abajo: la mitad de la información. En rampa secuencial el
 * neutro caería donde cayera el mínimo y esa mitad desaparecería — el fallo que el
 * proyecto hermano cometió con los kernels. El payload lo declara; aquí no se
 * adivina.
 *
 * **Mismo renderer que V1 y V2** (`LayerMaps`), y por la misma razón: un kernel, un
 * feature map y una atribución se diferencian en qué significan los números, no en
 * qué son. Aquí además todos los mapas son `n×n` pase lo que pase con los pools,
 * así que las capas se comparan entre sí y con el patch sin reescalar nada.
 */
export function Deconvolution({
  run,
  patchDataset,
  patchIdx,
}: {
  run: string;
  patchDataset: string;
  patchIdx: number;
}) {
  const data = useAsync(
    () => getDeconvolution(run, { patch_dataset: patchDataset, index: patchIdx }),
    [run, patchDataset, patchIdx]
  );
  const dec = data.data;
  const [layerIdx, setLayerIdx] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);

  const layer = dec?.layers[Math.min(layerIdx, dec.layers.length - 1)];
  const peak = layer && selected !== null ? layer.peaks[selected] : undefined;

  return (
    <section className="view">
      <Declares
        view="V16"
        title="Deconvolución"
        fixes={`el run y el patch #${patchIdx}`}
        varies="el filtro"
        measures="qué píxeles del patch lo activaron"
      >
        El gradiente de la activación de cada filtro respecto de la <strong>entrada</strong>.
        Divergente en 0: <strong>rojo y azul son signos opuestos</strong> — píxeles que empujan al
        filtro hacia arriba y hacia abajo—, y el neutro es «este píxel le da igual».
      </Declares>

      {data.loading && !dec && <Loading what="la deconvolución" />}
      {data.error && <ErrorNote problem={data.error} />}

      {dec && layer && (
        <>
          {/* Same control as V2, on purpose: two views of one patch that switch
              layers differently would read as two unrelated screens. */}
          <div className="row-actions">
            <label className="field field--inline">
              <span className="field__label">Capa</span>
              <select
                value={layerIdx}
                onChange={(e) => {
                  setLayerIdx(Number(e.target.value));
                  setSelected(null);
                }}
              >
                {dec.layers.map((l, i) => (
                  <option key={l.layer} value={i}>
                    capa {l.layer} · {l.count} filtros
                  </option>
                ))}
              </select>
            </label>
          </div>

          <LayerMaps
            layer={layer}
            size={96}
            selected={selected}
            onSelect={setSelected}
          />

          {peak && (
            <p className="card__hint">
              Filtro <strong>#{peak.filter}</strong>: se deriva su{" "}
              <strong>activación máxima</strong> ({peak.activation.toFixed(3)}), que cayó en ({peak.x},{" "}
              {peak.y}) del mapa de {layer.width}×{layer.height}.{" "}
              {peak.activation <= 0 && (
                <strong>
                  Este filtro no disparó en este patch, así que su gradiente es 0 en todas partes: el
                  cuadrado plano es correcto, no un fallo del dibujo.
                </strong>
              )}
            </p>
          )}

          <p className="card__hint">
            Se deriva el <strong>máximo</strong> de la activación, no su suma: sumar mezcla todos los
            campos receptivos y devuelve una mancha que cubre el patch entero — cierta e ilegible.
            {layer.silent > 0 && (
              <>
                {" "}
                <strong>
                  {layer.silent} de {layer.count} filtros no dijeron nada en este patch
                </strong>{" "}
                (activación 0 ⇒ gradiente 0 ⇒ mapa plano). «En este patch»: otro patch puede
                despertarlos, así que esto no dice que el filtro esté muerto.
              </>
            )}
          </p>
        </>
      )}
    </section>
  );
}
