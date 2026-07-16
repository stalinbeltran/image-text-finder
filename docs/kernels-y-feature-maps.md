# Visualización de kernels y feature maps — resumen para reproducir

Resumen de cómo lo hace `C:\Desarrollo\sliding-window-NIST-ocr` (CNN sobre MNIST, pestaña
**Features** de su web app), y qué habría que cambiar para traerlo a `image-text-finder`.
Documento de referencia: **no hay nada implementado aquí todavía**.

## Qué ofrece la pantalla original

Dos vistas, ambas alimentadas por la misma NN ya entrenada (se elige un experimento con
checkpoint `best.pt`):

1. **Kernels aprendidos** — los pesos de cada capa conv, pintados como mapas de calor.
   Vista estática: solo depende del modelo, no de ninguna entrada.
2. **Feature maps** — las activaciones de cada capa sobre *una entrada concreta*: cada
   kernel produce una matriz H×W = "el efecto de ese kernel sobre esta imagen".

En ambas, **click en cualquier mapa → su representación matricial completa**: una tabla con
los números, cada celda con fondo sombreado según su valor. Es la parte que da el "ver los
números y no solo el color".

Encima hay un editor del dígito con **vista en vivo**: cada retoque re-predice (con
debounce configurable en ms) y repinta los feature maps, así se ve en directo qué le hace
al reconocimiento. Es un extra: no hace falta para la visualización básica.

## Arquitectura de la solución (3 piezas)

### 1. El modelo expone dos métodos

En `src/swnist/models/features_cnn.py`. La CNN guarda sus bloques en un `nn.ModuleList`
donde cada bloque es un `nn.Sequential(Conv2d, activación, [pool])`:

```python
def feature_maps(self, x):
    """Salida de cada capa (activación + pooling): (N, kernels, H, W) por capa."""
    maps = []
    for block in self.blocks:
        x = block(x)
        maps.append(x)
    return maps

def kernels(self):
    """Pesos aprendidos de cada conv: (kernels, in_channels, k, k) por capa."""
    return [block[0].weight.detach() for block in self.blocks]
```

Toda la idea está aquí: como cada bloque es un `Sequential` cuyo **elemento 0 es el
Conv2d**, los pesos salen con `block[0].weight` y las activaciones simplemente reaplicando
los bloques uno a uno y quedándose con cada salida intermedia. No hay hooks ni
`register_forward_hook`.

### 2. El backend serializa a matrices de números

En `src/swnist/webapp/inference.py`. Nada de imágenes: se mandan **los números**, y el
navegador decide el color. Un mapa se serializa así:

```python
def map_payload(array, index):
    return {
        "index": index,
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "matrix": np.round(array.astype(float), 4).tolist(),
    }
```

`min`/`max` viajan con cada mapa porque **la normalización del color es por mapa**: cada
feature map se pinta contra su propio rango, si no los mapas apagados se ven todos negros.

Payload por capa (igual para kernels y para feature maps, de ahí que el front reuse el
mismo renderer):

```jsonc
{
  "layer": 1,
  "kernels": 8,          // nº de filtros de la capa
  "shown": 8,            // cuántos se mandan (truncado por max_maps)
  "truncated": false,
  "height": 26, "width": 26,   // solo en feature maps
  "kernel_size": 3, "in_channels": 1,  // solo en kernels
  "spec": { /* la capa tal cual del config: activation, pool, ... */ },
  "maps": [ { "index": 0, "min": …, "max": …, "mean": …, "matrix": [[…]] } ]
}
```

Dos detalles que valen la pena:

- **`MAX_MAPS_PER_LAYER = 64`**: trunca los mapas por capa para no reventar el navegador
  (una capa de 256 filtros × una matriz de 40×40 en JSON es enorme), y avisa con
  `truncated: true` para que la UI lo diga.
- **De los kernels solo se pinta el primer canal de entrada** (`array[k, 0]`). Un kernel de
  la capa 2 tiene forma `(filters, in_channels, k, k)` — con 32 canales de entrada no hay
  forma honesta de pintarlo como una sola matriz, así que se enseña el corte del canal 0 y
  ya. Es la simplificación que hace viable la vista; conviene decidir si nos sirve.

El modelo se cachea en memoria (`_model_cache` por `exp_id`), porque la vista en vivo
re-predice a cada trazo del pincel.

### 3. La API: dos endpoints

```
POST /api/predict                    → predicción + layers[] con los feature maps
GET  /api/experiments/{id}/kernels   → layers[] con los kernels aprendidos
```

`POST /api/predict` acepta la entrada de dos formas, y esa dualidad es lo que hace posible
el editor en vivo:

- `{experiment, dataset, split, index}` → una muestra del dataset;
- `{experiment, image, label}` → una imagen 28×28 suelta (píxeles en el body), que no está
  guardada en ningún sitio.

Ambas caen en el mismo `_analyze()`, que devuelve `layers` idéntico y además la predicción
(`pred`, `probs`, `top`, `margin`, `thumb`).

### 4. El front pinta canvas + tabla

En `static/app.js`. Un mapa = un `<canvas>` con un rect por celda, tamaño de celda
calculado para que el mapa quepa en ~72 px:

```js
function drawMap(canvas, matrix, min, max) {
  const cell = Math.max(2, Math.round(72 / Math.max(height, width)));
  canvas.width = width * cell; canvas.height = height * cell;
  const range = max - min || 1;             // ← || 1 evita dividir por cero (mapa plano)
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const norm = (matrix[y][x] - min) / range;
      const shade = Math.round(255 * norm);
      ctx.fillStyle = `rgb(${shade}, ${shade * 0.85}, ${60 + shade * 0.6})`;
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }
}
```

Con `image-rendering: pixelated` en el CSS, para que un mapa de 3×3 se vea como 3×3 celdas
nítidas y no como un borrón interpolado.

`renderLayers(layers, kind)` es **el mismo renderer para las dos vistas** — `kind` solo
cambia el rótulo (`"kernel"` vs `"feature map"`) y si el tamaño mostrado es
`kernel_size×kernel_size` o `height×width`. Al hacer click, `renderMatrix()` pinta la tabla
de números con el mismo esquema de sombreado (más apagado, para que los números se lean).

Detalle bueno de la vista en vivo: el mapa abierto se recuerda en
`state.selectedMap = {kind, layer, index}`, así al re-predecir **la matriz abierta se
actualiza en lugar de cerrarse**.

## Qué habría que cambiar para traerlo aquí

`image-text-finder` no es el mismo caso, y estas son las diferencias reales
(`src/itf/models/builder.py`):

| | sliding-window-NIST-ocr | image-text-finder |
|---|---|---|
| Bloques | `nn.ModuleList` de `Sequential` | `self.backbone` = un `nn.Sequential` de `Sequential` |
| Config | `layers[].kernels`, `pool: {type,size}` | `backbone[].filters`, `pool: <int>` |
| Entrada | 28×28, 1 canal | 40×40 (`input_size`), `in_channels` |
| Salida | 10 logits (softmax, dígito) | `CornerHead` → `(B, 4, 3)` |
| Front | JS vanilla, `app.js` | React + Vite (`web/`) |

Lo que **se copia casi tal cual**:

- `feature_maps()` / `kernels()` en `ConfigurableCNN`. Iterar `self.backbone` (que ya es
  `Sequential`, así que se itera igual) y `block[0].weight` sigue siendo el `Conv2d`
  **también con `batchnorm: true`** — el conv siempre va primero en `_conv_block()`. Ojo:
  `feature_maps()` no necesita el tensor `border`, porque `border_features` solo entra en
  la cabeza, no en el backbone.
- `map_payload()` y el truncado por `max_maps`, sin cambios.
- `drawMap()` y el normalizado por mapa, portado a un componente React con `useRef` sobre
  el canvas.

Lo que **hay que decidir**:

1. **Qué es "la predicción"** en el payload. Aquí no hay 10 probabilidades ni `top`/`margin`
   de clasificación: la salida es `(4, 3)` por esquina. La parte `layers` del payload es
   reutilizable tal cual; la de la predicción hay que rediseñarla.
2. **De dónde sale la entrada**. Allí era "una muestra del dataset por índice". Aquí el
   equivalente natural es un **patch** (`src/itf/patches/`), y habría que ver si se elige por
   índice del dataset de patches, o subiendo/recortando una imagen.
3. **El corte del canal 0 en los kernels**: con `filters: 32` en la primera capa y 64 en la
   segunda, la vista de la capa 2 en adelante se vuelve poco informativa. Alternativas:
   enseñar todos los canales de entrada de un filtro, o dejarlo en el canal 0 como allí.
4. **Editor en vivo sí/no**. Es la mitad del trabajo (`editor.js`, 395 líneas: pincel
   suave, transformaciones no destructivas, deshacer) y es independiente: la visualización
   funciona sin él.

## Ficheros de referencia en el proyecto original

```
src/swnist/models/features_cnn.py     ← feature_maps() y kernels() (203 líneas, todo el modelo)
src/swnist/webapp/inference.py        ← map_payload, _analyze, kernels, encoder PNG propio
src/swnist/webapp/main.py             ← POST /api/predict, GET /api/experiments/{id}/kernels
src/swnist/webapp/static/app.js       ← drawMap, renderLayers, renderMatrix (líneas ~1100-1222)
src/swnist/webapp/static/style.css    ← .maps/.map/.matrix (líneas ~104-114)
CLAUDE.md                             ← reglas 20-22 y sección "Visualización de los features"
```
