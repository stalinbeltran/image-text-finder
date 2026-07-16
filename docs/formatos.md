# Formatos en disco

Los artefactos que este proyecto escribe y lee, y **cómo evolucionan sin romper lo ya
generado**.

Estos ficheros son **contratos**: los escribe un módulo y los leen varios, a veces meses
después. Un cambio descuidado no da una excepción — da resultados peores sin que nadie se entere.

---

## 1. Por qué hace falta este documento

**El proyecto ya ha evolucionado sus formatos tres veces, cada una con una técnica distinta y
ninguna documentada:**

| Cambio | Cómo se resolvió |
|---|---|
| Se añadió `border` al `.npz` | *Feature sniffing*: `if "border" in data.files` → si no, ceros |
| Se añadió `border_features` a la config de red | Valor por defecto `False`, "so configs from before this feature rebuild an identical network" (comentario en `builder.py`) |
| La fase 4 añadirá procedencia al `config.json` del run | **Sin resolver todavía** |

Ninguno de los artefactos tiene **campo de versión**. Y la primera de esas tres migraciones
**está mal hecha**, con una trampa cargada y apuntando (§2).

---

## 2. La regla que faltaba: **ausente ≠ cero**

`patches/dataset.py:27-28`:

```python
border = data["border"] if "border" in data.files \
    else np.zeros((X.shape[0], NUM_BORDERS), dtype=np.uint8)
```

El docstring dice *"those patches fall back to all-zeros (no border known)"*. Pero **el modelo no
puede distinguir "no se sabe" de "no toca ningún borde"**: cero es un **valor con significado**,
no un nulo. Rellenar con ceros no marca el dato como ausente — **lo falsifica**.

### La trampa está cargada, y es el ejemplo del README

Medido ahora mismo en este repo:

- Los **tres** `.npz` de `data/patch-datasets/` (`clear-paragraphs-02`, `clear-paragraphs-02-1`,
  `reducido-40`) **no tienen `border`**: son anteriores a la feature.
- `configs/model.example.yaml` trae **`border_features: true`**.
- El README documenta exactamente: `itf-train --config configs/model.example.yaml --data data/patch-datasets/reducido-40`.

Qué pasa si lo corres, hoy:

1. `PatchDataset` rellena `border` con ceros, en silencio.
2. El modelo entrena creyendo que **ninguna ventana toca jamás un borde de la imagen**. Como la
   entrada es constante, aprende a ignorar esas 4 entradas.
3. En inferencia, `detect_corners` calcula los flags **de verdad** (`predict.py:69-70`) y le mete
   unos.
4. **El modelo recibe una distribución que nunca vio, justo en los bordes de la imagen** — que es
   exactamente donde el flag de borde debía ayudar.

Nada avisa. No hay excepción: solo predicciones peores en los bordes. *(De momento nadie ha sido
mordido: los cinco runs de `runs/` tienen `border_features` ausente ⇒ `False`. Y `reducido-40`
además tiene `val` vacío — ver protocolo.md §1.3. El ejemplo del README acumula **dos** fallos
silenciosos.)*

### La regla

> **Rellenar un campo ausente solo es legal si el consumidor no lo usa.**

Aplicado: `border` ausente **+** `border_features: False` → relleno inocuo, el modelo lo ignora.
`border` ausente **+** `border_features: True` → **error**, con su razón:

```
el dataset 'reducido-40' se construyó antes de los flags de borde y no los tiene.
Reconstrúyelo, o entrena con border_features: false.
```

Generalizado: **un lector que necesita un campo ausente falla; nunca lo inventa.** Y el
contrato ② de tests.md gana un caso que hoy no cubre — `test_patch_dataset_border_backfill`
prueba que el relleno ocurre, pero no que **se niegue** cuando la red sí lo usa.

---

## 3. Política de versionado

Cada artefacto lleva **`format_version`** (entero). Reglas:

| Cambio | ¿Bump? | Qué hace el lector |
|---|---|---|
| **Añadir** un campo cuyo default reproduce el comportamiento viejo | **No** | Lee ambos; el default preserva la semántica |
| Añadir un campo que el consumidor **necesita** | **No**, pero **falla si falta** (§2) | Error con razón, no relleno |
| **Cambiar el significado** de un campo existente, o su unidad, orden o escala | **Sí** | Rechaza el viejo, o migra explícitamente |
| Quitar un campo | **Sí** | — |

**No se bumpea por costumbre.** El sniffing aditivo funciona y es barato; la versión se gana el
sitio solo cuando hay que **rechazar o migrar**, que es cuando el sniffing no basta. Lo que no es
opcional es la §2: distinguir ausente de relleno.

Los `.npz` que ya existen no tienen `format_version` ⇒ **ausente significa versión 1**.

---

## 4. Los artefactos

### 4.1 `data/patch-datasets/<name>/` — el dataset de patches (B)

**`patches.npz`** — seis arrays paralelos, todos de largo N (el nº de patches). Verificado en el
repo:

| Array | Forma | dtype | Qué es |
|---|---|---|---|
| `X` | (N, n, n, 1) | `uint8` | **El patch: la entrada real de la CNN**. 0–255; `PatchDataset` lo pasa a `(N,1,n,n)` float en [0,1] |
| `y` | (N, 4, 3) | `float32` | `[exists, x, y]` por esquina, en el orden de `corner_order`. `x,y` **normalizados a [0,1]** dentro del patch |
| `border` | (N, 4) | `uint8` | Flags de borde en el orden de `border_order`. **Puede faltar** → §2 |
| `sample_idx` | (N,) | `int32` | De qué imagen de A salió |
| `patch_xy` | (N, 2) | `int32` | `(x0, y0)` de la esquina superior izquierda en esa imagen |
| `split` | (N,) | `int8` | **0 train, 1 val, 2 test** — el orden de `SPLIT_NAMES` |

Tres cosas que **no** se pueden deducir del array y por eso viven en el manifest: el **orden** de
las esquinas, el **orden** de los bordes y el mapeo **entero→split**. Son semántica pura: si se
pierden, los datos siguen cargando y significan otra cosa.

> `sample_idx` y `patch_xy` **están y no los usa nadie**. Son la procedencia del patch (V15 de
> ui.md) y el enlace B→A. Ya pagados.

**`manifest.json`** — la descripción. `patches.npz` es la carga; **el manifest es el contrato**:

```jsonc
{ "format_version": 2,                  // ← falta hoy
  "fingerprint": "sha256:…",            // ← falta hoy, contrato ⑧
  "source_id": "clear-paragraphs-02-8ea1ac04",
  "config": { … },                      // el PatchExtractConfig entero: n, stride, split, seed…
  "num_samples": 200, "num_patches": 9800,
  "patch_shape": [40, 40, 1], "label_shape": [4, 3],
  "corner_order": ["TL","TR","BR","BL"],      // semántica, no adorno
  "border_order": ["top","right","bottom","left"],
  "patches_per_split": {"train": 7840, "val": 980, "test": 980},
  "positives_per_corner": {"TL": 2012, …} }
```

- **`fingerprint`** (contrato ⑧): huella del contenido del `.npz`. Sin ella, **un dataset
  reconstruido bajo el mismo nombre es indistinguible del anterior**, y un barrido a medias queda
  incomparable en silencio. Es lo que el run copia para poder decir "me entrené con *este* B".
- **Invariante**: el manifest debe cuadrar con el `.npz` (`num_patches == X.shape[0]`,
  `patch_shape == X.shape[1:]`). Si no, hay corrupción — y eso se testea.

**`split.json`** — `{"train": [índices de A], "val": [...], "test": [...]}`. Son índices de la
**fuente**, no del `.npz`. Es lo que permite el cruce A×B de la pestaña Predecir.

> **El split es por imagen, no por patch** (`_assign_splits` baraja *samples*). Es correcto y no
> es evidente: si se repartiera por patch, ventanas solapadas de la misma imagen caerían en
> splits distintos y el val estaría contaminado. **Este invariante se testea.**

### 4.2 `runs/<name>/` — el run (E)

| Fichero | Qué |
|---|---|
| `config.json` | El `RunConfig` congelado: `data`, `out`, `model`, hiperparámetros, `seed`, `device` |
| `metrics.jsonl` | Una línea JSON por época: `{epoch, train_loss, val:{…}, seconds}` |
| `best.pt`, `last.pt` | `{"model": state_dict, "config": dict, "epoch": int}` — verificado |
| `summary.json` | `{run, epochs, best_val_loss, final, corner_order}` |

- **El checkpoint es autodescriptivo** (contrato ④): lleva la config entera, así que
  `load_model()` reconstruye la red sin ningún YAML. **Es la mejor propiedad del formato** y no
  se toca.
- **`metrics.jsonl` es append-only** y se lee **incrementalmente** (`?since=N`, R5 de api.md).
  Nunca se reescribe: un run vivo se está leyendo mientras se escribe.
- **Lo que cambia en la fase 4** (contrato ③): `config.json` gana `format_version`, procedencia
  (`network`/`recipe` **por nombre** + huella de B), commit de git y entorno; y `device` sale de
  la identidad (contrato ⑩). **Los cinco runs de `runs/` no tienen nada de eso** → el lector
  degrada (`name: null`), **no revienta**. Es la trampa más probable del plan y tiene test propio
  *antes* de la fase 4 (tests.md §5).
- **Falta estado explícito**: hoy `_run_status()` lo deduce de qué ficheros existen, así que un
  crash queda "running" para siempre. Va a `status.json`, como en el proyecto hermano.

### 4.3 `configs/` — las definiciones (C y D)

`configs/models/*.yaml` (C, hoy huérfano) y `configs/recipes/*.yaml` (D, por crear). YAML, no
JSON: **los escribe una persona**. Ambos con `format_version`.

Contraste deliberado con todo lo demás: **estos se versionan en git y no son artefactos**. Son
fuente.

### 4.4 La tabla por patch (E×B) — **un caché**, no un artefacto

*(D1, decidido 2026-07-16.)* Es **función pura de (run, huella de B, split, knobs por defecto)**,
y los cuatro ya tienen identidad ⇒ **se puede recalcular exacta ⇒ es un caché**, por el mismo
criterio de §5. No se nombra, no se lista, no se versiona, no tiene CRUD ni pantalla — y las
cuatro vistas de diagnóstico (V6–V9) salen igual.

- **Clave**: `(run, fingerprint de B, split, knobs)`. Si cambia cualquiera, se recalcula.
- **Ubicación**: un directorio de caché, gitignoreado. Borrarlo no pierde nada.
- **Coste de rehacerla**: segundos (~10⁴ forwards por lotes). Por eso puede ser síncrona (R3 de
  api.md).

> Lo que **no** se puede recalcular es el **criterio humano** — que a ti te interesaran "los
> patches donde falla el TL". Si algún día se quiere volver a una búsqueda guardada (y de ahí
> construir un dataset con esos fallos para reentrenar, como el proyecto hermano), lo que se
> guarda es **el filtro**: `{run, split, outcome, corner}`, cuatro campos, no 10⁵ filas. **Hoy no
> se construye** (D1: solo caché).

`.npz` columnar, porque es el idioma del proyecto y numpy agrega ~10⁵ filas al instante
(librerias.md descartó Parquet/DuckDB por eso):

| Array | Forma | Qué |
|---|---|---|
| `patch_idx` | (M,) int32 | Fila en el `.npz` de B |
| `score` | (M, 4) float32 | `p(exists)` por esquina |
| `xy_pred` | (M, 4, 2) float32 | Posición predicha, normalizada |
| `err_px` | (M, 4) float32 | Error en px; `NaN` donde no hay esquina real |

La clave del caché (run + huella de B + split + knobs) va **en el nombre del fichero o en un
sidecar**, para poder invalidarlo sin abrirlo. Con la tabla en mano, re-umbralizar es filtrar
columnas y **no vuelve a correr el modelo** (V8 de ui.md) — que es de dónde salen las horas de
CPU ahorradas.

`err_px = NaN` donde no hay esquina, **no 0**: es §2 otra vez. Un 0 diría "acertó exacto".

### 4.5 Fronteras: lo que no es nuestro formato

- **`labels.jsonl`** (A) lo define `image-text-sample-generator` en su `SAMPLE_FORMAT.md`. **No
  lo especificamos: especificamos qué consumimos** — `index`, `labels.{width,height,has_overlap}`
  y `blocks[].{block_id,kind,angle,quad}`, con `quad` (4,2) **horario desde TL**. Es una
  dependencia entre proyectos: si cambia allí, `datasets/loader.py` rompe aquí, y nada lo detecta
  salvo un test con un `labels.jsonl` de muestra.
- **El storage del barrido** es de `optuna` (SQLite). Nuestro es `sweeps/<name>/spec.json` (lo
  fijo, el espacio, el objetivo, el presupuesto). **La frontera importa**: los `trials` de optuna
  no son nuestros runs; un trial lanza un run y guarda su referencia (librerias.md).

---

## 5. Qué se versiona en git — una decisión pendiente

`.gitignore` tiene `/data/` y `/runs/` **enteros**. Consecuencia medida: `git ls-files runs`
está **vacío**. Los configs, las métricas y los manifests de los cinco runs reales **no están en
git**.

En un proyecto de investigación eso significa: **el registro de lo que has corrido no tiene
historia y está a un `rm -rf` de desaparecer.** No puedes volver a ver con qué entrenaste hace un
mes.

El proyecto hermano decidió lo contrario, y su regla 4 es explícita: `experiments/<id>/` va
versionado (config, métricas, estado) y **solo `checkpoints/` se ignora**. Es el patrón sensato:
**versiona la descripción, ignora la carga.**

Aplicado aquí sería:

```gitignore
/data/patch-datasets/*/patches.npz     # la carga: MB
!/data/patch-datasets/*/manifest.json  # la descripción: KB
!/data/patch-datasets/*/split.json
/runs/*/*.pt                           # la carga
# config.json, metrics.jsonl, summary.json → versionados
```

Tamaños: los `.npz` y los `.pt` son MB; los manifests y `metrics.jsonl` son KB. **El coste es
despreciable y lo que se compra es el registro de la investigación.**

### Decidido: se versiona la descripción, se ignora la carga

*(D5, decidido 2026-07-16.)* Medido en este repo: **descripción 105 KB vs carga 38,5 MB** — el
**0,3 %** del peso, y es el registro de la investigación.

El criterio, que es el mismo que decide D1 (§4.4):

> **Se versiona lo que no se puede recalcular; se ignora lo que sí.**
>
> `best.pt` sale de config + semilla + código, y aunque no saliera, git es la herramienta
> equivocada para 2 MB de binario: eso son backups. `metrics.jsonl` **es una medición** —
> recalcularla cuesta horas y solo sale igual si el código no ha cambiado. No es derivable: es un
> registro.

Contras asumidos: un run vivo reescribe `metrics.jsonl` cada época ⇒ `git status` sucio mientras
entrenas. Y un barrido son decenas de runs; si el historial se ensucia demasiado, se ignora
`sweeps/` (no se revierte la decisión).

---

## 6. Qué se testea de aquí

En `test_contracts.py` (tests.md), porque todo esto son contratos:

1. **`border` ausente + `border_features: True` → error**, no relleno silencioso (§2). *El más
   urgente: la trampa está cargada.*
2. **Manifest y `.npz` cuadran** (`num_patches`, `patch_shape`).
3. **El split es por imagen**: ningún `sample_idx` aparece en dos splits.
4. **Retrocompatibilidad**: un `config.json` sin procedencia se lee degradando (antes de la fase 4).
5. **`corner_order` / `border_order` viajan** en el manifest y coinciden con las constantes.
6. **La huella cambia** si cambia el contenido, y no si solo cambia el nombre (contrato ⑧).
