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

Generalizado: **un lector que necesita un campo ausente falla; nunca lo inventa.**

### La afinación: **dato** ausente ≠ **declaración** ausente

Bajando al detalle aparece una asimetría que la regla de arriba no distingue, y es la que hace
que esto se resuelva **sin migrar nada**:

| Tipo de campo | Ejemplo | Ausente significa | Comportamiento |
|---|---|---|---|
| **Dato** | `border` (N,4) | No se sabe — y **no se puede inventar** | **Fallar**, alto y claro |
| **Declaración de capacidad** | `has_border: true` en el manifest | "No lo tengo" | **Por defecto "no"**: seguro, rechaza de más y nunca de menos |

Por eso los manifests viejos no necesitan migración: no llevan `has_border` ⇒ se lee `False` ⇒ si
la red pide bordes, se niega. **El default correcto sale solo.**

### Cómo se resuelve

Tres piezas, ninguna grande:

1. **El manifest declara**: `has_border: true`. Hoy hay que *inferirlo* de que falte
   `border_order` — y **inferir no es declarar**. *(Comprobar el `.npz` directamente también
   valdría: `np.load` es perezoso y saber si el array está cuesta ~34 ms. Pero el manifest **es
   el contrato**, §4.1.)*
2. **`PatchDataset` enuncia el hecho, no dicta política**: una propiedad `has_border`. Sigue
   rellenando ceros —inocuo cuando nadie los usa— pero **deja de mentir en el docstring**: hoy
   dice *"no border known"*; debe decir *"ceros; consulta `has_border` antes de usarlos"*. Un
   lector no puede decidir esto solo: **no conoce el modelo**.
3. **Lo decide el validador de compatibilidad**, que es el mismo del contrato ① — porque son la
   misma pregunta (organizacion.md §2, recuadro tras ②). No hace falta un mecanismo para
   `border`: entra gratis en el validador que ① ya pedía.

Y el contrato ② de [tests.md](tests.md) gana el caso que hoy no cubre:
`test_patch_dataset_border_backfill` prueba que el relleno **ocurre**, pero no que **se niegue**
cuando la red sí lo usa.

> **Lo que NO se hace: migrar los tres `.npz` de hoy.** Se van a regenerar igualmente (D6: ~2000
> imágenes + el holdout) y además son **derivables** de fuente + config + semilla. **No se escribe
> código de migración para datos que estás a punto de tirar.** El arreglo es para el futuro.

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

**`format_version` nace en 1, y no hay v0.** Con los datos borrados (D18) no queda ningún `.npz`
anterior a los campos de §4.1: todo dataset nuevo trae `fingerprint`, `has_border` y `border`
desde el primer día. La versión empezará a hacer trabajo el día que haya que **rechazar o
migrar** algo — hoy no lo hay, y por eso está en 1 y quieta.

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
{ "format_version": 1,                  // nace en 1: con los datos borrados, no hay v0 del que venir
  "fingerprint": "sha256:…",            // contrato ⑧
  "has_border": true,                   // lo que el validador de ② consulta (§2)
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
| `config.json` | `format_version`, la receta (D), la red por valor (C) y la **procedencia** (§4.2.1). **`device` no está** *dentro de la receta*: es X (contrato ⑩), y va en `execution` |
| `metrics.jsonl` | Una línea JSON por época: `{epoch, train_loss, val:{…}, lr, seconds}` |
| `best.pt` | `{"model": state_dict, "config": dict, "epoch": int}` — es el **entregable**: lo que F carga (contrato ④) |
| `last.pt` | Lo anterior **más el estado de reanudación** (§4.2.2): es el **punto de guardado**, no un entregable |
| `summary.json` | `{run, epochs_run, epochs_requested, stopped_early, cancelled, monitor, best, final, corner_order}` |
| `status.json` | El estado **explícito**: `queued \| running \| done \| error \| cancelled`. Sin él, un crash queda "running" para siempre |
| `stop.json` | La **petición** de parada: `{requested_at, reason}`. Existe solo si alguien la pidió *(fase 4)* |

#### 4.2.2 `last.pt` — el estado de reanudación

> **Diseño, NO construido** *(2026-07-19)*. Hoy `last.pt` sigue siendo `{model, config, epoch}`, y
> **la tabla de abajo es lo que haría falta** para reanudar dentro de un trial sin mentir. Está
> escrito porque el camino barato —arrancar de `last.pt` tal cual— es el que sale natural y es el
> que rompe el contrato ⑪. Decisión de si se hace: **D21**.

**`best.pt` y `last.pt` dejan de tener el mismo formato, y es a propósito**: responden a preguntas
distintas. `best.pt` es el **entregable** —lo que F carga, lo que contrato ④ exige que se describa
solo— y **no cambia**: meterle estado de optimizador engordaría cada checkpoint que sirve inferencia
con datos que la inferencia no mira. `last.pt` es el **punto de guardado**, y lleva todo lo que hace
falta para que reanudar sea *bit-exacto*:

| Clave | Qué | Por qué, si falta |
|---|---|---|
| `model` | Los pesos | — |
| `config` | La red, como en `best.pt` (contrato ④) | — |
| `epoch` | La última época **completada** | Se reanudaría desde 0 o se saltaría una |
| `optimizer` | `optimizer.state_dict()` | Adam pierde sus momentos y SGD su inercia (`momentum=0.9`): la trayectoria **cambia** |
| `scheduler` | `scheduler.state_dict()`, o `null` si `scheduler: "none"` | El lr reanudaría en el valor de la época 0 |
| `rng` | `{torch, numpy}` — los estados de los generadores | Cambian el **barajado** y las **máscaras de dropout** de lo que queda |
| `best_monitor` | El mejor valor de `monitor` visto | La 1ª época tras reanudar se cree la mejor y **machaca un `best.pt` superior** |

**`null` en `scheduler` significa "esta receta no tiene", y es distinto de que falte la clave**
(§2). Un `last.pt` **sin** las claves de reanudación es un checkpoint del formato viejo: no se
reanuda desde él inventando un optimizador a cero —eso es exactamente el fallo silencioso del
contrato ⑪—, se **empieza de nuevo** diciéndolo.

**`metrics.jsonl` es append-only, así que reanudar lo trunca a `epoch` líneas antes de seguir.** Si
no, las épocas de la primera vuelta que ya se habían escrito quedan **duplicadas** con las de la
segunda, y V14 dibuja una curva que retrocede en el tiempo — cierta línea a línea, falsa como curva.

> **`best` es `null` si el monitor no midió nunca — jamás `±inf`** *(fase 4)*. Un centinela infinito
> no es una medición: es su ausencia (§2). Y no sobrevive al viaje: `json.dumps` escribe `Infinity`,
> que **no es JSON válido y ningún navegador puede parsear**, así que un run cuyo monitor no llegó a
> disparar tumbaría el `GET /runs` de *todos* los demás. El camino es real, no teórico:
> `monitor: val_pos_err_px` sobre un val sin esquinas devuelve `None` cada época.

> **Los JSON de un run se escriben con `os.replace`, y en Windows eso exige reintento en los DOS
> lados** *(fase 4)*. `status.json` se reescribe cada época mientras la UI sondea, así que
> `write_text` —que trunca primero y escribe después— deja una ventana real con el fichero vacío: el
> lector que cae ahí ve un `JSONDecodeError` y concluye que el run está **corrupto**, que es justo lo
> que no está. Pasó: `GET /runs/{name}` contestaba **404 «no tiene un config.json legible»** sobre un
> run sano, y un run corriendo podía parpadear a `error`. Lo cazó **un test que fallaba 1 de cada 3
> veces**, no el razonamiento.
>
> La cura es fichero temporal + `os.replace` (atómico), **pero `os.replace` no basta en Windows**:
> Windows no deja reemplazar un fichero que otro handle tiene abierto, y CPython abre para leer sin
> `FILE_SHARE_DELETE`. Medido en este repo, un lector y un escritor peleándose 4 s dieron **5111
> `os.replace` fallidos y 1130 lecturas fallidas** — el escritor moría *dentro del hilo del
> entrenamiento*. Por eso `write_json_atomic` y `read_text_retrying` reintentan **con deadline** (5 s,
> invisible al lado de una época de 20 s): la pregunta no es «¿cuántas veces lo he intentado?» sino
> «¿ha habido ya un hueco?». **El patrón de POSIX no porta**, y el `.tmp` se borra si aun así falla.
>
> `metrics.jsonl` es la excepción y no necesita nada de esto: es **append-only**, así que nunca se
> reemplaza. Lo que sí necesita es que el lector **descarte la última línea si está a medias** — un
> run vivo se lee mientras se escribe, y una línea rota ahí es lo normal, no corrupción.

> **Por qué la parada es un fichero y no un evento en memoria** *(fase 4)*. El estado de un run es
> del run (§4.2), y eso vale igual para lo que se le pide: así el CLI se para como el API, y una
> parada sobrevive a un reinicio — que en CPU, con runs de horas, pasa. Es **cooperativa**: la marca
> se lee al **final de la época**, que es el punto seguro (métricas escritas, checkpoint guardado).
> No se mata el hilo; matarlo a mitad de batch dejaría un `last.pt` a medias. Se versiona con el
> resto de la descripción (§5): dice *quién* pidió parar y *cuándo*, que `summary.cancelled` no dice.

#### 4.2.1 `provenance` — la forma exacta

*(D2, decidido 2026-07-16.)* Es el contrato ③. Lo escribe la fase 4 y lo lee todo lo demás; el
barrido no existe sin él, porque agrupar por red o por receta **es** esta estructura.

```jsonc
{ "provenance": {
    "patch_dataset": {"name": "…", "fingerprint": "sha256:…"},
    "network":       {"name": "…", "value": { … }},   // nombre para agrupar, valor para reproducir
    "recipe":        {"name": "…", "value": { … }},
    "sweep":         null,                            // o el nombre del barrido padre
    "git_commit":    "…",
    "environment":   {"python": "3.12.10", "torch": "2.13.0+cpu", "platform": "win32"}
} }
```

El **nombre y el valor van los dos, y no es redundancia** — es justo lo que el contrato ③
descubrió que faltaba: el valor reproduce, el nombre agrupa. Con solo el valor hay que comparar
diccionarios a mano para preguntar *"¿qué runs usaron la red X?"*, que es la pregunta que un
barrido hace todo el rato.

`environment` cierra el hueco que `git_commit` deja: el commit fija **el código**, no **el
intérprete**. Cambiar de torch mueve los resultados sin mover el commit, y al llegar la GPU
cambia entero (contrato ⑩). Los runs de CPU de hoy son exactamente los que se compararán con los
de GPU mañana.

**Ningún campo se rellena si falta** (§2): sin git, `git_commit` lleva la razón, no `null`.

- **El checkpoint es autodescriptivo** (contrato ④): lleva la config entera, así que
  `load_model()` reconstruye la red sin ningún YAML. **Es la mejor propiedad del formato** y no
  se toca.
- **`metrics.jsonl` es append-only** y se lee **incrementalmente** (`?since=N`, R5 de api.md).
  Nunca se reescribe: un run vivo se está leyendo mientras se escribe.
- **No hay lector que degrade, y es una simplificación de D18**: `runs/` está vacío, así que
  **todo run nace en la fase 4 con la procedencia completa**. Un `config.json` sin ella no es un
  caso legado: es un run corrupto, y se falla con la razón (§2). El camino de degradación
  (`name: null`) que este documento pedía **murió con D3** — era código de migración para datos
  que ya no existen.
- **El estado es explícito, desde el primer run**: `status.json`, no deducido de qué ficheros
  hay. Deducirlo es lo que dejaba un crash en "running" para siempre (organizacion.md §3).

### 4.3 `configs/` — las definiciones (C y D)

`configs/networks/*.yaml` (C) y `configs/recipes/*.yaml` (D). YAML, no JSON: **los escribe una
persona**. Ambos con `format_version`.

Contraste deliberado con todo lo demás: **estos se versionan en git y no son artefactos**. Son
fuente.

> **`networks/`, no `models/`** *(fase 3, 2026-07-16)*. Este documento decía `configs/models/`
> mientras glosario.md §1 fijaba «**no se usa "model" a secas, nunca**» y api.md R2 hacía
> desaparecer `/models` del vocabulario a propósito — la palabra es ambigua: significa C o E según
> quién hable. Era una contradicción entre documentos, no una decisión: el directorio estaba
> **vacío** (C era «hoy huérfano», que es justo lo que la fase 3 arregla), así que el renombrado
> costó cero. Lo lee `Settings.networks_root`.

**`format_version` es del fichero, no de la red.** Al congelar un C dentro de `runs/<name>/config.json`
se le quita: si viajara dentro, quedaría fosilizado en el checkpoint y en la procedencia, donde no
significa nada. Lo hace `itf-train` antes de construir el `RunSpec`.

### 4.4 La tabla por patch (E×B) — **un caché**, no un artefacto

*(D1, decidido 2026-07-16.)* Es **función pura de (run, huella de B, split, knobs por defecto)**,
y los cuatro ya tienen identidad ⇒ **se puede recalcular exacta ⇒ es un caché**, por el mismo
criterio de §5. No se nombra, no se lista, no se versiona, no tiene CRUD ni pantalla — y las
cuatro vistas de diagnóstico (V6–V9) salen igual.

- **Clave**: `(run, fingerprint de B, split, checkpoint + su mtime, knobs)`. Si cambia
  cualquiera, se recalcula.
- **Ubicación**: `data/cache/diagnostics/`, gitignoreado. Borrarlo no pierde nada.
- **Coste de rehacerla**: segundos (~10⁴ forwards por lotes). Por eso puede ser síncrona (R3 de
  api.md). **Medido** (fase 5, 980 patches de val): **1,0 s** la primera vez, **0,025 s** leyendo
  el caché, **0,014 s** otro agregado sobre la misma tabla.

> **El `mtime` del checkpoint entra en la clave, y este documento no lo pedía** *(fase 5)*. «Run»
> solo identifica una tabla si un run es **inmutable**, y no lo es mientras entrena: `best.pt` se
> reescribe en cada época que mejora. Sin el mtime, abrir Diagnóstico en la época 5 y otra vez en
> la 20 contesta **la tabla de la época 5 las dos veces**, sin que nada chirríe. El caché de
> modelos del código viejo se invalidaba por mtime por exactamente esta razón (organizacion.md
> §2-④).

> **`knobs` está vacío, y a propósito** *(fase 5)*. El único que podría entrar es `threshold`, y
> **no debe**: se aplica al agregar, sobre los scores guardados, que es lo que hace que V8 sea
> gratis. Metido en la clave, el caché se re-keyaría en cada punto de la curva y V8 costaría una
> pasada por umbral — o sea, justo las horas de CPU que esta tabla existe para no gastar. El campo
> está para que un knob futuro que **sí** cambie los números tenga dónde ir: no encontrarlo es
> como no se acabaría añadiendo.

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
  **Matiz desde D19**: ahora también lo *escribimos* (§4.6). Eso no nos hace dueños del formato —
  nos hace un segundo productor obligado a seguir al primero. Si el generador cambia su esquema,
  las derivadas viejas quedan igual de rotas que los originales viejos, y está bien así: la
  alternativa es un formato nuestro que diverge en silencio.
- **El storage del barrido** es de `optuna` (SQLite). Nuestro es `sweeps/<name>/spec.json` (lo
  fijo, el espacio, el objetivo, el presupuesto). **La frontera importa**: los `trials` de optuna
  no son nuestros runs; un trial lanza un run y guarda su referencia (librerias.md).

### 4.6 `data/sources/<name>/` — la fuente derivada (A′)

*(D19, 2026-07-18.)* Lo que escribe el resize. **Es el mismo formato de A** — `labels.jsonl`,
`dataset.json`, las imágenes — porque el consumidor es el mismo `SourceDataset` y no hay ninguna
ventaja en que no lo sea.

```
data/sources/<name>/
├── dataset.json      # metadatos + el bloque `derived` (abajo)
├── labels.jsonl      # una línea por muestra, mismo esquema que A
└── images/…          # las imágenes redimensionadas
```

**Cruza la frontera que acaba de describir §4.5, y conviene decirlo en voz alta**: hasta hoy solo
*leíamos* `labels.jsonl`. Ahora lo producimos. Eso no nos hace dueños del formato — nos hace un
**segundo productor obligado a seguir al primero**. La regla que lo mantiene sano:

> **Los campos que no consumimos se copian tal cual y no se inventan si faltan** (§2). No
> "mejoramos" el esquema: una derivada que no se lea con el mismo parser que un original es una
> derivada rota. Si el generador cambia su esquema, las derivadas viejas quedan igual de rotas
> que los originales viejos — y está bien así: la alternativa es un formato nuestro que diverge
> en silencio.

**Y aquí hay una excepción que no es opcional: "tal cual" no puede incluir píxeles.** El formato
anida geometría que nosotros **no leemos** — `blocks[].box`, `blocks[].lines[].quad`,
`lines[].words[].box` (SAMPLE_FORMAT.md §3.1) — y copiarla sin escalar produciría un dataset con
el `quad` a la resolución nueva y el `box` a la vieja. **Cargaría sin quejarse y dibujaría mal.**

> **El resize es todo o nada**: si no puede mover *todas* las coordenadas, no debe mover ninguna.
> Se reescalan `quad` y `box` **donde aparezcan, a cualquier profundidad**, y el recorrido es
> recursivo a propósito: una versión que solo mirase `labels.blocks` sería correcta en
> `clear-paragraphs` (que no trae `lines`) y silenciosamente incorrecta en `mixed-layout` — que es
> el peor sitio donde estar en lo cierto por casualidad.

Esto **amplía** lo que §4.5 dice que especificamos: como productores dependemos de dos campos más
(`box` y el anidamiento), no solo de los que consumimos como lectores. Es el precio de escribir el
formato de otro, y está aquí escrito para que se vea.

El bloque propio, y es el único añadido:

```jsonc
// dataset.json
{ "id": "clear-paragraphs-02-reducidos-w80",
  "derived": {
    "from": "clear-paragraphs-02-reducidos",       // el id DIRECCIONABLE del padre
    "from_declared_id": "clear-paragraphs-02-8ea1ac04",  // lo que el padre dice de sí mismo
    "op": "resize",
    "request": {"width": 320},               // lo que se pidió: width XOR height
    "size": [320, 240],                      // lo que salió
    "scale": [0.5, 0.5],                     // sx, sy REALES (out/in), no el factor pedido
    "resample": "lanczos",                   // "nearest" para las máscaras
    "created": "2026-07-18T…" } }
```

- **`from` y `from_declared_id` son dos campos porque no coinciden, y eso está medido.**
  `clear-paragraphs-02-reducidos` y `clear-paragraphs-02-8ea1ac04` **declaran el mismo `id`
  dentro de su propio `dataset.json`** — la reducida se quedó con el de la grande. Son justo las
  dos fuentes de la trampa del 14,5× de área (organizacion.md §3), así que fiarse del id
  declarado hace que `from` nombre **al padre equivocado**, en silencio y precisamente en el caso
  que el proyecto ya sabe que es peligroso. `from` es el id **direccionable** (lo que escribirías
  en `--source`); `from_declared_id` es la palabra del generador, guardada aparte. *Juntarlos es
  lo que perdía la información.*
- **`scale` son dos números y sale de la imagen resultante**, no del factor pedido
  (organizacion.md §1-A′). Es el dato con el que se reescalaron los quads; escribir el pedido
  sería documentar la intención en lugar del hecho.
- **`derived` ausente ⇒ es una fuente original**, no "una derivada de la que no se sabe el
  padre". Es la afinación de §2: *dato* ausente ≠ *declaración* ausente. Aquí la ausencia
  **significa** algo, y por eso es legal.
- **Una derivada de una derivada encadena**: `from` apunta al padre inmediato. No se aplana el
  historial, porque aplanarlo obligaría a recomponer escalas, y es ahí donde se pierde el
  redondeo.

**Qué NO lleva**: nada de patches, nada de `n`. A′ es A — no sabe que la CNN existe.

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
