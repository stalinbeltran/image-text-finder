# Organización del proyecto por dominios

Separa qué es **la red**, qué es **el dato**, qué es **el entrenamiento** y qué es **el
modelo entrenado**; y dónde se tocan. Pensado como base para reorganizar la UI.

Regla de lectura: cada dominio tiene **identidad propia** (algo que se puede nombrar,
listar, borrar) y **una sola razón para cambiar**. Donde dos dominios comparten un valor,
eso es un **contrato** y está listado abajo con nombre y número.

---

## 1. Los dominios

| # | Dominio | Qué es | Identidad | Código | Almacén |
|---|---------|--------|-----------|--------|---------|
| **A** | **Fuente** | Imágenes + geometría de párrafos. Lo produce *otro* proyecto | `id` = ruta relativa a `DATASETS_ROOT` | `datasets/loader.py` | externo, solo-lectura |
| **B** | **Dataset de patches** | El dato que la CNN consume de verdad | `<name>` = subdir | `patches/extract.py`, `patches/dataset.py` | `data/patch-datasets/<name>/` |
| **C** | **Red** | La arquitectura. Config puro, cero datos | `<name>.yaml` | `models/builder.py`, `models/heads.py`, `models/store.py` | `configs/networks/*.yaml` |
| **D** | **Receta** | Los hiperparámetros que definen el resultado | `<name>.yaml` | `training/recipe.py`, `training/loop.py`, `training/losses.py` | `configs/recipes/*.yaml` |
| **E** | **Run** | El modelo entrenado: pesos + métricas + procedencia | `<name>` = subdir | `training/loop.py` escribe, `api/app.py` lee | `runs/<name>/` |
| **H** | **Barrido** | Un espacio de D explorado con B y C fijos → muchos E | `<name>` | **no existe** | **falta** |
| **F** | **Inferencia** | Aplicar un E a una imagen completa | — (es una operación, no una cosa) | `inference/predict.py` | — |
| **G** | **Vocabulario** | Lo que *todos* comparten: nombres de esquina/borde y la geometría de la ventana | — | hoy disperso (§3) | — |
| **X** | **Ejecución** | `device`, `num_workers`, concurrencia. **Cuesta tiempo, no cambia el resultado** | — (transversal) | `api/jobs.py` | en memoria (§3) |

Tres observaciones que ordenan todo lo demás:

- **D es un sustantivo** (decisión de este proyecto, por el barrido de hiperparámetros): una
  receta se nombra, se guarda, se compara y se reutiliza. **Desde la fase 3 lo es de verdad**:
  `configs/recipes/*.yaml`, `RecipeStore`, `/recipes` y su pantalla. Antes solo vivía aplanada
  dentro de `runs/<name>/config.json`.
- **F sí es un verbo**: no tiene almacén ni identidad; es una llamada sobre un E. En la UI es
  un panel de resultados, no una entidad listable.
- **C también es un sustantivo, y desde la fase 3 se comporta como tal**: `configs/networks/*.yaml`,
  `NetworkStore`, `/networks` (+ `/networks/validate`) y su pantalla, que la UI **sí** llama. Antes
  el almacén y los endpoints existían pero estaban muertos —`web/src/api.ts` no tenía un solo
  método— y la arquitectura solo existía incrustada en el formulario de entrenamiento y congelada
  dentro de cada run.

**X (ejecución) es el eje que hace falta separar antes de que llegue la GPU.** `device` y
`num_workers` viven hoy dentro de `RunConfig`, mezclados con la receta y congelados en
`config.json`. Consecuencia: la *misma* receta entrenada en CPU y en GPU produce dos
`config.json` distintos y parece dos recetas distintas. Cuando llegue la GPU, eso rompe
cualquier comparación con lo ya entrenado en CPU.

### A — Fuente

`SourceDataset` → `Sample` (index, width, height, has_overlap, image_path, blocks) →
`Block` (block_id, kind, angle, `quad` (4,2) horario desde TL).

- **Define**: la verdad de campo. Nada aquí depende de la CNN.
- **No posee**: nada de patches. No sabe qué es `n`.
- **Ojo**: vive **fuera del repo** (`ITF_DATASETS_ROOT`, por defecto apunta a
  `image-text-sample-generator`). Es una dependencia externa disfrazada de carpeta local.

#### A′ — la fuente derivada (resize)

*(D19, 2026-07-18.)* Una fuente redimensionada **es una fuente**: imágenes + geometría, mismo
`SourceDataset`, mismo `labels.jsonl`. No es un dominio nuevo y no merece uno — lo único que
cambia es **quién la escribió**.

Y eso es justo lo que hay que resolver, porque A es externa y solo-lectura:

- **Dos raíces, no una.** La externa (`ITF_DATASETS_ROOT`) se sigue leyendo y **nunca se
  escribe**; las derivadas van a `data/sources/<name>/`, que es nuestro y está gitignored como
  el resto de `data/`. `discover_sources` recorre las dos y el `id` lleva prefijo de raíz, así
  que una derivada no puede sombrear a una original por colisión de nombre.
- **Solo reduce.** Ampliar devuelve `400`. Interpolar un render sintético no añade información:
  añade interpolador. Un B extraído de una fuente ampliada mediría LANCZOS, no el modelo — y el
  fallo sería silencioso, que es el patrón de §3.
- **La proporción se mantiene por construcción**: la entrada es **el ancho o el alto**, nunca
  los dos. La otra dimensión se deriva.
- **Dos factores de escala, no uno.** Los quads se escalan con `sx = out_w/W`, `sy = out_h/H`
  medidos de la imagen **ya redimensionada**, no con el factor pedido. Con un solo factor, el
  redondeo del lado derivado dejaría la geometría desplazada respecto al píxel — poco, y por
  eso peligroso.
- **La máscara, si la hay, va con NEAREST.** Interpolar una máscara de etiquetas fabrica clases
  que no existen. Es `ausente ≠ cero` (formatos.md §2) en versión continua.

**La separación que importa, y es lo que pidió el encargo**: el mecanismo de píxeles
(`itf.imageops`, sin un solo import de `itf`) **no sabe qué es un quad**, y el de coordenadas
(`itf.geometry.scale_quad`) **no sabe qué es un fichero**. La fuente derivada es la composición
de los dos. Por eso el primero sirve tal cual para una imagen cualquiera de prueba, que es la
razón de que estén separados y no una elegancia.

**No se extrae como librería** (librerias.md §0: *se extrae en la segunda vez*): existe en un
solo proyecto. Se escribe library-shaped y se anota como candidato.

### B — Dataset de patches

Lo definen sus parámetros de extracción: `source`, `patch_size` (n), `stride`,
`target_kinds`, `drop_overlap`, `split{train,val,test}`, `seed`.

Produce `patches.npz` con seis arrays paralelos:

| array | forma | qué es |
|---|---|---|
| `X` | (N, n, n, 1) uint8 | **el patch: la entrada real de la CNN** |
| `y` | (N, 4, 3) float32 | etiqueta `[exists, x, y]` por esquina |
| `border` | (N, 4) uint8 | flags de borde `top,right,bottom,left` |
| `sample_idx` | (N,) int32 | de qué imagen de A salió |
| `patch_xy` | (N,) int32 ×2 | dónde estaba en esa imagen |
| `split` | (N,) int8 | 0 train / 1 val / 2 test |

Más `manifest.json` (procedencia: `source_id` + config completa + conteos) y `split.json`
(índices de A por split).

- **Define**: `n`. Este es *el* número que ata B con C (contrato ①).
- **Es autocontenido**: tras construirse, el `.npz` tiene los píxeles; ya no necesita a A
  para entrenar. Solo la UI de Predict vuelve a cruzarlo con A (contrato ⑤).
- **`split` es por imagen, no por patch** (`_assign_splits` baraja *samples*): patches de la
  misma imagen nunca caen en splits distintos. Correcto, y no evidente.

### C — Red

`ModelConfig`: `input_size`, `in_channels`, `border_features`, `backbone[]`, `head{}`.
Cada bloque del backbone: `filters, kernel, stride, padding, batchnorm, activation, pool,
dropout`. La cabeza (`CornerHead`) saca **(B, 4, 3)** — 4 esquinas × `[exists, x, y]`.

- **Define**: qué transformación se aplica al patch. Nada más.
- **No posee**: pesos (eso es E), ni lr/epochs (eso es D), ni de dónde salen los patches.
- **Es config puro**: `build_model(dict)` construye desde un diccionario. Se puede listar,
  comparar y versionar sin tocar un dataset ni entrenar nada. **Esa propiedad es la que la
  UI hoy desaprovecha.**

### D — Receta de entrenamiento

Pérdida: `L = Σ_c [ BCE(exists_c) + λ · exists_c · smoothL1(x_c, y_c) ]`.

**Define**: el ajuste, no la red. Cambiar `lr` no cambia la red; cambiar `filters` sí.

**El criterio de pertenencia a D**: *si cambiarlo cambia los pesos resultantes, es D.* Si
solo cambia cuánto tarda, es X. Si cambia la forma del modelo, es C. Tres fronteras finas
que conviene fijar ahora, porque en un formulario se mezclan solas:

- `lambda_pos` y `pos_weight` **son D** (pesan términos de la pérdida), aunque suenen a
  arquitectura.
- `dropout` y `batchnorm` **son C** (cambian el módulo), aunque suenen a entrenamiento.
- `border_features` **es C** (cambia la forma de la cabeza), aunque suene a dataset.

#### Catálogo de hiperparámetros

**Todo lo de aquí existe desde la fase 3** en `itf.training.recipe.Recipe`, con su definición al
lado (un hiperparámetro sin definición en el catálogo no está terminado) y en la pantalla Recetas,
que enseña esa misma definición **en línea**. Las dos únicas ausencias son `augment` y `sampler`, y
**son decisiones, no huecos**: llevan su razón abajo.

Lo que decía la columna antes de la fase 3: `hoy` = existía en `RunConfig`; **`falta`** = no
existía. Se conserva porque **el «falta» era casi siempre un *default* que nadie eligió**, y esa es
la parte que hay que recordar: `momentum` a 0 y `smooth_l1_beta` a 1.0 no se decidieron, se
heredaron. Hoy están puestos a propósito en `configs/recipes/baseline.yaml`.

**Optimización** — cómo se da cada paso.

| Hiperparámetro | | Definición | Notas para el barrido |
|---|---|---|---|
| `lr` | hoy | Tamaño del paso del optimizador. | El más influyente. Barrer en **escala log** (p. ej. 1e-4…3e-2), nunca lineal. |
| `optimizer` | hoy | `adam \| adamw \| sgd \| rmsprop`. | Ver aviso de `momentum` abajo: hoy la comparación está sesgada. |
| `momentum` | **falta** | Inercia del paso; solo aplica a `sgd`/`rmsprop`. | **`_make_optimizer` solo pasa `lr` y `weight_decay`** → SGD corre con `momentum=0`. Barrer `optimizer` hoy compara Adam contra un SGD lisiado: siempre perderá, y no por serlo. Típico: 0.9. |
| `weight_decay` | hoy | Penalización L2 sobre los pesos; regulariza. | Log, 0…1e-2. En `adam` es L2 acoplada; en `adamw`, desacoplada (no son lo mismo aunque el campo se llame igual). |
| `batch_size` | hoy | Muestras por paso de gradiente. | **Acoplado a `lr`**: doblar el batch suele pedir subir el lr. Barrerlos juntos o fijar uno. Ojo al pasar a GPU (contrato ⑩). |
| `grad_clip` | **falta** | Recorta la norma del gradiente antes del paso. | Barato y estabiliza lr altos. Sin él, el extremo alto del barrido de `lr` diverge y no aprendes nada de esa zona. |

**Duración y programación** — cuánto y cómo se recorre.

| | | Definición | Notas |
|---|---|---|---|
| `epochs` | hoy | Pasadas completas sobre el train. | Hoy **siempre corre las N**: no hay parada temprana. En un barrido en CPU, esto es donde se quema el tiempo. |
| `scheduler` | **falta** | Cómo decae `lr` con las épocas (`none\|cosine\|step\|plateau`). | **La omisión más cara.** Hoy `lr` es constante. Con schedule, el `lr` inicial óptimo cambia — barrer `lr` sin schedule optimiza para un régimen que luego no usarás. |
| `warmup_epochs` | **falta** | Épocas subiendo `lr` desde ~0 al inicial. | Solo importa con batch grande o lr alto. Baja prioridad hasta que haya GPU. |
| `patience` / `min_delta` | **falta** | Parada temprana: épocas sin mejorar antes de cortar. | Per-run, mirando *su propia* curva de val. **Distinto del pruning del barrido** (contrato ⑨). |

**Pérdida — específicos de esta tarea.** Aquí está lo propio del problema:

| | | Definición | Notas |
|---|---|---|---|
| `lambda_pos` | hoy | Peso del término de posición (smoothL1) frente al de existencia (BCE). | Arbitra **"¿hay esquina?" vs "¿dónde exactamente?"**. Es el hiperparámetro más interesante del proyecto y el que más cuidado pide al rankear (contrato ⑨). |
| `pos_weight` | hoy | Peso de la clase positiva en la BCE. | El desbalance real, medido en `clear-paragraphs-02`: **20,5 % de positivos ⇒ 3,9:1**. Modesto, no brutal. Empieza el barrido en ese ratio y sácalo de `manifest.positives_per_corner / num_patches`, que cambia por dataset. |
| `smooth_l1_beta` | **falta** | Umbral donde smoothL1 pasa de cuadrático a lineal. | **Hoy es el default de PyTorch, `beta=1.0`, y las coords van normalizadas a [0,1]** → `\|error\| < 1` *siempre* → nunca se sale de la rama cuadrática. **La pérdida de posición es MSE pura; la robustez de Huber jamás se activa.** Para que haga algo: ~0.05–0.1 (≈2–4 px en un patch de 40). |

**Datos en tiempo de entrenamiento** — **no existen, y su ausencia es una decisión de la fase 3**,
no un olvido. `PatchDataset` devuelve el tensor crudo.

| | | Definición | Notas |
|---|---|---|---|
| `augment` | **fuera a propósito** | Perturbaciones aleatorias del patch en cada época. | **Trampa grave, ver aviso abajo.** Implementarlo mal es **peor que no tenerlo**: entrena, la loss baja, y aprende esquinas cambiadas sin que nada avise. Entra el día que alguien remapee las 4 cabezas **y** los 4 flags de borde **con su test**, no como añadido al cierre de una fase. |
| `sampler` | **fuera a propósito** | `random` vs muestreo balanceado de patches con/sin esquina. | Alternativa a `pos_weight` para el mismo problema; **elegir uno**, no ambos, o el desbalance se corrige dos veces. Como `pos_weight` ya está y **el desbalance real es modesto (3,9:1)**, no es urgente: entra si el barrido de `pos_weight` toca techo. |

> **Aviso sobre augmentation.** Los flips y las rotaciones de 90° **no son válidos aquí sin
> reetiquetar**: la etiqueta es por *tipo* de esquina. Un flip horizontal convierte una TL en
> una TR visualmente, así que habría que permutar `TL↔TR`, `BL↔BR` **y** hacer `x → 1-x`. Un
> `RandomHorizontalFlip` puesto sin pensar enseña basura y el fallo es silencioso: entrena,
> baja la loss y detecta las esquinas cambiadas. Además invalidaría los flags `border`
> (contrato ②), que son direccionales. Lo seguro sin tocar etiquetas: brillo, contraste,
> ruido. Las traslaciones exigen recalcular `x,y` **y** los flags de borde.

**Selección y reproducibilidad.**

| | | Definición | Notas |
|---|---|---|---|
| `monitor` | **hoy fijo** | Qué métrica elige `best.pt`. | Hardcoded a val loss (`loop.py:166`). Debe ser explícito, y **no puede ser la loss si barres λ** (contrato ⑨). |
| `seed` | hoy | Semilla de init de pesos + shuffle. | **No es un hiperparámetro a optimizar: es el eje de réplica.** Fijar todo y variar solo `seed` mide el ruido, que es lo único que te dice si una diferencia entre dos recetas es real. Distinto del `seed` de B, que fija el split (contrato ⑧). |

#### Los knobs baratos: no son D, y no requieren reentrenar

`threshold`, `stride` de inferencia, el radio de NMS y `min_size` de reconstrucción son de
**F**, y se ajustan **post-hoc sobre un modelo ya entrenado**. Barrerlos cuesta minutos, no
horas: se hace sobre el val de un solo run.

Distinguirlos importa **más en CPU que en GPU**: si `threshold` acaba metido en el barrido de
D, multiplicas horas de entrenamiento por algo que podías haber ajustado gratis al final.
Regla: **barre D entrenando; barre F evaluando.**

### E — Run

`runs/<name>/`: `config.json` (congelado), `metrics.jsonl` (una línea por época, pollable),
`best.pt`, `last.pt`, `summary.json`.

`RunConfig` = **`data` (→B) + `model` (→C) + hiperparámetros (D)**, aplanados en una sola
dataclass. Es el punto de encuentro de tres dominios, y por eso concentra los contratos.

- **Estado derivado del disco**: `_run_status()` deduce running/done por qué ficheros
  existen (`summary.json` → done, `config.json` → running). No hay estado explícito: un run
  que crashea queda "running" para siempre.
- El checkpoint es **autodescriptivo**: guarda `config` entera, así que `load_model()`
  reconstruye C sin consultar ningún YAML (contrato ④).

### H — Barrido

**No existe todavía.** Es el dominio que justifica que D sea un sustantivo: un barrido es
*un espacio de recetas explorado con todo lo demás fijo*.

Lo define:

- **Lo fijo**: un B y un C concretos. Sin esto no hay comparación posible (contrato ⑧).
- **El espacio**: qué campos de D varían y en qué rango/lista. `lr: log(1e-4, 3e-2)`,
  `optimizer: [adam, adamw]`, …
- **La estrategia**: `grid` (exhaustiva) | `random` (muestreo) | secuencial/bayesiana.
  Con presupuesto igual, **random gana a grid** en cuanto un parámetro importa mucho más que
  los otros — que es justo el caso aquí con `lr`: grid gasta el presupuesto reevaluando los
  irrelevantes.
- **El objetivo**: la métrica escalar que ordena los runs, y su dirección (contrato ⑨).
- **El presupuesto**: nº de puntos, épocas por punto, y si se poda.
- **Los hijos**: los E que genera, cada uno con su punto del espacio.

Un barrido **es un sustantivo de primera clase**: se nombra, se guarda, se reanuda y se
compara. En CPU puede durar horas; su estado **no puede vivir en memoria** (§3).

**Coste, con números reales de este repo.** `runs/cnn-02-01` marca ~20 s/época en
`clear-paragraphs-02`. Un barrido de 20 puntos × 20 épocas = 400 épocas ≈ **2,2 h en CPU,
secuencial**. Es tolerable — pero escala lineal con las tres cosas a la vez, así que un
dataset 4× mayor con 40 puntos ya son ~18 h. Dos palancas, por orden de rentabilidad:

1. **Poda** (contrato ⑨): la mayoría de puntos se ven malos en la época 3. Cortarlos ahí
   recorta el barrido a la mitad o menos. Es lo más rentable en CPU y hoy no existe nada.
2. **Concurrencia**: en CPU **no ayuda** — torch ya usa todos los núcleos en un run, así que
   N entrenamientos a la vez se pelean por los mismos cores y cada uno va ~N× más lento
   (§3). El límite de workers en CPU es **1**.

### F — Inferencia

`detect_corners` (ventana deslizante → NMS por distancia) → `reconstruct_boxes` (emparejado
voraz TL→BR) → `{corners, paragraphs, image_size}`.

- **Define**: cómo se pasa de "predicciones por patch" a "párrafos en la imagen".
- `stride` y `threshold` son de F, **no de B**: son de tiempo de uso, se eligen por llamada.
  Que `stride` también exista en B (contrato ⑤) es una coincidencia de nombre peligrosa.
- `reconstruct_boxes` es una **heurística**, no red: mejora sola si el modelo mejora, y es
  el sitio donde tocar si los párrafos salen mal pero las esquinas salen bien.

---

## 2. Dónde interactúan (los contratos)

### ① B ↔ C — `patch_size == input_size` ← **el contrato crítico**

La CNN se construye con `input_size` y calcula su dimensión aplanada con un tensor dummy de
ese tamaño (`_infer_flat_features`). Si los patches reales miden otra cosa, el flatten da
otra dimensión y el `Linear` de la cabeza revienta.

- **Se declara** en dos sitios independientes: `manifest.config.patch_size` (B) y
  `model.input_size` (C). Nada los une salvo la buena voluntad.
- **Se valida hoy solo en el front**: `RunsPanel.tsx:14` marca datasets `✗ incompatible`, y
  `TrainPanel.tsx:23-24` autorrellena `input_size` desde el manifest.
- **El backend no lo valida.** `POST /runs` acepta el par sin mirar. Un mismatch no falla al
  entrar: falla **dentro del hilo del job**, a mitad del primer batch, con un
  `mat1 and mat2 shapes cannot be multiplied`. El job queda en `error` con un mensaje que no
  dice nada del verdadero problema.
- **Al mover a la UI**: si B y C pasan a ser dos pantallas separadas, esta validación deja de
  ser cosmética y pasa a ser lo único que impide combinaciones imposibles. Debería vivir en
  el backend (400 con la razón), no en un `.tsx`.

**① y ② son la misma pregunta, y comparten validador** — ver el recuadro tras ②.

### ② B ↔ C — `border_features`

B escribe `border` (N,4). C decide si lo usa: con `border_features: true`, la cabeza recibe
`flat + 4`. Es un contrato **opcional y unidireccional**: el dataset ofrece, la red decide.

**Pero sí hay mismatch posible, y está vivo**: si el `.npz` es anterior a la feature y **no
trae** el array, `PatchDataset` **rellena ceros** — y cero significa "no toca ningún borde", no
"no se sabe". Con `border_features: true` la red entrena creyendo que ninguna ventana toca un
borde, y en inferencia `detect_corners` le mete los flags reales: **ve una distribución que nunca
entrenó, justo en los bordes**. Sin excepción. Los tres `.npz` del repo están en ese caso y el
ejemplo del README pide `border_features: true` ([formatos.md](formatos.md) §2).

(Ligado: `in_channels: 1` ↔ `patch_shape: [n, n, 1]`.)

---

> ### ① y ② son la misma pregunta: un solo validador
>
> | | Lo que **ofrece** B | Lo que **necesita** C |
> |---|---|---|
> | ① | `patch_size: 40` | `input_size: 40` |
> | ② | tiene `border` / no | `border_features: true` |
> | (③) | `patch_shape: [n,n,1]` | `in_channels: 1` |
>
> Las tres son **"¿puede esta red entrenar sobre este dataset?"**: mismo momento (antes de
> entrenar), mismo sitio (donde B y C se encuentran), mismo tipo de error (400 con razón). **No
> hace falta un mecanismo para `border`**: hace falta el validador que ① ya pedía, y ② entra
> gratis.
>
> **`itf/validation.py` — función pura de dos diccionarios** (manifest × config de red). Sin
> torch, sin entrenar, milisegundos. Devuelve una lista de problemas con `code`, `message` y
> `hint` (R4 de [api.md](api.md)).
>
> **Se llama dos veces, a propósito**: en `POST /runs` → **400 antes de crear el job**; y dentro
> de `train()` como red de seguridad, porque **`itf-train` no pasa por el API** y sin eso el CLI
> se salta la puerta. Cuesta nada: es pura. Es el patrón del proyecto hermano (`validation.py`
> usado por el manager *y* por el bucle).
>
> Que sea pura y barata **es la prueba de que está en la capa correcta**: si validar exigiera
> entrenar, la validación estaría en el sitio equivocado ([tests.md](tests.md) §3).

### ③ B + C + D → E — referencia vs. copia, y el nombre que se pierde

Al congelar el run:

- **`data` se guarda como ruta** → E *referencia* a B.
- **`model` se guarda como valor** → E *copia* a C.
- **Los hiperparámetros se guardan aplanados** → E copia a D **sin nombre**.

Consecuencias, y son asimétricas:

- Editar/renombrar la arquitectura **no afecta** a un run ya entrenado. Bien: reproducible.
- Borrar o reconstruir el dataset de patches **rompe la procedencia** de todos los runs que
  lo apuntaban: `_run_source()` sigue la cadena `run.config.data` → `manifest.config.source`
  → `source_id`, y si el directorio no existe devuelve `None` en silencio. La pestaña
  Predict pierde el dataset preseleccionado sin decir por qué.
- **E copia el *valor* de C y D, pero no su *identidad*.** No se puede preguntar "¿qué runs
  usaron la red X?" ni "¿qué runs usaron la receta Y?" — hay que comparar diccionarios a
  mano. **Un barrido necesita exactamente esa pregunta** (agrupar, ordenar, deduplicar).

**La regla que sale de aquí, y que H hace obligatoria**: un run debe guardar, de C y de D,
**el valor (copia, para reproducir) *y* el nombre de origen (referencia, para agrupar)**.
Y de B, la ruta **más una huella del contenido** — porque hoy nada distingue un B
reconstruido bajo el mismo nombre (contrato ⑧).

### ⑧ H ↔ B, C — la comparabilidad, y la regla de medir

Un barrido de D **solo es comparable si todos sus runs comparten el mismo B y el mismo C**.
Si a mitad de barrido se reconstruye el dataset de patches con otro `stride`, o se toca la
arquitectura, los puntos dejan de ser comparables entre sí — y **nada en el sistema lo
detecta**: `data` es una ruta, y una ruta reconstruida sigue apuntando igual.

**Generalizado**: *lo que no barres, se queda fijo*. Y de ahí sale la regla que gobierna el
proyecto entero:

> **El instrumento de medida no puede ser parte del experimento.**

Importa porque **los parámetros de B son variables de investigación** (`num_images`, las
fracciones del split — decidido en [protocolo.md](protocolo.md) §3): se van a barrer. Y en cuanto
barres el split, **cada punto tiene un val y un test distintos** ⇒ comparas mediciones hechas con
reglas distintas ⇒ mides la regla, no el modelo. El `test` de hoy no salva nada: **sale del mismo
`_assign_splits` que estarías barriendo**.

La salida es un **holdout** por encima de B: imágenes apartadas una vez que ninguna configuración
de B toca jamás, contra las que se mide todo. Y lo que lo hace viable es que la **F1 de párrafo
se mida por imagen y no por patch** — así el mismo holdout sirve aunque cambien `n`, el `stride`
o las fracciones. Con métricas de patch sería imposible: cambiar `n` cambia qué es un patch, y no
habría nada que comparar.

Lo que hace falta para cerrarlo: que B tenga una **huella de contenido** (hash del `.npz` o
del manifest) registrada en el manifest y copiada en cada run. Entonces un barrido puede
verificar que sus hijos son comparables, en vez de asumirlo.

**El resize añade un eje, y es de los caros de detectar** *(D19)*. Con `n` fijo, redimensionar
la fuente **cambia cuánto texto cabe en un patch**: es la misma palanca que hizo que las dos
`clear-paragraphs-02` dieran 49 y 713 patches por imagen (§3), pero ahora **la accionamos
nosotros a propósito**. Dos consecuencias que hay que sostener a mano:

- **La derivada declara su padre y su escala** en `dataset.json` (`derived_from`, `scale`), y B
  las arrastra a su manifest como cualquier `source_id`. Sin eso, dos B a resoluciones distintas
  tienen procedencias que se leen idénticas salvo por el nombre del directorio — y el nombre no
  es un dato.
- **El holdout (D16) se redimensiona con la misma orden o con ninguna.** Medir un modelo
  entrenado a 320 px contra un holdout a 640 no es un resultado malo: es un resultado que no
  significa nada, y tiene toda la cara de significar algo.

Ojo con los **dos `seed`**, que ya están bien separados y conviene no mezclar:

- el `seed` de **B** fija el reparto train/val/test (`_assign_splits`, por *imagen*);
- el `seed` de **D** fija init de pesos y shuffle.

En un barrido, el de B se queda **fijo** (mismo dataset, mismo split ⇒ comparable) y el de D
es el **eje de réplica**. Confundirlos hace que cada punto se evalúe sobre un split distinto
y el barrido mide ruido de split, no calidad de receta.

### ⑨ H ↔ D — el objetivo, y la trampa de λ

Un barrido necesita **una métrica escalar** que ordene los runs. Hoy no hay tal cosa
declarada: `loop.py:166` elige `best.pt` con `monitor = val_metrics.get("loss", train_loss)`,
hardcoded.

**Y la val loss no puede ser el objetivo de un barrido que varíe `lambda_pos`**:

> `loss = cls_loss + λ · pos_loss`. Si λ es parte del espacio, cada punto se está midiendo
> **con una función de pérdida distinta**. Bajar λ baja la loss *por definición*, sin mejorar
> nada. Un barrido de λ rankeado por val loss "descubre" que **λ=0 es lo mejor** — es decir,
> que lo óptimo es no predecir posiciones. Y el fallo es silencioso: sale un ganador con
> buena cara.

El objetivo tiene que ser **independiente de λ**.

**La respuesta es la F1 de párrafo** ([protocolo.md](protocolo.md) §2), y resuelve este contrato
entero:

- Es **independiente de λ por construcción**: no contiene la pérdida. La trampa de arriba
  desaparece.
- **Integra las dos métricas en tensión.** `f1` de existencia y `pos_err_px` tiran en direcciones
  opuestas —detectar vs. localizar, que es lo que λ arbitra— y por eso parecía hacer falta un
  frente de Pareto o una restricción del tipo *"el mejor `pos_err_px` entre los que tengan
  `f1 ≥ 0.9"*. Con la F1 de párrafo no hace falta: **detectar mal rompe el IoU y localizar mal
  también**. Es un escalar, y es el que de verdad quieres.
- Es **barata** (~10⁴ forwards por lotes sobre val: segundos), así que puede rankear cada punto.

**Pero hoy no existe: no hay ninguna métrica de párrafo en el código** (§3). Hasta que se
escriba, cualquier barrido optimiza un proxy de fidelidad desconocida. Las métricas de patch
siguen valiendo para elegir `best.pt` dentro de un run (baratas, por época) y para diagnosticar
(V7/V8/V9 de ui.md) — **si el paso 2 del protocolo confirma que predicen la F1 de párrafo**.

*(El frente de Pareto — V12 de ui.md — sigue siendo útil para **mirar** qué compra λ. Ya no es lo
que decide el ganador.)*

Relacionado: **parada temprana ≠ poda**. La parada temprana (`patience`, en D) mira la curva
de *un* run. La poda (en H) compara *entre* runs y mata a los que van peor que la mediana a
la misma época. La segunda es la que salva horas de CPU, y necesita que H vea a todos sus
hijos en vivo.

### ⑩ X ↔ D — el device no es una receta (y `batch_size` es la trampa)

`device` y `num_workers` viven hoy **dentro de `RunConfig`** y se congelan en `config.json`.
Cuando llegue la GPU, la misma receta entrenada en CPU y en GPU dará dos `config.json`
distintos: parecerán dos recetas y no lo son. **X debe salir de la identidad de D** antes de
ese momento, o todo lo entrenado hoy en CPU queda incomparable con lo de mañana.

La trampa concreta al migrar: **`batch_size` es D, no X**, aunque sea lo primero que apetece
subir cuando hay VRAM ("ahora caben 512"). Cambiarlo cambia el resultado — y además está
acoplado a `lr`. Subirlo al pasar a GPU **invalida la comparación** con todo el barrido de
CPU. Si se quiere batch mayor en GPU, es un **punto nuevo del espacio**, no el mismo punto
más rápido.

`num_workers` sí es X puro: hoy 0 (Windows); con GPU habrá que subirlo para alimentarla.
No cambia los pesos.

### ④ E → F — el checkpoint se describe solo

`load_model()` lee `ckpt["config"]["model"]` y reconstruye la red. F **nunca** necesita el
YAML de C ni el dataset B. Es la dependencia más limpia del proyecto: un `.pt` es portable.

El caché de modelos (`_MODEL_CACHE`, clave `(run, checkpoint, device, mtime)`) vive en
`api/app.py` y se invalida por `mtime` — y a mano en rename/delete (`_drop_model_cache`).

### ⑤ B ↔ F — **la geometría duplicada**

La ventana deslizante de inferencia debe ser *idéntica* a la de extracción, o el modelo ve
patches con una geometría que nunca entrenó. Hoy eso se sostiene con dos costuras:

1. `inference/predict.py:25` importa **`_positions`** — una función **privada** — desde
   `patches/extract.py`.
2. Los flags de borde se **recalculan a mano** en `predict.py:69-70`, duplicando
   literalmente `extract.py:127-132`.

Si alguien toca una de las dos, la otra no se entera y no hay test que lo note. **Esto es G
(vocabulario compartido) disfrazado de dependencia B→F.** Antes de tocar la UI, esa
geometría debería ser un módulo propio que B y F importen.

### ⑥ A ↔ B ↔ F — el cruce de la pestaña Predict

El único sitio donde A vuelve después de la extracción:
`GET /datasets/{id}/samples?patch_dataset=<B>` anota cada imagen de A con su split leído del
`split.json` de B (`_split_map`), para poder predecir "solo el test". Une los tres dominios
en una sola llamada, y es legítimo: es una vista, no un acoplamiento estructural.

### ⑦ G → todos — las constantes viven en el sitio equivocado

`CORNER_NAMES`, `BORDER_NAMES`, `NUM_CORNERS`, `NUM_BORDERS` están declaradas en
**`datasets/loader.py`** (dominio A). Pero las importan:

- `models/builder.py` (C) → `NUM_BORDERS`
- `patches/dataset.py` y `patches/extract.py` (B) → las cuatro
- `training/loop.py` (D) y `inference/predict.py` (F) → `CORNER_NAMES`

Es decir: **la red importa del cargador del dataset fuente** solo para saber que hay 4
bordes. Dirección equivocada — C no debería saber que A existe. Es vocabulario de proyecto
(G) y quiere su propio módulo.

---

## 3. Trampas: lo que rompió la separación, y volvería a romperla

**Esta lista cambió de significado con el borrado del código (2026-07-16).** Ya no describe lo
que está roto: describe **lo que se rompió una vez y volvería a romperse sola**. Sigue siendo la
sección más útil del documento, y ahora más:

> **Casi todas eran *defaults*.** `smooth_l1_beta=1.0` es el default de PyTorch. `momentum=0` es
> el default de SGD. Meter lógica de dominio en `app.py` es lo que sale natural. Un hilo por job
> es lo que hace `threading.Thread`. **Nadie eligió estos fallos: aparecieron por no elegir.**
> Reconstruir desde cero no protege de ellos — los invita.

Las citas (`jobs.py:67`, `dataset.py:35`…) son del código anterior y resuelven contra el tag
**`pre-rediseno`**. Son la evidencia de que la trampa es real, no una hipótesis.

De mayor a menor. Los marcados **[barrido]** son bloqueantes para H.

1. **C y D no existen como entidades.** — ✅ **arreglada en la fase 3 (2026-07-16).** C tenía
   almacén y endpoints (`configs/models/*.yaml`, `GET/POST /models`) pero estaban **muertos**:
   `web/src/api.ts` nunca los llamaba. D no tenía ni eso. Ninguna de las dos se podía listar,
   nombrar ni reutilizar sin entrenar. **[barrido]**
   *Hoy*: `configs/networks/*.yaml` + `configs/recipes/*.yaml`, `/networks` y `/recipes` con sus
   dos pantallas, y `itf-train --network <nombre> --recipe <nombre>`, que es lo que hace que la
   procedencia por nombre (③) se sostenga sola.
2. **`JOBS` no es una cola: es un `Thread` por job, sin límite.** — ✅ **arreglada en la fase 7.**
   El `JobQueue` nació con `max_workers=1` en la fase 2 (era la trampa, no una feature que se añade
   luego) y la fase 7 le puso persistencia y cancelación encima. Un barrido de 20 puntos = 20
   entrenamientos simultáneos peleándose por los mismos núcleos (torch ya usa todos), cada uno ~20×
   más lento y **cada uno con su `PatchDataset` entero en RAM** — `dataset.py:35` carga todo `X` a
   un tensor float32 residente (40×40 float32 = 6,4 KB/patch ⇒ 100k patches ≈ 640 MB, ×2 por
   train+val, ×N runs). **En CPU el límite es 1.**
3. **El estado de los jobs vive en memoria** (`self._jobs`, hilos daemon). — ✅ **arreglada en la
   fase 7.** Con `persist_dir`, cada transición se escribe y se recarga al arrancar; un job vivo al
   morir el proceso recarga como `interrupted`. Y lo durable de verdad —el barrido— **se reanuda**:
   `spec.json` + `optuna.db` + los runs están en disco, así que el `lifespan` re-encola lo que quedó
   a medias. Reiniciar la API ya no borra la investigación.
4. **No hay cancelar, ni parada temprana, ni poda.** — ✅ **arreglada en la fase 7.** Cancelación
   cooperativa (`cancel` callback + `POST /jobs/{id}/cancel`, corta en el punto seguro), parada
   temprana per-run (`patience`, fase 3) y **poda** entre runs (`optuna` `MedianPruner`, la palanca
   nº1 en CPU): en un barrido de 30 puntos, 14 se podaron.
5. **`POST /runs` sobrescribe en silencio.** **[barrido]** No comprueba si el run existe
   (`retrain` sí: 409), y `train()` hace `mkdir(exist_ok=True)` + trunca `metrics.jsonl`. Un
   barrido que autogenere nombres puede machacar sus propios resultados sin avisar.
6. **La geometría de la ventana está duplicada** entre extracción e inferencia (contrato ⑤),
   una de las dos copias vía import de una función privada.
7. **El vocabulario compartido vive en el dominio A** (contrato ⑦).
8. **El contrato crítico ① se valida solo en el front.** El backend acepta combinaciones
   imposibles y falla tarde y mal.
9. **`RunConfig` aplana cuatro dominios** (B, C, D, X) en una dataclass. El YAML
   (`model.example.yaml`) sí separa `model:` / `train:`, y el CLI los reaplana
   (`cli.py:24-29`); la API también (`TrainRequest`). La separación existe en el fichero y se
   pierde en el código.
10. **El estado de un run es inferido del disco**, no explícito: `_run_status()` mira qué
    ficheros hay, así que un crash deja "running" para siempre. En un barrido de 20, los
    muertos no se distinguen de los vivos.
11. **No existe ninguna métrica de párrafo.** **[barrido]** `evaluate()` es todo a nivel de
    patch. **Nadie ha medido nunca si los párrafos salen bien en la imagen completa** — que es el
    objetivo real del proyecto y, según el contrato ⑨, el objetivo correcto del barrido. Ver
    [protocolo.md](protocolo.md) §2.
12. **El val de `clear-paragraphs-02` son 20 imágenes**, no 980 patches: los patches de una
    imagen están correlacionados, así que el tamaño de muestra efectivo es ~20 y **diferencias
    de f1 bajo ~5 % no son resolubles**. El dato es sintético y el generador está al lado: es una
    elección por defecto, no una restricción (protocolo.md §1.1).
13. **`border` no está en ninguno de los `.npz`, y el ejemplo del README pide
    `border_features: true`.** El relleno silencioso con ceros falsifica el dato y el modelo ve
    en inferencia una distribución que no entrenó ([formatos.md](formatos.md) §2).
14. **Un dataset sin val elige `best.pt` por train loss**, sin avisar (`monitor =
    val_metrics.get("loss", train_loss)`). Le pasa a `reducido-40`, que es el ejemplo del README
    (protocolo.md §1.3).
15. **`smooth_l1_beta` por defecto anula el Huber** (§1, D): la pérdida de posición es MSE
    pura sin que nadie lo haya decidido.
16. **SGD corre sin momentum** (§1, D): cualquier barrido de `optimizer` está sesgado a favor
    de Adam.

---

## 4. Cómo se traslada a la UI

Hoy: `1 · Patches` → `2 · Train` → `3 · Runs` → `4 · Predict`. Cruzado con los dominios:

| Pestaña actual | Dominios que mezcla | Problema |
|---|---|---|
| 1 · Patches | **A + B** | Elegir fuente y definir patches en un solo formulario |
| 2 · Train | **C + D + X** (+ ref a B) | Arquitectura, receta y device indistinguibles en un mismo form |
| 3 · Runs | **E** (+ reabre C y D) | `ModelConfigForm` con `lockArchitecture` ya intuye la separación C/D |
| 4 · Predict | **F** (+ navega A y B) | Correcto: es una vista que cruza |
| — | **C sola** | **No existe** |
| — | **D sola** | **No existe** |
| — | **H** | **No existe** |

El detalle revelador: el modo *"retrain same"* fija la arquitectura y deja editar solo
dataset + hiperparámetros. Eso **ya es** la frontera C↔D, descubierta a mano y resuelta con
una prop (`lockArchitecture`) en vez de con una separación real. La UI ya quiere esta
organización; solo que la implementa con un booleano.

Dirección natural, si se acepta el mapeo un dominio ↔ una pantalla:

- **Fuentes** (A) — solo lectura: qué datasets hay, cuántas muestras, ejemplos.
- **Patches** (B) — construir y listar datasets de patches. Muestra `n`, splits, positivos
  por esquina. *Es el sitio donde se decide `n`.*
- **Redes** (C) — **pantalla que falta**: CRUD de arquitecturas, sin datos y sin entrenar.
  Los endpoints ya existen.
- **Recetas** (D) — **pantalla que falta**: CRUD de conjuntos de hiperparámetros, con nombre.
  Sin red y sin dataset: una receta es reutilizable entre arquitecturas. Es lo que convierte
  el barrido en "elegir un espacio" en vez de "rellenar un formulario 20 veces".
- **Entrenar** — elegir una red (C) + un dataset (B) + una receta (D) → lanza E. La pantalla
  donde el contrato ① *se hace visible*: al elegir B y C, o casan o no. `device` va aquí como
  opción de ejecución (X), **fuera** de la receta.
- **Barridos** (H) — **pantalla que falta**: fijar B y C, definir el espacio sobre D, elegir
  objetivo y presupuesto, lanzar, ver la tabla de puntos ordenada por el objetivo, podar,
  reanudar. Es la pantalla que da sentido a que D tenga nombre.
- **Runs** (E) — listar, métricas en vivo, renombrar, borrar, re-entrenar. Un run del barrido
  enlaza a su padre.
- **Predecir** (F) — un run + una imagen → párrafos. Y es el sitio de los **knobs baratos**
  (`threshold`, `stride`, NMS): se ajustan aquí, post-hoc, sin reentrenar.

Nota de orden: **C/D/H son tres pantallas que no existen y un solo cambio de fondo** —
darles identidad (nombre + almacén) a la arquitectura y a la receta. Sin eso, H no se puede
construir: un barrido es literalmente "una lista de D con B y C fijos", y hoy ninguna de las
tres cosas se puede nombrar.

Nota para la visualización de kernels: **encaja en dos pantallas distintas, y no es lo
mismo**. Los kernels aprendidos son de **E** (necesitan pesos → un run entrenado). Los
feature maps son de **E aplicado a un patch de B**. Una red sin entrenar (C) no tiene nada
que enseñar salvo su forma. Por eso la entrada de esa vista es un **patch** (contrato ①: es
la entrada real de la CNN), no una imagen completa — la imagen completa pertenece a F.
