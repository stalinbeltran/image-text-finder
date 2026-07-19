# Organización de la UI

Cómo se estructura la interfaz aplicando los dominios de
[organizacion.md](organizacion.md). **Ese documento manda**; este solo lo proyecta sobre
pantallas. Si algo aquí contradice a aquel, gana aquel.

Este es un proyecto de **investigación sobre reconocimiento visual**: la UI no es un panel de
control para lanzar entrenamientos, es el **instrumento de medida**. Las pantallas de análisis
(§4) son el producto, no un extra.

---

## 0. Las dos reglas

**Regla 1 — una pantalla, un dominio.** Cada sustantivo (A, B, C, D, E, H) tiene su pantalla,
donde se lista, se crea, se nombra y se borra. Un formulario que mezcla dos dominios es un bug
de organización: es lo que hoy hace `TrainPanel` con C + D + X.

**Regla 2 — toda vista de análisis declara `(qué fija, qué varía, qué mide)`.** Una vista que
cruza dominios no es una excepción a la regla 1: es un **experimento**, y un experimento sin
control declarado no mide nada. Un barrido fija B y C, varía D y mide el objetivo (contrato
⑧). Un mapa de activaciones fija E y el patch, y varía la capa. **Es la misma estructura**, y
por eso el catálogo de §4 está tabulado así.

Esta regla es la que impide que la UI degenere en "pantallas con cosas bonitas": si no puedes
decir qué fija una vista, la vista no sabe lo que enseña.

### Librerías

Hoy `web/package.json` solo trae `react` y `react-dom`, y todo se dibuja a mano
(`LineChart.tsx` es SVG dependency-free). **Se pueden añadir librerías**; el criterio es que
paguen su peso, no que existan.

#### Se añaden

**`optuna`** (backend, para **H**) — **la más rentable del proyecto, y con diferencia.** No es
una librería de gráficos: resuelve de golpe tres de los problemas listados en §3 de
organizacion.md, que son justo los bloqueantes del barrido:

| Problema | Lo que aporta |
|---|---|
| No hay poda ni parada temprana (§3-4) | `MedianPruner`, `SuccessiveHalvingPruner` (ASHA). **La palanca nº1 en CPU** |
| El estado de los jobs vive en memoria (§3-3) | Storage en SQLite: un barrido de horas sobrevive a un reinicio y se reanuda |
| La estrategia del espacio (grid/random/bayes) | Samplers: `GridSampler`, `RandomSampler`, `TPESampler` |
| El objetivo doble en tensión (contrato ⑨) | **Multiobjetivo nativo**: `directions=["maximize", "minimize"]` sobre (`f1`, `pos_err_px`) → **da el frente de Pareto de V12 hecho** |

> **Optuna va *dentro* de H, no *en lugar* de H.** Es el motor del espacio y la poda; la
> organización (B y C fijos, objetivo declarado, procedencia por nombre) la sigue mandando
> organizacion.md. No dejar que la librería dicte la estructura: sus `trials` no son nuestros
> runs — un trial *lanza* un run y guarda su referencia.

**`@observablehq/plot`** (front, para V8, V12, V13, V14) — **entró en la fase 5** (0.6.17).
Gramática de gráficos concisa, hecha para exploración: un histograma o un scatter con leyenda de
color son 3–5 líneas. Cubre lo que hoy no existe y sería tedioso a mano, sobre todo **V13
(coordenadas paralelas)**. Se monta con `useEffect` + `ref` + `replaceChildren` (patrón estándar,
Plot devuelve un nodo DOM) — en `components/PlotFigure.tsx`, que además le pasa la tinta de
tokens.css (Plot trae sus propios grises y desaparecen en oscuro) y lo repinta al cambiar de modo.

> **Dos cosas que Plot hace y hay que saber, las dos medidas en la fase 5.** Devuelve un `<svg>`
> pelado, pero un **`<figure>`** cuando le pides `legend: true` — y un `figure` se lleva los
> márgenes por defecto del navegador, que descuadró el eje alineado de R4 en el único panel con
> leyenda. Y en escala **log**, un `rectY` (que va desde un y=0 implícito) **desaparece sin
> avisar**, dejando los ejes puestos y la gráfica vacía con cara de gráfica.

#### No se añaden, y por qué

- **Nada para V1/V2/V7 (matrices y mapas de calor).** El `drawMap` del proyecto hermano son ~15
  líneas de canvas y hace exactamente esto, con control fino sobre la normalización por mapa
  (§5). Una librería de charting aquí es más código y menos control, no menos.
- **PyTorch Lightning.** Daría scheduler y early stopping "gratis", pero exige reescribir
  `loop.py` entero (180 líneas que funcionan) y se lleva por delante el control del bucle. El
  `scheduler` que falta ya está en `torch.optim.lr_scheduler`: son ~10 líneas.
- **Parquet / DuckDB para la tabla por patch (§3).** Tentador —V7/V8/V9 son agregados
  columnares— pero a esta escala (~10⁵ filas) numpy en memoria es instantáneo, y **`.npz` ya es
  el idioma del proyecto** (`patches.npz`). Se reconsidera solo si la tabla deja de caber en
  memoria.
- **Virtualización de tablas** (`@tanstack/react-virtual`): innecesaria mientras las galerías
  paginen, como hacen hoy.
- **`captum`** (occlusion, saliency, GradCAM): opcional y de baja prioridad. V4 a mano son ~20
  líneas y las necesitamos por cabeza de esquina, que es forma propia. Paga solo si más adelante
  queremos integrated gradients o GradCAM.

#### Qué dibuja qué

Tres tecnologías, según la forma (§4.0), no según el gusto:

| Tecnología | Vistas | Por qué |
|---|---|---|
| **Canvas** a mano (`drawMap`) | V1, V2, V4, V7 | Matrices densas: 40×40 = 1600 celdas. Como SVG pesa; en canvas es un `fillRect` por celda y `image-rendering: pixelated` hace el resto |
| **Observable Plot** | V8, V9, V10, V12, V13, V14 | Ejes, leyendas y escalas hechos. V9 es 4×4 → `Plot.cell` con ejes gratis |
| **HTML/CSS** | V3, V5 (meters), V6 (grid), V11/V15 (overlays) | No son gráficas: son medidores y capas sobre una imagen |

#### Deudas que crean

- Entrar Plot deja **dos formas de dibujar gráficas** (Plot y el `LineChart.tsx` a mano). No es
  urgente, pero la dirección es migrar `LineChart` a Plot cuando se toque —y al hacerlo, V14
  pasa a small multiples por R4—, no mantener las dos.
- **La paleta ya está fijada** *(D12, fase 1, 2026-07-16)*. Vive en
  **`web/src/theme/tokens.css`** y **solo ahí**; la valida `npm run validate:palette`, que parsea
  ese fichero —así valida lo que de verdad se sirve, no una copia que deriva— y pasa en claro y
  en oscuro. Su forma es la que este documento pedía: categórica de **4 slots** (R1), **una**
  secuencial (R3) y **una** divergente (R2). Lo que **sí** quedaba decidido aquí, porque es de
  dominio y no de estética: la normalización de los mapas va **por mapa** (§5), los kernels van
  **centrados en 0** (R2) y **nunca hay doble eje** (R4).

---

## 1. El mapa de pantallas

Cuatro grupos, que siguen la dependencia entre dominios (no el gusto): no puedes entrenar sin
dato y red, no puedes analizar sin run.

| Grupo | Pantalla | Dominio | Hoy |
|---|---|---|---|
| **Datos** | Fuentes | A | dentro de `ExtractPanel` |
| | Patches | B | `ExtractPanel` |
| **Modelo** | Redes | C | **no existe** (endpoints muertos) |
| | Recetas | D | **no existe** |
| **Entrenar** | Entrenar | B×C×D + X → E | `TrainPanel` (mezcla C+D+X) |
| | Barridos | H | **no existe** |
| | Runs | E | `RunsPanel` |
| **Analizar** | Diagnóstico | E × B | **no existe** ← *el corazón* |
| | Predecir | F | `PredictPanel` |

Cambio de fondo respecto a hoy: **la UI actual es un pipeline de 4 pasos numerados**
(`1 · Patches → 2 · Train → 3 · Runs → 4 · Predict`). Eso asume que el usuario recorre el
flujo una vez, de izquierda a derecha. En investigación no se recorre: se **itera sobre un
punto** y se vuelve. Los números se quitan; los grupos no ordenan pasos, agrupan dominios.

---

## 2. Pantallas de dominio

Para cada una: qué posee y — más importante — **qué no toca**. La frontera es el contenido.

### Fuentes (A) — solo lectura, salvo el resize

Lista de datasets bajo `DATASETS_ROOT`, con nº de muestras y miniaturas. Ver una muestra con
sus párrafos dibujados (la verdad de campo).

- **No toca**: nada de patches. Aquí no se elige `n`.
- **Endpoints**: `GET /datasets`, `GET /datasets/{id}`, `GET /datasets/{id}/samples`, `GET /image`.
- *Hoy la fuente se elige dentro del formulario de extracción; separarla la hace inspeccionable
  por sí sola, que es lo que hace falta para juzgar si el dataset es bueno.*

#### Las fuentes derivadas caen aquí, y **no llevan pantalla propia** *(D19)*

Una fuente redimensionada (A′) **es una fuente**, así que sale en esta tabla, en esta galería y
en este visor, sin código nuevo. Eso no es una comodidad: es la comprobación de que la derivación
no inventó un formato. **Si hiciera falta una pantalla de «datasets redimensionados», la
reutilización habría sido mentira** — y además rompería la regla de arriba, *una pantalla, un
dominio*, metiendo una segunda pantalla para A.

Y el visor que ya existe resulta ser **la herramienta de depuración correcta para un resize**: el
overlay SVG dibuja el `quad` **encima de los píxeles redimensionados**. Si la geometría no hubiera
seguido a la imagen, se ve de un vistazo. No hay que construir nada para eso.

**Lo único que la pantalla no puede decir hoy, y sí debe**: cuáles de esas filas son derivadas, de
qué padre y a qué escala. Sin eso, una derivada y una original **se leen igual salvo por el
nombre** — y el nombre del directorio **no es un dato** (organizacion.md ⑧). La columna
«Imágenes» tampoco distingue: una derivada tiene *las mismas* muestras que su padre; lo que cambia
es el tamaño, que no está en la tabla.

- Una **columna de procedencia** que declare `← <padre> ×<escala>`, tomada del bloque `derived`
  que `GET /sources` ya manda.
- **Ausente ⇒ original**, y se pinta como tal (un guion), no como «derivada desconocida»: la
  ausencia significa algo (formatos.md §2).
- **`from`, no `from_declared_id`**: se enseña el id direccionable, el que sirve para volver a la
  fuente. El declarado no es único — dos `clear-paragraphs-02` comparten el suyo — así que
  enseñarlo invitaría a confundir precisamente el par que ya causó el error del 14,5×.
#### El formulario de resize — **aquí, no en Patches**

Redimensionar produce una fuente, así que vive en la pantalla de fuentes. Ponerlo en Patches sería
volver al pecado original de esta UI: elegir la fuente *dentro* del formulario de extracción, que
es lo que hacía imposible mirar un dataset por sí solo.

Es el mismo patrón que «Construir un dataset de patches»: **POST → job → polling** (R3), y al
terminar se refresca la lista. Lo propio de este formulario:

- **La fuente de origen es la seleccionada en la tabla**, no un `select` aparte. Ya hay una fila
  elegida y su galería debajo; un segundo selector permitiría redimensionar A mientras miras B, que
  es una forma barata de equivocarse de fuente — y equivocarse de fuente **no falla**, produce un
  dataset válido que mide otra cosa (organizacion.md §3).
- **Ancho o alto, uno de los dos**, con un radio. No dos campos que se puedan rellenar a la vez: la
  proporción se mantiene por construcción y pedir las dos sería pedir una deformación. La regla la
  hace cumplir `check_resize` (400); el radio es lo que evita **llegar** al 400.
- **Enseña el tamaño resultante en vivo** — `160×160 → 80×80` — calculado de la muestra
  seleccionada. Es lo que convierte «80» en una decisión en vez de una apuesta, y **es donde se ve
  que solo se reduce** antes de enviar.
  > Ese cálculo **duplica** `itf.imageops.target_size`, y eso es contrato ⑤ otra vez. Se tolera
  > porque es un *preview*: la autoridad es `check_resize`, que rechaza con 400, y nada de aquí
  > puede colar una petición mala. **Pero un preview que miente es peor que ninguno**, así que el
  > espejo es exacto, redondeo incluido: `round` de Python es **al par**, `Math.round` no
  > (`100×50` con ancho 5 da alto 2 en el servidor y 3 con `Math.round`). Verificado sobre 25.200
  > combinaciones, cero discrepancias. Si algún día decide en vez de previsualizar, se va al
  > servidor.
- **El 400 se enseña con su `hint`**, como el resto (R4). `upscale_not_allowed` dice por qué no y
  qué pedir en su lugar; tragárselo y enseñar «error» tiraría la mitad útil.
- **No hay borrar**, todavía. Una derivada se borra a mano (`data/sources/<name>`). Cuando lo haya,
  tendrá que avisar de qué datasets de patches la referencian — es el contrato ③ otra vez, y esa es
  razón suficiente para no improvisarlo ahora.

### Patches (B)

CRUD de datasets de patches. Por cada uno: `n`, `stride`, splits, `positives_per_corner`, nº de
patches. Construir uno nuevo desde una fuente.

- **Es el sitio donde se decide `n`** — y por tanto donde nace el contrato ①.
- **Debe mostrar el desbalance**: `positives_per_corner / num_patches`. Es el número que
  gobierna `pos_weight` (§4-D de organizacion.md) y hoy está en el manifest sin que nadie lo
  mire.
- **Falta borrar**, y borrar debe avisar de qué runs lo referencian (contrato ③).
- **Endpoints**: `GET/POST /patch-datasets`, `GET /patch-datasets/{name}`. Falta `DELETE`.

### Redes (C) — pantalla que falta

CRUD de arquitecturas **con nombre**, sin datos y sin entrenar. Los endpoints
(`GET/POST /models`) ya existen y están muertos: `web/src/api.ts` no los llama.

- **No reutiliza `ModelConfigForm.tsx`**: ese componente es **C + D + X en un solo formulario**
  (su propio comentario lo admite: *"the conv backbone, the head, and the training
  hyperparameters"*), y `lockArchitecture` es el booleano que tapa la frontera C/D en vez de
  resolverla. Es el fichero más mezclado del front. Se parte en dos formularios nuevos —uno
  para C, uno para D— y `device` se va a Entrenar.
- Muestra la **traza espacial**: `40 → 20 → 10 → 5` con el nº de canales por capa, y los
  parámetros totales. Es gratis (no necesita pesos) y es lo único que una red sin entrenar
  puede enseñar de sí misma.
- **No toca**: hiperparámetros, dataset, pesos.

### Recetas (D) — pantalla que falta

CRUD de conjuntos de hiperparámetros **con nombre**. Sin red y sin dataset: una receta es
reutilizable entre arquitecturas, y ese es justo el punto.

- El formulario sigue el catálogo de §1-D de organizacion.md, **agrupado igual**:
  optimización / duración y schedule / pérdida / datos en train / selección y semilla.
- **`device` y `num_workers` NO van aquí** (son X, contrato ⑩). Van en la pantalla de
  Entrenar, como opción de ejecución.
- Cada campo lleva su definición del catálogo como ayuda en línea. Un hiperparámetro sin
  definición no se añade al formulario.

### Entrenar (B × C × D + X → E)

Elegir un dataset, una red, una receta; `device` aparte. Lanza un run.

- **Es donde el contrato ① se hace visible**: al elegir B y C, o casan (`patch_size ==
  input_size`) o no. Hoy eso lo hace `RunsPanel.tsx:14` marcando `✗ incompatible`; con B y C
  ya separados, esa comprobación **debe estar en el backend** (400 con la razón), no solo en
  el `.tsx`.
- Debe **estimar el coste** antes de lanzar: `metrics.jsonl` ya guarda `seconds` por época, así
  que hay con qué. En CPU eso es información, no adorno.

### Barridos (H) — pantalla que falta

Fijar B y C. Definir el espacio sobre D (qué campos varían, en qué rango, escala log o lineal).
Elegir estrategia (`grid | random`), objetivo y presupuesto. Lanzar, ver, podar, reanudar.

- **El objetivo se declara aquí y no puede ser la val loss si λ varía** (contrato ⑨). La UI
  debe **impedirlo activamente**: si `lambda_pos` está en el espacio y el objetivo es `loss`,
  se bloquea con la razón. Es un error que produce un ganador de buena cara, así que no puede
  quedar en una nota al pie.
- Tabla de puntos ordenada por el objetivo + las vistas V12/V13 de §4.
- **El estado va en disco, no en memoria** (§3 de organizacion.md): un barrido en CPU dura
  horas y hoy `JOBS` se pierde al reiniciar.
- Muestra el **límite de workers** y por qué en CPU es 1.

### Runs (E)

**Son dos pantallas, no una** (2026-07-19): `/runs` es la **lista** —una fila por run: estado,
procedencia por nombre, mejor valor del monitor, épocas, s/época— y `/runs/:name` el **detalle**
—procedencia entera, ejecución (X), curvas (V14) y **todas** las épocas—. Con las curvas y la
tabla desplegadas dentro de cada tarjeta, ver *qué runs hay* costaba una pantalla de scroll por
run, y un barrido deja doce de golpe: la lista dejaba de responder la pregunta que es su razón de
ser. La URL lleva el nombre, así que un run concreto se enlaza y se recarga — y renombrar **mueve
la pantalla** (`navigate`), porque quedarse en la vieja da un 404 al siguiente refresco.

Lo de hoy (listar, curvas en vivo, renombrar, borrar, re-entrenar) más:

- **De qué B, C y D salió, por nombre** — hoy imposible: el run copia el valor y pierde la
  identidad (contrato ③).
- Enlace a su barrido padre, si lo tiene.
- Entrada a Diagnóstico.
- El toggle `lockArchitecture` de hoy **desaparece**: era la frontera C/D descubierta a mano.
  Con C y D separados, "re-entrenar la misma red" es *elegir otra receta con la misma C*, que
  es lo que siempre fue.

### Predecir (F)

Un run + una imagen completa → esquinas y párrafos. Lo de hoy (dataset/carpeta/upload, grid de
miniaturas, popup) más V11 (etapas del pipeline).

- **Es el sitio de los knobs baratos**: `threshold`, `stride` de inferencia, radio de NMS,
  `min_size`. Se ajustan aquí, post-hoc, **sin reentrenar** (§1-D de organizacion.md). Deben
  ser sliders con repintado en vivo, no campos de un formulario que se envía.

---

## 3. El substrato: la evaluación por patch

**Las vistas V6–V9 de §4 son la misma pasada sobre el val.** No pueden recalcularla cada una.

Hace falta una operación **E × split de B → tabla por patch**, materializada: una fila por
patch con, por cada tipo de esquina, `score`, `(x, y)` predicho, `(x, y)` real, `exists` real,
error en px — más `sample_idx` y `patch_xy`, que **ya están en el `.npz` y hoy nadie usa**.

Esto es exactamente lo que el proyecto hermano tenía como entidad de primera clase
(`evaluations/<id>/results.jsonl` + filtros + matriz de confusión), y es la pieza suya más
valiosa después de los mapas — más que los kernels.

Lo que compra:

- **Todas las vistas de diagnóstico leen de aquí**: una pasada, muchas vistas.
- **Ajustar `threshold` sale gratis**: los scores ya están guardados; re-umbralizar es filtrar
  una tabla, no correr el modelo. En CPU esa diferencia son minutos contra horas.
- **Filtrar por resultado** (aciertos / fallos / por tipo de esquina) sin recomputar.

> **Decidido (D1, 2026-07-16): es un caché, no una entidad.** Se puede recalcular exacta a partir
> de cosas que ya tienen identidad (run, huella de B, split, knobs), y **lo que se puede
> recalcular no se guarda** ([formatos.md](formatos.md) §4.4). Consecuencia para la UI: **no hay
> pantalla de Evaluaciones** ni un dominio nuevo — la tabla se calcula al abrir Diagnóstico y se
> invalida sola. Las cuatro vistas salen igual.
>
> Lo único que no es recalculable es el **criterio** ("los fallos del TL"). Si algún día se quiere
> volver a una búsqueda guardada —y de ahí construir un dataset con esos fallos para reentrenar,
> como el hermano— se guardaría **el filtro** (4 campos), nunca la tabla. Hoy no se construye.

---

## 4. Catálogo de visualizaciones

### 4.0 Reglas de forma y color

Se eligen **antes** que la paleta, y la paleta se **valida con un script, nunca a ojo**. Seis
reglas, todas con consecuencia concreta en este proyecto:

**R1 — Los 4 tipos de esquina son 4 slots categóricos fijos.** `TL, TR, BR, BL` en el orden de
`CORNER_NAMES`, **los mismos en toda la app**: el mismo TL en V3, en V8, en V9 y en el overlay
de V11. El color sigue a la entidad, nunca a su rango: filtrar o reordenar **no repinta** a los
supervivientes. Son 4 series, que es justo donde entra el suelo de daltonismo → **etiquetado
directo obligatorio**, no cortesía. Y 4 está bajo el techo de 8: nunca se generan hues nuevos.

> **Corolario, y hacía falta escribirlo** *(fase 5)*: hay vistas con una dimensión categórica de
> **dos** clases que **no es la esquina** — positivo/negativo en V8, train/val en V14 — y **no
> pueden coger prestado un slot de esquina**. Si "positivo" se pinta del verde de TR mientras el
> selector de esquina está en la misma pantalla, un color significa dos cosas. Se usan **los dos
> extremos de la rampa divergente**: ya están en la paleta fija, son tintas opuestas, y el
> validador los aprueba el uno contra el otro en ambos modos — que es literalmente el trabajo de
> una divergente. **Solo para dos clases**: una tercera pediría un hue nuevo, y eso es de D12.

> **Y aquí las 4 esquinas se miden con la lista dura: `--pairs all`, no adyacentes.** La
> comprobación por defecto solo mira pares vecinos, y vale cuando lo único que se toca son
> vecinos (una pila, unas barras). Aquí **cualquiera de las cuatro puede quedar pegada a
> cualquier otra**: 4 meters en fila (V3), 4 overlays sobre una imagen (V11), una rejilla 4×4
> (V9). Con la lista de adyacentes, un choque entre dos no vecinas **no se vería**.
>
> Esa exigencia es la que fija la paleta (D12): de las **70** formas de escoger 4 de los 8 hues
> documentados, **solo 2 pasan todos los gates en ambos modos** con all-pairs. No es una
> elección estética con margen; es un hueco de dos.
>
> Y cierra el círculo con el etiquetado directo: en oscuro el peor par (TR↔BL) queda en ΔE 6,9,
> dentro de la banda 6–8, que es **legal solo con codificación secundaria**. R1 ya la exigía por
> su cuenta — y R5 (la tabla de números) descarga el WARN de contraste de BR y BL en claro. Las
> dos reglas no son cortesía: **son lo que hace legal esta paleta**. Una vista que se las salte
> convierte los WARN en fallos reales.

**R2 — Los kernels son datos con signo: paleta divergente centrada en 0.** Corrección al
proyecto hermano, y no es cosmética: allá se normalizaba `min→max` con una rampa continua, así
que el **cero caía en cualquier sitio** y la estructura de signo —qué excita y qué inhibe, que
es *lo que un kernel es*— quedaba invisible. Aquí: dos tintas opuestas (cálida/fría), **gris
neutro en el 0**, rango simétrico `±max|w|`. Nunca una tinta en el punto medio.

**R3 — Los feature maps tras ReLU son magnitud: paleta secuencial de una tinta**, clara→oscura.
Nunca arcoíris. **Ojo a la activación**: `relu`/`sigmoid` dan valores no negativos → secuencial;
`tanh` da valores con signo → divergente centrada en 0, como R2. La vista debe mirar
`spec.activation`, no asumir.

**R4 — Jamás doble eje. En este proyecto es una trampa real y concreta.** `metrics.jsonl`
guarda, en la misma línea, `loss ≈ 0.28`, `f1 ≈ 0.77` y `pos_err_px ≈ 11`: **tres escalas
distintas**. Superponerlas en un plot con dos eje-y inventa una correlación que no está en los
datos. Van en **gráficas separadas o small multiples**, apiladas y con el eje x (época)
alineado. Mismo criterio en V8: precision y recall comparten escala 0–1 (una gráfica), pero los
**conteos** del histograma no van con ellas.

**R5 — La tabla de números *es* el "table-view twin".** El click-en-un-mapa → matriz de números
que se trae del proyecto hermano (§5) no es un extra bonito: es el equivalente accesible que
todo mapa de calor debe tener, porque un mapa codifica **solo con color**. Se construye por
accesibilidad, y de paso resulta ser la vista más útil para depurar. Aplica a V1, V2, V7 y V9.

**R6 — Marcas finas, rejilla discreta, etiquetas selectivas.** Nada de un número sobre cada
punto; leyenda presente siempre que haya ≥2 series; el texto lleva tinta de texto, no el color
de la serie. La paleta (categórica de 4, secuencial, divergente) se pasa por el validador de
daltonismo **antes** de darse por buena, en claro y en oscuro.

### 4.1 El catálogo

Cada vista declara su control (regla 2). `€` = coste en CPU.

| | Vista | Fija | Varía | Mide | Forma | Color | € |
|---|---|---|---|---|---|---|---|
| **V1** | Kernels de capa 1 | E | — | los pesos | grid de heatmaps | **divergente ±0** (R2) | gratis |
| **V2** | Feature maps por capa | E, patch | la capa | activación | grid de heatmaps | **secuencial** (R3) | 1 forward |
| **V3** | Predicción del patch | E, patch | — | 4×`[p,x,y]` | **4 meters** + overlay | categórica ×4 (R1) | 1 forward |
| **V4** | Occlusion sensitivity | E, patch | posición de la máscara | caída de `p` | heatmap 40×40 | secuencial | ~300 fw |
| **V5** | Scrubber de la ventana | E, imagen | el recorte `(x0,y0)` | predicción y su **estabilidad** | overlay + 4 meters | categórica ×4 | 1 fw/mov |
| **V6** | Galería peor-primero | E, split | el patch | error por patch | grid de miniaturas | — | tabla §3 |
| **V7** | Error por posición | E, split | posición real | error px | heatmap 40×40 | secuencial | tabla §3 |
| **V8** | Scores + PR | E, split | `threshold` | precision/recall | histograma + línea, **2 gráficas** (R4) | categórica: pos/neg | tabla §3 |
| **V9** | Co-activación de tipos | E, split | tipo real | qué cabeza dispara | heatmap 4×4 | secuencial | tabla §3 |
| **V10** | Test del flag de borde | E, patch | los 4 flags | cambio en la predicción | dumbbell (antes→después) | 1 tinta, 2 tonos | 5 fw |
| **V11** | Etapas del pipeline | E, imagen | la etapa | qué se pierde y dónde | overlay conmutable | categórica ×4 (R1) | 1 predict |
| **V12** | Pareto f1 vs pos_err_px | B, C | la receta (D) | las dos métricas | scatter | **secuencial por λ** | gratis |
| **V13** | Coordenadas paralelas | B, C | la receta (D) | el objetivo | líneas normalizadas | secuencial por objetivo | gratis |
| **V14** | Curvas de entrenamiento | B, C, D | la época | loss y métricas | **small multiples** (R4) | categórica: train/val | gratis |
| **V15** | Procedencia del patch | B | — | de qué imagen salió | overlay | 1 acento | gratis |

Nota sobre V12: **λ es magnitud continua, no identidad** → rampa secuencial, no 4 colores
categóricos. Y sobre V3: 4 probabilidades contra un umbral son **meters** (razón contra un
límite, con el `threshold` marcado en la pista), no un gráfico de 4 barras — y desde luego no
una tarta.

Las que merecen detalle:

**V7 — error por posición dentro del patch.** Mapa de calor: para cada esquina positiva, dónde
caía de verdad y cuánto erró el modelo. **Es la vista que dice qué dominio arreglar**: si el error
se concentra en los bordes del patch (esquinas medio visibles), la respuesta es **bajar el
`stride` de B**, no meter filtros en C. Sin esta vista, ese diagnóstico se confunde
sistemáticamente con "la red es pequeña". La más valiosa del catálogo para la pregunta real del
proyecto — y **lo fue la primera vez que se miró**: sobre `fase4-ui`, **16,4 px en el borde contra
9,1 px en el centro**.

> **La resolución es un control, no 40×40** *(fase 5, 2026-07-17)*. Este documento decía «40×40», y
> **el dato no da para eso**: ~200 esquinas de un tipo repartidas en 1600 celdas son **0,1 muestras
> por celda**, así que el mapa sale **moteado — cierto e ilegible**, que es peor que ilegible a
> secas porque el moteado parece estructura. A **10×10** (celdas de 4 px, ~8 esquinas cada una) el
> borde-vs-centro se ve de un vistazo. No es una corrección al documento sino una consecuencia de
> los datos: la resolución legible **crece con el dataset** (D6 traerá ~10× más), y `bins =
> patch_size` sigue dando exactamente el mapa de aquí. Lo que sí es regla: la vista **enseña cuántas
> esquinas hay detrás de cada celda**, porque una celda de 2 muestras y una de 200 se pintan igual.
> Y el ratio ~2× sale idéntico a 10×10 y a 40×40 ⇒ es real, no un artefacto del binning.

**V8 — histograma de scores + curva PR.** Separabilidad de positivos vs negativos, por tipo de
esquina. El desbalance es de **3,9:1** (20,5 % de positivos en `clear-paragraphs-02`): bastante
para que la accuracy sea engañosa —acertar "no hay esquina" siempre ya da 80 %— y la PR sea la
que informa. **Es el barrido gratis**: elegir `threshold` sin reentrenar.

**V9 — co-activación de tipos.** Ojo con el nombre: **no es una matriz de confusión clásica**.
Las 4 cabezas son binarias independientes, no un softmax: un patch puede activar dos a la vez, o
ninguna. Lo que se tabula es *dado que la verdad era TL, qué cabezas dispararon*. La confusión
**TL↔TR es el fallo de manual aquí** (son espejo), y esta es la única vista que lo enseña.

**V4 — occlusion sensitivity.** Deslizar una máscara de ~5×5 sobre el patch y medir cuánto cae
`p(exists)`. Sobre 40×40 con stride 2 son ~324 forwards: **menos de un segundo**. Es la versión
rigurosa de "toco el dígito y veo qué cambia" del proyecto hermano — misma intuición, pero
sistemática y sin salirse de la distribución.

**V5 — scrubber de la ventana.** Arrastrar el recorte de 40×40 sobre una imagen real y ver las 4
predicciones en vivo. Además de probar el modelo con entradas **en distribución** (a diferencia
de un editor de píxeles, §5), mide algo que no se ve de otra forma: **cuánto tiembla la
predicción al mover la ventana 1 px**. Esa estabilidad es exactamente lo que decide el `stride`
de inferencia y el radio de NMS.

**V11 — etapas del pipeline.** El fallo puede nacer en tres sitios: predicción del patch, NMS, o
reconstrucción voraz TL→BR. Superponer las tres capas sobre la imagen (crudas pre-NMS / esquinas
post-NMS / cajas), conmutables, dice cuál perdió el párrafo. Hoy `predict_image` ya devuelve
`corners` y `paragraphs`; **las crudas pre-NMS habría que exponerlas**. Sin esto, "el párrafo
salió mal" no es diagnosticable.

**V12 — Pareto.** Los runs de un barrido en el plano (`f1`, `pos_err_px`), **coloreados por λ**:
enseña la tensión detectar-vs-localizar que λ arbitra.

**Es diagnóstico, no el ranking.** El ganador lo decide la **F1 de párrafo** (contrato ⑨,
protocolo.md §2), que es λ-independiente y ya integra las dos métricas de este plano — detectar
mal rompe el IoU y localizar mal también. V12 sirve para **entender qué compró λ**, no para
elegir.

**V1 — kernels, y hasta dónde llegan.** Con `in_channels: 1`, los kernels de la **capa 1 son
exactos e interpretables**: se aplican al patch mismo, y deberían salir detectores de borde
orientados. Si tras entrenar parecen ruido, la red no aprendió — se ve en un vistazo. **De la
capa 2 en adelante no hay proyección honesta** (32, 64, 128 canales de entrada): ahí la
información está en los **feature maps** (V2), no en los pesos. Conclusión práctica: mostrar la
capa 1 completa y **no invertir en vistas de kernels profundos**.

### Prioridad

1. **V3, V8, V7, V6** — la tabla de §3 y lo que se lee de ella. Responden la pregunta del
   proyecto (¿detecta o localiza?, ¿dónde falla?) y V8 ahorra horas de CPU desde el primer día.
2. **V2 + V1, V11, V9** — los mapas portados del proyecto hermano, y el diagnóstico del pipeline.
3. **V12, V13** — cuando exista H.
4. **V4, V5, V10, V15** — sondas finas; V5 es la más rentable de las cuatro.

---

## 5. Qué viene del proyecto hermano, y qué cambia

Detalle completo en [kernels-y-feature-maps.md](kernels-y-feature-maps.md).

### Se copia tal cual

- **`map_payload`**: la matriz como números + `min`/`max`/`mean`. El backend manda números, el
  navegador decide el color.
- **Normalización del color por mapa** (contra su propio `min`/`max`, no global): sin esto los
  mapas de activación baja se ven todos negros.
- **Truncado `max_maps`** (64) con aviso `truncated` — aquí hace más falta: 128 filtros en la
  capa 3.
- **`drawMap`** en canvas + `image-rendering: pixelated`, y **click en un mapa → tabla de
  números** (que resulta ser el table-view accesible que exige R5, no solo una comodidad).
- **El mapa abierto sigue abierto** al re-predecir (`state.selectedMap`): su matriz se actualiza
  en vez de cerrarse. Es lo que hace usable cualquier vista en vivo, y aplica igual a V5.
- **Debounce configurable** para las vistas en vivo.

### Cambia, y por qué

| Allá | Acá | Motivo |
|---|---|---|
| Pestaña *Features* = NN + muestra + editor + banco, todo junto | Se reparte: kernels/maps→**E**, patch→**B**, sondas→Diagnóstico | Regla 1. Aquella pestaña mezclaba cuatro cosas |
| Orden *Entrenar → Experimentos → Probar → Datasets → Features* | Datos → Modelo → Entrenar → Analizar | Allá el dato era uno (MNIST) y la app iba de la red. **Acá el dato es un pipeline** (fuente→patches) y pesa tanto como la red |
| Entrada = una muestra del dataset | Entrada = **un patch** | Contrato ①: el patch es la entrada real de la CNN. La imagen completa es de F |
| Salida = 10 probs, top-3, margen | **4×`[exists, x, y]`** | La cabeza es `CornerHead`, no un softmax. V3 la sustituye |
| **Editor de píxeles** con pincel suave | **Scrubber de ventana** (V5) + **occlusion** (V4) | Ver abajo |
| Kernels: corte del canal 0, todas las capas | Capa 1 completa; profundas vía feature maps | Con 1 canal de entrada la capa 1 es exacta; el corte del canal 0 en las profundas es cosmético |
| Kernels con rampa continua `min→max` | **Divergente centrada en 0** (R2) | Los pesos tienen signo. Normalizar `min→max` deja el cero en cualquier sitio y **esconde qué excita y qué inhibe**, que es lo que un kernel es |
| Regla 21: pintar en 112×112 y promediar | **No aplica** | MNIST es suave porque nace de reescalar. Acá el extractor lee las imágenes ya pixeladas *tal cual* |
| Evaluaciones + filtros + guardar filtro como dataset | La tabla por patch de §3 | Misma idea, mejor pieza. Es lo más valioso que se trae |

**Sobre el editor, que es el cambio de fondo.** Allá tenía todo el sentido: MNIST *es* un dígito
dibujado a mano, así que dibujar uno cae dentro de la distribución. Acá un patch es **un recorte
de texto renderizado y pixelado**; pintarlo a mano produce una entrada que el modelo no vio
nunca, y la predicción resultante no dice nada del modelo — dice del artefacto. La intuición que
hacía valioso el editor ("toco algo y veo qué cambia") se conserva **mejor** en dos vistas:

- **V5** mueve el recorte por una imagen real → entradas en distribución, y de paso mide la
  estabilidad que fija el `stride`.
- **V4** ocluye regiones sistemáticamente → la misma pregunta ("¿qué pasa si quito esto?") pero
  barriendo todas las posiciones en vez de a ojo.

Un editor de patches queda como opción de baja prioridad; V4 y V5 lo cubren mejor.

---

## 6. Orden de construcción

**El orden lo manda [plan-ui.md](plan-ui.md), que es el plan de ejecución. Este documento no lo
duplica**: dos documentos dueños del mismo orden se desincronizan siempre — y de hecho lo
hicieron (§6 llegó a poner H antes que los mapas, al revés que el plan).

Lo que sí es de aquí, porque son razones de diseño y no de calendario:

- **Las tres pantallas que faltan (Redes, Recetas, Barridos) son un solo cambio de fondo**: darle
  identidad —nombre y almacén— a C y a D. Sin eso H no se puede construir, porque un barrido es
  literalmente "una lista de D con B y C fijos". De ahí que sea la fase 3 del plan.
- **V8 antes que H.** Ajustar `threshold` es gratis y post-hoc; si se entra al barrido sin eso,
  se gastan horas de CPU buscando en D lo que estaba en F.
- **La prioridad entre vistas** está en §4.1, y el plan la respeta fase a fase.
