# image-text-finder — instrucciones para Claude

Detección de esquinas de párrafo por patches: se trocea la imagen en patches `n×n` y una CNN
configurable responde, por patch, **¿cae aquí una esquina de párrafo y dónde?** (una cabeza
por tipo: `TL, TR, BR, BL`). En inferencia, una ventana deslizante recompone los párrafos.
Ver [README.md](README.md) para montar y correr.

---

## Estado actual — léelo primero

> **Fases 0, 0.5, 1, 2, 3, 4, 5, 6, 7 y 8 hechas (2026-07-17). El plan está completo**: la fase 8
> montó las **cuatro sondas** (V4 occlusion, V5 scrubber, V10 flag de borde, V15 procedencia del
> patch — V9 ya la trajo la 6). **Están las nueve pantallas, los diez contratos y las quince
> vistas** — **no queda ningún xfail**. **No quedan decisiones bloqueando**; lo abierto en
> [decisiones.md](docs/decisiones.md) §2–§3 se responde al llegar a su fase (**D7** —bbox vs.
> rotación— bloquea la métrica de párrafo, que es lo que le falta al barrido). Lo que queda es
> **investigación** (barrer con el instrumento ya montado) y las **extracciones** de librería
> (`matrixview`, `jobq`, D9/D10), no fases de plan.
>
> **La fase 8 son sondas, no contratos**: no quitó ningún xfail (no le tocaba). Las tres de patch
> (V4, V10) y la de B (V15) cuelgan del **mismo clic en la galería** que ya abría V2/V3; V5 vive en
> **Predecir** porque su entrada es una imagen. **Un patrón nuevo que respetar**: `itf.geometry`
> gana `window_at` (una ventana suelta) y `windows` lo usa — es la fórmula de los flags de borde en
> **un solo sitio**, así que el scrubber (V5, ventana off-grid) y la extracción (B) los calculan
> igual por construcción, no por reteclearlos (contrato ⑤ otra vez). Y una costura que se sostuvo en
> vivo: **V4 baseline, V5 corner y V10 baseline son la misma predicción** — el atajo de recalcularla
> en cada sonda mentiría, así que las tres leen el mismo forward (`_patch_from_body` en `app.py`
> resuelve el patch una vez para V2/V4/V10).
>
> **La fase 7 cerró el contrato ⑨**: `POST /sweeps` rechaza con 400 (`objective_varies_with_space`)
> rankear por `loss` mientras `lambda_pos` está en el espacio — la validación es **pura**
> (`itf.sweeps.spec.check_sweep`), así que contesta sin cargar `optuna`. Antes la 6 cerró el ⑤.
> **No queda ningún xfail.**
>
> **El flujo completo funciona: dato → red → receta → run → diagnóstico → predicción → barrido**, y
> desde la UI. Por CLI: `itf-train --name <run> --patch-dataset <B> --network <C> --recipe <D>
> --device cpu`. Toma **nombres**, no valores: es lo que hace que la procedencia se sostenga sola.
>
> **Ya existen**: `itf.geometry` (G), `itf.metrics` (las definiciones de los números),
> `itf.matrixview` (matriz→payload, **sin importar nada de `itf`**, lista para extraer — ver abajo),
> `itf.datasets` (A), `itf.patches` (B), `itf.models` (C), `itf.validation` (①② + `check_run`),
> `itf.training` (D+E), **`itf.inference` completo** (④ `load_model`, F `predict_image` con las tres
> etapas, V1 `kernels`, V2 `feature_maps`, `ModelCache` con clave de mtime), **`itf.diagnostics`**
> (E×B: tabla por patch, agregados —`pr`, `error_map`, `rows`, **`coactivation`** V9— y caché),
> **`itf.sweeps`** (H: `spec` puro con ⑨, `store` con `spec.json` nuestro + `optuna.db` del motor,
> `runner` con optuna —espacio, poda, reanudación—) e `itf.api` (todo + **`/sweeps`** GET/POST/`/trials`/`/stop`,
> **`POST /jobs/{id}/cancel`**, y el resume de barridos en el `lifespan`). La cola de verdad vive en
> `itf.api.jobs` (**límite=1, cancelación cooperativa, persistencia** con clave por `persist_dir`).
> El código anterior sigue en el tag **`pre-rediseno`** — consúltalo para **algoritmos**, no para
> estructura.
>
> **`itf.matrixview` y `itf.sweeps`/`jobq` están aislados pero NO extraídos** (fases 6 y 7):
> `claude-libs/` no existe —las fases 3 y 4 tampoco extrajeron `convspec` ni `exp-registry`— y
> extraer cierra D9/D10. `matrixview` no importa nada de `itf`; **`itf.sweeps` está escrito
> library-shaped** (librerias.md §2: el barrido se extrae en el 2º proyecto, no en el 1º) y la cola
> es el germen de `jobq`. Si tocas `matrixview`, **no le metas dependencias de `itf`**: es lo único
> que lo mantiene extraíble.
>
> **Las tres puertas de entrenar son una**: `POST /runs`, `itf-train` y **el barrido** (cada trial)
> preguntan a `itf.validation.check_run` y reservan con `RunStore.create`. **La puerta que queda más
> laxa es por la que entra el barrido** — y ya son tres, así que la regla dejó de ser hipotética.
>
> **`tests/` es la barra de progreso del plan**: un test por contrato, los que faltan en
> `xfail(strict=True)`. `.\.venv\Scripts\python -m pytest -q` → *133 passed, 0 xfailed*, en verde.
> **Cada fase debe quitar los suyos** (§3 de [tests.md](docs/tests.md) dice cuáles); si los deja
> puestos, el XPASS estricto pone la suite en rojo y la fase no está terminada. **Ya no queda ningún
> xfail**: los diez contratos están implementados.
>
> **La tabla por patch es un caché y no una entidad** (D1): no se nombra, no se lista, no hay
> pantalla de Evaluaciones. Vive en `data/cache/diagnostics/` y **borrarla no pierde nada** — hay un
> test que lo afirma. Su clave lleva el **mtime del checkpoint**, que D1 no pedía: sin él, un run
> vivo (que reescribe `best.pt` cada época que mejora) serviría para siempre la tabla de la primera
> vez que lo miraste.
>
> **`runs/fase3-01` no tiene procedencia** y el API lo dice en voz alta: es de la fase 3, anterior
> al contrato ③, y no puede decir de qué red salió. **No se construye ningún lector que degrade**
> (eso es lo que mató D3): o se borra y se reentrena, o se queda como está y la pantalla lo marca.
> Diagnóstico se **niega** sobre él con la razón — sin procedencia no hay B contra el que medir.
>
> **Arrancar**: `.\.venv\Scripts\python -m itf.api` (8000) y `cd web && npm run dev` (5173). El
> front proxya `/api` al backend. **La paleta vive en `web/src/theme/tokens.css` y solo ahí** —
> `npm run validate:palette` la valida parseando ese fichero. Si tocas un color, córrelo: no se
> elige a ojo (D12). **Observable Plot entró en la fase 5** (V8, V14); las matrices (V1, V2, V7, V9)
> siguen en canvas a mano (`MatrixCanvas`), y eso es deliberado (ui.md §0). **V1 y V2 comparten
> renderer** (`components/LayerMaps.tsx`): un kernel y un feature map se diferencian en qué
> significan los números, no en qué son. **V2 abre con V3** (mismo `(run, patch)`, mismo clic en la
> galería); **V1 y V9 son de Diagnóstico** y **V11 (las tres etapas + los sliders de F) vive en
> Predecir**. **V12 (Pareto) y V13 (paralelas) entraron en la fase 7**, con Plot, en la pantalla
> **Barridos** (`screens/sweeps/`). El **trabajo de color lo declara el payload**, nunca se adivina:
> un peso es divergente ±0 (R2), una activación mira `spec.activation` (R3) — deducirlo del dato
> pinta una capa `tanh` de dos formas según el patch.
>
> El resto de `docs/` son **especificaciones, no descripciones**: lo no construido no está
> ejecutado ni verificado. Cuando un documento cita un fichero y una línea (`app.py:61`,
> `dataset.py:27-28`), habla del **código anterior** — resuelve contra el tag. Son los hallazgos
> que motivaron el diseño.

> **Fuera de plan (2026-07-18): el resize de fuentes, D19.** Redimensionar una fuente manteniendo
> proporción, reescalando su geometría. **Tres piezas separadas a propósito**: `itf.imageops` (los
> píxeles, **sin un import de `itf`** — por eso sirve para una imagen cualquiera),
> `itf.geometry.scale_quad` (las coordenadas) y `itf.datasets.resize` (la composición). Escribe una
> **fuente derivada A′** en `data/sources/`, **segunda raíz** — A sigue siendo externa y
> solo-lectura. Ids con prefijo `derived/`; `POST /sources/{id}/resize` → job; `itf-resize`.
> **Solo reduce** (`400 upscale_not_allowed`, comprobado contra *todas* las muestras, no la
> primera). Tres cosas que solo aparecieron al correrlo y que conviene no re-aprender:
> **(1)** el formato anida geometría que no leemos (`box`, `lines[]`, `words[]`) y copiarla sin
> escalar deja el dataset internamente incoherente — el resize es **todo o nada**, y el recorrido es
> recursivo; **(2)** `dataset.json.id` **no es único**: `clear-paragraphs-02-reducidos` declara el
> id de `-8ea1ac04`, así que la procedencia guarda `from` (id **direccionable**) y
> `from_declared_id` aparte; **(3)** reducir mucho **borra el texto** — a 80 px la tinta bajo umbral
> cae del 3,4 % al 0,2 % aunque la geometría siga siendo correcta. La resolución de A es un **eje de
> investigación** (⑧), no una forma de ahorrar disco. Y la resolución de ids vive en
> **`itf.datasets.roots`, una sola vez**: la usan `GET /sources`, `itf-extract` e `itf-resize`.

> **Fuera de plan (2026-07-19): el índice de offsets de A.** Una fuente de 20 000 imágenes tiene un
> `labels.jsonl` de **522 MB**, y todo lector de A lo parseaba entero (30 s) — por petición.
> `itf.datasets.index` guarda, por imagen, **dónde empieza su línea** más los cinco escalares que
> un listado necesita, en `data/cache/sources/`. Es un **caché como las tablas por patch (D1)**:
> recomputable, borrable sin pérdida, y **con la fecha+tamaño del fichero en la clave** — un offset
> contra un fichero cambiado no falla, decodifica **otra imagen**, que es el único fallo silencioso
> que esto podía introducir (y por eso el resize, D19, lo invalida solo). Tres consecuencias:
> `GET /sources` **no cuenta solapes** si no hay índice (devuelve `null`, no `0`: ausente ≠ cero) y
> la pantalla dice «sin contar»; `_sample` es un `seek`; y `SourceDataset.samples()` queda para el
> extractor, su único consumidor legítimo. Medido: `/samples` 30 s → **0,05 s**, 12 miniaturas
> ~6 min → **0,13 s**.

**Al terminar una fase, actualiza estas líneas.** Es lo único que le dice a la siguiente sesión
dónde está.

---

## Regla permanente: la organización por dominios manda

**[docs/organizacion.md](docs/organizacion.md) es la fuente de verdad sobre cómo está
organizado este sistema. Léelo antes de cualquier cambio y respeta sus fronteras.** No es
documentación descriptiva: es la estructura que el proyecto sostiene a propósito.

Aplica a todo cambio, por pequeño que parezca — un campo nuevo en un formulario o una clave
nueva en un config son exactamente donde estas fronteras se rompen.

**[docs/ui.md](docs/ui.md) proyecta esa organización sobre la interfaz** — pantallas, catálogo
de visualizaciones y reglas de forma y color. Léelo antes de tocar `web/`. Sus dos reglas:
*una pantalla, un dominio*, y *toda vista de análisis declara qué fija, qué varía y qué mide*.
Si contradice a organizacion.md, gana organizacion.md.

**[docs/plan-ui.md](docs/plan-ui.md) es el plan de ejecución** de ese rediseño, por fases
verticales (backend + front por dominio). Consúltalo para saber en qué fase estamos y qué toca.
Cada fase acaba con la app arrancando, los tests pasando y un commit.

**[docs/kernels-y-feature-maps.md](docs/kernels-y-feature-maps.md)** es material de origen, no una
regla: cómo el proyecto hermano `sliding-window-NIST-ocr` muestra kernels y feature maps, y qué
se porta. Léelo solo al construir V1/V2; lo vigente está destilado en ui.md §5.

**[docs/decisiones.md](docs/decisiones.md) lista lo que está sin decidir y qué bloquea.**
Consúltalo antes de empezar una fase, y **no tomes tú una decisión que esté ahí**: pregunta. Al
cerrarse, la decisión se escribe en el documento que le toca y en decisiones.md queda solo un
puntero. Una decisión que no se ve se acaba tomando sola — así nacieron el CORS abierto, los 20
imágenes de val y `/runs/` gitignoreado.

**[docs/glosario.md](docs/glosario.md) fija las palabras que significan dos cosas.** Cada entrada
ya ha causado un error. Las que más: **`sample` es una imagen, no un ejemplo de entrenamiento**
(el ejemplo es el patch — de ahí el malentendido de "980 patches de val" que en realidad son 20
imágenes); **`model`** es red (C) o run (E), nunca a secas; **`stride`** y **`seed`** significan
dos cosas cada uno y en prosa y en la UI van **siempre cualificados**.

**[docs/formatos.md](docs/formatos.md) define los artefactos en disco y cómo evolucionan.**
Léelo antes de añadir o cambiar un campo del `.npz`, del manifest o del `config.json` de un run.
La regla que más se incumple: **ausente ≠ cero** — rellenar un campo que falta solo es legal si el
consumidor no lo usa; si lo necesita, se falla con la razón, nunca se inventa el dato.

**[docs/tests.md](docs/tests.md) define qué se testea: los contratos de organizacion.md §2 **son**
el plan de pruebas.** Un contrato sin test es un comentario. Un contrato aún roto lleva su test
con `xfail(strict=True)` citando el documento — así "lo que está roto" es una lista ejecutable y
no prosa que envejece: cuando alguien lo arregla, el XPASS estricto pone la suite en rojo y
obliga a actualizar §3. **Cada fase de plan-ui.md debe quitar sus xfails**; si los deja puestos,
no está terminada. Y la frontera: los tests afirman **invariantes**; los resultados de
investigación (`f1 > 0.75`) van al protocolo, nunca a pytest.

**[docs/protocolo.md](docs/protocolo.md) define cuándo un resultado es creíble.** Léelo antes de
sacar cualquier conclusión de un entrenamiento, y antes de lanzar un barrido. Lo esencial: un run
aislado no es un resultado, es una anécdota (van N semillas, media ± sd); toda diferencia dentro
de la banda de ruido es un empate; el test se toca **una vez, al final, solo el ganador**; y dos
runs solo son comparables con el mismo commit de git y la misma huella del dataset.

**[docs/api.md](docs/api.md) define el contrato del API REST** — un recurso por dominio, reglas
R1–R7 (nombres, síncrono vs job, errores con razón y arreglo, polling incremental, agregados en
el servidor) y dónde el API hace cumplir los contratos. Léelo antes de tocar `src/itf/api/`.
Regla mecánica: **si una función de `app.py` no menciona HTTP, no es del API** — es dominio y va
en `itf`.

**[docs/librerias.md](docs/librerias.md) define qué se extrae como librería reutilizable**
(`exp-registry`, `jobq`, `convspec`, `matrixview`), qué **no**, y con qué obligaciones. Dos
reglas: *se extrae en la segunda vez, no en la primera* — nada que exista en un solo proyecto se
extrae, por general que parezca — y *la librería posee el mecanismo, el proyecto posee el
significado*. **Toda librería nace con su propio CLAUDE.md**; sin él no está terminada. Cada
extracción va enganchada a su fase de plan-ui.md, nunca como refactor aparte.

### Los dominios (resumen; el detalle está en el doc)

| | Dominio | Es | Vive en |
|---|---|---|---|
| **A** | Fuente | Imágenes + geometría de párrafos (proyecto externo, solo-lectura) | `datasets/loader.py` |
| **B** | Dataset de patches | El dato que la CNN consume de verdad | `patches/`, `data/patch-datasets/` |
| **C** | Red | La arquitectura. Config puro, cero datos | `models/`, `configs/networks/` |
| **D** | Receta | Hiperparámetros que **definen el resultado** | `training/` |
| **E** | Run | Modelo entrenado: pesos + métricas + procedencia | `runs/<name>/` |
| **H** | Barrido | Espacio de D con B y C fijos → muchos E | *no existe aún* |
| **F** | Inferencia | Aplicar un E a una imagen completa | `inference/predict.py` |
| **G** | Vocabulario | Nombres de esquina/borde, geometría de la ventana | *disperso* |
| **E×B** | Diagnóstico | La tabla por patch y lo que se lee de ella. **Un caché, no un dominio** (D1) | `diagnostics/`, `data/cache/` |
| **—** | matrixview | Matriz de números → payload (números + min/max/mean + trabajo de color). **Aislada, sin importar `itf`; lista para extraer** | `matrixview/` |
| **X** | Ejecución | `device`, `num_workers`, concurrencia. **Cuesta tiempo, no cambia el resultado** | `api/jobs.py` |

### Antes de tocar nada, pregúntate a qué dominio pertenece

El criterio, en orden:

1. ¿Cambia **la forma del modelo**? → **C** (`dropout`, `batchnorm`, `border_features` son C,
   aunque suenen a entrenamiento o a datos).
2. ¿Cambia **los pesos resultantes**? → **D** (`lambda_pos`, `pos_weight` son D, aunque
   suenen a arquitectura).
3. ¿Solo cambia **cuánto tarda**? → **X**. Nunca dentro de la identidad de D.
4. ¿Se ajusta **sin reentrenar**, sobre un modelo ya hecho? → **F** (`threshold`, `stride` de
   inferencia, NMS). Barrer esto no cuesta horas; no lo metas en D.

Si un cambio necesita tocar dos dominios, eso es un **contrato**: está numerado en §2 del
doc. Respétalo explícitamente o actualiza el doc — no lo dejes implícito.

### Contratos que se rompen solos si no se miran

- **① `patch_size` (B) == `input_size` (C)** — **cerrado en la fase 4**, pero sigue aquí porque lo
  que lo sostiene es una costumbre: **toda puerta que entrene pregunta a `itf.validation.check_run`
  antes de reservar el nombre**. Hoy son dos (`POST /runs`, `itf-train`); el barrido será la
  tercera. Una puerta que se salte el validador vuelve a reventar dentro del hilo del job.
- **⑤ La geometría de la ventana** — **cerrado en la fase 6.** Vive en `itf.geometry.windows` y la
  importan B (`patches/extract.py`) y F (`inference/predict.py`); F usa `win.border` **tal cual**.
  Sigue aquí porque lo que lo sostiene es una costumbre: **reteclear las seis líneas de los flags de
  borde es lo que sale natural** al tocar la inferencia. `test_contract_05` afirma
  `extract.windows is predict.windows` y que los flags escritos == los que dice `geometry` — no
  «¿es correcta `positions`?», sino «¿ven B y F la misma ventana?».
- **⑨ El objetivo de un barrido no puede ser la val loss si varía `lambda_pos`** — **cerrado en la
  fase 7.** `check_sweep` (puro, `itf.sweeps.spec`) devuelve `objective_varies_with_space` y `POST
  /sweeps` lo convierte en 400 **antes de reservar nada**. Sigue aquí porque lo que lo sostiene es
  una costumbre: es un barrido nuevo (una cuarta puerta) el que reintroduce el atajo de rankear por
  `loss`. Cada punto se mediría con una pérdida distinta y "ganaría" λ=0.
- **⑩ `batch_size` es D, no X.** Subirlo al pasar a GPU invalida la comparación con lo
  entrenado en CPU.

### Al añadir un hiperparámetro

Va al catálogo de §1-D del doc **con su definición**, clasificado (C / D / X / F) y con nota
de barrido (rango, escala log o lineal, acoplamientos). Un hiperparámetro sin definición en
el catálogo no está terminado.

---

## Contexto de trabajo

- **Hoy solo CPU.** Habrá GPU más adelante para procesamiento masivo. Por eso X debe estar
  separado de D *antes* de que llegue: si no, lo entrenado en CPU queda incomparable.
- **El objetivo es barrer hiperparámetros** (dominio H). Es la razón por la que D es un
  sustantivo con nombre y almacén, y no un formulario suelto.
- En CPU, **el límite de workers concurrentes es 1**: torch ya usa todos los núcleos, y cada
  run carga su `PatchDataset` entero en RAM. Lanzar N entrenamientos a la vez no acelera
  nada y se queda sin memoria.

## Convenciones

- **Idioma**: el usuario se comunica en español; documentación de alto nivel en español. El
  código (identificadores, docstrings) va en inglés, como está hoy.
- **Commits**: cada tarea terminada acaba en un commit descriptivo.
- **Stack**: Python 3.12 (PyTorch no tiene wheels para 3.14) + FastAPI + Vite/React. En
  Windows el intérprete es `.\.venv\Scripts\python.exe`.
- **Tests**: `.\.venv\Scripts\python -m pytest -q` desde la raíz del repo, antes de commitear
  cambios de código.
- **Hay Playwright y Chromium en esta máquina: la UI SE PUEDE ver.** *(Verificado 2026-07-18:
  lanza y renderiza, Chromium 149.)* El driver es `playwright 1.61.0` y vive en
  `C:\Users\User\AppData\Roaming\Python\Python314\` y en el venv del hermano
  (`..\image-text-sample-generator\.venv\Scripts\python.exe`); los navegadores en
  `%LOCALAPPDATA%\ms-playwright\`. **No está en el venv de este proyecto** (Python 3.12), así que
  `import playwright` falla aquí — para usarlo, `pip install playwright` en el venv (los
  navegadores ya están: **no** hace falta `playwright install`) o invocar el intérprete del
  hermano. **No tener un tool de navegador no significa no poder abrir un navegador**: se comprueba
  el entorno, no la lista de herramientas. Se escribió esto porque se entregó UI dos veces diciendo
  «no puedo verlo» sin haber mirado.
- `data/` y `runs/` son artefactos gitignored.

## Trampas: no las reproduzcas (lista completa en organizacion.md §3)

**Esto ya pasó una vez.** Se midió sobre el código anterior (recuperable en `pre-rediseno`) y por
eso no son hipótesis. Y lo importante: **casi todas eran *defaults*** — nadie las eligió,
aparecieron por no elegir. Construir desde cero **no protege de ellas: las invita.**

- **`smooth_l1_beta`**: el default de PyTorch es **1.0**, y con coordenadas normalizadas a [0,1]
  el error nunca supera 1 ⇒ la pérdida de posición es **MSE pura** y el Huber no se activa jamás.
  Ponlo a propósito (~0.05–0.1).
- **SGD sin momentum**: si al optimizador solo le pasas `lr` y `weight_decay`, SGD corre a
  momentum 0 ⇒ comparar optimizadores queda sesgado a favor de Adam.
- **Un hilo por job**: es lo que hace `threading.Thread`. Sin límite de workers, un barrido de 20
  puntos son 20 entrenamientos peleándose por los mismos núcleos, cada uno con su dataset entero
  en RAM. **En CPU el límite es 1.**
- **`POST /runs` sobrescribiendo en silencio**: `mkdir(exist_ok=True)` + truncar `metrics.jsonl`
  machaca resultados sin avisar. Un barrido que autogenera nombres es quien lo pisa.
- **Lógica de dominio dentro de `app.py`**: es lo que sale natural. Regla mecánica: si una
  función de `app.py` no menciona HTTP, no es del API.
- **`border` relleno con ceros**: cero significa "no toca ningún borde", no "no se sabe". Si la
  red usa `border_features` y el dataset no los trae, **se falla** (formatos.md §2). Con los datos
  borrados, todos los datasets nuevos los traerán — así que el camino de relleno **no se
  construye**.
- **Un dataset sin val**: `monitor = val_metrics.get("loss", train_loss)` cae al train loss sin
  avisar, y `best.pt` acaba siendo el checkpoint más sobreajustado. Un dataset sin val **no sirve
  para medir**: falla o avisa.
- **Augmentation con flips o rotaciones**: convierte una TL en TR e invalida los flags de borde.
  Sin reetiquetar, enseña basura y el fallo es silencioso.
- **Medir con una regla de 20 imágenes**: los patches de una imagen están correlacionados, así
  que "980 patches de val" eran 20 imágenes de muestra efectiva. El dato es sintético y el
  generador está al lado: **generar más es gratis** (protocolo.md §1).
- **Coger la fuente equivocada por el sufijo, y no enterarte**: hay **dos** `clear-paragraphs-02`
  — `-reducidos` (160×160) y `-8ea1ac04` (640×480). Con la misma ventana, la de 640×480 da **713
  patches por imagen en vez de 49**, el desbalance se va de 3,9:1 a **~67:1** y la época de 20 s a
  **319 s**. **Equivocarse no falla**: construye un B válido que mide otra cosa. Le pasó a la
  verificación de la fase 3, que llegó a "demostrar" que protocolo.md §1 estaba mal por 16× —
  no lo estaba. **Nombra la fuente entera**; `itf-extract` las lista si te equivocas.
- **Optimizar un proxy sin validarlo**: no existía ninguna métrica de párrafo — todo era a nivel
  de patch. La F1 de párrafo es el objetivo real y **es barata** (protocolo.md §2).
- **Definir un número dos veces**: `pos_err_px` lo escribían `evaluate()` (por época) y la tabla
  por patch (por patch), con la fórmula copiada. Es el contrato ⑤ con otro nombre — **dos copias
  que tienen que coincidir y nada que lo compruebe** — y si divergen, V7 y la curva del run
  describen cosas distintas con el mismo nombre, en silencio. Vive en **`itf.metrics`**, y el test
  no pregunta si la función es correcta (los dos lados llaman a la misma: no puede divergir) sino
  **si la tabla mide lo que el run reportó**. *(Fase 5.)*
- **Una gráfica vacía tiene cara de gráfica**: en escala log, un `rectY` de Plot va desde un y=0
  implícito, `log(0)` no existe y **Plot descarta todas las barras sin avisar**, dejando los ejes.
  Se caza contando marcas en el SVG, no mirando. Y dos series de `rect` sobre el mismo rango x **se
  tapan**, no se agrupan. *(Fase 5.)*
- **Releer la fuente entera para mirar una imagen**: `SourceDataset.samples()` parsea el
  `labels.jsonl` **completo**, y ahí dentro va toda la geometría anidada (`blocks[]`, `lines[]`,
  `words[]`). En `dirty-paragraphs-80ancho` eso son **522 MB y 30 s**. `GET /sources`, `GET
  /samples` y **`_sample()` —una vez por miniatura, por predicción y por arrastre del scrubber—**
  lo hacían cada uno: una galería de 24 miniaturas eran ~12 minutos de CPU. **No fallaba: parecía
  roto.** Lo arregla `itf.datasets.index` (offsets por imagen, cacheados en
  `data/cache/sources/`), y la costumbre que lo sostiene es: **para mirar UNA imagen no se llama a
  `samples()`** — se pide el offset al índice y se usa `sample_at`. `samples()` es para el único
  consumidor que necesita todos los bloques, el extractor. *(2026-07-19.)*
- **Una respuesta lenta sin acuse de recibo se lee como un clic perdido**: V11 y V5 dejan el
  fotograma anterior en pantalla a propósito (atenuado) para que un slider en vivo no parpadee —
  pero atenuar **no es un mensaje**, y a 30 s por respuesta la pantalla parecía ignorar el clic.
  `Working` (components/Async.tsx) lo dice con palabras. *(2026-07-19.)*
- **Un mapa moteado parece estructura**: el 40×40 de V7 con ~200 esquinas son 0,1 muestras por
  celda — **cierto e ilegible**, que es peor que ilegible. Enseña siempre cuántas muestras hay
  detrás de una celda. *(Fase 5.)*
