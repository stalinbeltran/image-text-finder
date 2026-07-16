# image-text-finder — instrucciones para Claude

Detección de esquinas de párrafo por patches: se trocea la imagen en patches `n×n` y una CNN
configurable responde, por patch, **¿cae aquí una esquina de párrafo y dónde?** (una cabeza
por tipo: `TL, TR, BR, BL`). En inferencia, una ventana deslizante recompone los párrafos.
Ver [README.md](README.md) para montar y correr.

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
| **C** | Red | La arquitectura. Config puro, cero datos | `models/`, `configs/models/` |
| **D** | Receta | Hiperparámetros que **definen el resultado** | `training/` |
| **E** | Run | Modelo entrenado: pesos + métricas + procedencia | `runs/<name>/` |
| **H** | Barrido | Espacio de D con B y C fijos → muchos E | *no existe aún* |
| **F** | Inferencia | Aplicar un E a una imagen completa | `inference/predict.py` |
| **G** | Vocabulario | Nombres de esquina/borde, geometría de la ventana | *disperso* |
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

- **① `patch_size` (B) == `input_size` (C)** — hoy solo lo valida el front (`RunsPanel.tsx`).
  El backend acepta el mismatch y revienta dentro del hilo del job.
- **⑤ La geometría de la ventana** está **duplicada** entre `patches/extract.py` e
  `inference/predict.py` (que además importa la privada `_positions`). Si tocas una, toca la
  otra: no hay test que lo pille.
- **⑨ El objetivo de un barrido no puede ser la val loss si varía `lambda_pos`** — cada punto
  se mediría con una pérdida distinta y "ganaría" λ=0.
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
- `data/` y `runs/` son artefactos gitignored.

## Estado conocido (ver §3 del doc para la lista completa)

Cosas que ya sabemos que están rotas o a medias — no las redescubras, y no las des por buenas:

- **C y D no existen como entidades en la UI.** `GET/POST /models` está implementado y
  **muerto**: `web/src/api.ts` no lo llama nunca.
- **`JOBS` no es una cola**: un hilo por job, sin límite ni cancelación, estado en memoria.
- **`POST /runs` sobrescribe en silencio** si el nombre existe.
- **`smooth_l1_beta` usa el default de PyTorch (1.0)** con coordenadas en [0,1] ⇒ la pérdida
  de posición es **MSE pura**, el Huber nunca se activa.
- **SGD corre sin momentum** (`_make_optimizer` solo pasa `lr` y `weight_decay`) ⇒ comparar
  optimizadores hoy está sesgado a favor de Adam.
- **Augmentation**: flips y rotaciones **no son válidos sin reetiquetar** (un flip convierte
  TL en TR e invalida los flags de borde). El fallo sería silencioso.
- **No existe ninguna métrica de párrafo.** `evaluate()` es todo a nivel de patch. Nadie ha
  medido nunca si los párrafos salen bien en la imagen completa — que es el objetivo real del
  proyecto. Ver protocolo.md §2.
- **Ninguno de los `.npz` de `data/patch-datasets/` tiene el array `border`** (son anteriores a
  la feature), y `configs/model.example.yaml` trae `border_features: true`. El ejemplo del README
  entrena con flags de borde **todos a cero** rellenados en silencio, y luego en inferencia
  `detect_corners` calcula los reales: el modelo ve en los bordes una distribución que nunca
  entrenó. Sin excepción, solo peores predicciones. Ver formatos.md §2.
- **Un dataset de patches sin val rompe la selección en silencio.** Con `val` vacío,
  `loop.py` cae a `monitor = train_loss` y `best.pt` acaba siendo el checkpoint más
  sobreajustado, sin warning. Le pasa a `reducido-40` (5 imágenes → split 4/0/1), que es el
  ejemplo del README.
- **El val de `clear-paragraphs-02` son 20 imágenes**, no 980 patches: los patches de una imagen
  están correlacionados. Diferencias de f1 bajo ~5 % no son resolubles ahí. El dato es sintético
  y el generador está al lado: es una elección, no una restricción.
