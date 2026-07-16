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
