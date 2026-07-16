# API

Estructura y definiciones del API REST que expone este proyecto a la web app. Aplica los
dominios de [organizacion.md](organizacion.md) y sirve las pantallas de [ui.md](ui.md).

**Este documento define el contrato. organizacion.md sigue mandando: si el API necesita un
recurso que no es un dominio, o mezcla dos, el error está en el API.**

---

## 0. La capa: qué es y qué no

```
web app  ──HTTP──▶  API  ──llamadas──▶  itf (el dominio)  ──▶  librerías reutilizables
```

**El API posee HTTP y nada más**: rutas, códigos, serialización, validación de la petición,
convertir errores del dominio en respuestas. Toda la lógica está debajo, en `itf`, y **debe
poder usarse sin el API** (los CLI `itf-extract` e `itf-train` lo prueban: si algo solo funciona
por HTTP, está en la capa equivocada).

Regla mecánica: **si una función de `app.py` no menciona HTTP, no es del API.**

El `app.py` anterior (511 líneas) la incumplía en seis sitios, y es la lista con la que se
comprueba que la reconstrucción no repite el error:

| Estaba en `app.py` | Es dominio de | Vive ahora en |
|---|---|---|
| `_discover_datasets()` | A | `itf.datasets.loader.discover_sources` ✅ |
| `_dataset_samples()` | A | `itf.datasets.loader.SourceDataset` ✅ |
| `_split_map()` | B | `itf.patches.store.PatchDatasetStore.split_map` ✅ |
| `_run_source()` | E→B→A (procedencia) | `itf.training.registry.RunStore` ✅ |
| `_run_status()` | E | `itf.training.registry` / `exp-registry` — *fase 4* |
| `_MODEL_CACHE` | F | `itf.inference` — *fase 6* |

---

## 1. Reglas del API

### R1 — Un recurso por sustantivo del dominio

Los recursos son exactamente los dominios de organizacion.md. Ni más (no hay recursos
"de conveniencia") ni menos (D no puede seguir sin recurso).

### R2 — Los nombres actuales mienten; se renombran

Dos colisiones reales, y las dos confunden a quien lee el API sin contexto:

| Hoy | Problema | Pasa a ser |
|---|---|---|
| `/datasets` | Son los datasets **fuente** (A), pero `/patch-datasets` (B) también son datasets | **`/sources`** |
| `/models` | Son **configs de arquitectura** (C). En ML "model" es lo entrenado (E) | **`/networks`** |

Que `/models` desaparezca es parte del objetivo: **"model" es la palabra ambigua** — significa
C o E según quién hable. `/networks` (C) y `/runs` (E) no se confunden nunca.

Coste: `tests/test_api.py` fija `/datasets`, y el front actual lo llama. Como el front se
reescribe igual ([plan-ui.md](plan-ui.md)), el rename sale casi gratis **si se hace en su fase**.

### R3 — Síncrono o job, según el tiempo

**Todo lo que pueda tardar más de ~1 s devuelve un job**, no el resultado:

| Operación | | Por qué |
|---|---|---|
| Construir patches, entrenar, barrer, evaluar un split | **job** (`202`) | Minutos u horas |
| Todo lo demás (CRUD, kernels, feature maps, validar, predecir una imagen) | **síncrono** | Un forward o leer disco |

Los jobs devuelven **`202 Accepted`** con el job en el cuerpo (hoy devuelven `200`, que dice
"hecho" cuando no lo está). El progreso se sigue por `/jobs/{id}` y, si el job escribe un run,
por sus métricas.

### R4 — Un error dice **por qué** y **cómo se arregla**

Heredado del proyecto hermano, donde es regla explícita, y es la diferencia entre un API
usable y uno adivinable:

```jsonc
{ "detail": {
    "code": "patch_size_mismatch",          // slug estable, para la UI
    "message": "la red 'cnn-a' espera patches de 40x40 y 'reducido-60' los tiene de 60x60",
    "hint": "elige un dataset con patch_size 40, o crea una red con input_size 60"
} }
```

- **`code`** es contrato: la UI puede reaccionar a él. `message` y `hint` son para humanos.
- **Se valida antes, no durante.** Un `400` al entrar vale mil veces más que un stack trace
  dentro del hilo del job media hora después — que es lo que pasa hoy con el contrato ①.

| Código | Significa |
|---|---|
| `400` | La petición es imposible, y el cuerpo dice por qué |
| `404` | No existe |
| `409` | Choca con el estado: en uso, ya existe, está corriendo |
| `202` | Aceptado, corriendo en segundo plano |

### R5 — El polling es incremental

`GET /runs/{name}/metrics?since=N` → `{records: [...], next: M}`. **Nunca se reenvía el
historial entero.** Hoy `GET /runs/{name}` devuelve *todas* las métricas en cada llamada, y la
UI las pide en bucle: el coste crece con la época. El proyecto hermano ya lo resolvió con
`since`/`next`; se copia.

Corolario: `GET /runs/{name}` **no incluye métricas**. Devuelve estado, config y procedencia.

### R6 — Los agregados se calculan en el servidor

Una tabla por patch tiene ~10⁵ filas. **El navegador nunca las recibe.** La curva PR, el mapa
de error y la co-activación (V7, V8, V9) son **endpoints que devuelven el agregado ya hecho**;
la tabla cruda solo se sirve **filtrada y paginada** (V6).

### R7 — Si se entrenó con ello, tiene nombre

`POST /runs` acepta **nombres** de red y receta, no valores inline. Quien quiera algo a medida,
lo guarda primero. Suena rígido y es deliberado: es lo que hace que el **contrato ③** se cumpla
solo — todo run puede decir de qué C y qué D salió, que es justo lo que un barrido necesita para
agrupar y lo que hoy es imposible.

---

## 2. El mapa de recursos

| Dominio | Recurso | Hoy |
|---|---|---|
| **A** Fuente | `/sources` | `/datasets` |
| **B** Dataset de patches | `/patch-datasets` | igual (falta `DELETE`) |
| **C** Red | `/networks` | `/models` (muerto: el front no lo llama) |
| **D** Receta | `/recipes` | **no existe** |
| **E** Run | `/runs` | igual (sin procedencia ni stop) |
| **E×B** Diagnóstico | `/runs/{name}/diagnostics` — **caché, no entidad** (D1) | **no existe** |
| **H** Barrido | `/sweeps` | **no existe** |
| **F** Inferencia | `/runs/{name}/predict` | `/predict`, `/predict-path` |
| **X** Jobs | `/jobs` | igual (sin cancelar) |

---

## 3. Recurso por recurso

Solo lo que no es evidente. `→ job` = R3.

### `/sources` (A)

```
GET  /sources                              lista
GET  /sources/{id}                         metadatos + ejemplo
GET  /sources/{id}/samples                 listado por muestra
       ?patch_dataset=<B>                  anota cada muestra con su split (train/val/test)
GET  /sources/{id}/samples/{index}/image   la imagen
       ?w=<px>                             reducida (miniaturas)
```

El `?patch_dataset=` es el cruce legítimo A×B de la pestaña Predecir (ui.md §2): permite
"predice solo el test". **Sustituye a `/image?path=`** (ver §6).

### `/patch-datasets` (B)

```
GET    /patch-datasets                     lista + manifest
POST   /patch-datasets                   → job
GET    /patch-datasets/{name}              manifest + fingerprint + used_by
DELETE /patch-datasets/{name}              409 si algún run lo referencia
GET    /patch-datasets/{name}/patches      miniaturas de patches (paginado)
         ?split=&offset=&limit=
GET    /patch-datasets/{name}/patches/{i}  un patch: píxeles, label, border, procedencia
```

- **`fingerprint`**: huella del contenido (contrato ⑧). Sin ella nada distingue un B
  reconstruido bajo el mismo nombre, y un barrido a medias queda incomparable **en silencio**.
- **`used_by`**: qué runs lo referencian. Es lo que permite el `409` con razón.
- **`/patches/{i}`** es la entrada de los feature maps (contrato ①: la entrada real de la CNN es
  el patch). Devuelve también `sample_idx` y `patch_xy`, que **ya están en el `.npz` y hoy no
  usa nadie** — son V15.

### `/networks` (C)

```
GET    /networks                lista
POST   /networks                guarda
GET    /networks/{name}
PATCH  /networks/{name}         renombra
DELETE /networks/{name}
POST   /networks/validate       traza espacial + nº de params, SIN guardar
```

`POST /networks/validate` es puro, síncrono y barato, y hace de la validación previa una
**función del API**, no un efecto secundario de entrenar:

```jsonc
{ "valid": true,
  "trace": [ {"layer": 1, "in": 40, "conv": 40, "out": 20, "channels": 32}, … ],
  "num_params": 214531,
  "flat_features": 3200 }
```

Si no cabe, `400` con `code: "layer_does_not_fit"` diciendo **qué capa, con qué tamaño y cómo
arreglarlo**. Alimenta en vivo la pantalla Redes.

### `/recipes` (D)

CRUD simple. El cuerpo es el catálogo de §1-D de organizacion.md. **`device` y `num_workers` no
están** (son X, contrato ⑩): una receta que lleve dentro el device deja de ser comparable entre
CPU y GPU.

### `/runs` (E)

```
GET    /runs                          lista + estado
POST   /runs                        → job     {name, patch_dataset, network, recipe, device}
GET    /runs/{name}                   estado + config + procedencia + checkpoints (sin métricas, R5)
GET    /runs/{name}/metrics?since=N   incremental → {records, next}
PATCH  /runs/{name}                   renombra (409 si corre)
DELETE /runs/{name}                   409 si corre o si un barrido lo referencia
POST   /runs/{name}/stop              cancelación cooperativa
```

`POST /runs` es donde el API **gana su sueldo** (§4). El cuerpo lleva **nombres** (R7) y
`device` **aparte** de la receta.

Antes de crear el job llama a **`itf.validation.check_compatible(manifest, model_cfg)`** —
función pura de dos diccionarios, sin torch, milisegundos— y si devuelve problemas responde
**400** con ellos (R4). Cubre de una vez los contratos ① (`patch_size == input_size`), ②
(`border_features` sobre un dataset sin `border`) e `in_channels`: **son la misma pregunta**
(organizacion.md §2, recuadro tras ②).

> **El mismo validador se llama también dentro de `train()`**, y no es redundancia: **`itf-train`
> no pasa por el API**. Sin el segundo control, el CLI se salta la puerta y el fallo vuelve a
> aparecer a mitad de época. El API da el `400` temprano; `train()` es la red de seguridad. Que la
> validación viva en el **dominio** y no aquí es lo que permite las dos llamadas (§0).

La procedencia que devuelve `GET /runs/{name}` es contrato ③. *(D2, decidido 2026-07-16.)*

```jsonc
{ "provenance": {
    "patch_dataset": {"name": "reducido-40", "fingerprint": "sha256:…"},
    "network": {"name": "cnn-a", "value": { … }},   // nombre para agrupar, valor para reproducir
    "recipe":  {"name": "adam-lr1e-3", "value": { … }},
    "sweep": null,                                   // o el barrido padre
    "git_commit": "acaf34d…",
    "environment": {"python": "3.12.10", "torch": "2.13.0+cpu", "platform": "win32"}
} }
```

**Los cinco campos son obligatorios y ninguno se rellena**: si no se puede saber el commit
(árbol sucio, sin git), se escribe la razón, no `null` silencioso — es formatos.md §2 otra vez.

Sobre `environment`: la regla 1 de comparación del protocolo es *mismo commit de git*, pero **el
commit no captura el intérprete**. Subir de torch 2.13 a 2.14 mueve los resultados igual que
tocar la pérdida, y **el plan incluye pasar a GPU**, donde el entorno cambia entero. Sin este
campo, los runs de CPU de hoy no podrían decir contra qué se les compara mañana.

**No hay retrocompatibilidad que mantener**: `runs/` está vacío (D18). Todo run nace con la
procedencia completa, y un `config.json` sin ella es un error, no un caso legado.

### `/runs/{name}/diagnostics` (E×B) — el substrato de §3 de ui.md

**Es un caché, no una entidad** (D1): no hay `POST` que lo cree, ni `id`, ni recurso listable.
Todos los endpoints son **`GET` idempotentes** sobre `(run, split)`; la tabla se calcula al primer
GET y se invalida sola si cambian el run, la huella de B o los knobs.

```
GET /runs/{name}/diagnostics/patches      tabla filtrada y paginada  (V6)
      ?split=val&outcome=&corner=&order=error&offset=&limit=
GET /runs/{name}/diagnostics/pr           curva PR + histograma      (V8)
      ?split=val&corner=TL
GET /runs/{name}/diagnostics/error-map    mapa 40×40                 (V7)
GET /runs/{name}/diagnostics/coactivation matriz 4×4                 (V9)
```

- **Síncronos** (R3): una pasada sobre val son ~10⁴ forwards por lotes, segundos. El primer GET
  paga; los demás leen el caché.
- **Agregados en el servidor** (R6): `pr`, `error-map` y `coactivation` devuelven el resultado ya
  hecho. El navegador **nunca** recibe 10⁵ filas; `patches` va filtrado y paginado.
- **La curva PR se calcula sobre los scores cacheados**: barrer `threshold` **no vuelve a correr
  el modelo**. Ahí está el ahorro de horas de CPU.

### `/runs/{name}` — introspección (V1, V2, V4)

```
GET  /runs/{name}/kernels        pesos por capa, como matrices
POST /runs/{name}/feature-maps   {patch_dataset, index} | {patch: [[…]]}  → capas + predicción
POST /runs/{name}/occlusion      {patch…, mask_size, stride}              → mapa 40×40
```

Todos devuelven el payload de `matrixview` (`matrix` + `min`/`max`/`mean` + `truncated`), y
**declaran el trabajo de color** (`sequential | diverging`), porque el cliente no puede saber si
mira un peso con signo o una activación ≥0. Es la corrección de §5 de ui.md.

### `/sweeps` (H)

```
GET  /sweeps
POST /sweeps                → job   {name, patch_dataset, network, space, strategy, objective, budget}
GET  /sweeps/{name}                 spec + progreso
GET  /sweeps/{name}/trials          la tabla ordenada por objetivo (V12, V13)
POST /sweeps/{name}/stop
```

`POST /sweeps` **rechaza con `400`** si `objective` es `loss` y `lambda_pos` está en `space`
(contrato ⑨). No es un aviso: es un `400`.

### `/runs/{name}/predict` (F)

```
POST /runs/{name}/predict   {source, index} | {upload}  + threshold, stride, nms_radius, min_size
```

Devuelve **las tres etapas** (V11), no solo la última:

```jsonc
{ "raw": [ … ],          // detecciones por patch, pre-NMS   ← hoy no se expone
  "corners": [ … ],      // tras NMS
  "paragraphs": [ … ],   // tras reconstrucción
  "image_size": [w, h] }
```

Sin `raw`, "el párrafo salió mal" no es diagnosticable: no se sabe qué etapa lo perdió.

### `/jobs` (X)

```
GET  /jobs            GET /jobs/{id}            POST /jobs/{id}/cancel
```

`cancel` es **cooperativo**: marca el `stop_event`; el trabajo corta en el siguiente punto
seguro (fin de época). No mata el hilo.

---

## 4. Dónde el API hace cumplir los contratos

**Esta sección es la razón de ser de la capa.** Un contrato que no se comprueba en la frontera
se comprueba en producción:

| Contrato | Dónde | Qué pasa hoy |
|---|---|---|
| **① `patch_size == input_size`** | `POST /runs` → `400` | Solo lo mira `RunsPanel.tsx`. Por HTTP directo, revienta con `mat1 and mat2 shapes cannot be multiplied` **dentro del hilo del job** |
| **③ B en uso** | `DELETE /patch-datasets/{n}` → `409` con la lista | No hay `DELETE`; y borrar a mano deja runs sin procedencia, en silencio |
| **⑨ objetivo vs λ** | `POST /sweeps` → `400` | No existe. Produciría un ganador de buena cara |
| **⑩ X fuera de D** | `device` fuera de `/recipes` | Está dentro de `RunConfig` y se congela |
| **R7 procedencia** | `POST /runs` exige nombres | Se copia el valor y se pierde la identidad |

---

## 5. Qué pone la librería y qué pone el proyecto

Esto es lo que hace que el API sea **reaprovechable** ([librerias.md](librerias.md)), y parte
en dos limpiamente:

**Superficie de librería** — un proyecto nuevo la quiere idéntica, así que la librería puede
traer un **router de FastAPI opcional**:

- `/jobs` (listar, ver, cancelar) **es** la superficie de `jobq`.
- `/runs` (CRUD, estado, métricas con `since`) **es** la de `exp-registry`.

```python
from exp_registry.fastapi import router as runs_router
app.include_router(runs_router, prefix="/runs")
```

Condiciones para que esto no corrompa la librería:

- **Extra opcional** (`pip install exp-registry[api]`); el núcleo **no importa fastapi**. Un CLI
  debe poder usar la librería sin ver HTTP.
- El router es un **adaptador fino**: traduce, no decide.
- La librería expone `/runs` **genérico**. Lo específico —que un run de ITF salga de un
  `patch_dataset` y una `network`— lo añade el proyecto. Si el router necesita saber qué es un
  patch, la frontera está mal.

**Superficie de proyecto** — no se extrae nunca: `/sources`, `/patch-datasets`, `/networks`,
`/recipes`, `/sweeps`, `/predict`, `/diagnostics`. Son el **significado**, y el significado es
del proyecto (prueba de frontera de librerias.md §0).

Lo que **sí** se reaprovecha de esta capa, sin ser código, son las **reglas** R1–R7. Van al
CLAUDE.md del proyecto nuevo, como las convenciones de validación.

---

## 6. Las rutas arbitrarias: una decisión que hay que tomar

Hoy el API lee **cualquier fichero de imagen del disco por ruta absoluta**:

```
GET  /image?path=C:\lo\que\sea.png      _resolve_image_path: solo comprueba que sea imagen
GET  /folder?path=C:\lo\que\sea         lista imágenes de cualquier carpeta
POST /predict-path  {path: …}
```

Y `app.py:61` monta CORS con **`allow_origins=["*"]`**.

Combinados: **cualquier página web que visites mientras el API corre puede enumerar y leer
imágenes de tu disco** — `fetch("http://127.0.0.1:8000/folder?path=C:\\Users\\User\\Pictures")`
desde cualquier origen, y el navegador le deja leer la respuesta. Es un agujero modesto (solo
imágenes) y esperable en una herramienta local, pero **no es una decisión que se haya tomado:
salió sola**.

Y hay un plan que lo agrava: **la GPU**. En cuanto el API viva en una máquina accesible por red,
esto deja de ser modesto.

Por eso el §3 propone `GET /sources/{id}/samples/{index}/image` en vez de `/image?path=`: la
ruta **se resuelve dentro del dominio**, no la manda el cliente. Para la fuente "carpeta
arbitraria" de la pestaña Predecir hay tres salidas, y hay que **elegir una a conciencia**:

1. **Raíces permitidas**: una allowlist (`DATASETS_ROOT` y poco más); fuera de ahí, `403`.
2. **Subir el fichero** en vez de referenciarlo (ya existe `POST /predict` con upload).
3. **Dejarlo como está** y cerrar CORS a `localhost:5173` + no exponer el API en red — decisión
   válida, pero **escrita**, no heredada.

### Decidido: raíces permitidas + CORS cerrado

*(D4, decidido 2026-07-16.)*

- **Allowlist de raíces**: `DATASETS_ROOT` más las que se declaren explícitamente. Una ruta que no
  cuelgue de una raíz permitida → **403**. La resolución se hace **después** de `Path.resolve()`,
  o `..\..\` se salta la comprobación.
- **CORS cerrado** al origen del front (`http://localhost:5173`), no `*`.

Se conserva la comodidad de "apunta a esa carpeta" declarando raíces, y **el día de la GPU no hay
que rehacer nada**. Se implementa en la **fase 2** (§7), que es cuando se toca esta zona.

---

## 7. Migración

El API cambia dentro de las fases de [plan-ui.md](plan-ui.md); **no hay una fase "arreglar el
API"**:

| Fase | Al API |
|---|---|
| **2** | `/datasets` → `/sources`; `DELETE /patch-datasets` + `used_by`; `/sources/{id}/samples/{i}/image`; CORS — ✅ |
| **3** | `/models` → `/networks` + `DELETE` + `/validate`; **`/recipes` nuevo** — ✅ *(y el almacén con ellos: `configs/models/` → `configs/networks/`, formatos.md §4.3)* |
| **4** | `POST /runs` con nombres y contrato ①; procedencia; `202`; `/metrics?since=`; `/stop` |
| **5** | `/diagnostics` + los agregados (caché) |
| **6** | `/kernels`, `/feature-maps`; `raw` en predict |
| **7** | `/sweeps`; `/jobs/{id}/cancel` |

Los renombres de R2 rompen `tests/test_api.py` (fija `/datasets`) y el front actual. Como el
front se reescribe igual, **el coste real es actualizar los tests** — y hacerlo en su fase, no
todo junto al final.
