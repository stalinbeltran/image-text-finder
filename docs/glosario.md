# Glosario

Las palabras de este proyecto que **significan dos cosas**, y cuál usar.

No es un diccionario de cortesía. Cada entrada de aquí **ya ha causado un error o una
confusión documentada**; si una palabra no ha hecho daño, no está en esta lista.

Regla general: **cuando una palabra tiene dos significados, en prosa y en la UI se cualifica
siempre.** En el código pueden convivir si el contexto las separa (dos recursos distintos del
API, dos dataclasses distintas); en una frase o en un formulario, no.

---

## 1. Las colisiones que hacen daño

### `sample` — una **imagen**, no un ejemplo de entrenamiento

La peor de todas, porque contradice el uso normal en ML.

- `Sample` (`datasets/loader.py:53`) es **una imagen fuente** con su geometría.
- `manifest.num_samples: 200` = **200 imágenes**. `manifest.num_patches: 9800` = 9800 ejemplos.
- **El ejemplo de entrenamiento de esta CNN es el patch**, no el sample.

> **Es la raíz del malentendido central del proyecto**: "val tiene 980 patches" suena a muestra
> grande, pero son **20 imágenes** — y el tamaño de muestra efectivo es 20, porque los patches de
> una imagen están correlacionados (protocolo.md §1.1). Leer `num_samples` como "ejemplos de
> entrenamiento" es exactamente el error que ese número invita a cometer.

**Regla**: en prosa, **imagen** y **patch**. Nunca "muestra" a secas. `sample_idx` en el `.npz`
significa *índice de imagen*, y por eso confunde.

### `model` — la arquitectura (C) o lo entrenado (E)

- `configs/models/*.yaml`, `ModelConfig`, `/models` → **arquitectura** (C): config puro, sin pesos.
- "el modelo entrenado", `best.pt` → **run** (E).

**Regla**: **no se usa "model" a secas, nunca.** Es **red** (C) o **run** (E). Por eso `/models`
se renombra a `/networks` (api.md R2): la palabra ambigua desaparece del vocabulario.

### `dataset` — la fuente (A) o los patches (B)

- `/datasets`, `SourceDataset`, `labels.jsonl` → **fuente** (A): imágenes + geometría.
- `/patch-datasets`, `patches.npz`, `PatchDataset` → **dataset de patches** (B).

**Regla**: **fuente** (A) y **dataset de patches** (B). `/datasets` → `/sources` (api.md R2).

### `stride` — el de extracción (B) o el de inferencia (F)

Son **dos cosas distintas de verdad**, no un mal nombre:

| | Dónde | Qué es |
|---|---|---|
| **stride de extracción** | `BuildPatchRequest.stride` (schemas.py:18), `PatchExtractConfig` | Parte de la **identidad de B**. Fijo una vez construido |
| **stride de inferencia** | `PredictPathRequest.stride` (schemas.py:61), `detect_corners` | **Knob de F**: se elige por llamada, se ajusta post-hoc sin reentrenar |

Coinciden en el nombre y **no tienen por qué coincidir en el valor**. Que el de inferencia sea
gratis de barrer y el de extracción no, es justo la diferencia que el nombre común esconde
(organizacion.md §1-D, "los knobs baratos").

**Regla**: en prosa y en la UI, **siempre cualificado**. En el API pueden llamarse igual: son
recursos distintos.

### `seed` — la del split (B) o la del entrenamiento (D)

Igual de real, y más peligrosa porque afecta al protocolo:

| | Dónde | Qué fija |
|---|---|---|
| **semilla de split** | `BuildPatchRequest.seed` (schemas.py:22) | Qué imágenes caen en train/val/test (`_assign_splits`) |
| **semilla de entrenamiento** | `TrainRequest.seed` (schemas.py:44) | Init de pesos + shuffle |

**En un barrido, la de B se queda fija y la de D es el eje de réplica** (contrato ⑧). Confundirlas
hace que cada punto se evalúe sobre un split distinto: **medirías el ruido del split, no la
calidad de la receta** (protocolo.md §4).

### `kernel` — el tamaño (un int) o el tensor de pesos

- `backbone[].kernel: 3` (`builder.py:67`) → **el tamaño**, un entero.
- `kernels()`, "los kernels aprendidos", V1 → **los tensores de pesos**.
- `filters: 32` (`builder.py:66`) → **cuántos** hay.

Así que "un kernel de 3" y "ver los kernels" hablan de cosas distintas. *(El proyecto hermano usa
`kernel_size` y `kernels` para el tamaño y el número — más claro, y ojo al portar sus docs.)*

**Regla**: **`kernel_size`** para el tamaño (aunque el campo del YAML se llame `kernel`),
**filtros** para el número, **kernel** a secas solo para el tensor.

### `run` / `job` / `trial` / `experimento` — cuatro palabras vecinas

| Palabra | Qué es |
|---|---|
| **run** (E) | El **artefacto**: `runs/<name>/` con pesos, métricas y procedencia. Sobrevive al proceso |
| **job** (X) | La **ejecución** en segundo plano. Vive en memoria y hoy muere al reiniciar |
| **trial** | Un punto del espacio de un barrido, en vocabulario de **optuna**. **Lanza** un run |
| **experimento** | Palabra del **proyecto hermano** para lo que aquí es un run |

**Regla**: aquí se dice **run**. "Experimento" solo al hablar de NIST. Y **un trial no es un run**:
la frontera importa, porque optuna no debe dictar la organización (librerias.md).

### `evaluate` / `evaluación` — tres cosas

1. `evaluate()` (`training/loop.py:67`) → las **métricas de val por época**, dentro del bucle.
2. La **tabla por patch** (E×B, ui.md §3) → el substrato del diagnóstico.
3. `evaluations/<id>/` → una **entidad del proyecto hermano**.

**Regla**: **métricas de época** (1) y **tabla por patch** (2). "Evaluación" a secas, no.

### `best` — el checkpoint o el ganador del barrido

- `best.pt` → mejor **val loss** dentro de *un* run (`loop.py:166`).
- "el mejor run del barrido" → mejor **objetivo** (F1 de párrafo, protocolo.md §2).

**Son criterios distintos**, y hoy el primero está hardcoded. **Regla**: `best.pt` es
literal; para lo otro, **ganador**.

### `config` — tres

`PatchExtractConfig` (B), `ModelConfig` (C), `RunConfig` (B+C+D+X aplanado). Más
`runs/<name>/config.json`, que es el tercero congelado.

**Regla**: siempre cualificado — *config de extracción*, *config de red*, *config del run*.

---

## 2. Un concepto, dos nombres

El problema inverso, y es peor: dos nombres hacen creer que son dos cosas.

### `patch_size` (B) == `input_size` (C)

**El mismo número**, en dos sitios, sin nada que los una: es **el contrato ①**, el crítico. La
duplicación del nombre es la razón de que nadie note que deben cuadrar hasta que revienta dentro
del hilo del job.

No se unifican —cada dominio declara lo suyo, y eso es correcto— pero **quien escriba cualquiera
de los dos debe saber del otro**. Por eso es un contrato con test propio.

---

## 3. Términos propios, sin colisión

Para fijarlos, porque son del dominio y no se traducen:

| | |
|---|---|
| **patch** | Ventana `n×n` recortada de una imagen. **La entrada real de la CNN** |
| **corner / esquina** | Uno de los cuatro tipos: `TL, TR, BR, BL`. Orden fijo (`CORNER_NAMES`), horario desde arriba-izquierda |
| **border / flag de borde** | Si el patch está **pegado al borde de la imagen**: `top, right, bottom, left` (`BORDER_NAMES`). **No** es el borde de un párrafo |
| **quad** | Los 4 puntos de un bloque en la fuente, horario desde TL. Puede venir **rotado** (`Block.angle`) |
| **exists** | `p(hay una esquina de este tipo en este patch)`. La salida es `(4, 3)` = `[exists, x, y]` |
| **score** | `sigmoid(exists)`. En el proyecto hermano se llama `conf` |
| **receta** | Los hiperparámetros que definen el resultado (D). **No** incluye `device` |
| **huella / fingerprint** | Hash del contenido de un B. Distingue un dataset reconstruido bajo el mismo nombre (contrato ⑧) |
| **knob barato** | Parámetro de F ajustable **post-hoc sin reentrenar**: `threshold`, stride de inferencia, radio de NMS, `min_size` |
| **suelo de ruido** | La diferencia mínima creíble, medida con N semillas. Por debajo, empate (protocolo.md §4) |
