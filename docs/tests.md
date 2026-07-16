# Tests

Qué se testea en este proyecto, y por qué esa lista y no otra.

**La idea entera cabe en una frase:**

> Los contratos de [organizacion.md](organizacion.md) §2 **son** el plan de pruebas. Un contrato
> sin test es un comentario, y los comentarios se pudren.

---

## 1. Lo que ya hay, y lo que revela

11 tests, **11,5 s**, todos en verde:

| Fichero | Cubre |
|---|---|
| `test_extract.py` | geometría de la ventana, formas y etiquetas de B |
| `test_model.py` | forward, pérdida, un paso de entrenamiento, split, border |
| `test_api.py` | un end-to-end de humo |

Dos observaciones que motivan todo lo demás:

### 1.1 Ya hay tests de contrato, sin saberlo

`test_patch_dataset_border_backfill` y `test_border_features_change_head_and_forward` **son el
contrato ②** (el dataset siempre ofrece `border`; la red decide si lo usa; los `.npz` viejos
rellenan ceros). Están bien y pasan — solo que no se llaman así, así que cuando uno falle nadie
sabrá qué frontera se rompió.

### 1.2 Del contrato ⑤ se testea la mitad que no puede romperse

Y esto es la ilustración perfecta de por qué "testear funciones" no es "testear contratos".

`test_positions_cover_edges` y `test_positions_flush_when_not_divisible` prueban `_positions()`.
Pero `inference/predict.py:25` **importa** `_positions` de `patches/extract.py`: es la misma
función, no puede divergir. **Ese test no puede fallar nunca por la razón que importa.**

La mitad que sí puede romperse no la mira nadie: los **flags de borde** están **duplicados
literalmente**:

```python
# patches/extract.py:127-132
border = (int(y0 == 0), int(x0 + n >= w), int(y0 + n >= h), int(x0 == 0))

# inference/predict.py:69-70
borders.append((int(y0 == 0), int(x0 + n >= w), int(y0 + n >= h), int(x0 == 0)))
```

Hoy son idénticos. Si alguien toca uno, **nada se entera** y el modelo empieza a ver en
inferencia una geometría que no entrenó — con el fallo silencioso y difuso (predicciones algo
peores cerca de los bordes) en vez de una excepción.

**Regla que sale de aquí: testea la costura, no la función.** La pregunta no es "¿`_positions`
es correcta?" sino "**¿extracción e inferencia ven la misma ventana?**".

---

## 2. El mecanismo: `xfail(strict=True)` hace ejecutable el "lo que está roto"

La mayoría de los contratos **están rotos hoy** (organizacion.md §3). Un test para algo no
implementado no se omite ni se deja en rojo: se marca.

```python
@pytest.mark.xfail(strict=True, reason="contrato ①: no se valida en el backend, organizacion.md §3")
def test_contract_01_rejects_patch_size_mismatch():
    ...
```

Qué compra `strict=True`, que es la clave:

- Mientras el contrato siga roto → el test falla → **xfail esperado** → la suite pasa. Nadie
  convive con una suite roja (que es como se aprende a ignorarla).
- El día que alguien lo arregle → el test pasa → **XPASS estricto** → **la suite FALLA**, y
  obliga a quitar el marcador y actualizar §3.

> **Consecuencia: la documentación no puede desviarse de la realidad.** Hoy §3 es prosa, y la
> prosa envejece sola. Como xfails, no: la sección "lo que está roto" pasa a ser una lista
> ejecutable que el CI mantiene sincronizada en las dos direcciones.

Y sale gratis una barra de progreso: **el número de xfails que bajan es el avance de
[plan-ui.md](plan-ui.md)**.

---

## 3. Un test por contrato

En **`tests/test_contracts.py`**, nombrados por su número. `pytest tests/test_contracts.py -v`
debe leerse como el **parte de estado** de organizacion.md §2.

| | Contrato | El test afirma | Hoy |
|---|---|---|---|
| **①** | `patch_size == input_size` | `POST /runs` con desajuste → **400** (no una excepción dentro del hilo del job) | **xfail** — no hay validación |
| **②** | `border_features` | El `.npz` sin `border` carga ceros **si la red no lo usa**, y **falla si sí lo usa** | **parcial** — los dos tests existen, pero el caso que falla no está (formatos.md §2) |
| **③** | procedencia | El run registra `network`/`recipe` **por nombre** + huella de B; `DELETE` de un B en uso → **409** | **xfail** |
| **④** | checkpoint autodescriptivo | `load_model(ckpt)` reconstruye la red **sin** el YAML de C | ✅ probablemente — **falta escribirlo** |
| **⑤** | geometría compartida | **Los flags de borde de extracción == los de inferencia** para la misma ventana | **falta — y es el único que puede romperse en silencio** |
| **⑦** | dirección de dependencias | `itf.models` no importa nada de `itf.datasets` | **xfail** — hoy importa `NUM_BORDERS` |
| **⑧** | comparabilidad | Reconstruir B con otro contenido cambia su huella; los `seed` de B y de D son independientes | **xfail** — no hay huella |
| **⑨** | objetivo vs λ | `POST /sweeps` con `objective=loss` y `lambda_pos` en el espacio → **400** | **xfail** — no existe H |
| **⑩** | X fuera de D | Dos runs que solo difieren en `device` tienen la misma identidad de receta | **xfail** |

El ⑥ (el cruce A×B×F de la pestaña Predecir) se deja fuera a propósito: es una vista, no una
frontera estructural, y ya lo cubre el end-to-end del API.

**El ⑤ es el que hay que escribir primero.** Es el único de la lista que está *roto y en verde*:
no hay nada que avise, la duplicación es real y el fallo sería silencioso. Además es barato —
extraer un dataset diminuto, correr `detect_corners` sobre la misma imagen y comparar los flags
por `(x0, y0)`.

### La velocidad delata dónde está la validación

**Si un test de contrato necesita entrenar para detectar la violación, la validación está en la
capa equivocada.** El ① es testeable en milisegundos *precisamente porque* debe rechazarse antes
de entrenar. Que hoy solo se manifieste tras media hora de job es el síntoma.

Presupuesto: `test_contracts.py` por debajo de **~10 s**. Si sube, es una señal, no un problema
de la máquina.

---

## 4. Tests de arquitectura

El ⑦ no es una función: es una **dirección**. Se testea leyendo los imports, y es mecánico y
barato. La capa objetivo:

| Módulo | Puede importar de |
|---|---|
| `itf.geometry` *(no existe aún; la ventana compartida, contrato ⑤)* | — |
| `itf.datasets` (A) | — |
| `itf.models` (C) | **nada de `itf`** |
| `itf.patches` (B) | `datasets`, `geometry` |
| `itf.training` (D) | `patches`, `models` |
| `itf.inference` (F) | `models`, `geometry` — **no `patches`** |
| `itf.api` | todos |

Dos violaciones vivas: `models/builder.py` importa `NUM_BORDERS` de `datasets.loader` (⑦), e
`inference/predict.py` importa la **privada** `_positions` de `patches.extract` (⑤). Ambas se
vuelven verdes cuando exista `itf.geometry`, que es donde G debía estar desde el principio.

---

## 5. Reproducibilidad y retrocompatibilidad

**Reproducibilidad** (regla 1 de [protocolo.md](protocolo.md) §7): *misma semilla + misma config
⇒ mismos pesos*. El proyecto hermano tiene ese test; ITF **no**. Debería pasar hoy
(`manual_seed` + `num_workers=0` cubren el shuffle), pero **nadie lo ha comprobado**, y sin él
todo el protocolo se apoya en una suposición.

**Retrocompatibilidad de formatos** — no es hipotética, ya ha pasado y va a volver a pasar:

- `.npz` sin `border` → ceros. **Ya está testeado** (`test_patch_dataset_border_backfill`), y es
  el modelo a seguir.
- `config.json` sin procedencia (`network`/`recipe`/huella) → debe leer degradando, **no
  reventar**. Los runs de `runs/` son reales y la fase 4 les añade campos. Es la trampa más
  probable del plan y merece su test **antes** de la fase 4.

---

## 6. Lo que NO se testea

Tan importante como lo de arriba:

- **Resultados de investigación.** Un test que afirme `f1 > 0.75` **no es un test: es una
  afirmación de investigación disfrazada**. Rompe por razones legítimas —cambiar el dataset,
  arreglar `smooth_l1_beta`— y entrena a la gente a "arreglar" el umbral hasta que pase. Los
  resultados van al protocolo (baseline con N semillas), no a pytest.
  **La frontera: los tests afirman invariantes; el protocolo mide resultados.**
- **Valores numéricos exactos de la pérdida.** Frágiles y sin información. `test_training_step_reduces_loss` sí vale:
  afirma un invariante (los gradientes fluyen), no un número.
- **Que torch funcione.**
- **El render de la UI píxel a píxel.** Las reglas de forma y color de ui.md §4.0 se revisan, no
  se testean — salvo la paleta, que **sí** tiene un validador ejecutable y hay que correrlo.

---

## 7. Convenciones

- **Datasets sintéticos diminutos**, construidos en el test. Es lo que ya hace `test_extract.py`
  y por lo que la suite tarda 11 s. Nunca se toca `data/` ni `runs/` reales.
- Un test de contrato **se llama como su contrato** (`test_contract_05_...`): cuando falla, el
  nombre dice qué frontera se rompió, sin leer el código.
- El `reason` de un `xfail` **cita el documento y la sección**. Un xfail sin razón es un test
  apagado.
- **Antes de cada commit que toque código**: `.\.venv\Scripts\python -m pytest -q`.

---

## 8. Por dónde empezar

1. **⑤ — la geometría.** El único roto y en verde. Barato, y hoy nada avisa.
2. **Reproducibilidad.** Sostiene el protocolo entero y probablemente ya pasa; hay que probarlo.
3. **Renombrar y mover** los dos tests de ② a `test_contracts.py`. Coste cero, y arranca el
   fichero.
4. **Los xfails** del resto (①, ③, ⑦, ⑧, ⑨, ⑩). Escribirlos **ahora**, rotos: convierten §3 en
   una lista viva y le dan al plan su barra de progreso.
5. **④** — escribirlo; debería pasar.

Del 4 en adelante, cada fase de plan-ui.md tiene un deber añadido: **quitar sus xfails**. Una
fase que implementa un contrato y deja el xfail puesto no está terminada — y la suite lo dirá,
porque el XPASS estricto la pone en rojo.
