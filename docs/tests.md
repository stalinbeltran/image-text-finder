# Tests

Qué se testea en este proyecto, y por qué esa lista y no otra.

**La idea entera cabe en una frase:**

> Los contratos de [organizacion.md](organizacion.md) §2 **son** el plan de pruebas. Un contrato
> sin test es un comentario, y los comentarios se pudren.

---

## 1. La lección de la suite anterior

`tests/` se borró con el resto del código (2026-07-16); vive en el tag `pre-rediseno`. Eran 11
tests en 11,5 s, todos en verde. **Merece la pena entender por qué no bastaban**, porque la
suite nueva se escribe justo al revés.

### 1.1 Había tests de contrato, sin saberlo

`test_patch_dataset_border_backfill` y `test_border_features_change_head_and_forward` **eran el
contrato ②**. Bien escritos y pasando — pero no se llamaban así, y **les faltaba el caso que
importa**: que el sistema se **niegue** cuando la red pide `border` y el dataset no lo tiene
(formatos.md §2). Probaban que el relleno *ocurre*, no que el fallo *se detecte*.

### 1.2 Del contrato ⑤ se testeaba la mitad que no puede romperse

La ilustración perfecta de por qué "testear funciones" no es "testear contratos", y la razón de
que este documento exista.

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

Eran idénticos, y por eso funcionaba. Pero si alguien tocaba uno, **nada se enteraba**: el
modelo empezaba a ver en inferencia una geometría que no entrenó, con el fallo silencioso y
difuso (predicciones algo peores cerca de los bordes) en vez de una excepción.

**Regla que sale de aquí: testea la costura, no la función.** La pregunta no es "¿`_positions`
es correcta?" sino "**¿extracción e inferencia ven la misma ventana?**".

> En el diseño nuevo la costura desaparece: la ventana vive en **`itf.geometry`** y la importan
> B y F (§4). Pero **el test sigue haciendo falta** — precisamente porque duplicar esas seis
> líneas es lo que sale natural cuando escribes la inferencia y no quieres importar del
> extractor. El test es lo que impide que la duplicación vuelva.

---

## 2. El mecanismo: `xfail(strict=True)` convierte los contratos en la lista de tareas

Con el árbol vacío (2026-07-16), **ningún contrato está implementado todavía**. Un test para algo
que aún no existe no se omite ni se deja en rojo: se marca.

```python
@pytest.mark.xfail(strict=True, reason="contrato ①: sin implementar, plan-ui.md fase 4")
def test_contract_01_rejects_patch_size_mismatch():
    ...
```

Qué compra `strict=True`, que es la clave:

- Mientras el contrato no exista → el test falla → **xfail esperado** → la suite pasa. Nadie
  convive con una suite roja (que es como se aprende a ignorarla).
- El día que su fase lo implemente → el test pasa → **XPASS estricto** → **la suite FALLA**, y
  obliga a quitar el marcador.

> **Los tests de contrato son la lista de tareas del proyecto, y es ejecutable.** Escribe los diez
> **ahora**, todos en xfail: definen el destino antes de escribir una línea de `src/`. Cada fase
> de [plan-ui.md](plan-ui.md) se mide por **cuántos xfails quita**, y no puede olvidarse de
> hacerlo: el XPASS estricto la pone en rojo.
>
> Es lo más parecido a TDD que admite un proyecto de investigación: **los invariantes se fijan
> primero; los resultados, nunca** (§6).

Y como el `reason` cita el documento, la lista **no puede desviarse de la realidad**: la prosa
envejece sola, un xfail estricto no.

---

## 3. Un test por contrato

En **`tests/test_contracts.py`**, nombrados por su número. `pytest tests/test_contracts.py -v`
debe leerse como el **parte de estado** de organizacion.md §2.

La columna que lleva la información es **qué fase lo quita**: es la barra de progreso del plan.

| | Contrato | El test afirma | Lo quita | |
|---|---|---|---|---|
| **①** | `patch_size == input_size` | `check_compatible` con desajuste → `patch_size_mismatch`, con los dos números y su arreglo (no una excepción dentro del hilo del job) | **3** | ✅ |
| **①** | ídem, **por HTTP** | `POST /runs` con el desajuste → **400, y ni job ni run creados** | **4** | ✅ |
| **②** | `border_features` | El `.npz` sin `border` carga ceros **si la red no lo usa**, y **falla si sí lo usa** | **3** | ✅ |
| **②** | ídem, **por HTTP** | `POST /runs` con red que pide bordes sobre B que no los trae → **400** | **4** | ✅ |
| **③** | procedencia | El run registra `network`/`recipe` **por nombre** + huella de B | **4** | ✅ |
| **③** | B en uso | `DELETE` de un B que un run referencia → **409 con la lista** | **2** | ✅ |
| **④** | checkpoint autodescriptivo | `load_model(ckpt)` reconstruye la red **sin** el YAML de C | **4** | ✅ |
| **⑤** | geometría compartida | **Los flags de borde de extracción == los de inferencia** para la misma ventana (`extract.windows is predict.windows`, y los flags escritos == `geometry`) | **6** | ✅ |
| **⑦** | dirección de dependencias | `itf.models` importa de `itf.geometry` (G) pero **no** de `itf.datasets` (A); `itf.validation` no importa nada de `itf` | **3** | ✅ |
| **⑧** | comparabilidad | Reconstruir B con otro contenido cambia su huella; la semilla de B sola decide el split | **2** | ✅ |
| **⑨** | objetivo vs λ | `POST /sweeps` con `objective=loss` y `lambda_pos` en el espacio → **400** (`objective_varies_with_space`); control: el mismo espacio con `objective=f1` **pasa** el ⑨ | **7** | ✅ |
| **⑩** | X fuera de D | Dos runs que solo difieren en `device` tienen la misma identidad de receta | **3** | ✅ |

> **① y ② empezaron en la fase 3 y los cerró la 4** *(deuda pagada, 2026-07-16)*. La fase 3 los
> afirmaba sobre el **validador** (`check_compatible`), que es donde vive la regla, y `itf-train` lo
> llamaba antes del primer batch — así que el camino del **CLI** estaba cerrado de verdad, pero
> `POST /runs → 400` no existía y nada obligaba a la fase 4 a llamar al validador. Ahora los dos
> tienen su test por HTTP, y **lo que afirman no es solo el 400**: es que la negativa llega **antes
> de crear el job y antes de reservar el nombre**. Un 400 que dejara el `runs/<name>/` puesto
> cumpliría la letra del contrato y dejaría un cadáver en cada equivocación.
>
> Las dos puertas preguntan a **`check_run`**, una sola función (`check_compatible` +
> `check_measurable`). No es cortesía: dos comprobaciones separadas se desincronizan, y la puerta
> que quede más laxa es por la que entra un barrido.

> **Un xfail necesita un control, y por eso ③ son dos tests.** «Borrar un B en uso da 409» lo
> cumpliría también un `DELETE` que fallara siempre. El control —borrar uno libre da 204— es lo
> que hace que el 409 signifique *en uso* y no *roto*.

> **① y ② comparten validador** (`itf.validation`, organizacion.md §2): sus tests son dos casos de
> la misma función pura, no dos mecanismos. Y como es pura, **corren en milisegundos sin entrenar**
> — que es exactamente la señal de que la validación está en la capa correcta.

El ⑥ (el cruce A×B×F de la pestaña Predecir) se deja fuera a propósito: es una vista, no una
frontera estructural, y ya lo cubre el end-to-end del API.

**El ⑤ sigue siendo el más importante de la lista, y ahora por otra razón.** Antes era *el único
roto y en verde*. Hoy no hay código que romper — pero **duplicar esas seis líneas es lo que sale
natural** cuando escribes la inferencia en la fase 6 y no te apetece importar del extractor. El
test es lo que impide que la duplicación **vuelva**, y su xfail es lo que hace que no se olvide.

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
| `itf.geometry` (G: el vocabulario y la ventana) | — |
| `itf.datasets` (A) | `geometry`, `imageops`, `validation` — **solo el resize** (D19). El *lector* de A no importa nada, como siempre; quien importa es `datasets/resize.py`, que es la composición de los dos mecanismos y por definición conoce a ambos. Es la forma del arreglo de ⑦, no una excepción: A necesita coordenadas (G) y píxeles, y la alternativa —reescribir el escalado de quads dentro del loader— son dos fórmulas que pueden divergir |
| `itf.matrixview` (matriz → payload) | **nada de `itf`** — es puro: arrays. Y así se queda: es lo que la mantiene extraíble (librerias.md §5) |
| `itf.models` (C) | **solo `geometry`** |
| `itf.validation` (contratos ① y ②) | **nada de `itf`** — es puro: dos dicts |
| `itf.imageops` (píxeles: resize proporcional) | **nada de `itf`** — es puro: una imagen y una dimensión. No sabe qué es un quad, y eso es lo que lo hace servir para una imagen cualquiera (D19, librerias.md §2) |
| `itf.metrics` (qué significan los números) | **nada de `itf`** — es puro: arrays |
| `itf.patches` (B) | `datasets`, `geometry` |
| `itf.training` (D) | `patches`, `models`, `validation`, `metrics`, `geometry` |
| `itf.inference` (F) | `models`, `geometry`, `matrixview` — **no `patches`** |
| `itf.diagnostics` (E×B) | `patches`, `training`, `inference`, `metrics`, `geometry` |
| `itf.api` | todos |

> **`itf.metrics` nació en la fase 5, y por la razón de `itf.geometry`** — no importa nada de `itf`
> porque toma **arrays, no objetos**: no sabe qué es un run ni un dataset. `pos_err_px` se
> calculaba en dos sitios (`evaluate()` cada época; la tabla por patch para V7) y **tenían que
> coincidir sin que nada lo comprobara**: la forma exacta del ⑤ con otro nombre. El sitio correcto
> se construye cuando aparece el segundo lector, no cuando duele.
>
> **`itf.diagnostics` está arriba, y eso es correcto**: es el cruce de E y B, así que conoce a los
> dos por definición. Lo que no puede es que alguien importe **de** él, salvo `itf.api`.

> **`itf.models` importa de `geometry`, y eso *es* el arreglo del ⑦, no una excepción a él.**
> Una versión anterior de esta tabla decía «nada de `itf`» para C, y se contradecía con su propio
> párrafo de abajo y con organizacion.md ⑦, que dice que el vocabulario «quiere su propio módulo»
> — o sea, que **C lo importe de G**. Lo que ⑦ prohíbe es que la red importe del **cargador de la
> fuente** (`itf.datasets`) para saber que hay 4 bordes: eso es que C sepa que A existe. G es el
> suelo que todos comparten; que la cabeza saque `(B, 4, 3)` con un 4 escrito a mano en `models`
> **y** otro en `geometry` sería el mismo error, con más pasos. *(Corregido en la fase 3.)*

`itf.validation` no importa nada porque **compara diccionarios, no objetos**: el manifest de B
contra la config de C. Eso es lo que le permite correr en milisegundos y ser llamado tanto por
`train()` como por el API (api.md §3).

**Las dos violaciones que había ya no existen** —se borraron con el código— pero la tabla no está
para celebrarlo: está para que **no vuelvan**. Las dos nacieron de la misma comodidad, y las dos
volverían igual:

| Lo que pasó (en el tag) | Por qué sale solo |
|---|---|
| `models/builder.py` importaba `NUM_BORDERS` de `datasets.loader` (⑦) | La constante estaba ahí y el import funciona. **Nadie decide** que la red dependa del cargador de la fuente: se teclea |
| `inference/predict.py` importaba la **privada** `_positions` de `patches.extract` (⑤) | Reusar es la reacción correcta; el fallo fue que **el sitio correcto no existía** |

`itf.geometry` e `itf.validation` son ese sitio. Nacen ahora, no cuando duelan.

---

## 5. Reproducibilidad

**Regla 1 de [protocolo.md](protocolo.md) §7**: *misma semilla + misma config ⇒ mismos pesos*. El
proyecto hermano tiene ese test; ITF nunca lo tuvo. **Sin él, el protocolo entero se apoya en una
suposición** — y las cinco reglas de comparación empiezan por dar por hecho que un run repetido
se repite.

Debería pasar en cuanto exista el bucle (`manual_seed` + `num_workers=0` cubren el shuffle), pero
*debería pasar* no es *pasa*. Va con la fase 4, junto al resto de E.

> **Escrito, y son dos tests, no uno** *(fase 4, 2026-07-16)*. El barato afirma la
> **inicialización**; el que importa afirma el **entrenamiento**, que es lo que la regla dice. Entre
> los dos están el shuffle, el batching y el optimizador — que es justo donde se esconde una semilla
> olvidada. Los dos llevan su control con otra semilla: sin él, «se repite» lo cumpliría también un
> bucle que ignorara la semilla entera.
>
> Y de regalo, la verificación de la fase 4 lo enseñó **fuera de pytest**: el mismo run lanzado
> desde la UI y desde `itf-train` dio los mismos números hasta el último decimal (val loss
> `0.8602307364344597`). Dos puertas, un resultado.

### 5.1 La reanudación es un caso de reproducibilidad, no una utilidad

> **El test de abajo no existe todavía porque lo que describe no está construido** *(2026-07-19)*:
> reanudar el **barrido** sí está hecho y testeado (`test_resume_clears_the_stop_request`,
> `test_resume_refuses_a_finished_sweep_with_a_reason`); reanudar **dentro de un trial** es D21.

Reanudar un entrenamiento interrumpido **pretende ser X** —mismo resultado, menos tiempo perdido—,
y la implementación ingenua lo convierte en D en silencio: sin el estado del optimizador y del RNG,
el run reanudado tiene el mismo nombre y la misma procedencia que el no interrumpido, y **otros
pesos**. Ver organizacion.md ⑪ y formatos.md §4.2.2.

Por eso el test no pregunta «¿reanuda?» sino **si reanudar se nota**:

> **N épocas de tirón == N−k épocas + reanudar las k que faltan**, hasta el último bit.

Es la regla 1 de protocolo.md §7 aplicada a la interrupción, y **hereda su control**: con otra
semilla los pesos deben diferir, o «son iguales» lo cumpliría también un bucle que no entrena.
Conviene además afirmar los dos fallos silenciosos de al lado, porque ninguno rompe nada visible:
que `metrics.jsonl` tenga **N líneas y no N+k** (si no, V14 dibuja una curva que retrocede), y que
**`best.pt` no empeore** al reanudar (sin `best_monitor` la primera época de la segunda vuelta se
cree la mejor y machaca un checkpoint superior).

**Retrocompatibilidad: no hay ninguna que testear, y es una simplificación de D18.** Este
documento pedía un test para leer `config.json` sin procedencia degradando; **D3 lo mató**: no
queda ningún run viejo, así que todo run nace completo y un `config.json` sin procedencia es un
error, no un caso legado (formatos.md §4.2). Lo mismo con el `.npz` sin `border`: el caso que se
testea **no es el relleno, es la negativa** (§3, contrato ②).

> Que este documento haya perdido su sección de retrocompatibilidad es el efecto que D18 buscaba:
> **con la casa vacía, el código de migración no se escribe.**

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

- **Datasets sintéticos diminutos**, construidos en el test. Es lo que hacía `test_extract.py`
  (en el tag) y por lo que la suite tardaba 11 s. Nunca se toca `data/` ni `runs/` reales.
- Un test de contrato **se llama como su contrato** (`test_contract_05_...`): cuando falla, el
  nombre dice qué frontera se rompió, sin leer el código.
- El `reason` de un `xfail` **cita el documento y la sección**. Un xfail sin razón es un test
  apagado.
- **Antes de cada commit que toque código**: `.\.venv\Scripts\python -m pytest -q`.

---

## 8. Por dónde empezar

Con el árbol vacío ya no hay un orden que elegir: **se escriben los nueve de una vez, todos en
xfail, y son la fase 0.5 de [plan-ui.md](plan-ui.md)**. Ninguno puede pasar todavía —no hay
`src/`— y eso es exactamente lo que los hace útiles: definen el destino antes de la primera línea
de código.

Lo único que se recupera del tag es el **material**, no los tests:

```powershell
git show pre-rediseno:tests/test_extract.py    # cómo se construía el dataset sintético diminuto
```

Los dos tests de ② (`test_patch_dataset_border_backfill`,
`test_border_features_change_head_and_forward`) **estaban bien escritos y eran el contrato ② sin
saberlo** (§1.1). Se reescriben con el caso que les faltaba —que el sistema **se niegue** cuando
la red pide `border` y el dataset no lo tiene— y con su número por nombre.

Y a partir de aquí, cada fase de plan-ui.md tiene un deber añadido: **quitar sus xfails** (§3
dice cuáles). Una fase que implementa un contrato y deja el xfail puesto **no está terminada**, y
la suite lo dirá sola: el XPASS estricto la pone en rojo.
