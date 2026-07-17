# Plan de construcción

Plan paso a paso para construir el sistema con la organización de [organizacion.md](organizacion.md),
proyectada sobre HTTP en [api.md](api.md) y sobre pantallas en [ui.md](ui.md). **Este documento se
ejecuta; los otros mandan.**

Cada fase termina con: la app **arranca**, los tests **pasan**, y hay un commit. Ninguna fase
deja el árbol a medias.

---

## 0. Punto de partida: el árbol está vacío

**No hay código.** `src/`, `web/`, `tests/` y los `configs/*.example.yaml` se borraron
(2026-07-16) para construir desde el diseño sin nada viejo que imitar por error. Todo sigue
recuperable en el tag **`pre-rediseno`**:

```powershell
git show pre-rediseno:src/itf/patches/extract.py     # un fichero
git checkout pre-rediseno -- src/                    # el paquete entero
```

**Consúltalo cuando reconstruyas un algoritmo, no una estructura.** Lo que había estaba mal
*organizado* (dominio dentro de `app.py`, C+D+X en un formulario), pero varios algoritmos eran
correctos y tenían tests. Lo que solo vivía allí y no está en los docs:

| | Dónde, en el tag |
|---|---|
| El desempate por distancia al centro al etiquetar esquinas | `extract.py` |
| El borde *flush* de `_positions` (la ventana cubre `[0, size)`) | `extract.py` |
| El radio de NMS = `stride/2` | `predict.py` |
| La normalización de `pos_loss` por `mask.sum()` | `losses.py` |
| El emparejado voraz TL→BR de la reconstrucción | `predict.py` |

Los datos (`data/`, `runs/`) también se borraron: no estaban en git, D6 los regenera, y
quitarlos **simplifica el diseño** — mata D3 (migrar configs viejos), mata el camino de relleno
de `border` en `PatchDataset`, y `format_version` nace en 1 con todo presente.

> **Lo que las 161 citas de los docs referencian** (`app.py:61`, `dataset.py:27-28`,
> `predict.py:69-70`…) resuelve contra el tag. Son **hallazgos históricos que motivaron el
> diseño**, no descripciones del árbol de hoy.

---

## 1. Cómo se construye

**Vertical, dominio a dominio.** Cada fase se lleva backend y front del mismo dominio, y acaba
con algo que funciona de punta a punta. No hay una fase "el backend" y otra "el front": de las
nueve pantallas, tres dependen de backend que no existe, y un plan solo-front se atasca en la
fase 3.

Con el árbol vacío no hay convivencia con lo viejo, así que **no hay grupo `legacy` ni nada que
borrar fase a fase** (era D15, que queda sin objeto). A cambio, la regla se vuelve más
importante: **cada fase deja la app arrancando y los tests en verde.** Una fase que deja el árbol
roto "hasta la siguiente" es un big-bang disfrazado.

Y no hace falta la UI para entrenar: `itf-train` (fase 3) da el CLI mucho antes de que haya
pantallas.

---

## 2. Las fases

### Fase 0 — Cerrar decisiones abiertas *(sin código)* — ✅ **hecha (2026-07-16)**

Eran **dos**, y las dos están cerradas ([decisiones.md](decisiones.md) §4):

1. **D2 — forma de la procedencia** (contrato ③): nombre + valor de C y D, huella de B,
   `sweep`, `git_commit` y **`environment`**. La escribe la fase 4 y la lee todo lo demás.
   → **formatos.md §4.2.1**, api.md §3.
2. **D16 — el holdout**: 500 imágenes, fuente propia, misma config del generador, generado lo
   primero. Bloqueaba el **paso 0 del protocolo**, o sea todo. → **protocolo.md §3**.

*(D1 —tabla por patch— ya estaba cerrada: es un **caché**, así que no hay entidad ni pantalla.)*
*(La tercera pregunta —qué hacer con los `config.json` ya entrenados— **murió con el borrado**:
no queda ninguno. `device` sale de la identidad de D desde el primer run, sin migración ni
retrocompatibilidad.)*

### Fase 0.5 — Los contratos, en xfail *(tests, sin `src/`)*

**Antes de la primera línea de `src/`**, y es [tests.md](tests.md) §2 quien lo pide: los diez
tests de contrato se escriben **ahora**, todos en `xfail(strict=True)` citando su documento.

Esta fase existe porque faltaba: cada fase de abajo tiene el deber de **quitar sus xfails**, pero
ninguna era dueña de **crearlos**. Sin ella el plan no tiene barra de progreso — y con ella,
`pytest tests/test_contracts.py -v` **es** el parte de estado de organizacion.md §2.

**Entregable**: `tests/test_contracts.py`. **Verificación**: la suite pasa en verde, con los diez
como xfail esperados.

### Fase 1 — Esqueleto y paleta *(front, desde cero)* — ✅ **hecha (2026-07-16)**

1. Nav de 4 grupos (Datos / Modelo / Entrenar / Analizar), rutas, layout, convenciones de carga
   y error. **Sin números de paso.** → `web/src/nav.ts`, `App.tsx`, `components/Async.tsx`.
2. **La paleta** (D12) → **`web/src/theme/tokens.css`**, y solo ahí. La valida
   `npm run validate:palette`, que **parsea ese fichero** — valida lo que se sirve, no una copia
   que deriva.
3. Componentes base: `MatrixCanvas` (el `drawMap` portado, normalización **por mapa**), `Meter` y
   `NumberTable` (R5). Se miran en `/kitchen`, con datos sintéticos.

**Verificado**: `npm run dev` arranca en 5173; el validador pasa en claro y en oscuro (exit 0);
`tsc --noEmit` y `npm run build` limpios; y la app **se abrió en Chrome headless en los dos
modos** — React monta, la cascada resuelve los tokens del modo activo, y el toggle `data-theme`
gana sobre el ajuste del SO.

**Lo que se aprendió, y cambia una regla** *(escrito en ui.md §4.0 R1)*: las 4 esquinas hay que
validarlas con **`--pairs all`**, no con la lista de adyacentes — cualquiera de las cuatro puede
quedar pegada a cualquier otra (4 meters en V3, 4 overlays en V11, la rejilla de V9), así que un
choque entre no vecinas no se vería. Con esa exigencia, **de las 70 formas de escoger 4 de los 8
hues documentados solo 2 pasan** en ambos modos. La paleta no se eligió: se enumeró.

### Fase 2 — Datos: Fuentes (A) y Patches (B) — ✅ **hecha (2026-07-16)**

**Es la que crea `src/itf/`**, así que también es la que arregla el `pip install -e .` roto.

1. **Front**: Fuentes (solo lectura, con los párrafos dibujados en SVG sobre la imagen) y Patches
   (listar + construir + borrar).
2. Patches muestra **el desbalance**. Medido en `clean-paragraphs-01/reducido`: **21,6 % / 3,6:1**
   — muy cerca del 20,5 % / 3,9:1 documentado, o sea que la corrección de protocolo.md §1.4 se
   sostiene sola.
3. **Back**: `itf.geometry` (G: vocabulario + la ventana, contratos ⑤ y ⑦), `itf.datasets` (A),
   `itf.patches` (B, con la huella de ⑧), `itf.training.registry` (la mitad lectora de E, que es
   quien contesta `used_by`), `itf.api` con `/sources`, `/patch-datasets` y `/jobs`.
4. **D4 implementado**: CORS cerrado a `localhost:5173` y las rutas resueltas **dentro** del
   dominio — el cliente manda un id, nunca una ruta. `GET /image?path=` no vuelve.

**Verificado**: 12 tests pasan (10 siguen en xfail); dataset construido **desde la UI**; borrar
uno en uso da **409 con la lista de runs** y no borra nada. El `itf-extract` produce **la misma
huella** que la UI construyendo en otra ruta con otro nombre — que es exactamente lo que el
contrato ⑧ pide.

**Xfails que quita**: los dos de ⑧ (huella, semilla de split) y el `DELETE` de ③.

**Lo que apareció al construir y no estaba en el diseño**:

- **El aviso de val vacío tiene sitio: el manifest.** protocolo.md §1.3 pedía «falla o avisa» sin
  decir dónde. Ahora `manifest.warnings[]` lo lleva y la UI lo enseña en rojo. Se ve solo: el
  ejemplo del README (`reducido`, 5 imágenes) sale **4/0/1** y dispara el aviso. La negativa dura
  es de la fase 4, donde está el daño.
- **La cola nace con `max_workers=1`**, no con un hilo por job. Es la fase 7 quien la hace de
  verdad (persistencia, cancelar), pero el límite no es una feature que se añade después: es la
  trampa, y añadirlo luego sería tocar la cola con jobs corriendo.

### Fase 3 — Modelo: Redes (C) y Recetas (D) — ✅ **hecha (2026-07-16)** ← *era el desbloqueo*

La fase que lo condiciona todo: sin identidad para C y D no hay Entrenar limpio ni H.

1. **Back — D como entidad**: almacén (`configs/recipes/*.yaml`) + `GET/POST/GET{name}/DELETE
   /recipes`.
2. **Back — C**: `/networks` completo (CRUD + `POST /networks/validate`). El nombre `/models` no
   vuelve: era la palabra ambigua (api.md R2).
3. **Back — el catálogo de hiperparámetros entero** (§1-D de organizacion.md). **Dos son
   trampas por defecto** y vuelven solas si no se ponen a propósito (organizacion.md §3):
   - `momentum` — si al optimizador solo le pasas `lr` y `weight_decay`, **SGD corre a momentum
     0** y cualquier comparación de optimizadores queda sesgada a favor de Adam.
   - `smooth_l1_beta` — el default de PyTorch es 1.0, y con coordenadas en [0,1] eso hace la
     pérdida de posición **MSE pura**: el Huber nunca se activa.
   - Y los que hay que añadir: `scheduler` (el más rentable: sin él `lr` es constante),
     `grad_clip`, `patience`/`min_delta`, `monitor` explícito.
4. **Front**: pantalla Redes (solo arquitectura, + traza espacial `40→20→10→5` y nº de params,
   que es gratis y es lo único que una red sin entrenar puede enseñar) y pantalla Recetas (solo
   D, agrupada como el catálogo, **con la definición de cada campo en línea**).
5. `device` **no** aparece en Recetas: es X, va en Entrenar.

**Verificado**: 42 tests pasan (4 siguen en xfail). Red y receta **creadas desde la UI de verdad**
(Chrome por CDP, clic real en el formulario): la **traza se recalcula mientras escribes** —bajar
`input_size` a 32 la movió a `32 → 16 → 8` en vivo— y ambas aterrizan como YAML con
`format_version`. Y **se entrena por CLI**: `clear-paragraphs-02-reducidos` → 20 épocas en
**7,2 min** (21,7 s/época, contra los 6,7 min y ~20 s que predecía protocolo.md §1), **F1 0,80**,
`pos_err_px` **9,4**. `best.pt` salió de la **época 17**, no de la 20 — la selección por `val_loss`
no es decorativa. `config.json` congela red y receta por valor con `execution` **fuera**
(contrato ⑩) y **sin** `format_version` dentro de la red. `pos_weight: 3.9` llega vivo hasta la
BCE: el recall sube de 0,54 a **0,85** a costa de la precisión, que es justo lo que debe hacer.

**Xfails que quita**: ①, ② (×2), ⑦ (×2) y ⑩. Quedan ③ y ④ (fase 4), ⑤ (fase 6) y ⑨ (fase 7).
**Con una deuda explícita**: ① y ② los afirma hoy el **validador**, y `itf-train` lo llama — pero
`POST /runs → 400` no existe aún. **La fase 4 debe extender esos dos tests a HTTP** (tests.md §3);
mientras tanto nada obliga a que su `POST /runs` llame al validador.

**Lo que apareció al construir y no estaba en el diseño**:

- **`configs/models/` → `configs/networks/`.** formatos.md §4.3 decía `models/` mientras
  glosario.md §1 prohibía la palabra y api.md R2 la borraba del vocabulario: era una
  contradicción entre documentos, no una decisión. El directorio estaba vacío ⇒ coste cero.
- **`augment` y `sampler` se quedan fuera, y ahora está escrito por qué** (organizacion.md §1-D).
  No eran un hueco: un flip convierte una TL en TR e invalida los flags de borde, **y el fallo es
  silencioso**. Implementarlo mal es peor que no tenerlo.
- **La negativa por val vacío tiene tipo propio** (`NoValidationSplitError`). Salía como un
  *traceback*, y un traceback se lee como «la herramienta está rota» — que invita justo a
  rodearlo, y rodearlo **es** la trampa (best.pt por train loss). Ahora es un mensaje y un exit 2.
- **`itf-extract --source` admite id, ruta relativa o absoluta, y si falla lista las fuentes
  reales.** Nació de una trampa que se cobró esta misma verificación: hay **dos**
  `clear-paragraphs-02` (160×160 y 640×480) y **coger la equivocada no falla** — construye un B
  válido con 713 patches/imagen en vez de 49 y un desbalance de ~67:1 en vez de 3,9:1. La
  verificación llegó a «demostrar» que protocolo.md §1 estaba mal por 16×. **No lo estaba: era la
  fuente equivocada**, y toda la tabla reproduce exacta (protocolo.md §1). Una medición contra la
  fuente que no es no se parece a un error: se parece a un hallazgo.

### Fase 4 — Entrenar y Runs (E) — ✅ **hecha (2026-07-16)**

1. **Back**: validar el **contrato ①** en `POST /runs` (400 con la razón si `patch_size !=
   input_size`) llamando a `itf.validation` (organizacion.md §2): el mismo validador cubre ① y ②
   de una vez. Escribir la **procedencia por nombre** (fase 0.2). Sacar X de la identidad.
2. **Back**: `POST /runs` **no sobrescribe en silencio** un run existente → 409. (Era una trampa
   del código viejo: `mkdir(exist_ok=True)` + truncar `metrics.jsonl` machaca resultados sin
   avisar, y un barrido que autogenera nombres es justo quien la pisa.)
3. **Front**: pantalla Entrenar (elegir B + C + D; `device` aparte; **estimar el coste** con los
   `seconds` de `metrics.jsonl`). Pantalla Runs: procedencia por nombre. **"Re-entrenar la misma
   red" no es un modo**: es *elegir otra receta con la misma C*, que es lo que siempre fue — con
   C y D separadas sale gratis y no hace falta ningún `lockArchitecture`.

**Verificado**: 68 tests pasan (2 siguen en xfail). Run entrenado **desde la UI de verdad** (Chrome
por CDP, clic real): `runs/fase4-ui/config.json` lleva la procedencia completa —`fase3-red` +
huella, `cnn-a` y `corta-2ep` por nombre **y** por valor, `sweep: null`, commit `b89e5e15…+sucio`
y `environment`—, con `execution` fuera y **sin `data`**. Reusar el nombre da **409** con razón y
arreglo, y no toca lo que hay. La **parada** clicada en la época 2 cerró en la 3 como `cancelled`,
con `best.pt` en su sitio. Y el **coste estimado** apareció solo en cuanto hubo un run comparable:
25,5 s/época × 20 ≈ 8,5 min.

**La prueba de reproducibilidad que no estaba planeada**: el mismo run lanzado por la UI y por
`itf-train` dio **los mismos números hasta el último decimal** (val loss `0.8602307364344597`,
f1 `0.6734793187347932`). Dos puertas distintas, un solo resultado — que es la regla 1 de
protocolo.md §7 sostenida en la práctica, no solo en el test.

**Xfails que quita**: ③ (procedencia) y ④ (checkpoint autodescriptivo). Quedan ⑤ (fase 6) y ⑨
(fase 7). **Y paga la deuda de la fase 3**: ① y ② ahora tienen su test por HTTP — lo que afirman no
es solo el 400, es que **no se crea ni el job ni el run**.

**Lo que apareció al construir y no estaba en el diseño**:

- **`GET /runs/{name}` contestaba 404 «config.json ilegible» sobre un run sano** — y lo cazó **un
  test flaky, no el razonamiento**. `write_text` trunca y luego escribe, así que un sondeo que caía
  en esa ventana leía un JSON a medias y concluía que el run estaba corrupto; `status.json`, que se
  reescribe **cada época**, podía hacer parpadear a `error` un run que iba perfectamente. La cura es
  `os.replace`, **pero en Windows no basta**: no deja reemplazar un fichero que otro handle tiene
  abierto, y medido aquí un lector y un escritor peleándose 4 s dieron **5111 replaces fallidos y
  1130 lecturas fallidas** — el fallo aterrizaba *dentro del hilo del entrenamiento*, matando un run
  sano. Reintento con deadline en los dos lados (formatos.md §4.2). **La lección general**: el
  patrón «escritura atómica» de POSIX no porta a Windows, y aquí el ratio de sondeo lo destapa.
  *(Y la de proceso: la flakiness no era del test. Un test que falla 1 de cada 3 veces está
  diciendo algo.)*
- **`best` era `Infinity` y ningún navegador puede parsearlo.** El bucle arrancaba `best` en `±inf`
  y lo escribía tal cual: `json.dumps` emite `Infinity`, que **no es JSON válido**, así que un solo
  run cuyo monitor no llegara a disparar habría tumbado el `GET /runs` de *todos* los demás. El
  camino es real: `monitor: val_pos_err_px` sobre un val sin esquinas devuelve `None` cada época.
  Ahora es `None` — que es lo que significa: **no medido**, no «infinitamente malo» (formatos.md §2
  otra vez, y esta vez desde dentro).
- **Validar y reservar iban en el orden equivocado en el CLI.** `itf-train` reservaba el nombre y
  *luego* validaba, así que cada negativa dejaba un `runs/x/` muerto — y arreglar el dataset y
  reintentar con el mismo nombre contestaba «ese run ya existe» por un run que no vio un batch. El
  API no lo tenía porque validaba antes. La cura no fue reordenar el CLI: fue **`check_run`**, una
  función que las **dos puertas** preguntan. Dos comprobaciones separadas se desincronizan, y la
  puerta que queda más laxa es por la que entra un barrido.
- **Un crash antes de la primera época dejaba el run en `queued` para siempre** — la trampa del
  «crash que queda running» con otra palabra. `train()` marca sus propios fallos, pero solo desde
  la época 1; lo de antes (un `.npz` que no carga) rompe con `status.json` aún en `queued`. Lo
  cierra `RunStore.marking_failures`, que usan las dos puertas: **quien reservó el run es quien
  puede cerrarlo**.
- **La parada es un fichero, no un evento en memoria** (`stop.json`, formatos.md §4.2). El estado de
  un run es del run: así el CLI se para igual que el API y una parada sobrevive a un reinicio —
  que en CPU, con runs de horas, pasa.
- **La fase 3 dejó un run sin procedencia, y el diseño decía que eso no podía pasar.** D3 dio por
  hecho que `runs/` estaba vacío, pero la verificación de la fase 3 creó `fase3-01` **antes** de que
  la procedencia existiera. No se construye ningún lector que degrade —eso es justo lo que D3
  mató—: `GET /runs` lo **dice en voz alta** («no es comparable: bórralo y reentrénalo») y sigue
  listando el resto. Un run que no puede decir de qué red salió no es un caso legado que tolerar.

> A partir de aquí el flujo completo funciona: dato → red → receta → run. Lo que sigue **añade
> capacidad**.

### Fase 5 — La tabla por patch y el diagnóstico — ✅ **hecha (2026-07-17)** ← *la app se vuelve instrumento*

1. **Back**: la operación **E × split de B → tabla por patch** (`.npz`, el idioma del proyecto),
   con `score`, `(x,y)` predicho y real, error px, `sample_idx`, `patch_xy`.
2. **Front**: entra **Observable Plot**. V3 (predicción del patch: 4 meters + overlay), V6
   (galería peor-primero), V7 (error por posición), **V8 (scores + PR + `threshold`)**.
3. Las curvas de entrenamiento, en **small multiples** (R4): `loss`, `f1` y `pos_err_px` tienen
   escalas distintas (~0,28, ~0,77 y ~11 en las medidas del código viejo) y **no van en la misma
   gráfica** — superponerlas inventaría una correlación.

**Por qué aquí y no después**: V8 deja elegir `threshold` **gratis y post-hoc**. Entrar al
barrido sin ella es gastar horas de CPU buscando en D lo que estaba en F.

**Verificado**: 90 tests pasan (siguen 2 en xfail: ⑤ y ⑨). Diagnóstico abierto **en Chrome de
verdad** sobre `fase4-ui` × val de `fase3-red`, con clic real en la galería. **Y la promesa de la
fase se cumplió con números**:

- **El umbral es gratis y se nota**: f1 **0,673** en `threshold` 0,50 → **0,728** en 0,64, post-hoc
  y sin reentrenar. La tabla se calcula en **1,0 s**; los GET siguientes van a **0,025 s** y otro
  agregado sobre la misma tabla, a **0,014 s**. Por eso `/diagnostics` es síncrono (R3).
- **V7 dijo algo la primera vez que se miró**: error **16,4 px en el borde** del patch contra
  **9,1 px en el centro**. Es exactamente el diagnóstico que ui.md §4.1 le pide — apunta al
  `stride` de B, no a los filtros de C — y sin la vista se habría leído como «la red es pequeña».
- **El desbalance sale solo y cuadra**: **20,5 % de positivos, 3,88:1** sobre el val de
  `fase3-red`, contra el 20,5 % / 3,9:1 que documenta protocolo.md §1.

**Xfails que quita**: ninguno — no le tocaba ninguno. Quedan ⑤ (fase 6) y ⑨ (fase 7).

**Lo que apareció al construir y no estaba en el diseño**:

- **`pos_err_px` se calculaba en dos sitios, y ahora se calcula en uno** (`itf.metrics`). No era un
  problema visible: era la forma exacta del contrato ⑤ con otro nombre. `evaluate()` escribe
  `pos_err_px` cada época y la tabla lo escribe por patch para V7 — dos copias de una fórmula que
  **tienen que coincidir**, sin nada que lo comprobara. El test que lo cierra no pregunta «¿es
  correcta `position_error_px`?» (los dos lados llaman a la misma función: no puede divergir) sino
  **«¿mide la tabla lo mismo que reportó el run?»** — y sobre datos reales sale idéntico hasta el
  último decimal: `f1` 0,673479 por los dos caminos. Eso es tests.md §1.2 aplicado antes de que
  doliera.
- **Y unificarlas movió un número, medido y no supuesto.** Reentrenar `fase4-ui` tras el refactor da
  `loss`, `f1`, `precision` y `recall` **idénticos bit a bit** —o sea, los pesos no se tocan— y
  `pos_err_px` 12,427402796 contra 12,427402806: **asociatividad de float32**, porque `evaluate()`
  antes sumaba en unidades de patch y escalaba el total en float64, y ahora escala por elemento y
  suma valores 40× mayores en float32. El orden nuevo es **el correcto**: es lo que hace que el
  número del run sea exactamente la media de lo que la tabla guarda. ~1e-9 está muy por debajo de
  lo resoluble (protocolo.md §1), pero queda escrito en vez de esperando a ser un misterio.
- **El mapa 40×40 de V7 no se puede leer, y no es culpa de la vista: es del dato.** ui.md §4.1 pide
  40×40; con ~200 esquinas de un tipo repartidas en 1600 celdas son **0,1 muestras por celda** y
  sale moteado — **cierto e ilegible**, que es la peor combinación: parece estructura. A 10×10
  (celdas de 4 px, ~8 esquinas cada una) el borde-vs-centro se ve de un vistazo, y el ratio ~2×
  **sale igual a las dos resoluciones**, así que es real y no un artefacto del binning. La
  resolución es un control (`?bins=`), no una constante: la resolución legible **crece con el
  dataset**, y `bins = patch_size` sigue dando el mapa que el documento describe.
- **La clave del caché necesita el `mtime` del checkpoint, y D1 no lo pedía.** «Run + huella de B +
  split» solo identifica una tabla si un run es inmutable, y **no lo es mientras entrena**:
  `best.pt` se reescribe en cada época que mejora. Sin el mtime, abrir Diagnóstico en la época 5 y
  otra vez en la 20 contesta la tabla de la 5 las dos veces — un caché mintiendo con buena cara.
  El caché de modelos del código viejo ya usaba mtime por la misma razón (organizacion.md §2-④).
- **Una escala log dejó una gráfica sin barras, y con cara de gráfica.** El histograma de V8 nació
  en log «por el desbalance»: un `rectY` va desde un y=0 implícito, `log(0)` no existe, y Plot
  **descarta cada barra en silencio** dejando los ejes puestos. Se vio contando `rect` en el SVG
  (0 de 40), no mirando. Y la escala no hacía falta: el desbalance es 3,9:1, «modesto, no brutal»
  (organizacion.md §1-D), así que lineal se lee perfectamente. *(Dos series de `rect` sobre el
  mismo rango x tampoco se apilan: se tapan. Van media caja cada una.)*
- **El eje alineado de R4 no estaba alineado, y solo en el panel con leyenda.** Plot devuelve un
  `<svg>` pelado, pero un `<figure>` cuando le pides `legend: true` — y un `figure` se lleva los
  **márgenes por defecto del navegador (40 px)**. Medido: ese panel salía de 540 px empezando en
  x=287 y sus hermanos de 620 en x=247. R4 pide small multiples *apiladas y con el eje x alineado*
  porque **la alineación es el mecanismo entero**; rota, no se rompe nada — las columnas
  simplemente dejan de poder compararse, que es justo para lo que existen.
- **Un canvas no se repinta cuando cambia el tema, y `MatrixCanvas` lo arrastraba desde la fase 1.**
  La cascada se re-resuelve sola; un canvas ya pintado y un SVG ya construido, no. Se ve ahora
  porque V7 pone un mapa grande al lado del toggle. Lo cierra `useThemeVersion` (observa
  `data-theme` **y** la media query, que son las dos formas en que tokens.css cambia de modo).
- **Las dos clases de V8 y V14 no pueden pedir prestado un slot de esquina** (R1: el color sigue a
  la entidad, y el selector de esquina está en la misma pantalla). Usan **los dos extremos de la
  rampa divergente**: ya están en la paleta fija (D12 no genera hues nuevos), son tintas opuestas y
  el validador ya los aprobó el uno contra el otro. Solo para dos clases: una tercera pediría un
  hue, y los hues son de D12.

### Fase 6 — Mapas, kernels y el pipeline — ✅ **hecha (2026-07-17)** ← *cierra el ⑤*

1. **Back**: `GET /runs/{name}/kernels` y `POST /runs/{name}/feature-maps` (**entrada = un
   patch**, contrato ①). `map_payload` portado a **`itf.matrixview`**, aislado y sin importar `itf`.
2. **Front**: V1 (kernels de capa 1, **divergente centrada en 0**) y V2 (feature maps,
   secuencial/divergente según `spec.activation`), ambos con click → tabla de números.
3. **Back**: `itf.inference.predict` con las detecciones **crudas pre-NMS**; **Front**: V11 (etapas
   del pipeline en Predecir) + los knobs baratos como sliders con repintado en vivo.

Y de propina, porque su fase le tocaba (api.md §3): **V9** (`GET /diagnostics/coactivation`) y el
**caché de modelos** movido de `app.py` a `itf.inference.ModelCache` (api.md §0).

**Verificado**: 120 tests pasan (queda 1 xfail, ⑨). Los endpoints, **contra la API de verdad**
sobre un run entrenado: los kernels de capa 1 salen con **estructura de signo** (job `diverging`,
8 filtros 3×3), V2 devuelve las dos capas con su predicción, V11 da **18 crudas → 11 esquinas → 3
párrafos** y los knobs vuelven en el payload. Front: `tsc` y `build` limpios, la paleta valida, y
**las dos pantallas montan en Chrome headless** (Diagnóstico con V1/V9, Predecir con V11), cero
errores, con el run seleccionado y la procedencia en pantalla.

**Xfails que quita**: ⑤ (la geometría compartida). Queda ⑨ (fase 7).

**Lo que apareció al construir y no estaba en el diseño**:

- **La regla de D13 no es «capa 1», es `in_channels == 1`.** La capa 1 solo es la que la cumple: con
  un canal de entrada un filtro **es** una matriz y se aplica al patch. Una red con `in_channels: 3`
  tiene el mismo problema una capa antes, y servir `weight[:, 0]` ahí sería la proyección deshonesta
  que D13 rechaza con un número de capa más convincente. Así que `kernels` se **niega** (código
  `kernels_not_projectable`) sobre `in_channels != 1` y manda a V2, en vez de mirar solo el índice.
- **V9 miente sin su control, y el control es la co-ocurrencia real.** `matrix[TL][TR]` alto tiene
  dos causas opuestas —la cabeza TR confundida por un TL, **o** que esos patches de verdad llevan un
  TR— y la matriz sola no las distingue. Sobre el run de prueba salió justo el caso que lo enseña:
  `truth_rate` es 0 fuera de la diagonal y la cabeza TR dispara 0,36 dado un TL real ⇒ eso **es**
  confusión, no convivencia. Sin la matriz de al lado se leería como que los párrafos comparten
  esquina. Es la familia del moteado de V7: cierto, presentado para que la lectura obvia sea falsa.
- **El trabajo de color se lee de `spec.activation`, no del dato.** El atajo
  `"diverging" if maps.min() < 0 else "sequential"` es falso por patch: una capa `tanh` cuyas
  activaciones salgan todas positivas en **este** patch volvería secuencial, y divergente en el
  siguiente — el color significaría dos cosas en dos capturas de la misma red. Test incluido con una
  capa `tanh` y un patch plano.
- **El caché de modelos necesita el `mtime`, y por lo mismo que la tabla por patch (fase 5).**
  `best.pt` se reescribe cada época que mejora, así que un caché por ruta serviría los pesos de la
  época 5 para siempre — mirarías un run mejorar y sus kernels no cambiarían. Con la clave de mtime
  se invalida solo; el caché viejo necesitaba un `_drop_model_cache` a mano en rename/delete, éste
  no. Hay test que reescribe el checkpoint y comprueba que el modelo devuelto cambia.
- **`matrixview` se construyó pero no se extrajo** (D9/D10 siguen abiertas y `claude-libs/` no
  existe). Vive aislada en `src/itf/matrixview/`, sin un solo import de `itf` — la condición que
  librerias.md §5 pide **antes** de extraer. La extracción queda en un `git mv`, coherente con que
  las fases 3 y 4 tampoco extrajeron `convspec` ni `exp-registry`.

### Fase 7 — Cola de verdad y Barridos (H) — ✅ **hecha (2026-07-17)** ← *cierra el ⑨*

1. **Back — la cola** (`itf.api.jobs`): **límite de workers (=1 en CPU)**, cancelación cooperativa
   (un job lleva un `cancel` callback; `POST /jobs/{id}/cancel`) y persistencia (`persist_dir`: cada
   transición al disco, un job vivo al morir el proceso recarga como `interrupted`).
2. **Back — `optuna`** dentro de `itf.sweeps` (escrito library-shaped, **no extraído**): espacio,
   samplers, **pruners** (`MedianPruner`, la palanca nº1 en CPU) y storage SQLite. Sus `trials`
   **no son** nuestros runs: un trial lanza un run (con `provenance.sweep`) y guarda su nombre.
3. **Front**: pantalla Barridos + V12 (Pareto, secuencial por λ) y V13 (paralelas).
4. **La UI bloquea objetivo = `loss` si `lambda_pos` está en el espacio** (contrato ⑨), el mismo
   400 que el servidor — adelantado antes de enviar.

**Verificado contra la API de verdad** (2026-07-17, dataset sintético): 133 tests pasan, **0
xfailed** (⑨ cerrado). Un barrido de 4 puntos completa 4/4 por HTTP; en uno de 30, **14 puntos se
podaron**; y el de 40, **matado el proceso a 4/40 `running`**, al rearrancar la API el `lifespan`
lo **reanudó hasta 40/40** — el punto que corría al morir se repescó a `fail` y un trial nuevo ocupó
su hueco. El estado durable es `sweeps/<name>/spec.json` + `optuna.db` + los runs.

**Lo que apareció al construir y no estaba en el diseño**:

- **La API no puede abrir la SQLite de optuna mientras el worker la tiene abierta.** El worker
  sostiene el study todo el barrido; una segunda conexión desde el hilo de la API lo pisaba en la
  creación del esquema (`IntegrityError` en `alembic_version`) y en los write locks. Cura: el worker
  escribe un `progress.json` tras cada trial y la API lee **eso**, nunca la SQLite. Es un caché del
  study, derivable, y mantiene a optuna fuera de `app.py`.
- **La poda con solo 4 puntos no salta con el default.** `MedianPruner` necesita 5 trials para
  tener mediana; con `n_startup_trials=1, n_warmup_steps=1` puede podar desde el segundo punto, que
  es lo que un barrido corto necesita.
- **El objetivo son las métricas de patch, no la de párrafo, y es a propósito.** La F1 de párrafo
  (el objetivo *real*, protocolo.md §2) depende de **D7** (bbox vs. rotación), que sigue abierta; el
  barrido rankea por `f1`/`pos_err_px` (λ-independientes) hasta que D7 se cierre. No se tomó una
  decisión abierta.
- **`itf.sweeps` es library-shaped pero NO extraído**, coherente con `matrixview` (fase 6) y con que
  las fases 3–4 no extrajeran `convspec` ni `exp-registry`. La cola es el germen de `jobq`.

### Fase 8 — Sondas — ✅ **hecha (2026-07-17)** ← *el plan queda completo*

V5 (scrubber, la más rentable), V4 (occlusion), V10 (flag de borde), V15 (procedencia del patch).
**V9 no estaba aquí: la trajo la fase 6** (api.md §3), así que la 8 son cuatro sondas, no cinco.

1. **Back — dos sondas de patch** (`itf.inference.introspect`): `occlusion` (V4, ~361 forwards en
   un batch, mapa de `p(esquina|ocluido)` por esquina) y `border_test` (V10, 5 forwards, voltea
   cada flag). Endpoints `POST /runs/{name}/occlusion` y `/border-test`, con la misma forma de
   entrada que `feature-maps` — factorizada en `_patch_from_body`.
2. **Back — una sonda de imagen** (`itf.inference.predict.window_prediction`): V5, una ventana
   off-grid → 4 cabezas + la **estabilidad al mover 1 px**. `POST /runs/{name}/window`. Los flags de
   borde salen de **`itf.geometry.window_at`** (nuevo), que `windows` ahora usa: la fórmula del
   contrato ⑤ en un solo sitio.
3. **Front**: V4 (`Occlusion.tsx`, 4 heatmaps en canvas, secuencial), V10 (`BorderTest.tsx`,
   dumbbell con Plot, 1 tinta 2 tonos), V15 (`PatchProvenance.tsx`, overlay sobre la imagen fuente,
   **casi todo front** — los números ya estaban en el `.npz`) cuelgan del clic en la galería de
   Diagnóstico; V5 (`Scrubber.tsx`, arrastre + 4 meters) vive en **Predecir**.

**Verificado contra la API de verdad** (2026-07-17, `clear-paragraphs-02-reducidos` → `fase8-b`,
run `fase8-01` con `cnn-a`, que tiene `border_features`): 145 tests pasan (**0 xfailed**, +12 de la
fase). Las cuatro sondas contestan 200 y **V4 baseline, V5 corner TL y V10 baseline son el mismo
número** (TL 0,244) — la costura «una predicción, muchas vistas» sostenida en vivo. `tsc` y `build`
limpios, la paleta valida en claro y oscuro. Y **en Chrome de verdad**: Diagnóstico monta con V1/V6/
V7/V8/V9 y la galería lista, y **Predecir monta V11 y V5 juntos** — el Scrubber cargó su ventana,
sus 4 meters y la estabilidad sin un clic.

**Xfails que quita**: ninguno — las sondas no son contratos, son vistas. El mecanismo sigue en pie.

**Lo que apareció al construir y no estaba en el diseño**:

- **El mapa de occlusion es la probabilidad ocluida, no la caída, y por una razón de color.** La
  caída (`baseline − p`) es **con signo** —tapar una región inhibitoria *sube* el score, y sobre el
  patch 0 de `fase8-01` la TL saltó de 0,244 a **0,82**— y una cantidad con signo en rampa
  secuencial pone el neutro donde caiga el mínimo, el trampa exacta de R2/R3. `p(esquina|ocluido)`
  es una probabilidad, nunca negativa, así que secuencial es honesto: oscuro = tapar ahí mató el
  score. La caída queda como **una resta que el lector ve** (baseline al lado), no como un color que
  miente. Misma familia que el moteado de V7 y la matriz sola de V9.
- **V10 se niega si la red no usa `border_features`**, en vez de dibujar cuatro dumbbells planos.
  Voltear un flag que la red ignora no cambia nada, y «no cambia» dibujado cuatro veces se lee como
  «el borde no importa a este patch» — una conclusión sobre el dato cuando la verdad es que la
  arquitectura no lo mira. Es la forma de D13: negarse (`border_not_used`, 409) en vez de proyectar
  una vista sin sentido.
- **`window_at` es lo que impide que ⑤ vuelva por la puerta de la sonda.** Reteclear los seis flags
  de borde para una ventana suelta es lo que sale natural al escribir el scrubber; en su lugar
  `windows` y V5 pasan los dos por `window_at`, y el test afirma que la ventana on-grid da los
  mismos flags que `windows` — no «¿son correctos?», sino «¿ven B y el scrubber la misma ventana?».

---

## 3. Dónde se puede torcer

- **La fase 3 es el cuello.** Todo lo demás depende de ella. Si hay que recortar algo, recorta
  fases 6 y 8, nunca la 3.
- **`smooth_l1_beta` y `momentum` cambian los resultados.** Arreglarlos está bien, pero **lo ya
  entrenado deja de ser comparable** con lo nuevo. Que el cambio sea consciente y quede en el
  registro del run, no un ajuste silencioso.
- **La fase 7 toca la cola con jobs corriendo.** Migrar con la casa vacía.
- **No dejes que optuna dicte la organización.** Su modelo (studies/trials) es suyo; el nuestro
  es B/C/D/E/H. Si acaban mezclándose, el barrido deja de poder explicar de qué red y qué
  dataset salió cada punto.
