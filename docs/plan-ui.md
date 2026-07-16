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

### Fase 0 — Cerrar decisiones abiertas *(sin código)*

Queda **una** (D2 en [decisiones.md](decisiones.md)):

1. **Forma de la procedencia en el run** (contrato ③): los nombres de campo de `network`,
   `recipe` y la **huella de B**. Se deciden ahora porque los escribe la fase 4 y los lee todo lo
   demás.

*(D1 —tabla por patch— ya está cerrada: es un **caché**, así que no hay entidad ni pantalla.)*
*(La tercera pregunta —qué hacer con los `config.json` ya entrenados— **murió con el borrado**:
no queda ninguno. `device` sale de la identidad de D desde el primer run, sin migración ni
retrocompatibilidad.)*

**Entregable**: organizacion.md actualizado. **Verificación**: nada que correr.

### Fase 1 — Esqueleto y paleta *(front, desde cero)*

1. Nav de 4 grupos (Datos / Modelo / Entrenar / Analizar), rutas, layout, convenciones de carga
   y error. **Sin números de paso.**
2. **La paleta**, que bloquea todas las vistas: 4 slots categóricos fijos para `TL, TR, BR, BL`
   (R1), 1 secuencial (R3), 1 divergente con gris neutro en el 0 (R2). **Se pasa por el
   validador de daltonismo, en claro y en oscuro, y se corrige hasta que pase.** No se elige a
   ojo.
3. Componentes base: `MatrixCanvas` (el `drawMap` portado, con normalización **por mapa**),
   `Meter`, y la **tabla de números** (R5: es el equivalente accesible de todo heatmap).
**Entregable**: la app arranca con la nav nueva y las pantallas vacías.
**Verificación**: `npm run dev` carga; el validador de paleta pasa en ambos modos.

### Fase 2 — Datos: Fuentes (A) y Patches (B)

1. **Front**: pantalla Fuentes (solo lectura: datasets, muestras, párrafos dibujados) y pantalla
   Patches (CRUD + construir).
2. Patches muestra **el desbalance** (`positives_per_corner / num_patches`): está en el manifest
   y hoy no lo mira nadie, y es el número que gobierna `pos_weight`.
3. **Back**: `DELETE /patch-datasets/{name}`, que **avisa de qué runs lo referencian** antes de
   borrar (contrato ③).
**Verificación**: tests; construir un dataset de patches desde la UI; intentar borrar uno en uso
y ver la razón.

### Fase 3 — Modelo: Redes (C) y Recetas (D) ← **el desbloqueo**

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

**Verificación**: tests; crear una red y una receta desde la UI; **entrenar por CLI**
(`itf-train`) — a partir de aquí ya se puede entrenar sin esperar a la UI.

### Fase 4 — Entrenar y Runs (E)

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
**Verificación**: tests; entrenar un run corto desde la UI y comprobar que su `config.json`
lleva la procedencia completa (nombre de red, de receta y huella de B).

> A partir de aquí el flujo completo funciona: dato → red → receta → run. Lo que sigue **añade
> capacidad**.

### Fase 5 — La tabla por patch y el diagnóstico ← *la app se vuelve instrumento*

1. **Back**: la operación **E × split de B → tabla por patch** (`.npz`, el idioma del proyecto),
   con `score`, `(x,y)` predicho y real, error px, `sample_idx`, `patch_xy`.
2. **Front**: entra **Observable Plot**. V3 (predicción del patch: 4 meters + overlay), V6
   (galería peor-primero), V7 (error por posición), **V8 (scores + PR + `threshold`)**.
3. Las curvas de entrenamiento, en **small multiples** (R4): `loss`, `f1` y `pos_err_px` tienen
   escalas distintas (~0,28, ~0,77 y ~11 en las medidas del código viejo) y **no van en la misma
   gráfica** — superponerlas inventaría una correlación.

**Por qué aquí y no después**: V8 deja elegir `threshold` **gratis y post-hoc**. Entrar al
barrido sin ella es gastar horas de CPU buscando en D lo que estaba en F.
**Verificación**: correr la tabla sobre un run existente; V7 y V8 pintan.

### Fase 6 — Mapas, kernels y el pipeline

1. **Back**: `GET /runs/{name}/kernels` y `POST /runs/{name}/feature-maps` (**entrada = un
   patch**, contrato ①). Reusar `map_payload` del proyecto hermano tal cual.
2. **Front**: V1 (kernels de capa 1, **divergente centrada en 0**) y V2 (feature maps,
   secuencial), ambos con click → tabla de números.
3. **Back**: exponer las detecciones **crudas pre-NMS**; **Front**: V11 (etapas del pipeline en
   Predecir) + los knobs baratos como sliders con repintado en vivo.

**Verificación**: los kernels de capa 1 de un run entrenado deben parecer **detectores de borde
orientados**. Si parecen ruido, la red no aprendió — y eso es información, no un bug de la
vista.

### Fase 7 — Cola de verdad y Barridos (H)

1. **Back — la cola**, que es el bloqueante real: **límite de workers (=1 en CPU)**,
   persistencia y cancelación cooperativa. La trampa a no repetir: un hilo por job sin límite
   (lo que había) convierte un barrido de 20 puntos en 20 entrenamientos peleándose por los
   mismos núcleos, **cada uno con su `PatchDataset` entero en RAM**.
2. **Back — `optuna`**: espacio, samplers, **pruners** (la palanca nº1 en CPU) y storage SQLite.
   Sus `trials` **no son** nuestros runs: un trial lanza un run y guarda su referencia.
3. **Front**: pantalla Barridos + V12 (Pareto, secuencial por λ) y V13 (paralelas).
4. **La UI bloquea activamente objetivo = `loss` si `lambda_pos` está en el espacio** (contrato
   ⑨). No es una nota al pie: ese error produce un ganador de buena cara.

**Verificación**: un barrido corto (4 puntos × 3 épocas) completo, con poda, sobrevive a un
reinicio de la API.

### Fase 8 — Sondas

V5 (scrubber, la más rentable), V4 (occlusion), V9 (co-activación), V10 (flag de borde), V15
(procedencia del patch).

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
