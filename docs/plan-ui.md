# Plan de rediseño de la UI

Plan paso a paso para llevar la interfaz a la organización de [ui.md](ui.md), que a su vez
proyecta [organizacion.md](organizacion.md). **Este documento se ejecuta; los otros dos
mandan.**

Cada fase termina con: la app **arranca**, los tests **pasan**, y hay un commit. Ninguna fase
deja el árbol a medias.

---

## 0. Punto de partida: qué se descarta y qué no

La propuesta era **empezar de cero descartando lo hecho**. Tras auditar `web/src` (1442
líneas), la propuesta es correcta **en lo que importa** y hay un matiz que la refuerza:

> El componente que parecía más reutilizable —`ModelConfigForm.tsx`, 228 líneas— es en realidad
> **el más sucio del front**. Su propio comentario lo dice: *"input size, the conv backbone, the
> head, **and the training hyperparameters**"*. Es **C + D + X en un formulario**, y
> `lockArchitecture` es el booleano que tapa la frontera C/D en lugar de resolverla. No se
> salva: se parte en dos.

Auditoría honesta, fichero a fichero:

| Fichero | | Qué se hace | Por qué |
|---|---|---|---|
| `App.tsx` | 40 | **Descartar** | La nav *es* lo que cambia: 4 pasos numerados → 4 grupos por dominio |
| `ExtractPanel.tsx` | 151 | **Descartar** | Mezcla A + B. Se parte en Fuentes y Patches |
| `ModelConfigForm.tsx` | 228 | **Descartar** | Mezcla C + D + X. Se parte en dos formularios nuevos |
| `TrainPanel.tsx` | 74 | **Descartar** | Envoltorio fino del anterior |
| `RunsPanel.tsx` | 313 | **Reescribir en gran parte** | E ya es un dominio, pero pierde el modal de retrain y `lockArchitecture`, y gana procedencia |
| `PredictPanel.tsx` | 395 | **Adaptar** | F ya es un dominio. Navegar A y B desde aquí es legítimo (ui.md §2): es una vista que cruza, no una mezcla |
| `api.ts` | 170 | **Conservar y crecer** | Los endpoints de A, B, E y F no cambian. Reescribir un `fetch` que funciona no organiza nada |
| `LineChart.tsx` | 61 | **Conservar** | Limpio y funciona. Muere cuando entre Plot (fase 5), no antes |
| `main.tsx` | 10 | **Conservar** | Trivial |

Resultado: **~2/3 del front es código nuevo**, y el 100% de la estructura (nav, rutas, dónde
vive el estado, cómo se parten los formularios) se hace desde cero. Eso *es* empezar de cero.
Lo que no se hace es borrar por ceremonia un cliente HTTP correcto.

**Y el trabajo de verdad no está en el front.** De las nueve pantallas, tres no existen porque
**les falta el backend**: C tiene endpoints muertos, D no tiene nada, H no existe. Un plan
solo-front se atasca en la fase 3. Por eso las fases de abajo son **verticales**: backend y
front juntos, por dominio.

---

## 1. La decisión que gobierna el plan

**Recomendación: franjas verticales, no big-bang.** La nav nueva se levanta en la fase 1 y las
pantallas viejas conviven en un grupo `legacy` que se vacía fase a fase, hasta borrarse entero
en la fase 4.

Por qué, y es una razón concreta de este proyecto: **entrenas en CPU y un run tarda horas**. Un
big-bang deja la herramienta inservible justo mientras la necesitas, y la tentación de "lo
arreglo rápido con un parche" es exactamente lo que produjo `lockArchitecture`. Con franjas,
cada fase deja la app usable y el código viejo se borra **cuando su sustituto ya funciona**, no
antes.

Si prefieres big-bang, cambia solo la fase 1 (el grupo `legacy` no existe) y la 4 (no hay nada
que borrar): el resto del plan es idéntico.

---

## 2. Las fases

### Fase 0 — Cerrar decisiones abiertas *(sin código)*

Tres preguntas que, si se responden tarde, se responden con un parche:

1. **La tabla por patch (ui.md §3): ¿entidad guardada o caché?** Es E × B con identidad propia,
   así que por la regla 1 pide pantalla; el proyecto hermano la tenía como entidad
   (`evaluations/<id>/`). **Si es entidad, va a organizacion.md antes de implementarse** — lo
   exige CLAUDE.md.
2. **Forma de la procedencia en el run** (contrato ③): `model_name`, `recipe_name` y una
   **huella de contenido de B**. Decidir los nombres ahora, porque los escribe la fase 4 y los
   lee todo lo demás.
3. **Sacar X de la identidad de D** (contrato ⑩): los `config.json` ya entrenados llevan
   `device` dentro. Decidir si se migran o si se leen con retrocompatibilidad.

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
4. Grupo `legacy` con las 4 pestañas actuales, intactas.

**Entregable**: la app arranca con la nav nueva; lo viejo sigue accesible.
**Verificación**: `npm run dev` carga; el validador de paleta pasa en ambos modos.

### Fase 2 — Datos: Fuentes (A) y Patches (B)

1. **Front**: pantalla Fuentes (solo lectura: datasets, muestras, párrafos dibujados) y pantalla
   Patches (CRUD + construir).
2. Patches muestra **el desbalance** (`positives_per_corner / num_patches`): está en el manifest
   y hoy no lo mira nadie, y es el número que gobierna `pos_weight`.
3. **Back**: `DELETE /patch-datasets/{name}`, que **avisa de qué runs lo referencian** antes de
   borrar (contrato ③).
4. **Borra** `ExtractPanel.tsx`.

**Verificación**: tests; construir un dataset de patches desde la UI; intentar borrar uno en uso
y ver la razón.

### Fase 3 — Modelo: Redes (C) y Recetas (D) ← **el desbloqueo**

La fase que lo condiciona todo: sin identidad para C y D no hay Entrenar limpio ni H.

1. **Back — D como entidad**: almacén (`configs/recipes/*.yaml`) + `GET/POST/GET{name}/DELETE
   /recipes`.
2. **Back — C**: falta `DELETE /models`. Los demás endpoints existen y solo hay que llamarlos.
3. **Back — los hiperparámetros que faltan** (catálogo §1-D de organizacion.md), en `RunConfig`
   y `loop.py`. Dos de ellos **arreglan bugs reales**, no añaden features:
   - `momentum` — hoy `_make_optimizer` solo pasa `lr` y `weight_decay`, así que **SGD corre a
     momentum 0**: cualquier comparación de optimizadores está sesgada.
   - `smooth_l1_beta` — hoy es el default de PyTorch (1.0) con coordenadas en [0,1], así que la
     pérdida de posición **es MSE pura** y el Huber nunca se activa.
   - Y los que faltan de verdad: `scheduler` (la omisión más cara: `lr` es constante),
     `grad_clip`, `patience`/`min_delta`, `monitor` explícito.
4. **Front**: pantalla Redes (solo arquitectura, + traza espacial `40→20→10→5` y nº de params,
   que es gratis y es lo único que una red sin entrenar puede enseñar) y pantalla Recetas (solo
   D, agrupada como el catálogo, **con la definición de cada campo en línea**).
5. `device` **no** aparece en Recetas: es X, va en Entrenar.

**Verificación**: tests (incluidos los nuevos hiperparámetros); crear una red y una receta desde
la UI; `itf-train --config configs/model.example.yaml` sigue funcionando.

### Fase 4 — Entrenar y Runs (E) — *fin de la migración*

1. **Back**: validar el **contrato ①** en `POST /runs` (400 con la razón si `patch_size !=
   input_size`) — hoy solo lo mira `RunsPanel.tsx` y el mismatch revienta dentro del hilo del
   job. Escribir la **procedencia por nombre** (fase 0.2). Sacar X de la identidad.
2. **Back**: `POST /runs` no debe **sobrescribir en silencio** (hoy sí; `retrain` ya devuelve
   409).
3. **Front**: pantalla Entrenar (elegir B + C + D; `device` aparte; **estimar el coste** con los
   `seconds` que ya guarda `metrics.jsonl`). Runs reescrito: procedencia por nombre, sin
   `lockArchitecture` — "re-entrenar la misma red" pasa a ser *elegir otra receta con la misma
   C*, que es lo que siempre fue.
4. **Borra**: `TrainPanel.tsx`, `ModelConfigForm.tsx` y **el grupo `legacy` entero**.

**Verificación**: tests; entrenar un run corto desde la UI; **abrir un run antiguo** (los
`config.json` de `runs/` no tienen los campos nuevos y no pueden romper).

> A partir de aquí la app está migrada y organizada. Todo lo que sigue **añade capacidad**.

### Fase 5 — La tabla por patch y el diagnóstico ← *la app se vuelve instrumento*

1. **Back**: la operación **E × split de B → tabla por patch** (`.npz`, el idioma del proyecto),
   con `score`, `(x,y)` predicho y real, error px, `sample_idx`, `patch_xy`.
2. **Front**: entra **Observable Plot**. V3 (predicción del patch: 4 meters + overlay), V6
   (galería peor-primero), V7 (error por posición), **V8 (scores + PR + `threshold`)**.
3. Migrar las curvas a Plot y **partirlas en small multiples** (R4): `loss ≈ 0.28`,
   `f1 ≈ 0.77` y `pos_err_px ≈ 11` **no van en la misma gráfica**.

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

1. **Back — la cola**, que es el bloqueante real: hoy `JOBS` lanza **un hilo por job, sin
   límite**, con el estado en memoria. Hace falta **límite de workers (=1 en CPU)**,
   persistencia y cancelar. Sin esto, un barrido de 20 puntos son 20 entrenamientos peleándose
   por los mismos núcleos, cada uno con su `PatchDataset` entero en RAM.
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
- **Retrocompatibilidad de lo ya entrenado.** `runs/` tiene modelos reales. Los `config.json`
  viejos no llevan `model_name`, `recipe_name` ni huella de B. Leerlos debe degradar (mostrar
  "desconocido"), no reventar. Es la trampa más probable de la fase 4.
- **`smooth_l1_beta` y `momentum` cambian los resultados.** Arreglarlos está bien, pero **lo ya
  entrenado deja de ser comparable** con lo nuevo. Que el cambio sea consciente y quede en el
  registro del run, no un ajuste silencioso.
- **La fase 7 toca la cola con jobs corriendo.** Migrar con la casa vacía.
- **No dejes que optuna dicte la organización.** Su modelo (studies/trials) es suyo; el nuestro
  es B/C/D/E/H. Si acaban mezclándose, el barrido deja de poder explicar de qué red y qué
  dataset salió cada punto.
