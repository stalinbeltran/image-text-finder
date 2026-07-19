# Decisiones abiertas

Lo que está **sin decidir** y bloquea algo. Índice, no archivo histórico.

**Por qué existe**: al escribir las specs se acumularon decisiones repartidas por varios
documentos. Nadie —ni tú ni un Claude que llegue en tres meses— puede ver de un vistazo qué falta
por elegir. Una decisión que no se ve **se acaba tomando sola, por defecto y sin pensar**; así
nacieron las tres que arrastrábamos: el CORS abierto, los 20 imágenes de val y `/runs/`
gitignoreado.

**Esto no es un ADR.** Los documentos ya cargan el *por qué* de cada decisión tomada, que es el
90 % del valor de un ADR. Aquí solo vive lo **pendiente**, más un índice de lo cerrado (§4).

**Ciclo de vida**: al decidirse, la decisión **se escribe en el documento que le corresponde** y
aquí queda una línea en §4 apuntando allí.

---

## 1. Bloquean la siguiente fase

**Ninguna.** D2 y D16 se cerraron el 2026-07-16 (§4). La fase 0 de [plan-ui.md](plan-ui.md) está
terminada y la construcción puede empezar.

Lo que queda abierto (§2, §3) **no bloquea**: son preguntas que se responden al llegar a su fase,
o resultados de investigación que el barrido mismo contestará.

---

## 2. Pueden esperar

| | Decisión | Recomiendo | Dónde |
|---|---|---|---|
| **D7** | ¿La métrica de párrafo soporta rotación, o compara contra el *bbox* del `quad`? | Bbox: basta con `clear-paragraphs` (`angle≈0`). Con `mixed-layout`, no | protocolo.md §2 |
| **D8** | ¿Añadir `limit` de train a `PatchExtractConfig`? | **Sí**, ahora que `num_images` es un eje del barrido (D6): deja de ser un truco y pasa a ser un parámetro que se mide | protocolo.md §3 |
| **D9** | Nombres de las librerías (`exp-registry`, `jobq`, `convspec`, `matrixview`) | Provisionales; decidir al extraer la primera. **Sigue abierta**: las fases 6 y 7 construyeron `matrixview` y la cola (`jobq`) library-shaped pero **no los extrajeron** (`claude-libs/` no existe) | librerias.md §1 |
| **D10** | ¿Monorepo `claude-libs` o cuatro repos? | Monorepo: cuatro repos es más ceremonia que valor a esta escala. **Sigue abierta**: no se ha creado el repo (van cuatro candidatos sin extraer: `convspec`, `exp-registry`, `matrixview`, `jobq`) | librerias.md §4 |
| **D11** | ¿Backportear NIST a las librerías? | **No.** Funciona; su valor ya está cobrado como evidencia | librerias.md §4 |
| **D14** | ¿Editor de patches? | No: V4 (occlusion) y V5 (scrubber) lo cubren mejor y en distribución | ui.md §5 |
| **D20** | ¿La **maximización de activación** (V17) reabre D13? | Ver abajo — **es la única de §2 que contradice una decisión ya cerrada**, así que no se resuelve construyendo | ui.md §4.1 |
| **D21** | ¿Merece la pena reanudar **dentro** de un trial? (reanudar el barrido **ya está hecho**) | **Esperar y medir**: cuesta cambiar el formato de `last.pt` y sostener una invariante nueva, para un ahorro que nadie ha medido | organizacion.md ⑪ |

---

## 3. Abiertas por lo que decidimos

Consecuencias de §4 que aún no tienen respuesta:

### D17 — ¿Qué rango barre `num_images` y las fracciones?

**En juego**: es un eje nuevo del barrido y es **el más caro de todos** — dobla las imágenes y
doblas cada época de cada punto.
**Recomiendo**: **no meterlo en el barrido general**. Medirlo **aparte y primero**, con la receta
de la baseline fija, para saber la curva "más datos → cuánto mejora" antes de gastar el
presupuesto grande. Es una pregunta distinta a "qué receta es mejor", y mezclarlas multiplica el
coste sin necesidad.
**Dónde vivirá**: protocolo.md §3.

### D21 — ¿Merece la pena reanudar **dentro** de un trial?

*(Abierta 2026-07-19. Ojo al alcance: **reanudar el barrido ya está resuelto** —`POST
/sweeps/{name}/resume` y el botón «Continuar»—. Esto es lo que queda: las épocas del trial que
estaba en vuelo.)*

**Cómo se llegó aquí**: el 19/07 se perdieron dos puntos de `dirty-20-lambda_pos_1` al morir el
proceso, y el diagnóstico inicial fue «hay que poder reanudar». **Era medio cierto y el reparto
importa**: reanudar el *barrido* ya existía y solo le faltaba una puerta (estaba cableado al
`lifespan`, así que la única forma de continuar era reiniciar el backend). Lo que sigue perdiéndose
son las épocas del trial en vuelo: `_reap_running` lo marca `FAIL` y **borra su run**.

**En juego, y por qué no es gratis**: `last.pt` guarda hoy `{model, config, epoch}` y nada más.
Reanudar desde ahí sin el estado del optimizador y del RNG produce un run con el mismo nombre y la
misma procedencia que el no interrumpido y **pesos distintos**, en silencio — el contrato **⑪** lo
desarrolla, y formatos.md §4.2.2 dice qué habría que guardar para que fuese bit-exacto. O sea que
esto **no es "añadir un flag"**: es cambiar el formato de `last.pt` y sostener una invariante nueva.

**Y hay un segundo problema, independiente**: la identidad del trial en optuna. Un trial `RUNNING`
cuyo proceso murió no se continúa entre procesos; lo que hay es reencolar el punto
(`study.enqueue_trial`), y el trial nuevo trae **otro número** — pero el run se llama
`{sweep}-{trial.number:04d}`, así que continuar los pesos de `-0003` como `-0004` deja una
procedencia que no se sostiene sola.

**Recomiendo esperar y medir antes de construir.** Con poda activada la mayoría de los puntos mueren
en la época 3, y el coste real de perder *un* trial en vuelo puede ser minutos. Es un trabajo caro
(formato + invariante + optuna) para un ahorro que **nadie ha medido todavía**: la pregunta previa
es *cuánto tiempo se pierde al año por esto*, y sale de mirar los barridos reales, no de razonar.
**Dónde vivirá**: organizacion.md ⑪ y formatos.md §4.2.2 (ya escritos como diseño, **no construidos**).

### D20 — ¿La maximización de activación (V17) reabre D13?

**En juego**: **D13 se cerró en «kernels profundos: nada»** (§4), y el argumento fue que de la capa
2 en adelante no hay **proyección honesta** de un filtro (32, 64… canales) a una matriz, así que esa
información vive en los feature maps (V2). El argumento sigue en pie para los **pesos**. V17 no
toca los pesos: **sintetiza una entrada** que maximiza el filtro, y esa entrada sí es una matriz
`n×n` en unidades de píxel, comparable con un patch real. Es decir, no es un contraejemplo a D13 —
es un camino que D13 no evaluó.

**Lo que hay que decidir no es «¿se puede?» sino «¿dice la verdad?»**. Una maximización es un
**óptimo local del gradiente sobre ruido**: depende de la semilla, de la regularización y del número
de pasos, y sale distinta cada vez. Enseñada como «esto es lo que le gusta al filtro 37» tiene la
misma cara de vista honesta que tenía `weight[k, 0]` — el proyecto hermano ya se comió esa, y es
literalmente el fallo que motivó D13. La pregunta real: **¿qué tendría que enseñar V17 al lado del
resultado para que no se lea como un hecho?** (varias semillas en rejilla, la activación alcanzada,
la receta de la optimización).

**Recomiendo**: **V16 sí, V17 aparcada** hasta tener V16 en pantalla. V16 no toca D13 en absoluto —
es una atribución sobre un patch real, misma familia que V4 — y contrastarla con V4 (que ya existe)
dice gratis si estas vistas aportan algo sobre esta red. Si V16 no aporta, V17 tampoco lo hará, y
sale mucho más cara. Y si D20 se cierra en «sí», **la condición mínima es la de arriba**: V17 no se
sirve sin varias semillas y sin la activación alcanzada, o repite el error de D13 con mejor
tipografía.

**Dónde vivirá**: ui.md §4.1 (y, si se cierra en «sí», una nota en D13 diciendo qué parte de aquel
«nada» sobrevive: los **pesos** profundos siguen sin proyección honesta).

---

## 4. Ya decididas

| | Decisión | Vive en | Fecha |
|---|---|---|---|
| **D19** | El **resize de una fuente** produce una **fuente derivada A′** que se escribe en una **raíz local nueva** (`data/sources/`), no en `ITF_DATASETS_ROOT`: A sigue siendo externa y solo-lectura. **Solo reduce**; ampliar es `400` (`upscale_not_allowed`) — interpolar un render sintético inventa nitidez y un B extraído de ahí mediría el interpolador. El mecanismo de píxeles (`itf.imageops`) y el de coordenadas (`itf.geometry.scale_quad`) son **piezas separadas**, y por eso la primera sirve para una imagen cualquiera | organizacion.md §1-A y ⑧, formatos.md §4.6, api.md §3 (`/sources`) | 2026-07-18 |
| **D12** | La paleta: los 4 tipos de esquina son los **4 primeros slots** de la paleta documentada (blue/green/magenta/yellow), secuencial azul, divergente azul↔rojo con gris en el 0. **Elegida por enumeración, no a ojo**: de las 70 formas de escoger 4 de los 8 hues, solo 2 pasan todos los gates en ambos modos con `--pairs all`, y esta es la mejor. Vive en `web/src/theme/tokens.css`; la valida `npm run validate:palette` | ui.md §4.0, tokens.css | 2026-07-16 |
| **D2** | La procedencia lleva **nombre + valor** de C y D, huella de B, `sweep`, `git_commit` y **`environment`**. El nombre agrupa, el valor reproduce. `environment` cierra el hueco de `git_commit`: el commit fija el código, no el intérprete — y al llegar la GPU cambia entero. Ningún campo se rellena si falta | formatos.md §4.2.1, api.md §3 (`/runs`) | 2026-07-16 |
| **D16** | El holdout son **500 imágenes**, **fuente propia** (`…-holdout`), **misma config** del generador, **generado lo primero**. Fuente aparte ⇒ la fuga es físicamente imposible. Misma config porque otra mediría robustez, que es otra pregunta. 500 ⇒ sd ≈0,65 %, con margen para que el suelo real sea peor que la aritmética | protocolo.md §3 | 2026-07-16 |
| **D1** | La tabla por patch es un **caché**, no una entidad. Se puede recalcular exacta ⇒ no se guarda. Sin pantalla de Evaluaciones. Un **filtro** guardado (criterios, no filas) se añadiría solo si se quiere reentrenar sobre los errores | formatos.md §4.4, ui.md §3, api.md §3 (`/diagnostics`) | 2026-07-16 |
| **D4** | **Allowlist de raíces + CORS cerrado** a `localhost:5173`. La ruta se comprueba tras `resolve()`. Se implementa en la fase 2 | api.md §6 | 2026-07-16 |
| **D5** | **Se versiona la descripción, se ignora la carga** (105 KB vs 38,5 MB). Criterio: se versiona lo que no se puede recalcular. Si el historial se ensucia, se ignora `sweeps/` — no se revierte | formatos.md §5 | 2026-07-16 |
| **D18** | **Borrar el código y los datos y construir desde cero**, con tag `pre-rediseno` como red de seguridad. Los datos no estaban en git y D6 los regenera; borrarlos **simplifica el diseño** (mata D3 y el relleno de `border`). El código estaba mal organizado y los docs lo citan 161 veces — el tag conserva la evidencia | plan-ui.md §0, README.md | 2026-07-16 |
| **D3** | *Muerta con D18*: no quedan `config.json` viejos que migrar ni degradar | — | 2026-07-16 |
| **D15** | *Muerta con D18*: sin código viejo no hay franjas ni big-bang que elegir | — | 2026-07-16 |
| **D6** | El **tamaño del dataset y las fracciones son variables de investigación**, no ajustes: entran al barrido. Consecuencia: hace falta un **holdout fuera de B**, viable porque la F1 de párrafo se mide **por imagen** | protocolo.md §3, organizacion.md ⑧ | 2026-07-16 |
| **D13** | **Kernels profundos: nada.** La regla real es `in_channels == 1` (no «la capa 1»): con un canal un filtro **es** una matriz y es exacto. De la capa 2 en adelante (32, 64… canales) no hay proyección honesta a una matriz, y esa información está en los feature maps (V2). `GET /kernels` sirve solo la capa 1 y se **niega** (`kernels_not_projectable`) sobre `in_channels != 1` | ui.md §4.1, `itf.inference.kernels` | 2026-07-17 |
