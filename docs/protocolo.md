# Protocolo experimental

Cuándo un resultado de este proyecto es **creíble**, y qué hay que hacer antes de gastar CPU en
un barrido.

Esto no es burocracia: es lo que separa "la receta X es mejor" de "la receta X salió mejor esta
vez". Sin protocolo, un barrido produce un ranking en el que **no se distingue el ganador del
ruido**, y se elige a cara o cruz creyendo que se midió.

---

## 1. Los números reales, que son los que mandan

Medidos sobre `data/patch-datasets/clear-paragraphs-02`, que es donde se entrenó `cnn-02-01`:

| | |
|---|---|
| Imágenes | 200 → **160 train / 20 val / 20 test** |
| Patches | 9800 (49 por imagen; `n=40`, `stride=20`) → 7840 / **980** / 980 |
| Positivos por esquina | ~2012 de 9800 = **20,5 %** |
| Coste | ~20 s/época → **6,7 min por run** de 20 épocas |

> **La fuente es `clear-paragraphs-02-reducidos`, la de 160×160.** No es un detalle: hay **dos**
> fuentes que empiezan por `clear-paragraphs-02` y la otra (`-8ea1ac04`) es de **640×480**, o sea
> **14,5× más área**. Con la misma ventana (`n=40`, `stride=20`) la rejilla pasa de 7×7 = **49**
> patches por imagen a 31×23 = **713**, y como el número de esquinas lo ponen los párrafos y no la
> rejilla, **el denominador crece y el numerador no**: el desbalance se desploma de **3,9:1** a
> **~67:1** y la época pasa de ~20 s a **~319 s**. La aritmética que las separa es
> `((L-40)/20)+1` por eje.
>
> Toda esta tabla es de la de 160×160. **Reverificada entera el 2026-07-16** (fase 3) y reproduce
> exacta: 9800 patches, 7840/980/980, TL = 2012 ⇒ 20,53 % ⇒ 3,88:1.
>
> **Por qué está escrito aquí**: al verificar la fase 3 se midió la de 640×480 creyendo que era
> ésta, y sus números (1,5 %, ~67:1, 319 s/época) parecían demostrar que esta tabla estaba mal por
> 16× y que §1.4 era una corrección equivocada. **No lo estaba: era la fuente equivocada.** El
> nombre casi compartido es la trampa — nómbrala entera siempre.

De ahí salen tres cosas, y las tres cambian el plan:

### 1.1 El val son 20 imágenes, no 980 patches

980 patches suena a muestra grande. **No lo es**: salen de **20 imágenes**, y los patches de una
misma imagen están fuertemente correlacionados (ventanas solapadas al 50 %, mismo texto, mismo
layout, misma tipografía). El tamaño de muestra efectivo de val es **~20**.

Con ~201 positivos de TL en val, el ruido de muestreo del recall ya es

```
sd ≈ √(p(1-p)/n) = √(0,7·0,3/201) ≈ 3,2 %
```

y eso **asumiendo independencia, que aquí no se cumple** — así que el ruido real es peor.
Traducido: **diferencias de f1 menores de ~5 % no son resolubles con este val**, y los barridos
se ganan por 1–2 %.

> **El cuello de botella no es el modelo: es la regla de medir.**

### 1.2 El dato es sintético, así que esto es una elección

El generador (`image-text-sample-generator`) está al lado y el dataset más grande que hay son
200 imágenes. **20 imágenes de val no es una restricción: es una decisión por defecto que nadie
tomó.** Arreglarlo cuesta una corrida del generador.

### 1.3 `reducido-40` está roto — y es el ejemplo del README

5 imágenes → split 4 / **0** / 1 → **val con 0 patches**. Entonces, en `training/loop.py`:

```python
has_val = len(val_ds) > 0          # False
val_metrics = {}                   # nunca se evalúa
monitor = val_metrics.get("loss", train_loss)   # ← cae al train_loss
```

**`best.pt` se elige por pérdida de entrenamiento**: el checkpoint más sobreajustado, en
silencio y sin un warning. Y es el flujo que documenta `README.md` y `configs/extract.example.yaml`.

**Regla**: un dataset de patches **sin val no sirve para medir**. Construirlo debe fallar o
avisar, y entrenar sobre él debe negarse a elegir `best.pt` por train loss.

### 1.4 Corrección: el desbalance es 3,9:1, no "brutal"

`organizacion.md` decía que el desbalance era brutal. **No lo es**: 20,5 % de positivos ⇒
**3,9:1**. Modesto. `pos_weight ≈ 3.9` es el punto de partida del barrido, no 50. *(Corregido en
organizacion.md.)*

---

## 2. La métrica que falta — y que debería ser el objetivo

**No existe ninguna métrica de párrafo en el código.** `evaluate()` es todo a nivel de patch:
`f1`, `precision`, `recall`, `pos_err_px`. Lo que de verdad importa —**si los párrafos salen
bien en la imagen completa**— no se mide en ningún sitio.

Es decir: el barrido optimizaría un **proxy cuya fidelidad nadie ha verificado nunca**.

### Qué es

Precisión/recall/**F1 a nivel de párrafo**: emparejar cada caja de `reconstruct_boxes` con el
párrafo real (de `labels.jsonl`) por **IoU ≥ 0,5**, y contar. Más el IoU medio de los
emparejados.

Detalle a resolver: los `quad` de la verdad pueden venir **rotados** (`Block.angle`) y
`reconstruct_boxes` devuelve cajas **alineadas a los ejes**. Comparar una cosa con la otra exige
decidir: o se compara contra el *bounding box* del quad (fácil, y suficiente si `angle≈0`), o la
métrica soporta rotación. Con `clear-paragraphs` vale lo primero; con `mixed-layout`, no.

### Es barata

Val de 200 imágenes × ~49 patches = ~10 000 forwards, por lotes: **segundos**. No es cara — es
que **nunca se escribió**.

### Y por eso debería ser el objetivo, no solo un validador

Esto es lo que cambió al hacer los números. Si es barata, el barrido puede **rankear
directamente por F1 de párrafo**, y eso resuelve solo dos problemas:

- **Disuelve el contrato ⑨.** La F1 de párrafo es **independiente de λ por construcción**: no
  contiene la pérdida. Desaparece la trampa de que bajar λ baje la loss "gratis".
- **Disuelve la tensión detectar-vs-localizar.** `f1` y `pos_err_px` tiran en direcciones
  opuestas y λ arbitra; por eso hacía falta un frente de Pareto. Pero **la F1 de párrafo ya
  integra las dos**: detectar mal rompe el IoU, y localizar mal también. Es un escalar, y es el
  escalar que de verdad quieres.

El diseño queda así:

| | Métrica |
|---|---|
| **Entrenar** (la pérdida) | `BCE + λ·smoothL1` — tiene que ser diferenciable |
| **Elegir `best.pt`** dentro de un run | val loss de patch (barato, por época) |
| **Rankear el barrido** | **F1 de párrafo en val** ← el objetivo real |
| **Reportar** | F1 de párrafo en **test**, una vez, del ganador |

**Caveat honesto**: la F1 de párrafo depende de los knobs de F (`threshold`, `stride` de
inferencia, radio de NMS, `min_size`). No es función pura del modelo. Durante el barrido se
**fijan a un valor por defecto** para que todos los puntos se comparen igual, y se ajustan
**post-hoc sobre el ganador** (V8 de ui.md, que es gratis). Comparar cada punto "a su mejor
threshold" sería más justo pero es una optimización anidada: no compensa.

Y **el paso 2 sigue haciendo falta** aunque la F1 de párrafo sea el objetivo: hay que saber si
el `f1` de patch la predice, porque de eso depende que las vistas de diagnóstico (V7, V8, V9),
que son todas de patch, sirvan para algo.

---

## 3. Paso 0 — Arreglar el instrumento antes de medir con él

**Generar más imágenes.** Es lo primero, lo más barato y lo que condiciona todo lo demás.

Cuánto val hace falta, en función de la resolución que quieras (≈10 positivos por esquina y por
imagen de val):

| Val | Positivos/esquina | sd teórico del recall |
|---|---|---|
| **20 img** (hoy) | 201 | **3,2 %** |
| 210 img | ~2 100 | 1,0 % |
| 840 img | ~8 400 | 0,5 % |

Y lo que cuesta el train, que es la otra cara:

| Train | Patches | s/época | min/run (20 ép.) | Barrido 20 pts, sin poda |
|---|---|---|---|---|
| **160 img** (hoy) | 7 840 | 20 | 6,7 | 2,2 h |
| 400 img | 19 600 | 50 | 17 | 5,6 h |
| 1 600 img | 78 400 | 200 | 67 | 22 h |

> **La lección: el tamaño de train y el de val son knobs independientes.** Train manda en el
> coste; val manda en la resolución. El `split: 80/10/10` de hoy los acopla, y por eso subir val
> parece caro cuando no tiene por qué serlo.

### Decidido: el tamaño del dataset **es una variable de investigación**, no un ajuste

*(D6, decidido 2026-07-16.)* **Cuántas imágenes generar y con qué porcentajes repartirlas son
preguntas del proyecto, no decisiones de diseño**: hoy nadie sabe qué punto da mejor resultado al
menor coste, y las tablas de arriba son aritmética teórica que **asume independencia entre
patches — que es falsa**. Así que `num_images` y las fracciones del split **entran al barrido**,
como cualquier otro parámetro.

Punto de partida razonable mientras tanto: ~2000 imágenes, 80/10/10 → train 1600, val 200, test
200. Pero es un punto de partida **para medir**, no una conclusión.

### La consecuencia: hace falta un **holdout** fuera de B

Barrer los parámetros de B tiene una trampa que hay que cerrar antes de barrer nada:

> **El instrumento de medida no puede ser parte del experimento.** Si el split entra al barrido,
> cada punto tiene un val y un test **distintos**: estarías comparando mediciones hechas con
> reglas distintas. Medirías la regla, no el modelo. Es el contrato ⑧ llevado a su conclusión.

El `test` de hoy **no sirve** para esto: se deriva del mismo `_assign_splits` que estarías
barriendo. Hace falta algo por encima de B:

> **Holdout**: un conjunto de imágenes apartado **una sola vez**, que **ninguna configuración de
> B toca jamás**. Todos los puntos del barrido —de D *o* de B— se miden contra las mismas
> imágenes.

**Y aquí encaja la métrica de §2, que es lo que hace esto posible**: la **F1 de párrafo se mide
por imagen, no por patch**. Por eso un holdout de imágenes sirve para **cualquier** configuración
de B — incluso si cambia `n`, el `stride` o las fracciones. Con métricas de patch no se podría:
cambiar `n` cambia qué son los patches, y no habría nada que comparar entre puntos.

### Decidido: 500 imágenes, fuente propia, generado lo primero

*(D16, decidido 2026-07-16.)* Tres respuestas, y el orden entre ellas importa:

**Cuándo: el primero, antes que ningún dataset de entrenamiento.** No por ceremonia: un holdout
generado *después* es un holdout que ya sabes que necesitabas, y la tentación de elegirlo para
que el resultado salga bien no se puede descartar desde fuera. Generado primero, esa duda no
existe.

**Cómo: su propia fuente, `<nombre>-holdout`.** Una lista de índices apartada dentro de la misma
fuente dependería de que nadie la mire; una fuente separada de la que **nunca se extraen patches
de entrenamiento** hace la fuga **físicamente imposible**. Esa es la propiedad que se compra, y
es la razón de que sea más simple que la alternativa.

**Con qué config: la misma que el resto.** Un holdout más difícil a propósito mediría otra cosa —
la robustez a un cambio de distribución, que es una pregunta legítima y **no es esta**. Misma
receta del generador, misma distribución, otra semilla.

**Cuánto: 500 imágenes.** Da ~5000 positivos por esquina ⇒ sd teórico del recall **≈0,65 %**, por
debajo de la banda que el paso 1 va a medir. Y la aritmética de arriba **asume independencia
entre patches, que es falsa**, así que el suelo real será peor: 500 deja margen para esa
sorpresa. El coste es una corrida del generador y **cero CPU de entrenamiento** — el holdout se
mide una vez, al final, sobre el ganador (§7, regla 5). Ser tacaño aquí no compra nada.

| | Val (dentro de B) | Holdout (fuera de B) |
|---|---|---|
| Tamaño | Un eje del barrido (D6) | **500, fijo para siempre** |
| Quién lo toca | El run y H, todo el rato | **Solo el ganador, una vez** |
| Está sesgado | **Sí**, al alza (§7) | No: nada lo optimizó |

| | Qué es | Quién lo toca |
|---|---|---|
| **train** | Ajusta los pesos | El run |
| **val** | Elige `best.pt`, y rankea el barrido | El run y H — **está sesgado al alza** (§7) |
| **test** (dentro de B) | Sale del mismo split que barres | **Inútil si barres B** |
| **holdout** (fuera de B) | La regla fija de todo | **Solo el ganador, una vez, al final** |

### Si el barrido sale demasiado lento

Salida: **recortar el train del barrido** (tarea proxy: rankear con train pequeño, reentrenar al
ganador con todo). Funciona, pero tiene su riesgo y hay que decirlo: **los hiperparámetros que
ganan con poco train no son siempre los que ganan con mucho** — `lr`, `batch_size` y la
regularización interactúan con el tamaño del dataset. Con `num_images` ya dentro del barrido,
esto deja de ser un truco: **es uno de los ejes que estás midiendo**.

**Y ojo**: `PatchExtractConfig` **no tiene un `limit` de train** (el proyecto hermano sí lo
tiene). Si se quiere la tarea proxy, hay que añadirlo.

---

## 4. Paso 1 — Medir el suelo de ruido

**Toda la aritmética de §3 es teórica y asume independencia. Esto la sustituye por una medición.**

1. Elegir una **config de referencia** (la baseline del proyecto).
2. Correr **5 runs idénticos variando solo `seed`** (el de D: init de pesos y shuffle; el de B
   **no se toca**, contrato ⑧: mismo split o se mide otra cosa).
3. Reportar **media ± sd** de la F1 de párrafo y del `f1` de patch.

**Coste**: 5 × 6,7 min ≈ **33 min** con el dataset de hoy. Con el train de 1600 imágenes, 5 × 67
min ≈ 5,5 h — sigue valiendo la pena, y es de una sola vez.

El resultado es **un número: la diferencia mínima creíble**. A partir de ahí, la regla es dura:

> **Toda diferencia dentro de la banda de ruido es un empate.** No se rompe con "pero es que
> este subió". Si el barrido devuelve 20 puntos y los 6 primeros caben dentro de la banda, el
> resultado es *"seis empatados"*, no *"ganó el primero"*.

Este número **también decide si el paso 0 fue suficiente**: si el suelo medido sigue por encima
de las diferencias que te importan, genera más val. Es un bucle, y termina cuando la banda es
más estrecha que lo que buscas.

Y sale gratis una segunda cosa: **la baseline**, que es el punto de comparación de todo lo demás.

---

## 5. Paso 2 — Comprobar que el proxy sirve

Con la métrica de §2 escrita:

1. Coger **~8 runs** que cubran un rango ancho de calidad (los 5 de la baseline + unos cuantos
   deliberadamente malos: `lr` alto, sin `border_features`, pocas épocas).
2. Medir en cada uno **`f1` de patch** y **F1 de párrafo**.
3. Mirar la **correlación de rangos (Spearman)** entre las dos.

Cómo se lee:

- **Correlación alta** → el proxy sirve. Las vistas de patch (V7, V8, V9) diagnostican de verdad,
  y un barrido rankeado por patch-f1 no se aleja del objetivo.
- **Correlación baja** → **hallazgo grande**, y de los que cambian el proyecto: significaría que
  el eslabón débil no es la CNN sino la **reconstrucción** (el emparejado voraz TL→BR de
  `reconstruct_boxes`, que es una heurística), y que mejorar la red no mejora los párrafos. Eso
  redirige el esfuerzo entero.

Cualquiera de las dos respuestas vale su coste. Y es la única forma de saber cuál de las dos es.

---

## 5.5 Medido — la evidencia disponible por esquina *(2026-07-20)*

**Resultado de investigación, no invariante.** Los tests afirman que la medida significa una
cosa; los números de aquí son de este dato y estos runs, y envejecen. V18 los recalcula.

### El fenómeno

La ventana deslizante etiqueta un patch con toda esquina cuyo **punto** cae dentro, y no dice
nada de si el **párrafo** de esa esquina cae dentro. Una TL cerca del borde inferior-derecho del
patch tiene su párrafo *fuera*: la etiqueta pide algo que los píxeles no enseñan.

La medida (`itf.metrics.corner_evidence`) es la fracción del patch que el cuerpo del párrafo
*puede* ocupar: `(1-fx)(1-fy)` para TL, y sus tres simétricas. Geométrica, sin píxeles ni modelo,
y **cota superior** — lo que marca como ciego lo está.

### Cuántas son

Fracción de esquinas positivas con evidencia < 0,05:

| B | patch | positivos | <0,01 | **<0,05** | <0,10 | mediana |
|---|---|---|---|---|---|---|
| `fase3-red` | 40 | 8 088 | 4,6 % | **16,2 %** | 27,9 % | 0,226 |
| `dirty-10` | 10 | 395 866 | 4,5 % | **16,7 %** | 28,5 % | 0,221 |
| `dirty-20` | 20 | 5 191 745 | 3,7 % | **14,4 %** | 25,2 % | 0,247 |

**No es una anomalía del extractor.** Con esquinas repartidas uniformemente dentro del patch lo
esperado sería 20,0 % bajo 0,05 y mediana 0,186; sale algo **mejor** que uniforme, lo cual encaja
con que `extract.py` desempata por cercanía al centro. Es el precio estructural de una ventana
deslizante, y por eso **no depende del `stride`** (ver ui.md, V7).

### Qué le hacen al modelo

`dirty-20-lambda_pos_1-0005`, sobre los **tres** splits (5,03 M / 628 k / 628 k patches):

| | train | val | test |
|---|---|---|---|
| score medio, ciegas | 0,623 | 0,621 | 0,619 |
| score medio, visibles | 0,882 | 0,881 | 0,881 |
| score medio, negativos | 0,037 | 0,037 | 0,037 |
| cuota del error de posición, ciegas | 30,9 % | 30,7 % | — |
| err_px ciegas / visibles | 5,23 / 1,97 | 5,26 / 1,98 | — |

Y la banda entera, en val: el error por banda sube **monótonamente** según baja la evidencia
(1,76 → 1,64 → 2,14 → 2,91 → **5,26** px), o sea dosis-respuesta, no un artefacto del corte.

### Las dos conclusiones, que van en direcciones opuestas

**1. En clasificación no hay nada roto.** El hueco train↔val de las ciegas es **0,002**, el de las
visibles **0,001**. Cero memorización. Score alto en val es generalización por definición, y el
control que lo cierra es que los **negativos se quedan en 0,037**: el modelo *distingue* una
esquina ciega de un patch sin esquina, luego hay señal visual real (contexto, y la brizna del
párrafo que sí entra). La hipótesis natural —«etiquetar sin ver el párrafo enseña ruido»— **queda
refutada** para la cabeza de detección.

**2. En posición sí hay un coste, y es grande.** El 14 % de las esquinas se lleva el **31 %** del
error, con 5,3 px contra 2,0 px sobre un patch de 20 (26 % del patch frente al 10 %). Y como es
**idéntico en train y en val**, no es distorsión: es dificultad genuina. Localizar el píxel exacto
de una esquina que no se ve no se puede hacer, y el modelo tampoco lo hace donde entrenó.

**Consecuencia para la métrica**: un `pos_err_px` global promedia lo posible con lo imposible y
**esconde cuánto margen queda de verdad**. Por eso V18 reporta detección y posición por separado,
en vez de un número.

### Lo que NO se ha medido — no lo cites como si sí

- **Si quitar las ciegas de la pérdida de posición mejoraría las visibles.** Eso es contaminación
  de una población sobre otra, y el contraste train↔val no lo contesta. Hace falta una **ablación**:
  dos runs idénticos, uno enmascarando la pérdida de posición donde la evidencia es baja,
  comparados **sobre las visibles**. Con las N semillas de §7, que aquí no hay.
- **Varianza de semilla.** Los nueve runs de `dirty-20-lambda_pos_1-*` llevan `seed: 1`; solo varía
  `lambda_pos`, y en una banda estrecha (0,545–0,715). Dos de ellos dan cuota 30,4 % y 30,7 %, lo
  cual es estabilidad frente a λ, **no** frente a semilla.

### Hallazgo colateral, y probablemente el más accionable

**Este modelo no sobreajusta en absoluto.** El hueco train↔val es ~0 en *todas* las poblaciones,
no solo en las ciegas, con 5,03 M de patches y 5 épocas. Para el barrido eso significa que
**capacidad y épocas están infraexplorados**, y que la regularización (dropout, weight decay)
ataca un problema que hoy no existe. Merece comprobarse con más semillas antes de construir sobre
ello, pero es el tipo de cosa que reordena un espacio de búsqueda.

---

## 6. Paso 3 — Quitar los sesgos antes de barrer

Barrer antes de esto es **medir el bug** (todos están en organizacion.md §3):

| | Qué pasa si no |
|---|---|
| `momentum` (hoy `_make_optimizer` solo pasa `lr` y `weight_decay`) | SGD corre a momentum 0: barrer `optimizer` compara Adam contra un espantapájaros |
| `smooth_l1_beta` (hoy 1.0 con coords en [0,1]) | La pérdida de posición es **MSE pura**; barrer λ mueve el peso de un término que no es el que crees |
| `scheduler` (hoy `lr` constante) | Barrer `lr` optimiza para un régimen que luego no vas a usar |

**Y estos arreglos invalidan lo ya entrenado**: cambian los resultados. `cnn-02-01` y compañía
dejan de ser comparables con lo nuevo. Es correcto arreglarlos — son bugs — pero **la baseline
del paso 1 se mide después**, no antes.

---

## 7. Las reglas de comparación

Lo que hace falta para poder decir *"X es mejor que Y"*:

1. **Mismo commit de git.** Un cambio en la pérdida o en el optimizador mata las comparaciones
   anteriores. El run debe registrar el commit — el proyecto hermano lo hace (`repro.py`), ITF
   **no**, y es parte de `exp-registry`.
2. **Misma huella de B** (contrato ⑧). Un dataset reconstruido bajo el mismo nombre **no es** el
   mismo dataset, y hoy nada lo detecta.
3. **N semillas, y media ± sd.** Nunca un número suelto. Un run aislado no es un resultado: es
   una anécdota.
4. **La diferencia supera la banda de ruido** (§4). Si no, es un empate.
5. **El test se toca una sola vez, al final, y solo el ganador.**

Sobre la 5, que es la que más se incumple sin darse cuenta:

> Val hace **dos** trabajos: elegir `best.pt` dentro de cada run **y** elegir al ganador del
> barrido. Con 20 configs compitiendo por el mejor val, el val del ganador está **sesgado al
> alza** — más o menos por el tamaño de la banda de ruido. **Ese número no es el que se
> reporta.** Se reporta el de test, medido una vez.

Hoy ITF **nunca toca el test** (`loop.py` solo usa train y val), así que el test está virgen. Es
un buen accidente: la disciplina se establece ahora, gratis, antes de que aparezca la tentación.

---

## 8. Presupuesto y orden

```
Paso 0  generar el holdout (500 img)        ~min   ← lo primero, y no se vuelve a tocar
Paso 0  generar ~2000 img + extraer         ~1 h (generador + extracción)
Paso 3  arreglar momentum/beta/scheduler     fase 3 del plan
Paso 1  baseline + suelo de ruido (5 seeds)  ~5,5 h  (una vez)
Paso 2  métrica de párrafo + ~8 runs         ~1 h + escribir la métrica
────────────────────────────────────────────────────────────────
Barrido 20 puntos, con poda                  una noche
```

**El orden es 0 → 3 → 1 → 2 → barrido.** El paso 3 se adelanta al 1 porque los arreglos cambian
los resultados: medir la baseline antes sería medirla dos veces.

Los pasos 0–2 cuestan **menos de un día** entre todos, y son los que deciden si la noche de
barrido significa algo. Es la mejor relación coste/valor que hay ahora mismo en el proyecto.

---

## 9. Análisis pendientes — la cola

**Para empezar en frío.** Cada entrada trae la pregunta, por qué está sin contestar, cómo se
contesta y con qué se rompe. No es una lista de deseos: son análisis concretos que salieron de
mediciones ya hechas y que quedaron a medias por presupuesto, no por indecisión.

Al cerrarse uno, su resultado se escribe arriba (§5.5 y vecinos) y aquí queda tachado o borrado.

### P1 — Ablación: ¿la pérdida de posición de las esquinas ciegas perjudica a las visibles?

**La pregunta que §5.5 dejó abierta.** Allí quedó refutado que las etiquetas ciegas distorsionen
la **clasificación** (hueco train↔val de 0,002). Lo que **no** se midió es si su contribución a la
pérdida de *posición* —el 31 % del error, sobre casos irreducibles— roba capacidad o gradiente a
las esquinas que sí se ven. El contraste train↔val **no lo contesta**: eso detecta memorización,
y esto es contaminación de una población sobre otra.

**Cómo se contesta**: dos runs idénticos salvo enmascarar (o atenuar) la pérdida de posición donde
`corner_evidence < 0.05`, con N semillas (§4), y **comparados sobre las esquinas visibles**. Es
dominio **D**, no B: se pondera en el dataloader con `itf.metrics.corner_evidence`, que no toca la
huella de B, así que no invalida nada de lo ya entrenado.

**La trampa que hay que evitar** (y que confundiría el resultado): al bajar el peso de las ciegas,
el modelo dejará de predecirlas y la métrica **global** empeorará aunque el modelo sea mejor. Hay
que fijar **antes** cuál es el número que manda — el de las visibles — o la caída se lee como un
fracaso del pesado.

**Empezar por `dirty-10`**, que es 4× más barato que `dirty-20` y basta para ver el signo.

### P2 — Confirmar el no-sobreajuste con N semillas

**Por qué importa más que su tamaño**: si es real, reordena el espacio de búsqueda entero
(§5.5, hallazgo colateral). El hueco train↔val salió ~0 en *todas* las poblaciones con 5,03 M de
patches y 5 épocas ⇒ **capacidad y épocas están infraexplorados**, y dropout/weight decay atacan
un problema que hoy no existe.

**Por qué no está confirmado**: es **un run** (`dirty-20-lambda_pos_1-0005`). Y no vale mirar los
otros ocho: los nueve llevan `seed: 1`, solo varía `lambda_pos`. Hacen falta runs con semillas
distintas, que **no existen todavía**.

Se mide con el instrumento ya montado: V18 y el diagnóstico sobre `train` y `val` del mismo run.

### P3 — Barrer las **formas** de la red (dominio C)

*(Los detalles se discuten al abordarlo; esto fija la pregunta y lo que ya se sabe que estorba.)*

**Por qué está en la cola**: es la consecuencia directa de P2. Si no hay sobreajuste, lo que falta
es capacidad — y la capacidad vive en **C**, que hoy nadie ha barrido.

**Lo que hay que resolver antes de escribir código**, porque no es un barrido más:

- **H está definido como «espacio de D con **B y C fijos**»** (organizacion.md §1). Barrer C es un
  objeto **distinto**, no un parámetro nuevo. O se extiende la definición de H o nace otro
  sustantivo — y esa elección es de organizacion.md, no del código. Ver **D22**.
- **El contrato ① lo acota**: `patch_size` (B) == `input_size` (C). Un espacio de C que muestree
  `input_size` genera puntos inválidos contra un B dado, y `check_run` tiene que rechazarlos
  **antes de reservar el nombre** — es la cuarta puerta de entrenar.
- **Comparabilidad**: dos C con capacidad distinta sobre el mismo B son comparables por `f1`, pero
  el coste por época **no** es constante, así que la poda por tiempo favorece a las pequeñas sin
  decirlo. Decidir si el presupuesto se cuenta en épocas o en segundos.
