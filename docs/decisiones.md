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
| **D9** | Nombres de las librerías (`exp-registry`, `jobq`, `convspec`, `matrixview`) | Provisionales; decidir al extraer la primera | librerias.md §1 |
| **D10** | ¿Monorepo `claude-libs` o cuatro repos? | Monorepo: cuatro repos es más ceremonia que valor a esta escala | librerias.md §4 |
| **D11** | ¿Backportear NIST a las librerías? | **No.** Funciona; su valor ya está cobrado como evidencia | librerias.md §4 |
| **D13** | Kernels profundos: ¿corte del canal 0, todos los canales, o nada? | Nada: de la capa 2 en adelante la información está en los feature maps | ui.md §4.1 |
| **D14** | ¿Editor de patches? | No: V4 (occlusion) y V5 (scrubber) lo cubren mejor y en distribución | ui.md §5 |

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

---

## 4. Ya decididas

| | Decisión | Vive en | Fecha |
|---|---|---|---|
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
