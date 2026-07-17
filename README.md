# image-text-finder

Detección de esquinas de párrafo por patches: se trocea cada imagen en patches `n×n` y una CNN
configurable responde, por patch, **¿cae aquí una esquina de párrafo y dónde?** — una cabeza por
tipo (`TL, TR, BR, BL`). En inferencia, una ventana deslizante recompone los párrafos.

Las imágenes las produce
[image-text-sample-generator](../image-text-sample-generator) (ver su `SAMPLE_FORMAT.md`).

---

## Estado: fase 5 — la app se vuelve el instrumento

Hechas las fases **0** (decisiones), **0.5** (los contratos en xfail), **1** (esqueleto y paleta),
**2** (Fuentes y Patches), **3** (Redes y Recetas), **4** (Entrenar y Runs) y **5** (la tabla por
patch y el diagnóstico) de [docs/plan-ui.md](docs/plan-ui.md). La siguiente es la **fase 6**:
mapas, kernels y el pipeline.

**Se entrena desde la UI y el run sabe de dónde salió** (dato → red → receta → run), y desde la
fase 5 **se puede mirar qué hace ese run, patch a patch**: la tabla por patch —un caché— y las
vistas V3, V6, V7 y V8, más las curvas de entrenamiento en small multiples.

Lo que compró, medido sobre `fase4-ui` la primera vez que se usó el instrumento:

- **El umbral sale gratis**: f1 **0,673** con `threshold` 0,50 y **0,728** con 0,64 — post-hoc,
  sobre scores ya guardados, **sin reentrenar ni un batch**. Y ahí está por qué V8 va antes que el
  barrido: eso es F, y buscarlo en D cuesta horas de CPU por punto.
- **V7 dice qué dominio arreglar**: el error en el **borde** del patch es **16,4 px** contra
  **9,1 px** en el centro — casi el doble. Eso apunta a bajar el `stride` de B, no a meter
  filtros en C.

El código anterior sigue recuperable en el tag **`pre-rediseno`**:

```powershell
git show pre-rediseno:src/itf/training/losses.py     # un fichero
git checkout pre-rediseno -- src/                    # todo el paquete
```

### Montar

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[train,api,dev]"
cd web; npm install
```

### Correr

Dos procesos. El backend:

```powershell
.\.venv\Scripts\python.exe -m itf.api          # http://127.0.0.1:8000
```

Y el front, que proxya `/api` al backend:

```powershell
cd web
npm run dev                                     # http://localhost:5173
```

Funcionan de verdad **Fuentes**, **Patches**, **Redes**, **Recetas**, **Entrenar**, **Runs** y
**Diagnóstico**: el flujo entero, de la imagen al modelo entrenado y de ahí a mirar qué hace. Las
dos que quedan (Predecir, Barridos) están vacías y cada una dice qué fase la construye.
`/kitchen` es donde se mira la paleta y los componentes base.

Las imágenes fuente salen de [image-text-sample-generator](../image-text-sample-generator) y se
buscan en `../image-text-sample-generator/data/datasets`. Para apuntar a otro sitio:

```powershell
$env:ITF_DATASETS_ROOT = "D:\mis\datasets"
```

### Construir un dataset de patches sin la UI

El CLI hace exactamente lo mismo, y no es una comodidad: **si algo solo funcionara por HTTP,
estaría en la capa equivocada** (docs/api.md §0).

`--source` admite **un id**, una ruta relativa o una ruta absoluta. **Usa el id**: es lo único que
no depende de desde dónde ejecutes el comando.

```powershell
.\.venv\Scripts\itf-extract.exe --source clean-paragraphs-01/reducido --out data\patch-datasets\prueba --patch-size 40 --stride 20
```

Sobre esa fuente de 5 imágenes avisa de que el split de val queda vacío — que es correcto y es el
aviso que faltaba: sin val, un dataset no sirve para medir.

Si el nombre no existe, el comando **lista las fuentes que sí hay, con su ruta absoluta**. No es
cosmético: hay dos que solo se distinguen por el sufijo —`clear-paragraphs-02-reducidos` (160×160)
y `clear-paragraphs-02-8ea1ac04` (640×480)— y **equivocarse no falla**: construye un dataset
perfectamente válido con **14,5× más patches por imagen** y un desbalance de ~67:1 en vez de 3,9:1.
La que miden [docs/protocolo.md](docs/protocolo.md) §1 y el ejemplo de abajo es **la de 160×160**.

### Entrenar

Hacen falta tres cosas con **nombre**: un dataset de patches (B), una red (C) y una receta (D).
`itf-train` **no acepta valores sueltos, solo nombres** — y esa rigidez es a propósito: es lo que
hace que la procedencia del run se sostenga sola (docs/api.md R7). ¿Quieres algo a medida?
Guárdalo antes.

```powershell
# 1. un dataset con val de verdad (200 imágenes → 160/20/20)
.\.venv\Scripts\itf-extract.exe --source clear-paragraphs-02-reducidos --out data\patch-datasets\fase3-red --patch-size 40 --stride 20

# 2. entrenar: cnn-a y baseline vienen versionadas en configs/
.\.venv\Scripts\itf-train.exe --name fase3-01 --patch-dataset fase3-red --network cnn-a --recipe baseline --device cpu
```

`--device` es una bandera y **no** un campo de la receta: es X, cuesta tiempo y no cambia el
resultado (contrato ⑩). Si viviera en la receta, lo entrenado en CPU y en GPU parecerían dos
recetas distintas.

Verificado de punta a punta el 2026-07-16 sobre `clear-paragraphs-02-reducidos`: **20 épocas en
7,2 min** (21,7 s/época), F1 de patch **0,80**, `pos_err_px` **9,4**. El run queda en
`runs/fase3-01/` con `config.json` (la receta y la red congeladas **por valor**, y `execution`
aparte), `metrics.jsonl` (una línea por época, apendable y consultable en vivo), `best.pt`,
`last.pt`, `summary.json` y `status.json`.

**`best.pt` no es la última época**: en esa corrida salió de la **17**, elegida por `val_loss`.
Ésa es justo la razón de que un dataset **sin val** no sirva para medir — y por la que `train()`
se niega a entrenar sobre uno, en vez de caer al train loss en silencio y quedarse el checkpoint
más sobreajustado.

Todo lo que puede negarse, se niega **antes del primer batch** y **antes de reservar el nombre**
— con la razón y el arreglo, no media hora después dentro del job. Sobre el `prueba` de arriba
(5 imágenes ⇒ val vacío):

```powershell
.\.venv\Scripts\itf-train.exe --name x --patch-dataset prueba --network cnn-a --recipe baseline
```

```
No se puede entrenar esto, y se ve antes del primer batch:

  [no_validation_split] el dataset no tiene patches de val, así que no hay con qué elegir
  best.pt ni con qué medir
    -> reconstruye el dataset con una fracción de val > 0: sin val, elegir checkpoint cae en
       la pérdida de entrenamiento y se queda el más sobreajustado, en silencio
```

Sale con **2** y **no deja `runs/x/` a medias** — que importa más de lo que parece: si el nombre
quedara reservado, arreglar el dataset y reintentar contestaría *«ese run ya existe»* por un run
que no llegó a ver un batch. Es una negativa, no un fallo: al construir, un val vacío solo
**avisa** (puede que solo quieras mirar patches); al entrenar, **se niega**, que es donde está el
daño.

El contrato ① sale igual, y con la misma forma — `cnn-a` espera 40, así que un dataset de 60 no
entra:

```powershell
.\.venv\Scripts\itf-extract.exe --source clear-paragraphs-02-reducidos --out data\patch-datasets\tmp-60 --patch-size 60 --stride 30
.\.venv\Scripts\itf-train.exe --name y --patch-dataset tmp-60 --network cnn-a --recipe baseline
```

```
  [patch_size_mismatch] la red espera patches de 40x40 y el dataset los tiene de 60x60
    -> elige un dataset con patch_size 40, o una red con input_size 60
```

**Las dos puertas —`itf-train` y `POST /runs`— preguntan a la misma función** (`itf.validation.
check_run`). No es cortesía: dos comprobaciones separadas se desincronizan, y la puerta que queda
más laxa es por la que entra un barrido.

### Entrenar desde la UI, y mirar el run

**Entrenar** (`/train`) elige tres nombres —B, C y D— y `device` **aparte**. Enseña si el dataset
y la red casan (contrato ①) y **estima el coste** con los `seconds` que otros runs ya midieron:

```
Coste estimado: 25.5 s/época × 20 épocas ≈ 8.5 min
  medido sobre 1 run(s) con el mismo dataset y la misma red (fase4-ui)
```

Solo estima con runs **comparables de verdad**: misma huella de B, misma red. Si no hay ninguno,
**lo dice** en vez de inventarse un número. Y por eso un run sin procedencia no sirve para
estimar, aunque tenga las métricas: no puede decir de qué dataset salió.

**Runs** (`/runs`) enseña de qué B, C y D salió cada uno **por nombre**, con la huella de B, el
commit y el entorno. Las métricas llegan **incrementalmente** (`?since=`): nunca se reenvía el
historial. Y desde la fase 5 van también como **curvas: tres paneles apilados** (V14) —
`loss ≈ 0.28`, `f1 ≈ 0.77` y `pos_err_px ≈ 11` son tres escalas, así que **nunca comparten
gráfica ni doble eje** (docs/ui.md R4). Apilados y con el eje de épocas **alineado**: eso es lo
que deja compararlos sin que la gráfica invente la correlación por ti.

**Parar** es cooperativo: el run **termina la época** en curso —métricas escritas, checkpoint
guardado— y cierra como `cancelled`, no como `done`. Verificado a mano: parado en la época 2, cerró
en la 3 con *«3 de 20 épocas · parado a mano»*. Se cierra como `cancelled` porque tiene pesos de
verdad: llamarlo `done` lo colaría en una comparación como si hubiera terminado.

**Un run no se sobrescribe jamás.** Reusar un nombre contesta **409** con la razón y el arreglo, y
no toca lo que hay:

```
ya existe un run llamado 'fase4-ui'
elige otro nombre, o borra ese run primero: no se sobrescribe nunca
```

Era una trampa medida del código anterior (`mkdir(exist_ok=True)` + truncar `metrics.jsonl`), y
quien la pisa es justo un barrido que autogenera nombres.

### Diagnóstico: qué hace el run, patch a patch

**Diagnóstico** (`/diagnostics`) elige un run y un split y enseña tres vistas que leen **una sola
pasada** sobre ese split. Esa pasada es una **tabla por patch** (`score`, posición predicha, error
en px, por esquina) y es un **caché**, no una entidad: se puede recalcular exacta a partir del run,
la huella de B y el split, así que no se nombra, no se lista y **borrarla no pierde nada**
(D1). Vive en `data/cache/diagnostics/`, gitignoreada.

También responde por HTTP, que es donde se ve lo que compra. Con la API corriendo:

```powershell
curl "http://127.0.0.1:8000/runs/fase4-ui/diagnostics/pr?split=val&corner=TL"
curl "http://127.0.0.1:8000/runs/fase4-ui/diagnostics/error-map?split=val&bins=10"
curl "http://127.0.0.1:8000/runs/fase4-ui/diagnostics/patches?split=val&outcome=fp&threshold=0.9"
```

Medido de punta a punta el 2026-07-17 sobre `fase4-ui` (980 patches de val):

| | |
|---|---|
| Primer GET (calcula la tabla) | **1,0 s** |
| Segundo GET (lee el caché) | **0,025 s** |
| Otro agregado sobre la misma tabla | **0,014 s** |

Por eso `/diagnostics` es **síncrono y no un job** (docs/api.md R3). El día que necesite un 202,
la tabla dejó de ser barata y el umbral gratis se fue con ella.

**V8 — el barrido gratis.** Los scores están guardados, así que mover el `threshold` **no vuelve a
correr el modelo**: es filtrar una columna. Sobre `fase4-ui`, f1 **0,673** en 0,50 y **0,728** en
0,64 — **+0,055 sin reentrenar nada**. Y el desbalance que sale solo es **20,5 % de positivos
(3,9:1)**, exactamente el que documenta [docs/protocolo.md](docs/protocolo.md) §1.

**V7 — qué dominio arreglar.** Sobre `fase4-ui`: **borde 16,4 px vs centro 9,1 px**. El error se
concentra en los bordes del patch —esquinas medio visibles— y eso se arregla **bajando el `stride`
de B**, no metiendo filtros en C. Sin esta vista, ese diagnóstico se confunde sistemáticamente con
«la red es pequeña».

> **`bins` no es 40, y es un hallazgo de la fase 5.** [docs/ui.md](docs/ui.md) §4.1 pedía un mapa
> 40×40; con ~200 esquinas de un tipo repartidas en 1600 celdas eso son **0,1 muestras por celda** y
> el mapa sale **moteado: cierto e ilegible**. A 10×10 (celdas de 4 px, ~8 esquinas cada una) la
> estructura borde-vs-centro se ve de un vistazo. La resolución es un control, y el ratio ~2× sale
> igual a 10×10 que a 40×40 — o sea que es real, no un artefacto del binning.

**V6 y V3.** La galería va **peor-primero** y se filtra por resultado (`fp`, `fn`, aciertos…) al
umbral que tengas puesto — otra vez, sin recalcular. Un clic abre **V3**: el patch con las 4
esquinas, cuatro *meters* contra el umbral y el error dibujado como la línea entre dónde estaba la
esquina (el anillo) y dónde la puso el modelo (el punto).

Todo lo que no se puede medir **se niega con la razón y el arreglo**, nunca con un número inventado:

```
GET /runs/fase3-01/diagnostics/pr    -> 409 run_without_provenance
  "no tiene procedencia: no puede decir de qué dataset salió, así que no hay contra qué
   diagnosticarlo" -> "es anterior a la fase 4. Bórralo y reentrénalo: no es comparable con nada"
```

Y si el dataset se reconstruyó bajo el mismo nombre desde que se entrenó el run, **la huella no
cuadra y el diagnóstico se niega** (contrato ⑧): su split ya no es el que ese `best.pt` usó para
elegirse, así que los números saldrían con buena cara y medirían otra cosa.

### La paleta se valida, no se opina

```powershell
cd web
npm run validate:palette
```

Debe decir **`→ PASA en claro y en oscuro`** (exit 0). El script parsea
`web/src/theme/tokens.css`, así que valida **lo que de verdad se sirve**, no una copia. Comprueba
la banda de luminosidad, el suelo de croma, la separación bajo daltonismo (protanopía y
deuteranopía, Machado-Oliveira-Fernandes 2009), el suelo de visión normal, el contraste contra la
superficie, la monotonía de la rampa secuencial y que el 0 de la divergente sea gris neutro.

Reporta dos **WARN**, y los dos son legales *solo* porque el diseño ya exige la mitigación:
etiquetado directo (R1) y la tabla de números (R5). Ver [docs/ui.md](docs/ui.md) §4.0.

### Los tests de contrato

La **barra de progreso del plan**: un test por contrato de `docs/organizacion.md` §2, todos en
`xfail(strict=True)` mientras no exista su implementación.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Salida esperada hoy: **`90 passed, 2 xfailed`**, en verde y en ~17 s. Que estén en xfail no es
deuda: es el mecanismo. Cuando una fase implementa su contrato, el test pasa, el **XPASS estricto
pone la suite en rojo** y obliga a quitar el marcador — así "lo que falta" es una lista ejecutable
en vez de prosa que envejece. Ver [docs/tests.md](docs/tests.md) §2.

Los 2 que quedan son los contratos ⑤ (fase 6) y ⑨ (fase 7), y **fallan por lo correcto**: no
existen `itf.inference.predict` ni `POST /sweeps`.

### El entorno

- **Python 3.12** (PyTorch aún no tiene wheels para 3.14). Verificado con **3.12.10**.
- El intérprete del proyecto es `.\.venv\Scripts\python.exe`.
- Los tres scripts que declara `pyproject.toml` —**`itf-extract`, `itf-api` e `itf-train`**—
  funcionan. `itf-train` llegó en la fase 3, y desde la fase 4 pasa por la misma puerta que el API:
  valida con `check_run` y reserva el nombre con `RunStore.create`.
- **Solo CPU hoy.** Habrá GPU para procesamiento masivo; por eso `device` ya está fuera de la
  identidad de la receta (contrato ⑩). Y por eso el límite de workers es **1**: torch ya usa
  todos los núcleos y cada run carga su `PatchDataset` entero en RAM, así que lanzar N
  entrenamientos a la vez no acelera nada y se queda sin memoria.

## Por dónde se empieza

Lee [CLAUDE.md](CLAUDE.md): abre con el estado y enlaza los once documentos. En orden:

| | |
|---|---|
| [docs/organizacion.md](docs/organizacion.md) | **La raíz.** Los dominios (A–H, X, G) y los contratos ①–⑩ donde se tocan |
| [docs/protocolo.md](docs/protocolo.md) | Cuándo un resultado es creíble. **Léelo antes de sacar conclusiones de un entrenamiento** |
| [docs/api.md](docs/api.md) · [docs/ui.md](docs/ui.md) | La organización proyectada sobre HTTP y sobre pantallas |
| [docs/plan-ui.md](docs/plan-ui.md) | El plan de ejecución, por fases |
| [docs/formatos.md](docs/formatos.md) · [docs/tests.md](docs/tests.md) | Los artefactos en disco; qué se testea |
| [docs/decisiones.md](docs/decisiones.md) | Lo que sigue sin decidir, y qué bloquea |
| [docs/glosario.md](docs/glosario.md) | Las palabras que significan dos cosas |
| [docs/librerias.md](docs/librerias.md) | Qué se extrae para reutilizar en otros proyectos |

## Estructura prevista

```
src/itf/
├── geometry/    # la ventana deslizante, compartida por extracción e inferencia (contrato ⑤)
├── metrics.py   # qué significan pos_err_px y la f1. Un sitio, dos lectores (D y el diagnóstico)
├── validation/  # compatibilidad B↔C: función pura de dos dicts (contratos ①②)
├── datasets/    # lee labels.jsonl (SAMPLE_FORMAT)
├── patches/     # extracción n×n -> .npz  +  torch Dataset
├── models/      # config -> CNN + cabeza de esquinas
├── training/    # pérdidas, bucle, checkpoints, métricas
├── inference/   # detección por ventana deslizante + reconstrucción de párrafos
├── diagnostics/ # E×B: la tabla por patch (un CACHÉ) y sus agregados — V6, V7, V8
└── api/         # FastAPI: un recurso por dominio
web/             # Vite + React
├── src/theme/    # tokens.css: LA PALETA, y solo aquí
├── src/components/  # MatrixCanvas, Meter, PatchCanvas, PlotFigure, TrainingCurves, Declares…
├── src/screens/diagnostics/   # V3, V6, V7, V8
└── scripts/      # validate-palette.mjs
configs/         # networks/*.yaml (redes)  ·  recipes/*.yaml (recetas)
tests/           # test_contracts.py: un test por contrato  ·  test_diagnostics.py: las costuras
docs/            # el diseño
```

`data/` y `runs/` son artefactos: se ignora la carga (`.npz`, `.pt`) y **se versiona la
descripción** (configs, métricas, manifests) — ver [docs/formatos.md](docs/formatos.md) §5.
`data/cache/` es **derivado entero**: se recalcula exacto, así que ni se versiona ni se echa de
menos.
