# image-text-finder

Detección de esquinas de párrafo por patches: se trocea cada imagen en patches `n×n` y una CNN
configurable responde, por patch, **¿cae aquí una esquina de párrafo y dónde?** — una cabeza por
tipo (`TL, TR, BR, BL`). En inferencia, una ventana deslizante recompone los párrafos.

Las imágenes las produce
[image-text-sample-generator](../image-text-sample-generator) (ver su `SAMPLE_FORMAT.md`).

---

## Estado: fase 3 — ya se puede entrenar

Hechas las fases **0** (decisiones), **0.5** (los contratos en xfail), **1** (esqueleto y paleta),
**2** (Fuentes y Patches) y **3** (Redes y Recetas) de [docs/plan-ui.md](docs/plan-ui.md). La
siguiente es la **fase 4**: Entrenar y Runs (E).

Desde la fase 3 hay **red** (C) y **receta** (D) como entidades con nombre, almacén y pantalla — y
`itf-train`, así que **se entrena por CLI sin esperar a la UI**.

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

**Fuentes** y **Patches** funcionan de verdad: listan, construyen y borran. El resto de pantallas
están vacías y cada una dice qué fase la construye. `/kitchen` es donde se mira la paleta y los
componentes base.

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

Todo lo que puede negarse, se niega **antes del primer batch** — con la razón y el arreglo, no
media hora después dentro del job. Sobre el `prueba` de arriba (5 imágenes ⇒ val vacío):

```powershell
.\.venv\Scripts\itf-train.exe --name x --patch-dataset prueba --network cnn-a --recipe baseline
```

```
Este dataset no sirve para medir:

  el dataset '...\data\patch-datasets\prueba' no tiene val, así que no se puede elegir
  best.pt por 'val_loss'. Reconstrúyelo con una fracción de val > 0: sin val, elegir
  checkpoint cae en la pérdida de entrenamiento y se queda el más sobreajustado, en silencio.
```

Sale con **2** y **no deja un run a medias**. Es una negativa, no un fallo: al construir, un val
vacío solo **avisa** (puede que solo quieras mirar patches); al entrenar, **se niega**, que es
donde está el daño. Lo mismo con los contratos ① y ② (`patch_size` ≠ `input_size`, o una red con
`border_features` sobre un dataset que no los trae): los ve el validador, son microsegundos, y
salen con la misma forma.

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

Salida esperada hoy: **`42 passed, 4 xfailed`**, en verde y en ~7 s. Que estén en xfail no es
deuda: es el mecanismo. Cuando una fase implementa su contrato, el test pasa, el **XPASS estricto
pone la suite en rojo** y obliga a quitar el marcador — así "lo que falta" es una lista ejecutable
en vez de prosa que envejece. Ver [docs/tests.md](docs/tests.md) §2.

Los 4 que quedan son los contratos ③ y ④ (fase 4), ⑤ (fase 6) y ⑨ (fase 7), y **fallan por lo
correcto**: no existen `itf.inference`, la procedencia por nombre ni `POST /sweeps`.

### El entorno

- **Python 3.12** (PyTorch aún no tiene wheels para 3.14). Verificado con **3.12.10**.
- El intérprete del proyecto es `.\.venv\Scripts\python.exe`.
- Los tres scripts que declara `pyproject.toml` —**`itf-extract`, `itf-api` e `itf-train`**—
  funcionan. `itf-train` llegó en la fase 3.
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
├── validation/  # compatibilidad B↔C: función pura de dos dicts (contratos ①②)
├── datasets/    # lee labels.jsonl (SAMPLE_FORMAT)
├── patches/     # extracción n×n -> .npz  +  torch Dataset
├── models/      # config -> CNN + cabeza de esquinas
├── training/    # pérdidas, bucle, checkpoints, métricas
├── inference/   # detección por ventana deslizante + reconstrucción de párrafos
└── api/         # FastAPI: un recurso por dominio
web/             # Vite + React — ya existe (fase 1)
├── src/theme/    # tokens.css: LA PALETA, y solo aquí
├── src/components/  # MatrixCanvas, Meter, NumberTable, Async
└── scripts/      # validate-palette.mjs
configs/         # networks/*.yaml (redes)  ·  recipes/*.yaml (recetas)
tests/           # test_contracts.py: un test por contrato — ya existe
docs/            # el diseño
```

`data/` y `runs/` son artefactos: se ignora la carga (`.npz`, `.pt`) y **se versiona la
descripción** (configs, métricas, manifests) — ver [docs/formatos.md](docs/formatos.md) §5.
