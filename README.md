# image-text-finder

Detección de esquinas de párrafo por patches: se trocea cada imagen en patches `n×n` y una CNN
configurable responde, por patch, **¿cae aquí una esquina de párrafo y dónde?** — una cabeza por
tipo (`TL, TR, BR, BL`). En inferencia, una ventana deslizante recompone los párrafos.

Las imágenes las produce
[image-text-sample-generator](../image-text-sample-generator) (ver su `SAMPLE_FORMAT.md`).

---

## Estado: el diseño está cerrado; el código, por escribir

**Este repo contiene hoy el diseño y los tests, no la implementación.** El código anterior se
borró para reconstruirlo desde cero siguiendo `docs/`; sigue recuperable en el tag
**`pre-rediseno`**:

```powershell
git show pre-rediseno:src/itf/patches/extract.py     # un fichero
git checkout pre-rediseno -- src/                    # todo el paquete
```

**No hay `src/` ni web app**, así que no hay app que arrancar. Las fases de
[docs/plan-ui.md](docs/plan-ui.md) las van creando, y esta sección se rellena —verificando cada
comando— según lleguen.

### Lo que sí se puede correr

La suite de contratos, que es la **barra de progreso del plan**: un test por contrato de
`docs/organizacion.md` §2, todos en `xfail(strict=True)` mientras no exista su implementación.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Salida esperada hoy: **`13 xfailed`**, en verde y en ~3,5 s. Que estén en xfail no es deuda: es
el mecanismo. Cuando una fase implementa su contrato, el test pasa, el **XPASS estricto pone la
suite en rojo** y obliga a quitar el marcador — así "lo que falta" es una lista ejecutable en vez
de prosa que envejece. Ver [docs/tests.md](docs/tests.md) §2.

### El entorno

- **Python 3.12** (PyTorch aún no tiene wheels para 3.14). Verificado con **3.12.10**.
- El intérprete del proyecto es `.\.venv\Scripts\python.exe`, y **ya está montado**: trae torch
  2.13.0+cpu, FastAPI y pytest. Con él, `pytest -q` funciona (arriba).

> **Caveat: hoy el proyecto no se puede instalar, y es esperable.** `pip install -e .` **falla**
> mientras no exista `src/`:
>
> ```
> error: error in 'egg_base' option: 'src' does not exist or is not a directory
> ```
>
> `pyproject.toml` declara `[tool.setuptools.packages.find] where = ["src"]`, y ese directorio se
> borró con el código (D18). No es tu entorno: **se arregla solo en la fase 2**, que es la que
> crea `src/itf/`. Hasta entonces no hay nada que importar de todos modos.
>
> Ojo con lo que dice `pip list`: sigue mostrando `image-text-finder 0.1.0` en editable, pero es
> un **puntero colgado** del install anterior al borrado — `python -c "import itf"` da
> `ModuleNotFoundError`. Los tests de contrato **cuentan con ello**: es justo por lo que fallan, y
> por lo que están en xfail.

`pyproject.toml` declara también los scripts `itf-extract` / `itf-train` / `itf-api`, que
**apuntan a módulos que aún no existen**.

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
web/             # Vite + React
configs/         # models/*.yaml (redes)  ·  recipes/*.yaml (recetas)
tests/           # test_contracts.py: un test por contrato — ya existe
docs/            # el diseño
```

`data/` y `runs/` son artefactos: se ignora la carga (`.npz`, `.pt`) y **se versiona la
descripción** (configs, métricas, manifests) — ver [docs/formatos.md](docs/formatos.md) §5.
