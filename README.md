# image-text-finder

Detección de esquinas de párrafo por patches: se trocea cada imagen en patches `n×n` y una CNN
configurable responde, por patch, **¿cae aquí una esquina de párrafo y dónde?** — una cabeza por
tipo (`TL, TR, BR, BL`). En inferencia, una ventana deslizante recompone los párrafos.

Las imágenes las produce
[image-text-sample-generator](../image-text-sample-generator) (ver su `SAMPLE_FORMAT.md`).

---

## Estado: el código está por escribir

**Este repo contiene hoy el diseño, no la implementación.** El código anterior se borró para
reconstruirlo desde cero siguiendo `docs/`; sigue recuperable en el tag **`pre-rediseno`**:

```powershell
git show pre-rediseno:src/itf/patches/extract.py     # un fichero
git checkout pre-rediseno -- src/                    # todo el paquete
```

**No hay nada que instalar ni que correr todavía**: no hay `src/`, ni `tests/`, ni web app. Esta
sección se rellena —y se verifica ejecutando cada comando— a medida que las fases de
[docs/plan-ui.md](docs/plan-ui.md) los vayan creando.

Lo único vigente del entorno:

- **Python 3.12** (PyTorch aún no tiene wheels para 3.14). En Windows, `py -3.12 -m venv .venv`.
- El intérprete del proyecto es `.\.venv\Scripts\python.exe`.
- `pyproject.toml` declara el paquete `itf` y los scripts `itf-extract` / `itf-train` /
  `itf-api`, que **apuntan a módulos que aún no existen**.

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
docs/            # el diseño
```

`data/` y `runs/` son artefactos: se ignora la carga (`.npz`, `.pt`) y **se versiona la
descripción** (configs, métricas, manifests) — ver [docs/formatos.md](docs/formatos.md) §5.
