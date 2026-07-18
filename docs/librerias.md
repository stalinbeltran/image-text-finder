# Librerías reutilizables

Qué se extrae de este proyecto para poder reaprovecharlo en proyectos futuros, qué **no**, y
qué obligaciones tiene crear una librería.

---

## 0. El criterio: se extrae en la segunda vez, no en la primera

Una librería sacada de un solo caso de uso no es reutilizable: es un caso de uso con
`pyproject.toml`. La generalidad no se adivina, se **observa**.

**Aquí ya la hemos observado**, y esa es toda la base de este documento: existen **dos
proyectos hermanos** —`image-text-finder` y `sliding-window-NIST-ocr`— que resolvieron los
mismos problemas **por separado y sin coordinarse**. Donde ambos construyeron la misma pieza,
el requisito está probado. Y como podemos ver las dos implementaciones a la vez, sabemos
también **dónde se equivocó cada una**: la librería es la unión de lo que cada uno acertó, no
una copia del que llegó primero.

Nada de lo que existe en **un solo** proyecto entra aquí, por evidente que parezca su
generalidad. Cuando un segundo proyecto lo pida, se extrae.

### La prueba de frontera

> **La librería posee el mecanismo. El proyecto posee el significado.**

- La librería sabe que *un run es un directorio con config congelada, métricas y checkpoints*.
  **No sabe** qué hay dentro de la config.
- La librería sabe *construir bloques conv desde un dict y mirar sus pesos*. **No sabe** qué
  significa la salida ni qué cabeza va encima.
- La librería sabe *pintar una matriz como mapa de calor*. **No sabe** si es un kernel o una
  activación — y por tanto **no elige el color**: eso lo decide quien conoce el dato.

Si para escribir la librería necesitas nombrar `TL`, `paragraph` o `MNIST`, la frontera está
mal puesta.

---

## 1. Las cuatro librerías

| Nombre *(provisional)* | Qué | ITF | NIST |
|---|---|---|---|
| **`exp-registry`** | Registro de experimentos reproducibles | `runs/` (flojo) | `experiments/` + `repro.py` (completo) |
| **`jobq`** | Cola de trabajos en proceso, con cancelación | `api/jobs.py` (sin límite ni stop) | `webapp/manager.py` (con stop) |
| **`convspec`** | Backbone conv declarativo + introspección | `models/builder.py` | `models/features_cnn.py` |
| **`matrixview`** | Matriz de números → mapa de calor + tabla | *especificado, sin construir* | `inference.py` + `app.js` |

### `exp-registry` — registro de experimentos reproducibles

**Posee**: la estructura de un run en disco y su ciclo de vida.

- Directorio por run: config **congelada**, `environment.json`, `status.json`, `metrics.jsonl`
  (append + lectura incremental para polling en vivo), checkpoints.
- **Estado explícito** (`created | running | completed | failed | stopped`), no deducido de qué
  ficheros existen.
- Captura de entorno y semillas: versiones de Python/torch/numpy, plataforma, **commit de git**,
  disponibilidad de CUDA; `set_seed`, `seeded_generator`.
- CRUD con **comprobación de referencias**: no borrar lo que otro usa; renombrar como alias
  (el id nunca cambia, porque otros lo referencian).

**No posee**: qué va dentro de la config, qué métricas se registran, qué es un checkpoint por
dentro. Todo eso son diccionarios opacos.

**Qué gana la unión**: NIST lo tiene casi entero y bien (estado explícito, `environment.json`,
reproducibilidad verificada por un test: *misma semilla + misma config ⇒ mismos pesos*). ITF
tiene los tres bugs que la librería debe hacer imposibles:

- **estado inferido del disco** (`_run_status` mira qué ficheros hay ⇒ un crash queda "running"
  para siempre);
- **sobrescritura silenciosa** (`POST /runs` no comprueba si el nombre existe);
- **sin captura de entorno**: `loop.py` hace `torch.manual_seed` y poco más, así que un run no
  se puede reproducir desde cero.

### `jobq` — cola de trabajos en proceso

**Posee**: lanzar trabajo en segundo plano dentro del proceso, y poder pararlo.

- **Límite de workers** (en CPU es 1: torch ya usa todos los núcleos, así que N entrenamientos
  a la vez se pelean y cada uno va ~N× más lento, además de multiplicar la RAM).
- **Cancelación cooperativa**: `Future.cancel()` no sirve —solo cancela lo que aún no arrancó—,
  así que la función de trabajo recibe un `stop_event` y lo consulta por época. **NIST ya lo
  hace así**; es el diseño correcto y hay que conservarlo.
- **Persistencia**: el estado sobrevive a un reinicio. Hoy ninguno de los dos lo hace: son
  dicts en memoria con hilos daemon.
- Metadatos, historial y polling.

**No posee**: qué es el trabajo. Recibe un callable.

**Honestidad sobre qué aporta**: `ThreadPoolExecutor(max_workers=1)` ya da el límite de
workers. Lo que no da —y es el 80% del valor— es la **cancelación cooperativa**, la
**persistencia** y los **metadatos**. Y lo de fuera (Celery, RQ) pide Redis: absurdo para una
herramienta local de un solo usuario. Son ~150 líneas que valen la pena.

### `convspec` — backbone conv declarativo + introspección

**Posee**: `dict de config → bloques conv`, y mirar dentro.

- Un bloque = `Conv2d` + [BatchNorm] + activación + [pool] + [dropout]. **El conv siempre el
  primero**, y eso no es un detalle de implementación: es *el contrato* que hace posible la
  introspección.
- **Traza espacial con validación previa**: si un conv o un pooling no cabe en el mapa que le
  llega, se rechaza **antes de entrenar**, diciendo **qué capa falla, con qué tamaño y cómo
  arreglarlo**. Esto es de NIST y es lo mejor que tiene.
- **Introspección**: `feature_maps(x)` (reaplicar los bloques guardando cada salida) y
  `kernels()` (`block[0].weight`). Sin hooks.

**No posee**: **la cabeza**. `CornerHead` (4×3) y el clasificador de 10 dígitos son de sus
proyectos. La librería entrega el backbone y la dimensión aplanada; encima pones lo tuyo.

**Qué gana la unión**: ITF aporta `batchnorm` y `dropout2d`; NIST aporta la **traza espacial
con razones claras** (que ITF no tiene: allí una arquitectura imposible revienta con un
`mat1 and mat2 shapes cannot be multiplied` dentro del hilo del job) y la introspección.

### `matrixview` — matriz de números → mapa de calor + tabla

**Dos paquetes, un contrato**: `matrixview` (Python, serializa) y `@matrixview/canvas` (TS,
pinta). Lo valioso **no son las líneas de código** —el serializador son 8 y el pintor 15—, sino
**las decisiones que llevan dentro**, que si no se re-deducen mal cada vez. La prueba: NIST
las re-dedujo y **falló una**.

Lo que codifica:

- **Se mandan números, no imágenes.** El backend serializa `matrix` + `min`/`max`/`mean`; el
  color lo decide el cliente.
- **Normalización por mapa**, no global: sin esto, los mapas de activación baja se ven todos
  negros.
- **Truncado con aviso** (`max_maps`, `truncated: true`): 128 filtros × una matriz en JSON
  revientan el navegador.
- **La tabla de números es el gemelo accesible** del mapa de calor, no un extra: un heatmap
  codifica **solo con color**.
- **El pintor no elige el color**: recibe el trabajo (`sequential | diverging`) de quien conoce
  el dato. Porque **NIST se equivocó justo aquí**: pintó kernels —que tienen signo— con una
  rampa `min→max`, así que el cero caía en cualquier sitio y la estructura de signo (qué excita,
  qué inhibe: *lo que un kernel es*) quedaba invisible. Los datos con signo van **divergentes
  centrados en 0**; la magnitud va secuencial. La librería no puede saber cuál es: lo pide.

---

## 2. Lo que NO se extrae, y por qué

Tan importante como la lista de arriba. Cada una parece reutilizable y no lo es:

| Candidato | Por qué no |
|---|---|
| **Codificador PNG sin dependencias** (NIST) | Resuelve un problema que ITF **no tiene**: ITF ya usa PIL. Era una restricción de NIST ("sin CDNs ni deps"), no una necesidad general |
| **Registro/catálogo de datasets** | Los modelos de dominio **divergen de verdad**: NIST tiene subconjuntos custom y muestras editadas; ITF tiene una derivación fuente→patches. Lo común (escanear un directorio) es trivial |
| **Las cabezas** (`CornerHead`, clasificador) | Son el significado, no el mecanismo. Frontera de `convspec` |
| **El shell de la UI** | Stacks distintos: JS vanilla vs React. Lo único común es `drawMap`, y ya está en `matrixview` |
| **El editor de dígitos** (NIST) | Específico de su dominio: MNIST *es* un dígito dibujado a mano. En ITF un patch es texto renderizado, y pintarlo a mano mide el artefacto, no el modelo |
| **`imageops`** (resize proporcional) | **Un solo proyecto lo tiene** *(D19, 2026-07-18)*. Es el candidato más tentador de la lista —redimensionar manteniendo proporción es universal y son 20 líneas— y por eso mismo es donde §0 se pone a prueba: no hay segunda implementación que observar, así que no sabemos qué generalidad hace falta. Se escribe **library-shaped** (`itf/imageops.py`, sin un import de `itf`, con sus tests) y se extrae cuando un segundo proyecto lo pida. Nota para entonces: lo valioso no es el `img.resize()` — es **que devuelva la escala real medida de la salida**, que es lo que impide que la geometría se desplace por redondeo |
| **El barrido (H)** | **Un solo proyecto lo necesita.** Es obviamente general, y por eso mismo es la trampa: se escribe pensando en ITF y se descubre en el proyecto 3 que no encajaba. Se escribe **en forma de librería** (módulo propio, sin importar nada de `itf`) y se extrae cuando un segundo proyecto lo pida |
| **Validación con razones** (`validation.py`) | Es una **convención**, no una librería: "toda restricción se valida antes, y el error dice el porqué y cómo arreglarlo". Va en el CLAUDE.md de cada proyecto, no en un paquete |

---

## 3. Obligación: cada librería nace con su documento para Claude

**Una librería sin su documento no está terminada.** No es documentación de cortesía: es lo que
permite que un Claude que nunca vio el proyecto la use bien y —sobre todo— **no la "arregle"
rompiendo lo que la hace valer**.

`<lib>/CLAUDE.md`, con estas siete secciones. Las tres primeras son las que de verdad importan:

1. **Qué posee y qué NO.** La frontera, explícita. Dónde acaba la librería y empieza el
   proyecto. Con ejemplos de lo que *no* debe entrar (`convspec` no sabe qué es una cabeza).
2. **Las decisiones que lleva dentro, y por qué.** La sección más importante y la que nunca se
   escribe. Cada decisión no obvia, con su razón y **su consecuencia si se revierte**:
   - *"La normalización es **por mapa**. Global ⇒ los mapas de activación baja se ven negros."*
   - *"El límite de workers es 1 en CPU. Subirlo no acelera: torch ya usa todos los núcleos."*
   - *"El `Conv2d` va **primero** en cada bloque. Es lo que hace posible `kernels()`."*
   
   Sin esto, el siguiente que pase "simplifica" la normalización a global y rompe la librería
   sin que nadie lo note.
3. **Qué NO hace, deliberadamente, y qué usar en su lugar.** Evita que se le pidan cosas que no
   son suyas, y que crezca hasta dejar de ser reutilizable.
4. **La API**, con firmas reales y un ejemplo mínimo **que funciona** (copiado de un consumidor
   real, no inventado).
5. **Cómo se enchufa a un proyecto nuevo**: instalación, wiring, y el comando exacto verificado.
6. **Modos de fallo y trampas**: qué falla, con qué mensaje, y qué significa.
7. **Versión y compatibilidad**: qué rompe a los consumidores y cómo cambiarlo sin romperlos.

Requisitos que valen para todas:

- **Tests propios**, independientes de cualquier proyecto. Si un test necesita `itf`, la
  frontera está mal.
- **Ni un import de un proyecto consumidor.** Es la comprobación mecánica de la frontera.
- El README del consumidor documenta cómo instalarla, **con el comando ejecutado y verificado**
  (regla global: nunca dar por buena una instrucción sin correrla).

---

## 4. Dónde viven y cómo se consumen

**Propuesta**: un repo `C:\Desarrollo\claude-libs\`, un paquete por subdirectorio, hermano de
los proyectos que ya están en `C:\Desarrollo\`. Cuatro repos separados es más ceremonia que
valor a esta escala.

```
C:\Desarrollo\
├── claude-libs\
│   ├── exp-registry\   (+ CLAUDE.md, tests)
│   ├── jobq\
│   ├── convspec\
│   └── matrixview\
├── image-text-finder\
└── sliding-window-NIST-ocr\
```

Consumo: `.\.venv\Scripts\python -m pip install -e ..\claude-libs\exp-registry`, en el paso de
setup del README del consumidor (no como dependencia con ruta absoluta en `pyproject.toml`: eso
se rompe en cuanto el repo cambia de sitio o de máquina).

**El peligro del editable install, y hay que decirlo alto**: con `-e`, tocar la librería cambia
**a la vez** todos los proyectos que la usan, en silencio y sin avisar. Es cómodo mientras se
desarrolla y es una bomba cuando hay resultados que defender. Por eso: **tests propios** en cada
librería, y **fijar por tag de git** en cuanto la librería se estabilice.

**No se backportea NIST.** Funciona, y reescribirlo para que consuma las librerías es riesgo
sin retorno. Su valor ya está cobrado: es la **evidencia** que sostiene el diseño. Que nunca las
importe no debilita nada — el requisito ya está probado por dos implementaciones
independientes.

---

## 5. Cuándo se extrae cada una

Ninguna extracción es una fase propia: **cada librería sale de la fase de
[plan-ui.md](plan-ui.md) donde ese código se toca de todas formas.** Extraer y reescribir a la
vez es hacer dos cosas difíciles a la vez; extraer *lo que acabas de dejar bien* es casi gratis.

| Librería | Sale en | Por qué ahí |
|---|---|---|
| **`convspec`** | fase 3 | Ahí se parte `ModelConfigForm` y se tocan los hiperparámetros: el modelo está abierto |
| **`exp-registry`** | fase 4 | Ahí se arreglan procedencia, sobrescritura silenciosa y estado del run: es exactamente su contenido |
| **`matrixview`** | fase 6 | Ahí se construyen V1 y V2, sus dos primeros consumidores |
| **`jobq`** | fase 7 | Ahí se reescribe la cola para el barrido: límite, persistencia y cancelación |

Regla de secuencia: **primero funcionando en el proyecto, después extraído.** Con el módulo ya
aislado (sin importar nada de `itf`) y con su CLAUDE.md escrito, la extracción es un `git mv`
más un `pyproject.toml`, no un rediseño.
