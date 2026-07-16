# Decisiones abiertas

Lo que está **sin decidir** y bloquea algo. Índice, no archivo histórico.

**Por qué existe**: al escribir las specs se acumularon decisiones repartidas por seis
documentos. Nadie —ni tú ni un Claude que llegue en tres meses— puede ver de un vistazo qué
falta por elegir. Una decisión que no se ve **se acaba tomando sola, por defecto y sin pensar**;
así nacieron las tres que arrastramos: el CORS abierto, los 20 imágenes de val y `/runs/`
gitignoreado.

**Esto no es un ADR.** Los documentos ya cargan el *por qué* de cada decisión tomada, que es el
90 % del valor de un ADR. Aquí solo vive lo **pendiente**.

**Ciclo de vida**: al decidirse, la decisión **se escribe en el documento que le corresponde** y
aquí queda una línea en §3 apuntando allí. Este fichero se mantiene corto o deja de leerse.

---

## 1. Bloquean la siguiente fase

### D1 — ¿La tabla por patch es entidad guardada o caché?

**En juego**: es E×B con identidad propia, así que por la regla 1 de ui.md pediría pantalla
propia. El proyecto hermano la tenía como entidad (`evaluations/<id>/`).
**Opciones**: entidad con nombre y CRUD · caché invalidable por huella.
**Recomiendo**: entidad. Es lo que permite comparar dos evaluaciones y guardar el filtro.
**Bloquea**: fase 5. **Si es entidad, va a organizacion.md antes de implementarse** — lo exige
CLAUDE.md.
**Dónde**: plan-ui.md fase 0.1, ui.md §3.

### D2 — La forma exacta de la procedencia en el run

**En juego**: contrato ③. Nombres de campos que escribe la fase 4 y lee todo lo demás.
**Propuesta**: `provenance: {patch_dataset:{name,fingerprint}, network:{name,value}, recipe:{name,value}, sweep}`.
**Bloquea**: fase 4, y el barrido entero (sin esto no se puede agrupar por red ni por receta).
**Dónde**: api.md §3 (`/runs`), formatos.md §4.2.

### D3 — Los `config.json` viejos: ¿migrar o leer degradando?

**En juego**: los cinco runs de `runs/` llevan `device` dentro y no tienen procedencia.
**Opciones**: migrar con un script · leer con retrocompatibilidad (`name: null`).
**Recomiendo**: leer degradando. Migrar reescribe un registro histórico, que es justo lo que no
se debe tocar.
**Bloquea**: fase 4. Es **la trampa más probable del plan** y tiene test propio.
**Dónde**: plan-ui.md fase 0.3 y §3, formatos.md §4.2.

---

## 2. Deciden solas si no se miran

### D4 — Rutas arbitrarias + CORS abierto

**En juego**: `/image?path=` y `/folder?path=` leen **cualquier imagen del disco**, y CORS está
en `allow_origins=["*"]`. Cualquier página que visites mientras el API corre puede enumerar y
leer imágenes de tu disco.
**Opciones**: raíces permitidas (allowlist) · solo upload · dejarlo y cerrar CORS.
**Recomiendo**: allowlist **+** CORS cerrado. **La GPU lo agrava**: en cuanto el API viva en red,
deja de ser modesto.
**Urgencia**: media hoy, alta el día de la GPU. **Dónde**: api.md §6.

### D5 — ¿Se versiona el registro de investigación?

**En juego**: `.gitignore` excluye `/runs/` entero ⇒ `git ls-files runs` está **vacío**. Los
configs y las métricas de tus cinco runs no tienen historia y están a un `rm -rf` de
desaparecer.
**Opciones**: versionar la descripción e ignorar la carga (patrón del proyecto hermano) · dejarlo.
**Recomiendo**: versionar descripción (KB), ignorar `.npz` y `.pt` (MB).
**Contra**: cientos de runs de barrido ensucian el historial (mitigable ignorando `sweeps/`), y
un `metrics.jsonl` vivo hace ruido en `git status`.
**Urgencia**: **sube con cada run que corres**. **Dónde**: formatos.md §5.

### D6 — Cuántas imágenes generar, y el punto train↔val

**En juego**: val = 20 imágenes hoy; el ruido tapa las diferencias que busca el barrido. Train
manda en el coste, val en la resolución, y **son knobs independientes**.
**Propuesta**: ~2000 imágenes, 80/10/10 → train 1600, val 200. Barrido ~22 h en bruto, una noche
con poda.
**Nota**: el paso 1 del protocolo **mide** el suelo real y dice si hiciste bastante. Es un bucle.
**Bloquea**: todo el protocolo, y por tanto que el barrido signifique algo. **Dónde**: protocolo.md §3.

---

## 3. Pueden esperar

| | Decisión | Recomiendo | Dónde |
|---|---|---|---|
| **D7** | ¿La métrica de párrafo soporta rotación, o compara contra el *bbox* del `quad`? | Bbox: basta con `clear-paragraphs` (`angle≈0`). Con `mixed-layout`, no | protocolo.md §2 |
| **D8** | ¿Añadir `limit` de train a `PatchExtractConfig`? | Solo si hace falta la tarea proxy (rankear con train pequeño). El hermano lo tiene | protocolo.md §3 |
| **D9** | Nombres de las librerías (`exp-registry`, `jobq`, `convspec`, `matrixview`) | Provisionales; decidir al extraer la primera | librerias.md §1 |
| **D10** | ¿Monorepo `claude-libs` o cuatro repos? | Monorepo: cuatro repos es más ceremonia que valor a esta escala | librerias.md §4 |
| **D11** | ¿Backportear NIST a las librerías? | **No.** Funciona; su valor ya está cobrado como evidencia | librerias.md §4 |
| **D12** | La paleta concreta (los hex) | Al construir la fase 1, **pasada por el validador** en claro y oscuro | ui.md §0 |
| **D13** | Kernels profundos: ¿corte del canal 0, todos los canales, o nada? | Nada: de la capa 2 en adelante la información está en los feature maps | ui.md §4.1 |
| **D14** | ¿Editor de patches? | No: V4 (occlusion) y V5 (scrubber) lo cubren mejor y en distribución | ui.md §5 |
| **D15** | Big-bang o franjas verticales en el rediseño | Franjas: entrenar tarda horas y necesitas la herramienta mientras | plan-ui.md §1 |

---

## 4. Ya decididas

*(Vacío. Al cerrar una, borra su entrada de arriba y deja aquí una línea con el documento donde
vive ahora y el commit.)*
