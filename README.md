# image-text-finder

Patch-based **paragraph-corner detection** over datasets produced by
[image-text-sample-generator](../image-text-sample-generator) (see its
`SAMPLE_FORMAT.md`). The pipeline slices each image into `n×n` patches and
trains a configurable CNN to answer, for every patch: **does a paragraph
corner fall here, and if so, where?** — one head per corner type
(`TL, TR, BR, BL`). At inference a sliding window reassembles corners into
paragraph boxes.

## Components

1. **Patch extractor** (`itf.patches`) — turns a source dataset into a packed
   `.npz` of `n×n` patches, each labelled `(4, 3)` = `[exists, x, y]` per corner.
2. **Configurable CNN** (`itf.models`, `itf.training`) — architecture and
   hyperparameters are pure config (YAML); loss is
   `BCE(exists) + λ·exists·smoothL1(xy)`.
3. **API** (`itf.api`) — FastAPI over the whole flow: datasets, patch builds,
   model configs, training runs (background jobs + live metrics), inference.
4. **Web app** (`web/`) — Vite + React UI to configure extraction, design the
   network, launch training, watch live curves, and preview inference.

## Setup

PyTorch has no wheels for Python 3.14 yet, so create the venv with **Python
3.12**. On Windows use the `py` launcher; the commands below are for PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[train,api,dev]"
```

This installs the `itf-extract`, `itf-train`, and `itf-api` console scripts into
`.venv\Scripts\`. Either activate the venv (`.\.venv\Scripts\Activate.ps1`) so
they're on your PATH, or call them with the `.\.venv\Scripts\` prefix as shown
below.

## CLI usage

Run these from the **repo root** (where `.venv` lives) — the `.\.venv\Scripts\`
prefix is relative, so it fails from other folders such as `web\`.

```powershell
# 1. Build a patch dataset (writes data/patch-datasets/reducido-40/patches.npz)
.\.venv\Scripts\itf-extract --config configs/extract.example.yaml

# 2. Train a model (writes checkpoints + metrics to runs/cnn-a/)
.\.venv\Scripts\itf-train --config configs/model.example.yaml `
          --data data/patch-datasets/reducido-40 --out runs/cnn-a

# 3. Serve the API (interactive docs at http://127.0.0.1:8000/docs)
.\.venv\Scripts\itf-api --port 8000
```

> The example `configs/extract.example.yaml` points `source` at a dataset under
> `image-text-sample-generator`. Edit that path (or pass `--source <dir> --out <dir>`)
> to build from your own datasets.

## Web app

Requires Node.js (tested with Node 24 / npm 11).

```powershell
cd web
npm install
npm run dev        # http://localhost:5173  (proxies /api -> http://127.0.0.1:8000)
```

The dev server proxies `/api/*` to the backend. If the backend is not on
`http://127.0.0.1:8000`, point the proxy at it before starting Vite:

```powershell
$env:ITF_API_URL = "http://127.0.0.1:8010"; npm run dev
```

**Port note:** if `8000` is already in use on your machine, run the API on
another port (`itf-api --port 8010`) and set `ITF_API_URL` accordingly. Verify
the whole toolchain any time (back in the repo root, not `web\`) with
`.\.venv\Scripts\python -m pytest -q`.

### Predict panel — batch inference

The **Predict** tab evaluates many images at once instead of one upload. Pick a
trained run (it auto-selects the source dataset and split it was trained on),
then choose an image source:

- **Dataset / subset** — browse a source dataset as a thumbnail grid, filtered
  by split (`all / train / val / test`, taken from the run's patch dataset).
- **Folder** — point at any folder of images on the server's disk.
- **Single upload** — the original one-off upload flow.

Click a thumbnail to run inference and see the corner/paragraph overlay in a
popup (close with ✕, click outside, or Esc); each tile shows how many
paragraphs were reconstructed. **Predict all** runs the whole visible set
(loaded models are cached, so batches stay fast).

### Runs panel — manage trained models

The **Runs** tab lists trained models with per-row actions: **rename** (✎),
**delete** (🗑), and **retrain** (↻). The retrain view loads a run's frozen
config and lets you train the **same network** again: the parameters that
define the network — input size, the full conv backbone, and the head — are
shown but **fixed**, while everything you *can* change is editable: the **patch
dataset** and every **training hyperparameter** (epochs, batch size, lr,
optimizer, λ, device). Datasets whose patch size doesn't match the network's
input size are flagged incompatible and block the run. The original run is left
untouched. Active runs are protected — rename/delete are disabled while a run
is still training.

The parameter form lives in a single reusable component (`web/src/components/
ModelConfigForm.tsx`), shared by the Train panel (fully editable) and this
retrain view (`lockArchitecture`).

New backend endpoints backing these: `GET /datasets/{id}/samples`,
`GET /image`, `GET /folder`, `GET /runs/{name}/source`, `POST /predict-path`,
`PATCH`/`DELETE /runs/{name}`, and `POST /runs/{name}/retrain`.

## Layout

```
src/itf/
├── datasets/    # read labels.jsonl (SAMPLE_FORMAT)
├── patches/     # n×n extraction -> .npz  +  torch Dataset
├── models/      # config -> CNN + corner head
├── training/    # losses, train loop, checkpoints, metrics
├── inference/   # sliding-window detection + box reconstruction
└── api/         # FastAPI app, jobs, settings
web/             # Vite + React front-end
configs/         # example extract/model YAMLs
tests/           # pytest (extract, model, end-to-end API)
```

## Notes

- Coordinates follow SAMPLE_FORMAT: origin top-left, `quad` clockwise from TL.
- The extractor reads already-pixelated images as-is (text is never recognized).
- `data/` and `runs/` are gitignored build artifacts.
