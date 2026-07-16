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

The **Runs** tab lists trained models with per-row actions: **retrain same**
(↻), **new network from config** (＋), **rename** (✎), and **delete** (🗑).
Both training actions load the run's frozen config; a toggle in the panel
switches between the two modes:

- **Retrain the same network** — the parameters that define the network (input
  size, border features, conv backbone, head) are shown but **fixed**; you change only the
  **patch dataset** and the **training hyperparameters** (epochs, batch size,
  lr, optimizer, λ, device).
- **New network from this config** — the same config is a starting point with
  **everything editable** (architecture, head, dataset, hyperparameters), to
  spin off a brand-new network.

In both cases the original run is left untouched, and datasets whose patch size
doesn't match the network's input size are flagged incompatible and block the
run. Active runs are protected — rename/delete are disabled while a run is
still training.

The parameter form lives in a single reusable component (`web/src/components/
ModelConfigForm.tsx`), shared by the Train panel and both retrain modes (via
its `lockArchitecture` prop).

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

- **Border features** — each patch also carries 4 flags marking which source-image
  edges it sits flush against (`top, right, bottom, left`). This context is lost
  when a patch is cropped, yet it constrains where a paragraph corner can be when
  the paragraph starts near the image edge. The extractor always stores them in
  `patches.npz` (`border`, shape `(N, 4)`); the model uses them only when
  `model.border_features: true`, concatenating the flags to the flattened conv
  features before the head. Patch datasets built before this feature backfill the
  flags with zeros, so old `.npz` files still load.
- Coordinates follow SAMPLE_FORMAT: origin top-left, `quad` clockwise from TL.
- The extractor reads already-pixelated images as-is (text is never recognized).
- `data/` and `runs/` are gitignored build artifacts.
