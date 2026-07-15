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

```bash
python -m venv .venv                     # Python 3.12 recommended (torch wheels)
./.venv/Scripts/python -m pip install -e ".[train,api,dev]"
```

## CLI usage

```bash
# 1. Build a patch dataset
itf-extract --config configs/extract.example.yaml

# 2. Train a model
itf-train --config configs/model.example.yaml \
          --data data/patch-datasets/reducido-40 --out runs/cnn-a

# 3. Serve the API (docs at http://127.0.0.1:8000/docs)
itf-api --port 8000
```

## Web app

```bash
cd web
npm install
npm run dev        # http://localhost:5173  (proxies /api -> http://127.0.0.1:8000)
```

Set `ITF_API_URL` before `npm run dev` if the backend runs elsewhere.

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
