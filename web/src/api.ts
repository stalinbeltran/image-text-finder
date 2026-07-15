// Thin client over the FastAPI backend. In dev, requests go to /api/* which
// Vite proxies to the backend (see vite.config.ts).

const BASE = "/api";

export interface Dataset {
  id: string;
  name: string;
  count: number | null;
  path: string;
}

export interface PatchManifest {
  num_samples: number;
  num_patches: number;
  patch_shape: number[];
  patches_per_split: Record<string, number>;
  positives_per_corner: Record<string, number>;
  config: Record<string, unknown>;
}

export interface PatchDataset {
  name: string;
  manifest: PatchManifest | null;
}

export interface Job {
  id: string;
  kind: string;
  status: "pending" | "running" | "done" | "error";
  error: string | null;
  result: unknown;
  meta: Record<string, unknown>;
}

export interface EpochMetrics {
  epoch: number;
  train_loss: number;
  seconds: number;
  val: {
    loss?: number;
    exists_acc?: number;
    precision?: number;
    recall?: number;
    f1?: number;
    pos_err_px?: number | null;
  };
}

export interface RunSummary {
  name: string;
  status: string;
  epochs_done: number;
  last: EpochMetrics | null;
}

export interface RunDetail {
  name: string;
  status: string;
  config: Record<string, unknown> | null;
  metrics: EpochMetrics[];
  summary: Record<string, unknown> | null;
  checkpoints: string[];
}

export interface Corner {
  corner: string;
  x: number;
  y: number;
  score: number;
}

export interface PredictResult {
  corners: Corner[];
  paragraphs: { box: number[]; score: number }[];
  image_size: [number, number];
  run: string;
  checkpoint: string;
  path?: string;
}

export interface SampleItem {
  index: number;
  name: string;
  path: string;
  width: number;
  height: number;
  num_paragraphs: number;
  split?: "train" | "val" | "test" | null;
}

export interface DatasetSamples {
  dataset: string;
  count: number;
  splits: Record<string, number> | null;
  samples: SampleItem[];
}

export interface FolderItem {
  name: string;
  path: string;
}

export interface FolderListing {
  path: string;
  count: number;
  images: FolderItem[];
}

export interface RunSource {
  run: string;
  patch_dataset: string | null;
  source: string | null;
}

export interface PredictPathBody {
  path: string;
  run: string;
  checkpoint?: string;
  threshold?: number;
  stride?: number | null;
  device?: string;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

const jsonBody = (method: string) => (body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
const jsonPost = jsonBody("POST");
const jsonPatch = jsonBody("PATCH");

// URL to (optionally downscaled) image bytes served straight from disk.
export const imageUrl = (path: string, w?: number) =>
  `${BASE}/image?path=${encodeURIComponent(path)}${w ? `&w=${w}` : ""}`;

export const api = {
  datasets: () => req<Dataset[]>("/datasets"),
  datasetSamples: (id: string, patchDataset?: string) =>
    req<DatasetSamples>(
      `/datasets/${id}/samples${patchDataset ? `?patch_dataset=${encodeURIComponent(patchDataset)}` : ""}`,
    ),
  folder: (path: string) => req<FolderListing>(`/folder?path=${encodeURIComponent(path)}`),
  patchDatasets: () => req<PatchDataset[]>("/patch-datasets"),
  buildPatches: (body: unknown) => req<Job>("/patch-datasets", jsonPost(body)),
  runs: () => req<RunSummary[]>("/runs"),
  run: (name: string) => req<RunDetail>(`/runs/${name}`),
  runSource: (name: string) => req<RunSource>(`/runs/${name}/source`),
  startRun: (body: unknown) => req<Job>("/runs", jsonPost(body)),
  renameRun: (name: string, newName: string) =>
    req<{ renamed: string; to: string }>(`/runs/${name}`, jsonPatch({ new_name: newName })),
  deleteRun: (name: string, force = false) =>
    req<{ deleted: string }>(`/runs/${name}${force ? "?force=true" : ""}`, { method: "DELETE" }),
  retrainRun: (name: string, body: unknown) =>
    req<Job>(`/runs/${name}/retrain`, jsonPost(body)),
  job: (id: string) => req<Job>(`/jobs/${id}`),
  predict: (form: FormData) =>
    req<PredictResult>("/predict", { method: "POST", body: form }),
  predictPath: (body: PredictPathBody) =>
    req<PredictResult>("/predict-path", jsonPost(body)),
};
