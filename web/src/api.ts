/** The HTTP client. One function per endpoint, no logic.
 *
 * The important part is `request`: it turns the API's error shape (api.md R4)
 * back into an object with `code`, `message` and `hint`, instead of flattening it
 * into a string. The hint is the half that says how to fix it, and
 * `throw new Error(await res.text())` throws it away.
 */

import type { ApiProblem } from "./components/Async";

const BASE = "/api";

export class ApiError extends Error {
  problem: ApiProblem;
  status: number;
  constructor(status: number, problem: ApiProblem) {
    super(problem.message);
    this.status = status;
    this.problem = problem;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: init?.body ? { "Content-Type": "application/json", ...init?.headers } : init?.headers,
  });
  if (!res.ok) {
    let detail: unknown;
    try {
      detail = (await res.json())?.detail;
    } catch {
      detail = undefined;
    }
    // FastAPI's own validation errors are a list, not our shape -- a body that
    // is not R4-shaped must still arrive as something the UI can render.
    const problem: ApiProblem =
      detail && typeof detail === "object" && "code" in (detail as object)
        ? (detail as ApiProblem)
        : { code: `http_${res.status}`, message: `${res.status} ${res.statusText}` };
    throw new ApiError(res.status, problem);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

// ── A: sources ───────────────────────────────────────────────────────────────

export interface Source {
  id: string;
  source_id: string;
  num_samples: number;
  num_overlapping: number;
}

export interface SampleInfo {
  index: number;
  width: number;
  height: number;
  has_overlap: boolean;
  num_blocks: number;
  split: string | null;
}

export interface Geometry {
  index: number;
  width: number;
  height: number;
  has_overlap: boolean;
  blocks: { block_id: string; kind: string; angle: number; quad: number[][] }[];
}

export const listSources = () => request<{ sources: Source[]; root: string }>("/sources");

export const listSamples = (sourceId: string, patchDataset?: string) =>
  request<{ samples: SampleInfo[] }>(
    `/sources/${sourceId}/samples${patchDataset ? `?patch_dataset=${encodeURIComponent(patchDataset)}` : ""}`
  );

export const sampleGeometry = (sourceId: string, index: number) =>
  request<Geometry>(`/sources/${sourceId}/samples/${index}/geometry`);

/** The image URL. The client never sends a path -- only an id (D4). */
export const sampleImageUrl = (sourceId: string, index: number, w?: number) =>
  `${BASE}/sources/${sourceId}/samples/${index}/image${w ? `?w=${w}` : ""}`;

// ── B: patch datasets ────────────────────────────────────────────────────────

export interface Manifest {
  format_version: number;
  fingerprint: string;
  has_border: boolean;
  source_id: string;
  config: {
    source: string;
    patch_size: number;
    stride: number;
    target_kinds: string[];
    drop_overlap: boolean;
    split: { train: number; val: number; test: number };
    seed: number;
  };
  num_samples: number;
  num_patches: number;
  patch_shape: number[];
  corner_order: string[];
  border_order: string[];
  patches_per_split: Record<string, number>;
  positives_per_corner: Record<string, number>;
  warnings: ApiProblem[];
}

export interface PatchDataset {
  name: string;
  manifest: Manifest;
}

export interface BuildBody {
  name: string;
  source: string;
  patch_size: number;
  stride: number;
  drop_overlap: boolean;
  split: { train: number; val: number; test: number };
  seed: number;
}

export const listPatchDatasets = () =>
  request<{ patch_datasets: PatchDataset[] }>("/patch-datasets");

export const getPatchDataset = (name: string) =>
  request<{ name: string; manifest: Manifest; fingerprint: string; used_by: string[] }>(
    `/patch-datasets/${encodeURIComponent(name)}`
  );

export const buildPatchDataset = (body: BuildBody) =>
  request<Job>("/patch-datasets", { method: "POST", body: JSON.stringify(body) });

export const deletePatchDataset = (name: string) =>
  request<void>(`/patch-datasets/${encodeURIComponent(name)}`, { method: "DELETE" });

// ── C: networks ──────────────────────────────────────────────────────────────

export interface BlockSpec {
  filters: number;
  kernel: number;
  stride: number;
  padding?: number | "same";
  batchnorm?: boolean;
  activation?: string;
  pool?: number;
  dropout?: number;
}

export interface NetworkConfig {
  input_size: number;
  in_channels: number;
  backbone: BlockSpec[];
  head: { hidden?: number[]; dropout?: number };
  border_features: boolean;
}

export interface TraceStep {
  layer: number;
  in: number;
  conv: number;
  out: number;
  channels: number;
}

export interface NetworkDescription {
  valid: boolean;
  trace: TraceStep[];
  num_params: number;
  flat_features: number;
}

export const listNetworks = () =>
  request<{ networks: { name: string; config: NetworkConfig }[] }>("/networks");

/** Pure, synchronous and cheap: validation as a FUNCTION of the API rather than
 *  a side effect of training. It is what feeds the Redes screen live. */
export const validateNetwork = (config: NetworkConfig) =>
  request<NetworkDescription>("/networks/validate", { method: "POST", body: JSON.stringify(config) });

export const createNetwork = (name: string, config: NetworkConfig) =>
  request<{ name: string; config: NetworkConfig }>("/networks", {
    method: "POST",
    body: JSON.stringify({ name, ...config }),
  });

export const deleteNetwork = (name: string) =>
  request<void>(`/networks/${encodeURIComponent(name)}`, { method: "DELETE" });

// ── D: recipes ───────────────────────────────────────────────────────────────

/** The catalogue of organizacion.md §1-D. **No `device`, no `num_workers`**:
 *  those are X (contract ⑩), and they live in Entrenar. */
export interface RecipeValues {
  lr: number;
  optimizer: string;
  momentum: number;
  weight_decay: number;
  batch_size: number;
  grad_clip: number;
  epochs: number;
  scheduler: string;
  warmup_epochs: number;
  patience: number;
  min_delta: number;
  lambda_pos: number;
  pos_weight: number | null;
  smooth_l1_beta: number;
  monitor: string;
  seed: number;
}

export const listRecipes = () =>
  request<{ recipes: { name: string; recipe: RecipeValues }[] }>("/recipes");

export const createRecipe = (name: string, recipe: RecipeValues) =>
  request<{ name: string; recipe: RecipeValues }>("/recipes", {
    method: "POST",
    body: JSON.stringify({ name, ...recipe }),
  });

export const deleteRecipe = (name: string) =>
  request<void>(`/recipes/${encodeURIComponent(name)}`, { method: "DELETE" });

// ── X: jobs ──────────────────────────────────────────────────────────────────

export interface Job {
  id: string;
  kind: string;
  state: "queued" | "running" | "done" | "error";
  detail: Record<string, unknown>;
  result: unknown;
  error: string | null;
  created_at: string;
  finished_at: string;
}

export const getJob = (id: string) => request<Job>(`/jobs/${id}`);
