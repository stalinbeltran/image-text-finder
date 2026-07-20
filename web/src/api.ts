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

/** The provenance of a derived source (A'), from `dataset.json` (formatos.md §4.6). */
export interface Derived {
  /** The ADDRESSABLE id of the parent -- what you would type to get it back. */
  from: string;
  /** What the parent calls itself. Not unique: two `clear-paragraphs-02` share one. */
  from_declared_id: string;
  op: string;
  request: { width?: number; height?: number };
  size: [number, number] | null;
  scale: [number, number] | null;
}

export interface Source {
  id: string;
  source_id: string;
  num_samples: number;
  /** `null` = not counted yet (the source has no index on disk), NOT zero.
   *  Counting overlaps means parsing every block of every image, so the picker
   *  refuses to pay it; opening the source builds the index and fills it in. */
  num_overlapping: number | null;
  /** `null` means an ORIGINAL, not "a derived one we lost track of" (D19). */
  derived: Derived | null;
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

export const listSources = () =>
  request<{ sources: Source[]; root: string; derived_root: string }>("/sources");

/** A' — resize a source (D19). `width` XOR `height`: the aspect ratio is kept.
 *
 * A job (R3): N images. The refusals (`upscale_not_allowed`,
 * `resize_needs_one_dimension`) arrive as a 400 BEFORE the job exists, so a
 * rejected request leaves nothing behind to clean up.
 */
export const resizeSource = (sourceId: string, body: { name: string; width?: number; height?: number }) =>
  request<Job>(`/sources/${sourceId}/resize`, { method: "POST", body: JSON.stringify(body) });

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

// ── E: runs ──────────────────────────────────────────────────────────────────

/** Contract ③ (D2): the NAME to group, the VALUE to reproduce.
 *
 * Both, and it is not redundancy — with only the value you have to diff
 * dictionaries by hand to ask "which runs used network X?", which is the
 * question a sweep asks all the time. */
export interface Provenance {
  patch_dataset: { name: string; fingerprint: string };
  network: { name: string; value: NetworkConfig };
  recipe: { name: string; value: RecipeValues };
  sweep: string | null;
  /** The sha, or **the reason it is unknown** — never null (formatos.md §2). */
  git_commit: string;
  environment: { python: string; torch: string; platform: string };
}

export type RunState = "queued" | "running" | "done" | "error" | "cancelled";

export interface RunStatus {
  state: RunState;
  epoch?: number;
  error?: string;
  updated_at?: string;
}

export interface RunSummary {
  run: string;
  epochs_run: number;
  epochs_requested: number;
  stopped_early: boolean;
  cancelled: boolean;
  monitor: string;
  /** `null` when the monitor never produced a number — not ±Infinity, which is
   *  not valid JSON and would break the parse of this whole response. */
  best: number | null;
  final: EpochRecord | Record<string, never>;
  corner_order: string[];
}

export interface RunRow {
  name: string;
  state: RunState;
  provenance: Provenance | null;
  /** `null`, never 0, when no epoch has finished: 0 would read as "instant". */
  seconds_per_epoch: number | null;
  summary: RunSummary | null;
  error?: string;
}

export interface RunDetail {
  name: string;
  state: RunStatus;
  config: {
    format_version: number;
    network: NetworkConfig;
    recipe: RecipeValues;
    execution: { device: string; num_workers: number };
    provenance: Provenance;
  };
  provenance: Provenance;
  checkpoints: string[];
  summary: RunSummary | null;
  seconds_per_epoch: number | null;
}

export interface EpochRecord {
  epoch: number;
  train_loss: number;
  val: {
    loss: number;
    cls_loss: number;
    pos_loss: number;
    exists_acc: number;
    precision: number;
    recall: number;
    f1: number;
    /** `null` when val held no corners: not measured, and 0 would read as perfect. */
    pos_err_px: number | null;
  };
  lr: number;
  seconds: number;
}

export interface CreateRunBody {
  name: string;
  patch_dataset: string;
  network: string;
  recipe: string;
  /** X (contract ⑩): costs time, not result. Never part of the recipe. */
  device: string;
  num_workers: number;
}

export const listRuns = () => request<{ runs: RunRow[] }>("/runs");

export const getRun = (name: string) => request<RunDetail>(`/runs/${encodeURIComponent(name)}`);

/** Incremental (R5): `since` is the `next` the last call handed back. The whole
 *  history is never re-sent — the UI polls this in a loop. */
export const getRunMetrics = (name: string, since: number) =>
  request<{ records: EpochRecord[]; next: number }>(
    `/runs/${encodeURIComponent(name)}/metrics?since=${since}`
  );

export const createRun = (body: CreateRunBody) =>
  request<Job>("/runs", { method: "POST", body: JSON.stringify(body) });

/** Cooperative: it cuts at the end of the current epoch, keeping what it did. */
export const stopRun = (name: string) =>
  request<{ name: string; stop_requested: boolean }>(`/runs/${encodeURIComponent(name)}/stop`, {
    method: "POST",
  });

export const renameRun = (name: string, newName: string) =>
  request<{ name: string }>(`/runs/${encodeURIComponent(name)}`, {
    method: "PATCH",
    body: JSON.stringify({ name: newName }),
  });

export const deleteRun = (name: string) =>
  request<void>(`/runs/${encodeURIComponent(name)}`, { method: "DELETE" });

// ── B: one patch (fase 2's endpoint, which V3 and V6 read) ───────────────────

export interface Patch {
  index: number;
  /** (n, n) greyscale 0–255. The REAL input of the CNN (contract ①). */
  patch: number[][];
  /** (4, 3) — `[exists, x, y]` per corner, in `corner_order`. */
  label: number[][];
  border: number[];
  /** V15: which image of A this came from, and where in it. */
  sample_idx: number;
  patch_xy: number[];
  split: string;
  corner_order: string[];
  border_order: string[];
}

export const getPatch = (dataset: string, index: number) =>
  request<Patch>(`/patch-datasets/${encodeURIComponent(dataset)}/patches/${index}`);

// ── E×B: diagnostics (D1: a CACHE, so every route is an idempotent GET) ──────

/** V8. The curve is 101 thresholds computed off STORED scores: choosing one
 *  never runs the model again, which is why this view comes before the sweep. */
export interface PrCurve {
  corner: string;
  positives: number;
  negatives: number;
  /** The imbalance. ~20 % positives is why accuracy misleads and PR informs. */
  positive_rate: number | null;
  curve: { threshold: number; precision: number; recall: number; f1: number }[];
  /** The best-F1 threshold, REPORTED not applied: picking one is a decision. */
  best: { threshold: number; precision: number; recall: number; f1: number } | null;
  default_threshold: number;
  histogram: { edges: number[]; positive: number[]; negative: number[] };
}

/** V7. `matrix[y][x]` is `null` where no corner ever landed — **not 0**, which
 *  would paint "perfect" over the parts of the patch the data never covered. */
export interface ErrorMap {
  corner: string;
  patch_size: number;
  /** Cells per side. ui.md says 40×40; with ~200 corners that is 0.1 samples per
   *  cell and the map is speckle, so the resolution is a knob (see the backend's
   *  `DEFAULT_ERROR_MAP_BINS`). `bins === patch_size` is the full-resolution map. */
  bins: number;
  /** Px of the patch per cell — what the axes are labelled in. */
  cell_px: number;
  matrix: (number | null)[][];
  counts: number[][];
  samples: number;
  /** The payload declares the colour work: the client cannot know (api.md §3). */
  job: "sequential" | "diverging";
}

/** V6. No pixels: a row says WHICH patch it is and B serves the patch. */
export interface PatchRow {
  patch_idx: number;
  sample_idx: number;
  patch_xy: number[];
  score: number[];
  xy_pred: number[][];
  xy_true: number[][];
  exists: boolean[];
  /** `null` where there is no real corner: nothing to localise, not zero error. */
  err_px: (number | null)[];
  /** V18 — how much of this corner's paragraph fits in the patch. `null` where
   *  there is no corner. A miss at ~0 is a corner whose paragraph is outside the
   *  window: the pixels do not show what the label asks for. */
  evidence: (number | null)[];
}

export interface PatchPage {
  total: number;
  offset: number;
  limit: number;
  threshold: number;
  rows: PatchRow[];
  corner_order: string[];
}

export type Split = "train" | "val" | "test";
export type Corner = "TL" | "TR" | "BR" | "BL";
export type Outcome = "all" | "tp" | "fp" | "fn" | "tn";

const query = (params: Record<string, string | number | undefined>) =>
  Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== "")
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`)
    .join("&");

export const getPrCurve = (run: string, split: Split, corner?: string) =>
  request<PrCurve>(
    `/runs/${encodeURIComponent(run)}/diagnostics/pr?${query({ split, corner })}`
  );

export const getErrorMap = (run: string, split: Split, corner?: string, bins?: number) =>
  request<ErrorMap>(
    `/runs/${encodeURIComponent(run)}/diagnostics/error-map?${query({ split, corner, bins })}`
  );

export const getDiagnosticPatches = (
  run: string,
  params: {
    split: Split;
    corner?: string;
    outcome?: Outcome;
    order?: string;
    threshold?: number;
    /** V18's filters. `max_evidence=0.05` is "the corners nobody could see". */
    max_evidence?: number;
    min_evidence?: number;
    offset?: number;
    limit?: number;
  }
) =>
  request<PatchPage>(
    `/runs/${encodeURIComponent(run)}/diagnostics/patches?${query(params)}`
  );

/** V9 — **not a confusion matrix** (ui.md §4.1). The 4 heads are independent
 *  binaries, not a softmax: a patch can fire two at once or none, so the rows sum
 *  to nothing. `matrix[i][j]` is `P(head j fires | corner i really there)`.
 *
 * `truth_rate` is the **control**, and without it the view lies: a high
 * `matrix[TL][TR]` means either "the TR head is confused by a TL" or "these
 * patches really do contain a TR too" — opposite problems. Where the two agree,
 * the corners simply co-occur; where `matrix` runs above `truth_rate`, that is
 * the confusion. */
export interface Coactivation {
  corner_order: string[];
  threshold: number;
  /** Rows = the corner really there; columns = the head that fired. `null` when
   *  no patch has that corner: not measured, and 0 would read as a measurement. */
  matrix: (number | null)[][];
  truth_rate: (number | null)[][];
  /** Patches behind each row. A row of 3 paints exactly like a row of 300. */
  counts: number[];
  job: "sequential" | "diverging";
}

/** V18 — the same measurements, split by how much of the corner's paragraph was
 *  actually inside the patch.
 *
 * A corner whose point falls near the far edge has its paragraph *outside* the
 * window: the label asks for something the pixels do not show. Measured on
 * `dirty-20`, those are 14 % of corners and carry 31 % of the position error.
 *
 * **What it does not say**: that they are unlearnable. Detection on them
 * generalises — identical scores on train, val and test — so the model is reading
 * real context, not memorising. The gap is in *position*, which is why detection
 * and error are reported side by side rather than folded into one number. */
export interface EvidenceSplit {
  corner: string;
  threshold: number;
  blind_cut: number;
  corners: number;
  patch_size: number;
  bands: EvidenceBand[];
  blind: EvidencePopulation;
  seen: EvidencePopulation;
}

export interface EvidencePopulation {
  corners: number;
  corner_share: number;
  /** `null` when the population is empty: not measured, and 0 would read as one. */
  err_px: number | null;
  /** Its slice of the TOTAL error. Next to `corner_share`, this is the finding:
   *  equal shares mean the band is unremarkable, and the blind one is not. */
  error_share: number | null;
  recall: number | null;
  score_mean: number | null;
}

export interface EvidenceBand extends EvidencePopulation {
  low: number;
  high: number;
}

/** Default cut for "blind". Mirrors `itf.metrics.BLIND_EVIDENCE`.
 *
 * It is restated here rather than fetched because the gallery filters by it
 * *before* V18 has answered, and a filter that has to wait for another view is a
 * filter that flickers. The value the backend actually used always rides back in
 * `EvidenceSplit.blind_cut`, and V18 renders that one — so if these two ever
 * drift, the screen shows the drift instead of hiding it. */
export const BLIND_EVIDENCE = 0.05;

export const getEvidenceSplit = (
  run: string,
  split: Split,
  corner?: string,
  threshold?: number
) =>
  request<EvidenceSplit>(
    `/runs/${encodeURIComponent(run)}/diagnostics/evidence?${query({
      split,
      corner,
      threshold,
    })}`
  );

export const getCoactivation = (run: string, split: Split, threshold: number) =>
  request<Coactivation>(
    `/runs/${encodeURIComponent(run)}/diagnostics/coactivation?${query({ split, threshold })}`
  );

// ── F: introspection (V1, V2) and the pipeline (V11) ─────────────────────────

/** One matrix, with the stats the client normalises against.
 *
 * `min`/`max` ride with **each** map because the normalisation is **per map**
 * (ui.md §5): compute them across a layer instead and the quiet feature maps all
 * render black. */
export interface MapPayload {
  index: number;
  min: number;
  max: number;
  mean: number;
  matrix: number[][];
}

/** A layer of matrices. One shape for V1 and V2 — a kernel and a feature map
 *  differ in what the numbers mean, not in what they are.
 *
 * **`job` comes from the server and is never guessed.** The client cannot know if
 * it holds a signed weight or a non-negative activation, and guessing is the one
 * thing the sibling project got wrong (api.md §3, ui.md §4.0 R2). */
export interface LayerPayload {
  layer: number;
  /** How many the layer really has. */
  count: number;
  /** How many are in this response — `max_maps` truncates at 64. */
  shown: number;
  truncated: boolean;
  job: "sequential" | "diverging";
  maps: MapPayload[];
  spec: BlockSpec;
  /** V2 only. */
  height?: number;
  width?: number;
  /** V1 only. */
  kernel_size?: number;
  in_channels?: number;
}

/** V1 — layer 1 only (D13, decidido 2026-07-17).
 *
 * The rule is `in_channels === 1`, not "the first layer": with one input channel
 * a filter **is** a matrix and applies to the patch itself, so what you see is
 * exact. From layer 2 on a filter is 32 or 64 matrices over channels that are not
 * the image, and there is no honest projection — that information is in the
 * feature maps (V2). */
export interface Kernels extends LayerPayload {
  layers_in_backbone: number;
  deep_layers_note: string;
}

export const getKernels = (run: string) =>
  request<Kernels>(`/runs/${encodeURIComponent(run)}/kernels`);

export interface FeatureMaps {
  input_size: number;
  layers: LayerPayload[];
  prediction: {
    corners: { corner: string; score: number; x: number; y: number }[];
    corner_order: string[];
  };
}

/** V2 — **the input is a patch** (contract ①): the patch is the real input of the
 *  CNN. A whole image is F's question and goes to `predict`. */
export const getFeatureMaps = (
  run: string,
  body: { patch_dataset: string; index: number } | { patch: number[][]; border?: number[] }
) =>
  request<FeatureMaps>(`/runs/${encodeURIComponent(run)}/feature-maps`, {
    method: "POST",
    body: JSON.stringify(body),
  });

export interface DetectionDto {
  corner: string;
  x: number;
  y: number;
  score: number;
}

/** V11 — **the three stages**, not just the last one.
 *
 * The failure is born in exactly one of them and they are fixed in different
 * places: stage 1 is the model (C, D); NMS and the pairing are knobs of F you
 * turn without retraining. Without `raw`, "the paragraph came out wrong" is not
 * diagnosable — you cannot tell a corner that was never detected from one NMS
 * ate. */
export interface Prediction {
  /** Pre-NMS: what the window found, above `threshold`. */
  raw: DetectionDto[];
  /** Post-NMS: what survived. */
  corners: DetectionDto[];
  /** Post-pairing: `[x, y, w, h]` per paragraph. */
  paragraphs: { box: number[]; score: number }[];
  image_size: number[];
  /** Echoed back. The sliders are live, so answers arrive out of order: without
   *  this, a slow response overwrites a newer one. */
  knobs: { threshold: number; stride: number; nms_radius: number; min_size: number };
  /** The patch this run's model eats. **Every knob is measured in units of it**,
   *  so a client that hardcodes a stride is choosing a different thing per run:
   *  20 px is "every other window" on a 40-px patch and "skip half the image" on
   *  a 10-px one. */
  patch_size: number;
  /** What the sliding window actually looked at. `has_gaps` is the silent
   *  failure: a stride above the patch size returns a perfectly well-formed
   *  answer with the corners of the unseen strips simply missing. */
  coverage: {
    stride: number;
    patch_size: number;
    has_gaps: boolean;
    unseen_columns: number;
    unseen_rows: number;
  };
  source: string;
  index: number;
}

/** The knobs are **F, not D** (organizacion.md §1-D): chosen per call over a
 *  model already trained. Sweeping them costs a forward pass, not an afternoon —
 *  which is why they are sliders and not a form you submit. */
export interface PredictKnobs {
  threshold: number;
  /** The stride of INFERENCE. **Not B's**, which is part of the dataset's
   *  identity — the shared name is a dangerous coincidence (glosario.md). */
  /** `null` is not "0" and not "20": it means **let F choose half the patch**
   *  and tell us what it chose. There is no constant a client can honestly
   *  default this to without knowing the model's patch size. */
  stride?: number | null;
  nms_radius?: number | null;
  min_size?: number | null;
}

export const predict = (run: string, source: string, index: number, knobs: PredictKnobs) =>
  request<Prediction>(`/runs/${encodeURIComponent(run)}/predict`, {
    method: "POST",
    body: JSON.stringify({ source, index, ...knobs }),
  });

// ── the probes (fase 8): V4, V10, V5 ─────────────────────────────────────────

/** How V2, V4 and V10 all name a patch (contract ①): by index of a B (whose
 *  border flags are then the real ones), or inline pixels for a crop stored
 *  nowhere. One shape, three views. */
export type PatchRef =
  | { patch_dataset: string; index: number }
  | { patch: number[][]; border?: number[] };

/** V4 — occlusion sensitivity. `maps[c].matrix[iy][ix]` is `p(exists | occlude
 *  the cell at (iy, ix))`: a genuine probability, never signed, so **sequential**
 *  is honest (dark = covering here kills the score = the region that mattered).
 *  The *drop* is `baseline[c] − matrix`, one subtraction the reader can see rather
 *  than a diverging colour that would put the neutral wherever the min fell. */
export interface OcclusionMap {
  corner: string;
  matrix: number[][];
  min: number;
  max: number;
  mean: number;
}

export interface Occlusion {
  corner_order: string[];
  patch_size: number;
  mask_size: number;
  mask_stride: number;
  /** Cells per side. */
  bins: number;
  /** Top-left offset in patch px of each cell (row and column share it). */
  offsets: number[];
  /** `p(exists)` with no occlusion, per corner. The drop is measured against it. */
  baseline: number[];
  maps: OcclusionMap[];
  job: "sequential" | "diverging";
}

/** V16 — one map per filter: the gradient of its peak activation w.r.t. the input.
 *
 * Same `(K, H, W)` shape as V1/V2, so `LayerMaps` renders it unchanged — a kernel,
 * a feature map and an attribution differ in what the numbers mean, not in what
 * they are. Two fields are V16's own: `peaks` (which activation was
 * differentiated, and where) and `silent` (filters that never fired on this
 * patch, so their gradient is 0 everywhere and their map is a flat neutral
 * square — correct, and indistinguishable from "tiny gradients" unless said).
 */
export interface DeconvLayer extends LayerPayload {
  peaks: { filter: number; activation: number; y: number; x: number }[];
  /** Of `count`, how many said nothing **on this patch**. Not "dead": one patch. */
  silent: number;
}

export interface Deconvolution {
  input_size: number;
  layers: DeconvLayer[];
}

/** V16 — deconvolución. `border` is accepted by the endpoint but unused: the
 *  gradient never leaves the backbone, and `border_features` only touches the head. */
export const getDeconvolution = (run: string, ref: PatchRef) =>
  request<Deconvolution>(`/runs/${encodeURIComponent(run)}/deconvolution`, {
    method: "POST",
    body: JSON.stringify(ref),
  });

export const getOcclusion = (run: string, ref: PatchRef) =>
  request<Occlusion>(`/runs/${encodeURIComponent(run)}/occlusion`, {
    method: "POST",
    body: JSON.stringify(ref),
  });

/** V10 — the border-flag test. One flip per flag (5 forwards total): `scores` is
 *  what the 4 heads say once that flag is flipped, `baseline` what they say with
 *  the patch's real flags. The dumbbell draws each corner's move before→after.
 *
 * Only meaningful for a network with `border_features`: the endpoint refuses
 * (409 `border_not_used`) otherwise, because flipping a flag the network ignores
 * changes nothing and would read as "the border does not matter to this patch". */
export interface BorderFlip {
  border: string;
  flag_from: number;
  flag_to: number;
  scores: number[];
}

export interface BorderTest {
  corner_order: string[];
  border_order: string[];
  border: number[];
  baseline: number[];
  flips: BorderFlip[];
}

/** By index only: the real flags have to come from the dataset for the flip to
 *  measure a real change (an inline patch could carry made-up flags). */
export const getBorderTest = (run: string, patch_dataset: string, index: number) =>
  request<BorderTest>(`/runs/${encodeURIComponent(run)}/border-test`, {
    method: "POST",
    body: JSON.stringify({ patch_dataset, index }),
  });

/** V5 — the scrubber: one off-grid 40×40 crop of an image → the 4 heads live.
 *
 * `stability` is what a single prediction cannot give: how much each head's score
 * moves when the window shifts 1 px, which is exactly what sets the inference
 * `stride` and the NMS radius (ui.md §4.1). */
export interface WindowCorner {
  corner: string;
  score: number;
  /** Fractions of the patch, as the head speaks. */
  x: number;
  y: number;
  /** And in image px, so the overlay can place the dot on the image. */
  image_x: number;
  image_y: number;
}

export interface WindowPrediction {
  x0: number;
  y0: number;
  patch_size: number;
  image_size: number[];
  border: number[];
  corner_order: string[];
  corners: WindowCorner[];
  stability: { per_corner: number[]; max: number };
  source: string;
  index: number;
}

export const getWindow = (run: string, source: string, index: number, x0: number, y0: number) =>
  request<WindowPrediction>(`/runs/${encodeURIComponent(run)}/window`, {
    method: "POST",
    body: JSON.stringify({ source, index, x0, y0 }),
  });

// ── H: sweeps ──────────────────────────────────────────────────────────────────

/** A distribution over one recipe field. `float`/`int` are a range (optionally
 *  log); `categorical` is a list of choices. Mirrors `itf.sweeps.spec`. */
export type Distribution =
  | { type: "float"; low: number; high: number; log?: boolean }
  | { type: "int"; low: number; high: number; log?: boolean }
  | { type: "categorical"; choices: (string | number)[] };

export type Objective = "f1" | "pos_err_px" | "loss";
export type Strategy = "random" | "tpe" | "grid";

/** Which way each objective is "better". The Pareto view needs it to know which
 *  corner is the good one. Mirrors `OBJECTIVE_DIRECTION`. */
export const OBJECTIVE_DIRECTION: Record<Objective, "maximize" | "minimize"> = {
  f1: "maximize",
  pos_err_px: "minimize",
  loss: "minimize",
};

export interface SweepBudget {
  points: number;
  epochs: number;
  pruning: boolean;
}

export interface CreateSweepBody {
  name: string;
  patch_dataset: string;
  network: string;
  recipe?: string | null;
  space: Record<string, Distribution>;
  objective: Objective;
  strategy: Strategy;
  budget: SweepBudget;
  seed?: number;
}

export interface Trial {
  number: number;
  state: "complete" | "pruned" | "running" | "fail" | "waiting";
  run: string | null;
  params: Record<string, number | string>;
  value: number | null;
  /** Recorded for every trial whatever the objective, so V12 can draw the
   *  (f1, pos_err_px) Pareto view no matter what this sweep ranked by. */
  f1: number | null;
  pos_err_px: number | null;
  epochs_run: number | null;
}

export interface SweepBest {
  number: number;
  run: string | null;
  value: number;
  params: Record<string, number | string>;
}

export interface SweepProgress {
  name: string;
  objective: Objective;
  direction: "maximize" | "minimize";
  budget: SweepBudget;
  space: Record<string, Distribution>;
  patch_dataset: string;
  network: string;
  trials: Trial[];
  completed: number;
  best: SweepBest | null;
}

export interface SweepDetail extends SweepProgress {
  spec: CreateSweepBody & { seed: number };
  state: Job["state"];
}

export interface SweepRow {
  name: string;
  state: Job["state"];
  objective: Objective;
  patch_dataset: string;
  network: string;
  completed: number;
  points: number;
  best: SweepBest | null;
}

export const listSweeps = () => request<{ sweeps: SweepRow[] }>("/sweeps");

export const getSweep = (name: string) =>
  request<SweepDetail>(`/sweeps/${encodeURIComponent(name)}`);

export const createSweep = (body: CreateSweepBody) =>
  request<Job>("/sweeps", { method: "POST", body: JSON.stringify(body) });

export const stopSweep = (name: string) =>
  request<{ name: string; stop_requested: boolean }>(
    `/sweeps/${encodeURIComponent(name)}/stop`,
    { method: "POST" }
  );

/** The counterpart of `stopSweep`. Picks a stopped or interrupted sweep back up.
 *
 * The machinery predates the button: the API resumes unfinished sweeps on
 * startup, so before this existed the only way to continue a stopped sweep was
 * **restarting the backend** — which reads as "there is no way to continue".
 * 409 if it already met its budget, or if it is already running.
 */
export const resumeSweep = (name: string) =>
  request<{ name: string; job: string; resumed: boolean }>(
    `/sweeps/${encodeURIComponent(name)}/resume`,
    { method: "POST" }
  );

// ── X: jobs ──────────────────────────────────────────────────────────────────

export interface Job {
  id: string;
  kind: string;
  /** `interrupted` is what a restart leaves; `cancelled` is a cooperative stop. */
  state: "queued" | "running" | "done" | "error" | "cancelled" | "interrupted";
  detail: Record<string, unknown>;
  result: unknown;
  error: string | null;
  created_at: string;
  finished_at: string;
}

export const getJob = (id: string) => request<Job>(`/jobs/${id}`);

export const cancelJob = (id: string) =>
  request<Job>(`/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
