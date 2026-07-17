import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ApiError,
  createRun,
  listNetworks,
  listPatchDatasets,
  listRecipes,
  listRuns,
  type Manifest,
  type NetworkConfig,
  type RecipeValues,
  type RunRow,
} from "../api";
import { ErrorNote, Empty, Loading, type ApiProblem } from "../components/Async";
import { useAsync } from "../useAsync";

/** Entrenar (B × C × D + X → E).
 *
 * ui.md §2. Pick a dataset, a network, a recipe; `device` apart. The old
 * `TrainPanel` was C + D + X in one form — this screen picks three NAMES and one
 * execution option, and that difference is the whole redesign: it is what makes
 * contract ③ hold by itself (api.md R7).
 *
 * **"Re-train the same network" is not a mode.** It is choosing another recipe
 * with the same C, which is what it always was. With C and D separated it comes
 * out free, and the old `lockArchitecture` toggle — a boolean papering over the
 * C/D border instead of resolving it — has nothing left to do.
 */
export function Train() {
  const datasets = useAsync(listPatchDatasets, []);
  const networks = useAsync(listNetworks, []);
  const recipes = useAsync(listRecipes, []);
  const runs = useAsync(listRuns, []);
  const navigate = useNavigate();

  const [name, setName] = useState("");
  // Derived, not defaulted in an effect: an effect runs AFTER the render that
  // already drew the options, so there is a window where the select shows a list
  // with nothing chosen and a submit posts an empty name.
  const [pickedDataset, setPickedDataset] = useState("");
  const [pickedNetwork, setPickedNetwork] = useState("");
  const [pickedRecipe, setPickedRecipe] = useState("");
  const [device, setDevice] = useState("cpu");
  const [numWorkers, setNumWorkers] = useState(0);
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [busy, setBusy] = useState(false);

  const dataset = pickedDataset || datasets.data?.patch_datasets[0]?.name || "";
  const network = pickedNetwork || networks.data?.networks[0]?.name || "";
  const recipe = pickedRecipe || recipes.data?.recipes[0]?.name || "";

  const manifest = datasets.data?.patch_datasets.find((d) => d.name === dataset)?.manifest ?? null;
  const networkConfig = networks.data?.networks.find((n) => n.name === network)?.config ?? null;
  const recipeValues = recipes.data?.recipes.find((r) => r.name === recipe)?.recipe ?? null;

  const loading = datasets.loading || networks.loading || recipes.loading;
  const missing =
    !loading &&
    ((datasets.data?.patch_datasets.length ?? 0) === 0 ||
      (networks.data?.networks.length ?? 0) === 0 ||
      (recipes.data?.recipes.length ?? 0) === 0);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setProblem(null);
    setBusy(true);
    try {
      await createRun({
        name,
        patch_dataset: dataset,
        network,
        recipe,
        device,
        num_workers: numWorkers,
      });
      // Straight to Runs: launching is not the interesting part, watching is.
      navigate("/runs");
    } catch (err) {
      setProblem(err instanceof ApiError ? err.problem : { code: "unknown", message: String(err) });
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="screen screen--wide">
      <h1 className="screen__title">Entrenar</h1>
      <p className="screen__lede">
        Un dataset de patches, una red y una receta — <strong>por nombre</strong>, nunca por valor.
        Suena rígido y es a propósito: es lo que hace que todo run pueda decir de qué B, qué C y qué
        D salió, que es justo lo que un barrido necesita para agrupar.{" "}
        <strong>
          <code>device</code> va aparte
        </strong>
        : cuesta tiempo, no resultado.
      </p>

      {loading && <Loading what="datasets, redes y recetas" />}
      {datasets.error && <ErrorNote problem={datasets.error} />}
      {networks.error && <ErrorNote problem={networks.error} />}
      {recipes.error && <ErrorNote problem={recipes.error} />}

      {missing && (
        <Empty>
          Para entrenar hacen falta las tres cosas: un dataset de patches (Datos → Patches), una red
          (Modelo → Redes) y una receta (Modelo → Recetas).
        </Empty>
      )}

      {!loading && !missing && (
        <form className="card card--form" onSubmit={submit}>
          {problem && <ErrorNote problem={problem} />}

          <div className="form__grid">
            <label className="field">
              <span className="field__label">
                Nombre del run <span className="field__hint">no se sobrescribe nunca: si ya existe, se niega</span>
              </span>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                placeholder="cnn-a-baseline-s1"
              />
            </label>

            <label className="field">
              <span className="field__label">
                Dataset de patches <span className="field__hint">B — el dato que la CNN consume</span>
              </span>
              <select value={dataset} onChange={(e) => setPickedDataset(e.target.value)}>
                {datasets.data?.patch_datasets.map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name} (n={d.manifest.config.patch_size}, {d.manifest.num_patches} patches)
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span className="field__label">
                Red <span className="field__hint">C — la arquitectura</span>
              </span>
              <select value={network} onChange={(e) => setPickedNetwork(e.target.value)}>
                {networks.data?.networks.map((n) => (
                  <option key={n.name} value={n.name}>
                    {n.name} (input_size={n.config.input_size})
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span className="field__label">
                Receta <span className="field__hint">D — lo que define el resultado</span>
              </span>
              <select value={recipe} onChange={(e) => setPickedRecipe(e.target.value)}>
                {recipes.data?.recipes.map((r) => (
                  <option key={r.name} value={r.name}>
                    {r.name} ({r.recipe.optimizer}, lr={r.recipe.lr}, {r.recipe.epochs} épocas)
                  </option>
                ))}
              </select>
            </label>
          </div>

          <p className="card__section">
            Ejecución{" "}
            <span className="card__hint">
              X — cuesta tiempo, no cambia los pesos. Fuera de la identidad de la receta (contrato ⑩)
            </span>
          </p>
          <div className="form__grid">
            <label className="field">
              <span className="field__label">
                device <span className="field__hint">hoy solo CPU; la GPU llegará</span>
              </span>
              <select value={device} onChange={(e) => setDevice(e.target.value)}>
                <option value="cpu">cpu</option>
                <option value="cuda">cuda</option>
              </select>
            </label>
            <label className="field">
              <span className="field__label">
                num_workers{" "}
                <span className="field__hint">
                  0 = carga en el proceso. Con workers, el orden de los lotes lo decide el SO y el run
                  deja de ser función de su semilla
                </span>
              </span>
              <input
                type="number"
                min={0}
                value={numWorkers}
                onChange={(e) => setNumWorkers(+e.target.value)}
              />
            </label>
          </div>

          <Compatibility manifest={manifest} network={networkConfig} />
          <CostEstimate
            manifest={manifest}
            networkName={network}
            device={device}
            recipe={recipeValues}
            runs={runs.data?.runs ?? []}
          />

          <div className="row-actions">
            <button className="button" type="submit" disabled={!name || busy}>
              {busy ? "Lanzando…" : "Entrenar"}
            </button>
          </div>
        </form>
      )}
    </section>
  );
}

/** Contract ①, made visible where `n` and `input_size` meet.
 *
 * **This is a heads-up, not the gate.** The old UI checked it here and ONLY
 * here, so any other client — or a curl — got `mat1 and mat2 shapes cannot be
 * multiplied` from inside the job thread half an hour later. The refusal now
 * lives in `POST /runs` (400 with the reason), which is why this does not
 * disable the button: if the two ever disagree, the server wins, visibly.
 */
function Compatibility({
  manifest,
  network,
}: {
  manifest: Manifest | null;
  network: NetworkConfig | null;
}) {
  if (!manifest || !network) return null;

  const sizeOk = manifest.config.patch_size === network.input_size;
  const borderOk = !network.border_features || manifest.has_border;
  const channelsOk = manifest.patch_shape[2] === network.in_channels;
  const valOk = (manifest.patches_per_split?.val ?? 1) > 0;

  if (sizeOk && borderOk && channelsOk && valOk) {
    return (
      <p className="card__foot">
        <span className="pair-ok">✓ casan</span> el dataset trae patches de{" "}
        <code>{manifest.config.patch_size}</code> y la red espera{" "}
        <code>{network.input_size}</code>.
      </p>
    );
  }

  return (
    <div className="async async--warning" role="alert">
      {!sizeOk && (
        <p className="async__message">
          El dataset trae patches de <code>{manifest.config.patch_size}</code> y la red espera{" "}
          <code>{network.input_size}</code>: no casan.
        </p>
      )}
      {!borderOk && (
        <p className="async__message">
          La red pide <code>border_features</code> y este dataset se construyó sin los flags de
          borde.
        </p>
      )}
      {!channelsOk && (
        <p className="async__message">
          Los patches tienen <code>{manifest.patch_shape[2]}</code> canal(es) y la red espera{" "}
          <code>{network.in_channels}</code>.
        </p>
      )}
      {!valOk && (
        <p className="async__message">
          Este dataset no tiene val: no hay con qué elegir <code>best.pt</code> ni con qué medir.
        </p>
      )}
      <p className="async__hint">
        El servidor lo va a rechazar con la razón y el arreglo. Elige otra combinación, o crea la
        pieza que falta.
      </p>
    </div>
  );
}

/** What this will cost, from the `seconds` other runs already measured.
 *
 * ui.md §2 asks for it by name, and on CPU it is information, not decoration: an
 * epoch on the real dataset is ~20 s, so 20 epochs is 7 minutes and a careless
 * choice of dataset is 5 hours.
 *
 * **It only estimates from runs that are actually comparable**: same B — by
 * FINGERPRINT, not by name, since a rebuilt dataset under the same name is a
 * different dataset (contract ⑧) — and the same network. With nothing to go on
 * it says so instead of inventing a number: an absent estimate is not a zero
 * (formatos.md §2). A run with no provenance therefore cannot estimate either,
 * however many metrics it has: it cannot say which dataset it came from.
 *
 * **It does not filter by `device`, and that is a known gap, not an oversight.**
 * `device` is X, so it is deliberately outside the provenance (contract ⑩) and
 * `GET /runs` does not carry it — there is nothing here to filter on. Today it
 * cannot bite: everything is CPU. The day the GPU lands it would, by averaging a
 * 2 s/epoch GPU epoch with a 25 s/epoch CPU one into a number that describes
 * neither, so whoever adds the GPU must add `execution.device` to the run row and
 * filter on it here. Until then the estimate names the runs it used, so a wrong
 * one is at least traceable.
 */
function CostEstimate({
  manifest,
  networkName,
  device,
  recipe,
  runs,
}: {
  manifest: Manifest | null;
  networkName: string;
  device: string;
  recipe: RecipeValues | null;
  runs: RunRow[];
}) {
  if (!manifest || !recipe) return null;

  const comparable = runs.filter(
    (r) =>
      r.seconds_per_epoch !== null &&
      r.provenance?.patch_dataset.fingerprint === manifest.fingerprint &&
      r.provenance?.network.name === networkName
  );

  if (comparable.length === 0) {
    return (
      <p className="card__foot">
        <span className="card__hint">
          Todavía no hay ningún run con este dataset y esta red, así que no hay con qué estimar el
          coste. Lo habrá en cuanto termine el primero.
        </span>
      </p>
    );
  }

  const perEpoch =
    comparable.reduce((sum, r) => sum + (r.seconds_per_epoch ?? 0), 0) / comparable.length;
  const total = perEpoch * recipe.epochs;
  const minutes = total / 60;

  return (
    <p className="card__foot">
      Coste estimado: <strong>{perEpoch.toFixed(1)} s/época</strong> ×{" "}
      <strong>{recipe.epochs} épocas</strong> ≈{" "}
      <strong>{minutes < 1 ? `${total.toFixed(0)} s` : `${minutes.toFixed(1)} min`}</strong>{" "}
      <span className="card__hint">
        medido sobre {comparable.length} run(s) con el mismo dataset y la misma red (
        {comparable.map((r) => r.name).join(", ")}). El <code>batch_size</code> de esta receta puede
        moverlo, y en {device} el límite de workers es 1.
      </span>
    </p>
  );
}
