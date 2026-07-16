import { useState } from "react";
import { ApiError, createRecipe, deleteRecipe, listRecipes, type RecipeValues } from "../api";
import { ErrorNote, Empty, Loading, type ApiProblem } from "../components/Async";
import { useAsync } from "../useAsync";

/** Recetas (D) — the hyperparameters that DEFINE the result.
 *
 * ui.md §2: no network, no dataset. A recipe is reusable across architectures,
 * and that is the point — it is what turns a sweep into "pick a space" instead
 * of "fill in this form 20 times".
 *
 * **`device` and `num_workers` are not here** (contract ⑩). They are X: they cost
 * time, not accuracy. A recipe carrying the device stops being comparable
 * between CPU and GPU, and the GPU is coming.
 */

interface FieldDef {
  key: keyof RecipeValues;
  label: string;
  /** The definition from the catalogue. ui.md: a hyperparameter without one in
   *  the catalogue is not finished, and the form shows it INLINE. */
  help: string;
  kind?: "number" | "select";
  options?: string[];
  step?: number;
  /** Marked in the UI: these came back on their own once, from a default. */
  trap?: string;
}

const GROUPS: { title: string; note: string; fields: FieldDef[] }[] = [
  {
    title: "Optimización",
    note: "cómo se da cada paso",
    fields: [
      {
        key: "lr",
        label: "lr",
        help: "Tamaño del paso. El más influyente. Barrer en escala log (1e-4…3e-2), nunca lineal.",
        step: 0.0001,
      },
      {
        key: "optimizer",
        label: "optimizer",
        help: "Cómo se calcula el paso a partir del gradiente.",
        kind: "select",
        options: ["adam", "adamw", "sgd", "rmsprop"],
      },
      {
        key: "momentum",
        label: "momentum",
        help: "Inercia del paso. Solo aplica a sgd y rmsprop. Típico: 0.9.",
        step: 0.05,
        trap:
          "El default de SGD en torch es 0. El código anterior solo le pasaba lr y weight_decay, " +
          "así que SGD corría sin inercia y barrer «optimizer» comparaba Adam contra un espantapájaros.",
      },
      {
        key: "weight_decay",
        label: "weight_decay",
        help:
          "Penalización L2 sobre los pesos; regulariza. Log, 0…1e-2. En adam es L2 acoplada; en " +
          "adamw, desacoplada — no son lo mismo aunque el campo se llame igual.",
        step: 0.0001,
      },
      {
        key: "batch_size",
        label: "batch_size",
        help:
          "Muestras por paso de gradiente. Acoplado a lr: doblar el batch suele pedir subir el lr. " +
          "Es D, no X: cambiarlo cambia los pesos. Subirlo al pasar a GPU invalida la comparación.",
        step: 1,
      },
      {
        key: "grad_clip",
        label: "grad_clip",
        help:
          "Recorta la norma del gradiente antes del paso; 0 lo apaga. Barato y estabiliza lr altos: " +
          "sin él, el extremo alto de un barrido de lr diverge y no aprendes nada de esa zona.",
        step: 0.1,
      },
    ],
  },
  {
    title: "Duración y programación",
    note: "cuánto y cómo se recorre",
    fields: [
      { key: "epochs", label: "epochs", help: "Pasadas completas sobre el train.", step: 1 },
      {
        key: "scheduler",
        label: "scheduler",
        help:
          "Cómo decae lr con las épocas. La omisión más cara: sin él lr es constante, y con schedule " +
          "el lr inicial óptimo es otro número — barrer lr sin schedule optimiza para un régimen que no vas a usar.",
        kind: "select",
        options: ["none", "cosine", "step", "plateau"],
      },
      {
        key: "warmup_epochs",
        label: "warmup_epochs",
        help: "Épocas subiendo lr desde ~0. Solo importa con batch grande o lr alto.",
        step: 1,
      },
      {
        key: "patience",
        label: "patience",
        help:
          "Parada temprana: épocas sin mejorar antes de cortar; 0 la apaga. Mira su propia curva de val. " +
          "Distinto de la poda del barrido, que compara entre runs.",
        step: 1,
      },
      { key: "min_delta", label: "min_delta", help: "Cuánto cuenta como «mejorar» para patience.", step: 0.001 },
    ],
  },
  {
    title: "Pérdida",
    note: "lo propio de este problema",
    fields: [
      {
        key: "lambda_pos",
        label: "lambda_pos",
        help:
          "Peso del término de posición frente al de existencia. Arbitra «¿hay esquina?» vs «¿dónde exactamente?». " +
          "El hiperparámetro más interesante del proyecto, y el que más cuidado pide al rankear.",
        step: 0.1,
      },
      {
        key: "pos_weight",
        label: "pos_weight",
        help:
          "Peso de la clase positiva en la BCE. El desbalance real es ~3.9:1 (20,5 % de positivos): modesto, no brutal. " +
          "Sácalo del manifest del dataset — cambia con cada uno. Vacío = sin peso.",
        step: 0.1,
      },
      {
        key: "smooth_l1_beta",
        label: "smooth_l1_beta",
        help: "Umbral donde smoothL1 pasa de cuadrática a lineal. ~0.05–0.1 ≈ 2–4 px en un patch de 40.",
        step: 0.01,
        trap:
          "El default de PyTorch es 1.0 y las coordenadas van normalizadas a [0,1], así que |error| < 1 " +
          "siempre: la pérdida nunca sale de la rama cuadrática. Era MSE pura y el Huber no se activaba jamás.",
      },
    ],
  },
  {
    title: "Selección y reproducibilidad",
    note: "qué checkpoint te quedas, y cómo se replica",
    fields: [
      {
        key: "monitor",
        label: "monitor",
        help:
          "Qué métrica elige best.pt. No puede ser val_loss si barres lambda_pos: cada punto se mediría con " +
          "una pérdida distinta y λ=0 «ganaría» por definición, o sea que lo óptimo sería no predecir posiciones.",
        kind: "select",
        options: ["val_loss", "val_f1", "val_pos_err_px"],
      },
      {
        key: "seed",
        label: "semilla de entrenamiento",
        help:
          "Init de pesos y shuffle. No es un hiperparámetro a optimizar: es el EJE DE RÉPLICA. Fijar todo y " +
          "variar solo esto mide el ruido, que es lo único que dice si una diferencia es real. " +
          "Distinta de la semilla del dataset, que fija el split.",
        step: 1,
      },
    ],
  },
];

const DEFAULTS: RecipeValues = {
  lr: 0.001,
  optimizer: "adam",
  momentum: 0.9,
  weight_decay: 0,
  batch_size: 64,
  grad_clip: 0,
  epochs: 20,
  scheduler: "none",
  warmup_epochs: 0,
  patience: 0,
  min_delta: 0,
  lambda_pos: 1,
  pos_weight: null,
  smooth_l1_beta: 0.05,
  monitor: "val_loss",
  seed: 1,
};

export function Recipes() {
  const recipes = useAsync(listRecipes, []);
  const [creating, setCreating] = useState(false);

  return (
    <section className="screen screen--wide">
      <h1 className="screen__title">Recetas</h1>
      <p className="screen__lede">
        Los hiperparámetros que <strong>definen el resultado</strong>. Sin red y sin dataset: una
        receta es reutilizable entre arquitecturas, y eso es justo lo que convierte un barrido en
        «elegir un espacio» en vez de «rellenar un formulario 20 veces».{" "}
        <strong>
          <code>device</code> no está aquí
        </strong>
        : cuesta tiempo, no resultado — va en Entrenar.
      </p>

      <div className="row-actions">
        <button className="button" onClick={() => setCreating((v) => !v)}>
          {creating ? "Cancelar" : "Crear una receta"}
        </button>
      </div>

      {creating && (
        <RecipeForm
          onDone={() => {
            setCreating(false);
            recipes.reload();
          }}
        />
      )}

      {recipes.loading && <Loading what="las recetas" />}
      {recipes.error && <ErrorNote problem={recipes.error} />}
      {recipes.data?.recipes.length === 0 && !creating && <Empty>Todavía no hay ninguna receta.</Empty>}

      {recipes.data?.recipes.map((r) => (
        <RecipeCard key={r.name} name={r.name} recipe={r.recipe} onChange={recipes.reload} />
      ))}
    </section>
  );
}

function RecipeCard({
  name,
  recipe,
  onChange,
}: {
  name: string;
  recipe: RecipeValues;
  onChange: () => void;
}) {
  const [problem, setProblem] = useState<ApiProblem | null>(null);

  async function remove() {
    try {
      await deleteRecipe(name);
      onChange();
    } catch (err) {
      setProblem(err instanceof ApiError ? err.problem : { code: "unknown", message: String(err) });
    }
  }

  return (
    <article className="card">
      <header className="card__head">
        <h2 className="card__title">{name}</h2>
        <button className="button button--quiet" onClick={remove}>
          Borrar
        </button>
      </header>
      {problem && <ErrorNote problem={problem} />}
      <dl className="facts facts--inline">
        {GROUPS.flatMap((g) => g.fields).map((f) => {
          const value = recipe[f.key];
          return (
            <div className="fact" key={f.key}>
              <dt title={f.help}>{f.label}</dt>
              <dd>{value === null ? "—" : String(value)}</dd>
            </div>
          );
        })}
      </dl>
    </article>
  );
}

function RecipeForm({ onDone }: { onDone: () => void }) {
  const [name, setName] = useState("");
  const [values, setValues] = useState<RecipeValues>({ ...DEFAULTS });
  const [problem, setProblem] = useState<ApiProblem | null>(null);

  const set = (key: keyof RecipeValues, value: string) =>
    setValues((v) => ({
      ...v,
      [key]:
        key === "optimizer" || key === "scheduler" || key === "monitor"
          ? value
          : value === ""
          ? null
          : Number(value),
    }));

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setProblem(null);
    try {
      await createRecipe(name, values);
      onDone();
    } catch (err) {
      setProblem(err instanceof ApiError ? err.problem : { code: "unknown", message: String(err) });
    }
  }

  // Warn where it matters: monitor=val_loss with a λ sweep is contract ⑨, and
  // the sweep screen must BLOCK it. Here λ is a single value, so it is only a
  // heads-up — but it is the same trap, and it produces a winner with a good face.
  const lambdaWarning = values.monitor === "val_loss";

  return (
    <form className="card card--form" onSubmit={submit}>
      <h2 className="card__title">Crear una receta</h2>
      {problem && <ErrorNote problem={problem} />}

      <div className="form__grid">
        <label className="field">
          <span className="field__label">Nombre</span>
          <input value={name} onChange={(e) => setName(e.target.value)} required placeholder="adam-lr1e-3" />
        </label>
      </div>

      {GROUPS.map((group) => (
        <div key={group.title}>
          <p className="card__section">
            {group.title} <span className="card__hint">{group.note}</span>
          </p>
          <div className="form__grid">
            {group.fields.map((f) => (
              <label className="field" key={f.key}>
                <span className="field__label">
                  {f.label}
                  {f.trap && (
                    <span className="field__trap" title={f.trap}>
                      trampa por defecto
                    </span>
                  )}
                  {/* The definition, inline. A hyperparameter without one is not
                      finished -- and these are the definitions that were only
                      ever in a doc nobody had open while filling the form. */}
                  <span className="field__hint">{f.help}</span>
                </span>
                {f.kind === "select" ? (
                  <select value={String(values[f.key])} onChange={(e) => set(f.key, e.target.value)}>
                    {f.options!.map((o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    step={f.step ?? "any"}
                    value={values[f.key] === null ? "" : String(values[f.key])}
                    onChange={(e) => set(f.key, e.target.value)}
                  />
                )}
              </label>
            ))}
          </div>
        </div>
      ))}

      {lambdaWarning && (
        <div className="async async--warning" role="alert">
          <p className="async__message">
            Con <code>monitor: val_loss</code>, ojo si algún día barres <code>lambda_pos</code>.
          </p>
          <p className="async__hint">
            <code>loss = cls + λ·pos</code>: cada punto se mediría con una pérdida distinta y λ=0
            ganaría por definición — o sea, «lo mejor es no predecir posiciones». La pantalla de
            Barridos lo bloqueará; aquí solo te avisa.
          </p>
        </div>
      )}

      <div className="row-actions">
        <button className="button" type="submit" disabled={!name}>
          Guardar
        </button>
      </div>
    </form>
  );
}
