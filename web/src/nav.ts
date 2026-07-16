/** The map of screens. ui.md §1.
 *
 * Four groups that follow the DEPENDENCY between domains, not taste: you cannot
 * train without data and a network, and you cannot analyse without a run.
 *
 * What changed, and it is the point of the whole redesign: the old UI was a
 * numbered pipeline (`1 · Patches → 2 · Train → 3 · Runs → 4 · Predict`). That
 * assumes you walk the flow once, left to right. Research does not walk: it
 * ITERATES ON A POINT and comes back. So there are NO STEP NUMBERS -- the groups
 * gather domains, they do not order steps.
 *
 * One screen, one domain (ui.md regla 1). Each entry names its domain because a
 * screen that cannot say which domain it owns is how TrainPanel ended up being
 * C + D + X in one form.
 */

export interface Screen {
  path: string;
  label: string;
  /** The domain letter of organizacion.md §1. */
  domain: string;
  /** For the empty screens of fase 1: which phase builds it. */
  phase: string;
}

export interface NavGroup {
  label: string;
  screens: Screen[];
}

export const NAV: NavGroup[] = [
  {
    label: "Datos",
    screens: [
      { path: "/sources", label: "Fuentes", domain: "A — Fuente", phase: "fase 2" },
      { path: "/patch-datasets", label: "Patches", domain: "B — Dataset de patches", phase: "fase 2" },
    ],
  },
  {
    label: "Modelo",
    screens: [
      { path: "/networks", label: "Redes", domain: "C — Red", phase: "fase 3" },
      { path: "/recipes", label: "Recetas", domain: "D — Receta", phase: "fase 3" },
    ],
  },
  {
    label: "Entrenar",
    screens: [
      { path: "/train", label: "Entrenar", domain: "B × C × D + X → E", phase: "fase 4" },
      { path: "/sweeps", label: "Barridos", domain: "H — Barrido", phase: "fase 7" },
      { path: "/runs", label: "Runs", domain: "E — Run", phase: "fase 4" },
    ],
  },
  {
    label: "Analizar",
    screens: [
      { path: "/diagnostics", label: "Diagnóstico", domain: "E × B", phase: "fase 5" },
      { path: "/predict", label: "Predecir", domain: "F — Inferencia", phase: "fase 6" },
    ],
  },
];

export const SCREENS: Screen[] = NAV.flatMap((g) => g.screens);
