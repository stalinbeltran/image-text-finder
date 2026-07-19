import type { ReactNode } from "react";

/** The loading and error conventions, in one place (plan-ui.md fase 1.1).
 *
 * They live here rather than in each screen because api.md R4 makes the shape of
 * an error a CONTRACT, not a formatting choice: `code` is stable and the UI may
 * react to it, `message` and `hint` are for people. A screen that renders
 * `String(err)` throws the hint away -- and the hint is the half that says how to
 * fix it, which is the whole reason R4 exists.
 */

export interface ApiProblem {
  /** Stable slug. Contract: the UI may switch on it. */
  code: string;
  message: string;
  /** How to fix it. R4 says an error says why AND how. */
  hint?: string;
}

export function isApiProblem(value: unknown): value is ApiProblem {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as ApiProblem).code === "string" &&
    typeof (value as ApiProblem).message === "string"
  );
}

export function Loading({ what }: { what: string }) {
  return (
    <p className="async async--loading" role="status">
      Cargando {what}…
    </p>
  );
}

/** A request is in flight but the PREVIOUS answer is still on screen.
 *
 * Distinct from `Loading`, which replaces the content: V11 and V5 keep the last
 * frame on purpose so a live slider or a drag reads as continuous instead of
 * strobing a blank. But dimming the stale frame is all that said so, and dimming
 * is invisible when the answer takes 30 s on a busy CPU -- the screen then looks
 * like it ignored the click. A click that was received must SAY so, however
 * long the answer takes. */
export function Working({ what }: { what: string }) {
  return (
    <p className="async async--working" role="status">
      <span className="async__spinner" aria-hidden="true" />
      Calculando {what}… (lo que ves es la respuesta anterior)
    </p>
  );
}

export function ErrorNote({ problem }: { problem: ApiProblem | Error }) {
  if (!isApiProblem(problem)) {
    return (
      <div className="async async--error" role="alert">
        <p className="async__message">{problem.message}</p>
      </div>
    );
  }
  return (
    <div className="async async--error" role="alert">
      <p className="async__message">{problem.message}</p>
      {problem.hint && <p className="async__hint">{problem.hint}</p>}
      <code className="async__code">{problem.code}</code>
    </div>
  );
}

/** Nothing here yet, and why -- distinct from "still loading" and from "it broke".
 *  Ausente ≠ cero is a rule about data (formatos.md §2); this is its UI shape. */
export function Empty({ children }: { children: ReactNode }) {
  return <p className="async async--empty">{children}</p>;
}

/** A screen that the plan has not built yet.
 *
 * Fase 1 delivers the nav and EMPTY screens, so most routes land here. It names
 * the phase on purpose: an empty screen that does not say why reads as a bug.
 */
export function NotBuiltYet({ domain, phase, children }: { domain: string; phase: string; children?: ReactNode }) {
  return (
    <div className="not-built">
      <p className="not-built__domain">{domain}</p>
      <p className="not-built__phase">La construye la {phase} de plan-ui.md.</p>
      {children && <div className="not-built__note">{children}</div>}
    </div>
  );
}
