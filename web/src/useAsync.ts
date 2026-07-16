import { useCallback, useEffect, useState } from "react";
import { ApiError } from "./api";
import type { ApiProblem } from "./components/Async";

/** Load, error, data — the convention, in one hook (plan-ui.md fase 1.1).
 *
 * It keeps `ApiProblem` intact rather than stringifying it, so the screen can
 * render the `hint` (R4). A hook that returned `error: string` would delete the
 * most useful half of every error the API sends.
 */
export interface Async<T> {
  data: T | null;
  loading: boolean;
  error: ApiProblem | null;
  reload: () => void;
}

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): Async<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiProblem | null>(null);
  const [tick, setTick] = useState(0);

  const reload = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    fn()
      .then((value) => live && setData(value))
      .catch((err: unknown) => {
        if (!live) return;
        setError(
          err instanceof ApiError
            ? err.problem
            : { code: "network_error", message: String(err), hint: "¿está corriendo itf-api?" }
        );
      })
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  return { data, loading, error, reload };
}
