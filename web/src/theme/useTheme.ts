import { useEffect, useState } from "react";

/** A counter that bumps whenever the ACTIVE palette changes.
 *
 * Anything that reads colours out of the cascade -- `MatrixCanvas`, every Plot
 * figure -- resolves them once at paint time via `getComputedStyle`. CSS repaints
 * itself when the mode flips; **a canvas does not**, and neither does an SVG Plot
 * already built. Without this they keep the palette of the mode they were born in, so
 * flipping to dark leaves the error map painted in light-mode blues on a dark
 * surface.
 *
 * It is not hypothetical and it is not new: `MatrixCanvas` has had this since
 * fase 1 (its effect depends on `[matrix, job, size]`, none of which change when
 * the theme does). Fase 5 is where it becomes visible, because V7 puts a big
 * canvas on screen next to the toggle.
 *
 * Two sources, because there are two ways the mode can change and tokens.css
 * honours both: the OS setting (the media query) and the explicit toggle (the
 * `data-theme` attribute, which is the scope that must win either way).
 */
export function useThemeVersion(): number {
  const [version, setVersion] = useState(0);

  useEffect(() => {
    const bump = () => setVersion((v) => v + 1);

    const observer = new MutationObserver(bump);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", bump);

    return () => {
      observer.disconnect();
      media.removeEventListener("change", bump);
    };
  }, []);

  return version;
}
