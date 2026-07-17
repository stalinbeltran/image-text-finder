import { useEffect, useState } from "react";

type Theme = "system" | "light" | "dark";

/** Stamps `data-theme` on <html>, which is the scope tokens.css lets win.
 *
 * Both modes are SELECTED, not derived: the dark steps are the same hues
 * re-stepped for the dark surface and validated as their own set. So the toggle
 * is not a nicety -- `npm run validate:palette` checks a mode this switch is the
 * only way to actually look at.
 */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
  }, [theme]);

  return (
    <label className="theme-toggle">
      <span className="theme-toggle__label">Tema</span>
      <select
        className="theme-toggle__select"
        value={theme}
        onChange={(e) => setTheme(e.target.value as Theme)}
      >
        <option value="system">Sistema</option>
        <option value="light">Claro</option>
        <option value="dark">Oscuro</option>
      </select>
    </label>
  );
}
