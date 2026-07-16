import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { NAV, SCREENS } from "./nav";
import { NotBuiltYet } from "./components/Async";
import { ThemeToggle } from "./components/ThemeToggle";
import { Kitchen } from "./screens/Kitchen";

/** The shell: four groups, no step numbers (ui.md §1). */
export function App() {
  return (
    <div className="app">
      <header className="app__bar">
        <span className="app__title">image-text-finder</span>
        <ThemeToggle />
      </header>

      <div className="app__body">
        <nav className="nav" aria-label="Secciones">
          {NAV.map((group) => (
            <div className="nav__group" key={group.label}>
              <p className="nav__group-label">{group.label}</p>
              <ul className="nav__list">
                {group.screens.map((s) => (
                  <li key={s.path}>
                    <NavLink
                      to={s.path}
                      className={({ isActive }) => `nav__link ${isActive ? "is-active" : ""}`}
                    >
                      {s.label}
                    </NavLink>
                  </li>
                ))}
              </ul>
            </div>
          ))}
          <div className="nav__group nav__group--foot">
            <NavLink
              to="/kitchen"
              className={({ isActive }) => `nav__link ${isActive ? "is-active" : ""}`}
            >
              Paleta y componentes
            </NavLink>
          </div>
        </nav>

        <main className="main">
          <Routes>
            <Route path="/" element={<Navigate to="/sources" replace />} />
            {SCREENS.map((s) => (
              <Route
                key={s.path}
                path={s.path}
                element={
                  <section className="screen">
                    <h1 className="screen__title">{s.label}</h1>
                    <NotBuiltYet domain={s.domain} phase={s.phase} />
                  </section>
                }
              />
            ))}
            <Route path="/kitchen" element={<Kitchen />} />
            <Route
              path="*"
              element={
                <section className="screen">
                  <h1 className="screen__title">No existe esa pantalla</h1>
                </section>
              }
            />
          </Routes>
        </main>
      </div>
    </div>
  );
}
