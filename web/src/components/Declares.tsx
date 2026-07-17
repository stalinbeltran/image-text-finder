/** Regla 2 of ui.md, made into a component instead of a good intention.
 *
 * > Toda vista de análisis declara **(qué fija, qué varía, qué mide)**.
 *
 * A view that crosses domains is not an exception to "one screen, one domain":
 * it is an **experiment**, and an experiment with no declared control measures
 * nothing. A sweep fixes B and C, varies D, measures the objective (contract ⑧).
 * A feature map fixes E and the patch, and varies the layer. Same structure --
 * which is why the catalogue in ui.md §4.1 is tabulated in these three columns.
 *
 * Making it a component is what keeps it honest: the rule's real job is to stop
 * the UI degenerating into "screens with pretty things", and it only does that
 * if a view cannot be added without answering the three questions. Prose in a
 * doc gets skipped; a required prop does not.
 */
export interface DeclaresProps {
  /** The vista's id in ui.md §4.1 — V3, V6, V7… */
  view: string;
  title: string;
  fixes: string;
  varies: string;
  measures: string;
  children?: React.ReactNode;
}

export function Declares({ view, title, fixes, varies, measures, children }: DeclaresProps) {
  return (
    <header className="declares">
      <h2 className="declares__title">
        <span className="declares__view">{view}</span> {title}
      </h2>
      <dl className="declares__control">
        <div>
          <dt>Fija</dt>
          <dd>{fixes}</dd>
        </div>
        <div>
          <dt>Varía</dt>
          <dd>{varies}</dd>
        </div>
        <div>
          <dt>Mide</dt>
          <dd>{measures}</dd>
        </div>
      </dl>
      {children && <p className="declares__note">{children}</p>}
    </header>
  );
}
