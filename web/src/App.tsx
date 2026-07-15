import { useState } from "react";
import ExtractPanel from "./components/ExtractPanel";
import TrainPanel from "./components/TrainPanel";
import RunsPanel from "./components/RunsPanel";
import PredictPanel from "./components/PredictPanel";

type Tab = "patches" | "train" | "runs" | "predict";

const TABS: { id: Tab; label: string }[] = [
  { id: "patches", label: "1 · Patches" },
  { id: "train", label: "2 · Train" },
  { id: "runs", label: "3 · Runs" },
  { id: "predict", label: "4 · Predict" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("patches");

  return (
    <>
      <header>
        <h1>image-text-finder</h1>
        <span className="sub">patch-based paragraph-corner detection</span>
      </header>
      <nav>
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>
      <main>
        {tab === "patches" && <ExtractPanel />}
        {tab === "train" && <TrainPanel onStarted={() => setTab("runs")} />}
        {tab === "runs" && <RunsPanel />}
        {tab === "predict" && <PredictPanel />}
      </main>
    </>
  );
}
