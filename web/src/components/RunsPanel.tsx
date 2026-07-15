import { useEffect, useState } from "react";
import { api, RunDetail, RunSummary } from "../api";
import LineChart from "./LineChart";

export default function RunsPanel() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);

  const refresh = () => api.runs().then(setRuns).catch(() => {});
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 1500);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!selected) return;
    let live = true;
    const tick = () => api.run(selected).then((d) => live && setDetail(d)).catch(() => {});
    tick();
    const t = setInterval(() => {
      tick();
      if (detail?.status === "done") clearInterval(t);
    }, 1500);
    return () => {
      live = false;
      clearInterval(t);
    };
  }, [selected, detail?.status]);

  const m = detail?.metrics ?? [];
  const xs = m.map((e) => e.epoch);
  const last = m[m.length - 1];

  return (
    <>
      <div className="card">
        <h2>Runs</h2>
        <table>
          <thead>
            <tr>
              <th>name</th>
              <th>status</th>
              <th>epochs</th>
              <th>last val loss</th>
              <th>last F1</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr
                key={r.name}
                className={`clickable ${selected === r.name ? "selected" : ""}`}
                onClick={() => { setSelected(r.name); setDetail(null); }}
              >
                <td className="mono">{r.name}</td>
                <td><span className={`tag ${r.status === "done" ? "done" : "running"}`}>{r.status}</span></td>
                <td>{r.epochs_done}</td>
                <td>{r.last?.val?.loss?.toFixed(4) ?? "—"}</td>
                <td>{r.last?.val?.f1?.toFixed(3) ?? "—"}</td>
              </tr>
            ))}
            {runs.length === 0 && (
              <tr><td colSpan={5} className="muted">no runs yet</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {detail && (
        <div className="card">
          <h2>{detail.name} <span className={`tag ${detail.status === "done" ? "done" : "running"}`}>{detail.status}</span></h2>
          {last && (
            <div className="metrics-grid" style={{ marginBottom: 16 }}>
              <Stat k="train loss" v={last.train_loss?.toFixed(4)} />
              <Stat k="val loss" v={last.val?.loss?.toFixed(4)} />
              <Stat k="exists acc" v={last.val?.exists_acc?.toFixed(3)} />
              <Stat k="precision" v={last.val?.precision?.toFixed(3)} />
              <Stat k="recall" v={last.val?.recall?.toFixed(3)} />
              <Stat k="F1" v={last.val?.f1?.toFixed(3)} />
              <Stat k="pos err (px)" v={last.val?.pos_err_px?.toFixed(2)} />
            </div>
          )}
          <h3>Loss</h3>
          <LineChart
            xs={xs}
            series={[
              { label: "train", color: "#4c8dff", points: m.map((e) => e.train_loss) },
              { label: "val", color: "#2ec7a8", points: m.map((e) => e.val?.loss) },
            ]}
          />
          <div className="legend">
            <span><span className="dot" style={{ background: "#4c8dff" }} /> train loss</span>
            <span><span className="dot" style={{ background: "#2ec7a8" }} /> val loss</span>
          </div>
          <h3>Detection quality (val)</h3>
          <LineChart
            xs={xs}
            series={[
              { label: "precision", color: "#ffbf47", points: m.map((e) => e.val?.precision) },
              { label: "recall", color: "#ff6b6b", points: m.map((e) => e.val?.recall) },
              { label: "f1", color: "#2ec7a8", points: m.map((e) => e.val?.f1) },
            ]}
          />
          <div className="legend">
            <span><span className="dot" style={{ background: "#ffbf47" }} /> precision</span>
            <span><span className="dot" style={{ background: "#ff6b6b" }} /> recall</span>
            <span><span className="dot" style={{ background: "#2ec7a8" }} /> f1</span>
          </div>
        </div>
      )}
    </>
  );
}

function Stat({ k, v }: { k: string; v: string | undefined }) {
  return (
    <div className="stat">
      <div className="v">{v ?? "—"}</div>
      <div className="k">{k}</div>
    </div>
  );
}
