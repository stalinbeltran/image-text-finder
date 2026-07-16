import { useEffect, useState } from "react";
import { api, RunDetail, RunSummary } from "../api";
import LineChart from "./LineChart";
import ModelConfigForm, {
  configToModelForm,
  modelFormHyper,
  modelFormToModel,
  ModelForm,
} from "./ModelConfigForm";

export default function RunsPanel() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // edit / retrain form (shows every parameter that defines the network)
  const [retrainFor, setRetrainFor] = useState<string | null>(null);
  const [retrainName, setRetrainName] = useState("");
  const [retrainForm, setRetrainForm] = useState<ModelForm | null>(null);
  const [retrainData, setRetrainData] = useState<string>("");
  const [loadingCfg, setLoadingCfg] = useState(false);

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

  const rename = async (name: string) => {
    const to = window.prompt(`Rename run "${name}" to:`, name);
    if (!to || to === name) return;
    setError(null); setMsg(null);
    try {
      await api.renameRun(name, to);
      if (selected === name) { setSelected(to); setDetail(null); }
      setMsg(`Renamed ${name} → ${to}`);
      refresh();
    } catch (e) { setError(String(e)); }
  };

  const remove = async (name: string) => {
    if (!window.confirm(`Delete run "${name}"? This removes its checkpoints and metrics.`)) return;
    setError(null); setMsg(null);
    try {
      await api.deleteRun(name);
      if (selected === name) { setSelected(null); setDetail(null); }
      setMsg(`Deleted ${name}`);
      refresh();
    } catch (e) { setError(String(e)); }
  };

  const openRetrain = async (name: string) => {
    setRetrainFor(name);
    setRetrainName(`${name}-v2`);
    setRetrainForm(null);
    setRetrainData("");
    setMsg(null); setError(null);
    setLoadingCfg(true);
    try {
      const d = await api.run(name);
      setRetrainForm(configToModelForm(d.config));
      setRetrainData(String((d.config as any)?.data ?? ""));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingCfg(false);
    }
  };

  const submitRetrain = async () => {
    if (!retrainFor || !retrainName || !retrainForm || !retrainData) return;
    setError(null); setMsg(null);
    try {
      await api.startRun({
        data: retrainData,
        name: retrainName,
        model: modelFormToModel(retrainForm),
        ...modelFormHyper(retrainForm),
      });
      setMsg(`Retraining started as ${retrainName} — see it below for live metrics.`);
      setRetrainFor(null);
      refresh();
      setSelected(retrainName);
      setDetail(null);
    } catch (e) { setError(String(e)); }
  };

  const m = detail?.metrics ?? [];
  const xs = m.map((e) => e.epoch);
  const last = m[m.length - 1];

  return (
    <>
      <div className="card">
        <h2>Runs (trained models)</h2>
        {msg && <p className="muted">{msg}</p>}
        {error && <p className="err">{error}</p>}
        <table>
          <thead>
            <tr>
              <th>name</th>
              <th>status</th>
              <th>epochs</th>
              <th>last val loss</th>
              <th>last F1</th>
              <th>actions</th>
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
                <td onClick={(e) => e.stopPropagation()} style={{ whiteSpace: "nowrap" }}>
                  <button className="btn ghost" onClick={() => openRetrain(r.name)} title="View / edit parameters & retrain">↻</button>{" "}
                  <button className="btn ghost" onClick={() => rename(r.name)} title="Rename"
                          disabled={r.status !== "done"}>✎</button>{" "}
                  <button className="btn ghost" onClick={() => remove(r.name)} title="Delete"
                          disabled={r.status !== "done"}>🗑</button>
                </td>
              </tr>
            ))}
            {runs.length === 0 && (
              <tr><td colSpan={6} className="muted">no runs yet</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {retrainFor && (
        <div className="card">
          <h2>Edit &amp; retrain <span className="mono">{retrainFor}</span></h2>
          <p className="muted">
            Every parameter that defines <span className="mono">{retrainFor}</span> (its architecture,
            head and training hyperparameters), loaded from its frozen config. Tweak any of them and
            train a new run — the original is left untouched.
          </p>
          {loadingCfg && <p className="muted">loading config…</p>}
          {retrainForm && (
            <>
              <div className="row">
                <div className="field">
                  <label>New run name</label>
                  <input value={retrainName} onChange={(e) => setRetrainName(e.target.value)} />
                </div>
                <div className="field">
                  <label>Patch dataset (from original run)</label>
                  <input className="mono" value={retrainData} readOnly />
                </div>
              </div>

              <ModelConfigForm value={retrainForm} onChange={setRetrainForm} />

              <button className="btn" onClick={submitRetrain} disabled={!retrainName || !retrainData}>Start retrain</button>{" "}
              <button className="btn ghost" onClick={() => setRetrainFor(null)}>Cancel</button>
            </>
          )}
        </div>
      )}

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
