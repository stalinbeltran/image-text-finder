import { useEffect, useRef, useState } from "react";
import { api, PredictResult, RunSummary } from "../api";

const CORNER_COLORS: Record<string, string> = {
  TL: "#4c8dff", TR: "#2ec7a8", BR: "#ffbf47", BL: "#ff6b6b",
};

export default function PredictPanel() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [run, setRun] = useState("");
  const [checkpoint, setCheckpoint] = useState("best");
  const [threshold, setThreshold] = useState(0.5);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    api.runs().then((rs) => {
      const done = rs.filter((r) => r.status === "done");
      setRuns(done);
      if (done[0]) setRun(done[0].name);
    });
  }, []);

  const draw = (img: HTMLImageElement, res: PredictResult | null) => {
    const c = canvasRef.current;
    if (!c) return;
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const ctx = c.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    if (!res) return;
    ctx.lineWidth = 1.5;
    for (const p of res.paragraphs) {
      const [x, y, w, h] = p.box;
      ctx.strokeStyle = "rgba(46,199,168,0.9)";
      ctx.strokeRect(x, y, w, h);
    }
    for (const cn of res.corners) {
      ctx.fillStyle = CORNER_COLORS[cn.corner] ?? "#fff";
      ctx.beginPath();
      ctx.arc(cn.x, cn.y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const run_predict = async () => {
    if (!file || !run) return;
    setBusy(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("run", run);
      form.append("checkpoint", checkpoint);
      form.append("threshold", String(threshold));
      form.append("file", file);
      const res = await api.predict(form);
      setResult(res);
      const img = new Image();
      img.onload = () => draw(img, res);
      img.src = URL.createObjectURL(file);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h2>Inference preview</h2>
      <div className="row">
        <div className="field">
          <label>Run (trained)</label>
          <select value={run} onChange={(e) => setRun(e.target.value)}>
            {runs.map((r) => (
              <option key={r.name} value={r.name}>{r.name}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Checkpoint</label>
          <select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
            <option value="best">best</option>
            <option value="last">last</option>
          </select>
        </div>
        <div className="field">
          <label>Threshold: {threshold.toFixed(2)}</label>
          <input type="range" min={0} max={1} step={0.05} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
        </div>
        <div className="field">
          <label>Image</label>
          <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </div>
      </div>
      <button className="btn" onClick={run_predict} disabled={!file || !run || busy}>
        {busy ? "Predicting…" : "Predict"}
      </button>

      {error && <p className="err">{error}</p>}

      {result && (
        <>
          <div className="legend" style={{ margin: "12px 0" }}>
            {Object.entries(CORNER_COLORS).map(([k, v]) => (
              <span key={k}><span className="dot" style={{ background: v }} /> {k}</span>
            ))}
            <span><span className="dot" style={{ background: "#2ec7a8" }} /> paragraph box</span>
          </div>
          <canvas ref={canvasRef} />
          <p className="muted">
            {result.corners.length} corners · {result.paragraphs.length} reconstructed paragraph(s)
          </p>
        </>
      )}
    </div>
  );
}
