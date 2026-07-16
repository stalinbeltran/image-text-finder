import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  Dataset,
  imageUrl,
  PredictResult,
  RunSummary,
  SampleItem,
} from "../api";

const CORNER_COLORS: Record<string, string> = {
  TL: "#4c8dff", TR: "#2ec7a8", BR: "#ffbf47", BL: "#ff6b6b",
};

type Mode = "dataset" | "folder" | "upload";
type Split = "all" | "train" | "val" | "test";

interface GridItem {
  path: string;
  name: string;
  split?: string | null;
}

export default function PredictPanel() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [run, setRun] = useState("");
  const [checkpoint, setCheckpoint] = useState("best");
  const [threshold, setThreshold] = useState(0.5);
  const [error, setError] = useState<string | null>(null);

  const [mode, setMode] = useState<Mode>("dataset");

  // dataset / subset source
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [dataset, setDataset] = useState("");
  const [patchDataset, setPatchDataset] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [splitCounts, setSplitCounts] = useState<Record<string, number> | null>(null);
  const [split, setSplit] = useState<Split>("all");

  // folder source
  const [folderPath, setFolderPath] = useState("");
  const [folderItems, setFolderItems] = useState<GridItem[]>([]);

  // upload source
  const [file, setFile] = useState<File | null>(null);

  // selection + results
  const [selected, setSelected] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, PredictResult>>({});
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(false);
  const [batch, setBatch] = useState<{ done: number; total: number } | null>(null);
  const cancelRef = useRef(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // result popup
  const [modalOpen, setModalOpen] = useState(false);
  const [modalUrl, setModalUrl] = useState<string | null>(null);
  const [modalTitle, setModalTitle] = useState("");

  // --- load runs + datasets once -----------------------------------------
  useEffect(() => {
    api.runs().then((rs) => {
      const done = rs.filter((r) => r.status === "done");
      setRuns(done);
      if (done[0]) setRun(done[0].name);
    });
    api.datasets().then(setDatasets).catch(() => {});
  }, []);

  // --- when the run changes, seed dataset + patch dataset from its config -
  useEffect(() => {
    if (!run) return;
    api.runSource(run).then((src) => {
      setPatchDataset(src.patch_dataset);
      if (src.source) setDataset(src.source);
      else setDataset((d) => d || datasets[0]?.id || "");
    }).catch(() => {
      setDataset((d) => d || datasets[0]?.id || "");
    });
  }, [run]);

  // --- load samples when dataset / patch dataset changes -----------------
  useEffect(() => {
    if (mode !== "dataset" || !dataset) return;
    setLoading(true);
    api.datasetSamples(dataset, patchDataset ?? undefined)
      .then((r) => { setSamples(r.samples); setSplitCounts(r.splits); })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [mode, dataset, patchDataset]);

  // --- results depend on run/checkpoint/threshold: invalidate on change --
  useEffect(() => {
    setResults({});
    setSelected(null);
    setModalOpen(false);
  }, [run, checkpoint, threshold]);

  // --- close popup on Escape ---------------------------------------------
  useEffect(() => {
    if (!modalOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setModalOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [modalOpen]);

  const items: GridItem[] = useMemo(() => {
    if (mode === "folder") return folderItems;
    if (mode === "dataset") {
      const s = split === "all" ? samples : samples.filter((x) => x.split === split);
      return s.map((x) => ({ path: x.path, name: x.name, split: x.split }));
    }
    return [];
  }, [mode, folderItems, samples, split]);

  const loadFolder = async () => {
    if (!folderPath) return;
    setError(null);
    setLoading(true);
    try {
      const r = await api.folder(folderPath);
      setFolderItems(r.images.map((im) => ({ path: im.path, name: im.name })));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  // --- drawing -----------------------------------------------------------
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

  // --- predict one server-side image -------------------------------------
  const predictOne = async (path: string): Promise<PredictResult | null> => {
    if (results[path]) return results[path];
    setBusy((b) => ({ ...b, [path]: true }));
    try {
      const res = await api.predictPath({ path, run, checkpoint, threshold });
      setResults((r) => ({ ...r, [path]: res }));
      return res;
    } catch (e) {
      setError(String(e));
      return null;
    } finally {
      setBusy((b) => ({ ...b, [path]: false }));
    }
  };

  const openItem = async (path: string) => {
    setSelected(path);
    setUploadRes(null);
    setError(null);
    setModalTitle(path.split(/[\\/]/).pop() ?? path);
    setModalUrl(imageUrl(path));
    setModalOpen(true);
    if (!results[path]) await predictOne(path);
  };

  const predictAll = async () => {
    if (!run || items.length === 0) return;
    cancelRef.current = false;
    setError(null);
    const todo = items.filter((it) => !results[it.path]);
    setBatch({ done: 0, total: todo.length });
    for (let i = 0; i < todo.length; i++) {
      if (cancelRef.current) break;
      await predictOne(todo[i].path);
      setBatch({ done: i + 1, total: todo.length });
    }
    setBatch(null);
  };

  // --- upload (single ad-hoc image) --------------------------------------
  const [uploadRes, setUploadRes] = useState<PredictResult | null>(null);
  const [uploadBusy, setUploadBusy] = useState(false);
  const predictUpload = async () => {
    if (!file || !run) return;
    setUploadBusy(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("run", run);
      form.append("checkpoint", checkpoint);
      form.append("threshold", String(threshold));
      form.append("file", file);
      const res = await api.predict(form);
      setSelected(null);
      setUploadRes(res);
      setModalTitle(file.name);
      setModalUrl(URL.createObjectURL(file));
      setModalOpen(true);
    } catch (e) {
      setError(String(e));
    } finally {
      setUploadBusy(false);
    }
  };

  const selRes = selected ? results[selected] ?? null : uploadRes;

  // --- (re)draw the popup canvas whenever it opens or the result arrives -
  useEffect(() => {
    if (!modalOpen || !modalUrl) return;
    const img = new Image();
    img.onload = () => draw(img, selRes);
    img.src = modalUrl;
  }, [modalOpen, modalUrl, selRes]);

  const modalBusy = selected ? busy[selected] : uploadBusy;

  return (
    <div className="card">
      <h2>Inference preview</h2>

      {/* model + threshold controls */}
      <div className="row">
        <div className="field">
          <label>Run (trained)</label>
          <select value={run} onChange={(e) => setRun(e.target.value)}>
            {runs.map((r) => (
              <option key={r.name} value={r.name}>{r.name}</option>
            ))}
            {runs.length === 0 && <option value="">no trained runs</option>}
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
      </div>

      {/* source picker */}
      <div className="chips">
        {(["dataset", "folder", "upload"] as Mode[]).map((mo) => (
          <button key={mo} className={`chip ${mode === mo ? "active" : ""}`} onClick={() => setMode(mo)}>
            {mo === "dataset" ? "Dataset / subset" : mo === "folder" ? "Folder" : "Single upload"}
          </button>
        ))}
      </div>

      {mode === "dataset" && (
        <>
          <div className="row">
            <div className="field">
              <label>Source dataset</label>
              <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>{d.id}</option>
                ))}
                {datasets.length === 0 && <option value="">no datasets</option>}
              </select>
            </div>
            <div className="field">
              <label>Splits from patch dataset</label>
              <input className="mono" value={patchDataset ?? ""} placeholder="(none — all samples)"
                     onChange={(e) => setPatchDataset(e.target.value || null)} />
            </div>
          </div>
          <div className="chips">
            {(["all", "train", "val", "test"] as Split[]).map((s) => {
              const n = s === "all" ? samples.length : splitCounts?.[s];
              const disabled = s !== "all" && !splitCounts?.[s];
              return (
                <button key={s} className={`chip ${split === s ? "active" : ""}`}
                        disabled={disabled}
                        onClick={() => setSplit(s)}>
                  {s}{n != null ? ` (${n})` : ""}
                </button>
              );
            })}
          </div>
        </>
      )}

      {mode === "folder" && (
        <div className="row">
          <div className="field" style={{ gridColumn: "1 / -1" }}>
            <label>Folder path (on the server)</label>
            <input className="mono" value={folderPath} placeholder="C:\\path\\to\\images"
                   onChange={(e) => setFolderPath(e.target.value)}
                   onKeyDown={(e) => e.key === "Enter" && loadFolder()} />
          </div>
          <div className="field" style={{ alignSelf: "end" }}>
            <button className="btn ghost" onClick={loadFolder} disabled={!folderPath || loading}>Load folder</button>
          </div>
        </div>
      )}

      {mode === "upload" && (
        <div className="row">
          <div className="field">
            <label>Image</label>
            <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </div>
          <div className="field" style={{ alignSelf: "end" }}>
            <button className="btn" onClick={predictUpload} disabled={!file || !run || uploadBusy}>
              {uploadBusy ? "Predicting…" : "Predict"}
            </button>
          </div>
        </div>
      )}

      {error && <p className="err">{error}</p>}

      {/* batch controls + thumbnail grid */}
      {mode !== "upload" && (
        <>
          <div style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 8, flexWrap: "wrap" }}>
            <button className="btn" onClick={predictAll} disabled={!run || items.length === 0 || batch != null}>
              {batch ? `Predicting… ${batch.done}/${batch.total}` : `Predict all (${items.length})`}
            </button>
            {batch && <button className="btn ghost" onClick={() => (cancelRef.current = true)}>Stop</button>}
            <span className="muted">{loading ? "loading…" : `${items.length} image(s)`}</span>
          </div>

          <div className="thumbs">
            {items.map((it) => {
              const res = results[it.path];
              return (
                <div key={it.path}
                     className={`thumb ${selected === it.path ? "selected" : ""}`}
                     onClick={() => openItem(it.path)}
                     title={it.name}>
                  {it.split && <span className="split-badge">{it.split}</span>}
                  {busy[it.path]
                    ? <span className="badge busy">…</span>
                    : res && <span className="badge done">{res.paragraphs.length}¶</span>}
                  <img src={imageUrl(it.path, 220)} loading="lazy" alt={it.name} />
                  <div className="cap">{it.name}</div>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* result popup */}
      {modalOpen && (
        <div className="modal-backdrop" onClick={() => setModalOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3 className="mono">{modalTitle}</h3>
              <button className="modal-close" onClick={() => setModalOpen(false)} title="Close (Esc)">✕</button>
            </div>
            <div className="legend" style={{ marginBottom: 8 }}>
              {Object.entries(CORNER_COLORS).map(([k, v]) => (
                <span key={k}><span className="dot" style={{ background: v }} /> {k}</span>
              ))}
              <span><span className="dot" style={{ background: "#2ec7a8" }} /> paragraph box</span>
            </div>
            <div className="modal-body">
              <canvas ref={canvasRef} />
            </div>
            <p className="muted">
              {modalBusy
                ? "Predicting…"
                : selRes
                  ? `${selRes.corners.length} corners · ${selRes.paragraphs.length} reconstructed paragraph(s)`
                  : "no result"}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
