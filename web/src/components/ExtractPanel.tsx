import { useEffect, useState } from "react";
import { api, Dataset, Job, PatchDataset } from "../api";

export default function ExtractPanel() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [patchSets, setPatchSets] = useState<PatchDataset[]>([]);
  const [form, setForm] = useState({
    source: "",
    name: "",
    patch_size: 40,
    stride: 20,
    train: 0.8,
    val: 0.1,
    test: 0.1,
    seed: 1,
    drop_overlap: false,
  });
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    api.datasets().then(setDatasets).catch((e) => setError(String(e)));
    api.patchDatasets().then(setPatchSets).catch(() => {});
  };
  useEffect(refresh, []);

  useEffect(() => {
    if (!job || job.status === "done" || job.status === "error") return;
    const t = setInterval(async () => {
      const j = await api.job(job.id);
      setJob(j);
      if (j.status === "done") refresh();
    }, 800);
    return () => clearInterval(t);
  }, [job]);

  const submit = async () => {
    setError(null);
    try {
      const j = await api.buildPatches({
        source: form.source,
        name: form.name,
        patch_size: Number(form.patch_size),
        stride: Number(form.stride),
        drop_overlap: form.drop_overlap,
        seed: Number(form.seed),
        split: { train: Number(form.train), val: Number(form.val), test: Number(form.test) },
      });
      setJob(j);
    } catch (e) {
      setError(String(e));
    }
  };

  const set = (k: string, v: string | number | boolean) => setForm((f) => ({ ...f, [k]: v }));

  return (
    <>
      <div className="card">
        <h2>Build patch dataset</h2>
        <div className="field">
          <label>Source dataset</label>
          <select value={form.source} onChange={(e) => set("source", e.target.value)}>
            <option value="">— select —</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.id} {d.count != null ? `(${d.count})` : ""}
              </option>
            ))}
          </select>
        </div>
        <div className="row">
          <div className="field">
            <label>Output name</label>
            <input value={form.name} onChange={(e) => set("name", e.target.value)} placeholder="reducido-40" />
          </div>
          <div className="field">
            <label>patch_size (n)</label>
            <input type="number" value={form.patch_size} onChange={(e) => set("patch_size", e.target.value)} />
          </div>
          <div className="field">
            <label>stride</label>
            <input type="number" value={form.stride} onChange={(e) => set("stride", e.target.value)} />
          </div>
          <div className="field">
            <label>seed</label>
            <input type="number" value={form.seed} onChange={(e) => set("seed", e.target.value)} />
          </div>
        </div>
        <div className="row">
          <div className="field">
            <label>split train</label>
            <input type="number" step="0.05" value={form.train} onChange={(e) => set("train", e.target.value)} />
          </div>
          <div className="field">
            <label>split val</label>
            <input type="number" step="0.05" value={form.val} onChange={(e) => set("val", e.target.value)} />
          </div>
          <div className="field">
            <label>split test</label>
            <input type="number" step="0.05" value={form.test} onChange={(e) => set("test", e.target.value)} />
          </div>
        </div>
        <button className="btn" onClick={submit} disabled={!form.source || !form.name}>
          Build
        </button>
        {job && (
          <p>
            Job <span className="mono">{job.id}</span>{" "}
            <span className={`tag ${job.status}`}>{job.status}</span>
          </p>
        )}
        {job?.error && <p className="err">{job.error}</p>}
        {error && <p className="err">{error}</p>}
      </div>

      <div className="card">
        <h2>Built patch datasets</h2>
        <table>
          <thead>
            <tr>
              <th>name</th>
              <th>patches</th>
              <th>splits</th>
              <th>positives / corner</th>
            </tr>
          </thead>
          <tbody>
            {patchSets.map((p) => (
              <tr key={p.name}>
                <td className="mono">{p.name}</td>
                <td>{p.manifest?.num_patches ?? "—"}</td>
                <td className="muted">
                  {p.manifest ? Object.entries(p.manifest.patches_per_split).map(([k, v]) => `${k}:${v}`).join("  ") : "—"}
                </td>
                <td className="muted">
                  {p.manifest ? Object.entries(p.manifest.positives_per_corner).map(([k, v]) => `${k}:${v}`).join("  ") : "—"}
                </td>
              </tr>
            ))}
            {patchSets.length === 0 && (
              <tr>
                <td colSpan={4} className="muted">none yet</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}
