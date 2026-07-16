import { useEffect, useState } from "react";
import { api, Job, PatchDataset } from "../api";
import ModelConfigForm, {
  defaultModelForm,
  modelFormToModel,
  modelFormHyper,
  ModelForm,
} from "./ModelConfigForm";

export default function TrainPanel({ onStarted }: { onStarted: () => void }) {
  const [patchSets, setPatchSets] = useState<PatchDataset[]>([]);
  const [data, setData] = useState("");
  const [name, setName] = useState("");
  const [form, setForm] = useState<ModelForm>(defaultModelForm());
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.patchDatasets().then((ps) => {
      setPatchSets(ps);
      if (ps[0]) {
        setData(ps[0].name);
        const n = (ps[0].manifest?.config as any)?.patch_size;
        if (n) setForm((f) => ({ ...f, input_size: n }));
      }
    });
  }, []);

  const start = async () => {
    setError(null);
    try {
      const j = await api.startRun({
        data, name,
        model: modelFormToModel(form),
        ...modelFormHyper(form),
      });
      setJob(j);
      onStarted();
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <div className="card">
      <h2>Configure & train a model</h2>
      <div className="row">
        <div className="field">
          <label>Patch dataset</label>
          <select value={data} onChange={(e) => setData(e.target.value)}>
            {patchSets.map((p) => (
              <option key={p.name} value={p.name}>{p.name}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Run name</label>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="cnn-a" />
        </div>
      </div>

      <ModelConfigForm value={form} onChange={setForm} />

      <button className="btn" onClick={start} disabled={!data || !name}>Start training</button>
      {job && (
        <p>
          Started run <span className="mono">{String(job.meta.name)}</span>{" "}
          <span className={`tag ${job.status}`}>{job.status}</span> — see the Runs tab for live metrics.
        </p>
      )}
      {error && <p className="err">{error}</p>}
    </div>
  );
}
