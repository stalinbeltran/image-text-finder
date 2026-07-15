import { useEffect, useState } from "react";
import { api, Job, PatchDataset } from "../api";

interface Block {
  filters: number;
  kernel: number;
  stride: number;
  pool: number;
  batchnorm: boolean;
  dropout: number;
  activation: string;
}

const defaultBlock = (): Block => ({
  filters: 32, kernel: 3, stride: 1, pool: 2, batchnorm: true, dropout: 0, activation: "relu",
});

export default function TrainPanel({ onStarted }: { onStarted: () => void }) {
  const [patchSets, setPatchSets] = useState<PatchDataset[]>([]);
  const [data, setData] = useState("");
  const [name, setName] = useState("");
  const [inputSize, setInputSize] = useState(40);
  const [blocks, setBlocks] = useState<Block[]>([
    { ...defaultBlock(), filters: 32 },
    { ...defaultBlock(), filters: 64 },
    { ...defaultBlock(), filters: 128, dropout: 0.1 },
  ]);
  const [headHidden, setHeadHidden] = useState("128");
  const [headDropout, setHeadDropout] = useState(0.3);
  const [hyper, setHyper] = useState({
    epochs: 20, batch_size: 64, lr: 0.001, optimizer: "adam", lambda_pos: 1.0, device: "cpu",
  });
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.patchDatasets().then((ps) => {
      setPatchSets(ps);
      if (ps[0]) {
        setData(ps[0].name);
        const n = (ps[0].manifest?.config as any)?.patch_size;
        if (n) setInputSize(n);
      }
    });
  }, []);

  const editBlock = (i: number, patch: Partial<Block>) =>
    setBlocks((bs) => bs.map((b, j) => (j === i ? { ...b, ...patch } : b)));

  const start = async () => {
    setError(null);
    try {
      const model = {
        input_size: Number(inputSize),
        in_channels: 1,
        backbone: blocks.map((b) => ({
          filters: Number(b.filters), kernel: Number(b.kernel), stride: Number(b.stride),
          batchnorm: b.batchnorm, activation: b.activation,
          ...(Number(b.pool) > 1 ? { pool: Number(b.pool) } : {}),
          ...(Number(b.dropout) > 0 ? { dropout: Number(b.dropout) } : {}),
        })),
        head: {
          hidden: headHidden.split(",").map((s) => Number(s.trim())).filter((n) => n > 0),
          dropout: Number(headDropout),
        },
      };
      const j = await api.startRun({
        data, name, model,
        epochs: Number(hyper.epochs), batch_size: Number(hyper.batch_size),
        lr: Number(hyper.lr), optimizer: hyper.optimizer,
        lambda_pos: Number(hyper.lambda_pos), device: hyper.device,
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
        <div className="field">
          <label>input_size (n)</label>
          <input type="number" value={inputSize} onChange={(e) => setInputSize(Number(e.target.value))} />
        </div>
      </div>

      <h3>Backbone (conv blocks)</h3>
      {blocks.map((b, i) => (
        <div className="block-row" key={i}>
          <div>
            <label>filters</label>
            <input type="number" value={b.filters} onChange={(e) => editBlock(i, { filters: Number(e.target.value) })} />
          </div>
          <div>
            <label>kernel</label>
            <input type="number" value={b.kernel} onChange={(e) => editBlock(i, { kernel: Number(e.target.value) })} />
          </div>
          <div>
            <label>stride</label>
            <input type="number" value={b.stride} onChange={(e) => editBlock(i, { stride: Number(e.target.value) })} />
          </div>
          <div>
            <label>pool</label>
            <input type="number" value={b.pool} onChange={(e) => editBlock(i, { pool: Number(e.target.value) })} />
          </div>
          <div>
            <label>dropout</label>
            <input type="number" step="0.05" value={b.dropout} onChange={(e) => editBlock(i, { dropout: Number(e.target.value) })} />
          </div>
          <div>
            <label>act</label>
            <select value={b.activation} onChange={(e) => editBlock(i, { activation: e.target.value })}>
              {["relu", "leaky_relu", "gelu", "elu", "tanh"].map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </div>
          <button className="btn ghost" onClick={() => setBlocks((bs) => bs.filter((_, j) => j !== i))}>✕</button>
        </div>
      ))}
      <button className="btn ghost" onClick={() => setBlocks((bs) => [...bs, defaultBlock()])}>+ block</button>

      <h3>Head</h3>
      <div className="row">
        <div className="field">
          <label>hidden (comma-sep)</label>
          <input value={headHidden} onChange={(e) => setHeadHidden(e.target.value)} />
        </div>
        <div className="field">
          <label>dropout</label>
          <input type="number" step="0.05" value={headDropout} onChange={(e) => setHeadDropout(Number(e.target.value))} />
        </div>
      </div>

      <h3>Training</h3>
      <div className="row">
        <div className="field">
          <label>epochs</label>
          <input type="number" value={hyper.epochs} onChange={(e) => setHyper({ ...hyper, epochs: Number(e.target.value) })} />
        </div>
        <div className="field">
          <label>batch_size</label>
          <input type="number" value={hyper.batch_size} onChange={(e) => setHyper({ ...hyper, batch_size: Number(e.target.value) })} />
        </div>
        <div className="field">
          <label>lr</label>
          <input type="number" step="0.0001" value={hyper.lr} onChange={(e) => setHyper({ ...hyper, lr: Number(e.target.value) })} />
        </div>
        <div className="field">
          <label>optimizer</label>
          <select value={hyper.optimizer} onChange={(e) => setHyper({ ...hyper, optimizer: e.target.value })}>
            {["adam", "adamw", "sgd", "rmsprop"].map((o) => (
              <option key={o}>{o}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>λ position</label>
          <input type="number" step="0.1" value={hyper.lambda_pos} onChange={(e) => setHyper({ ...hyper, lambda_pos: Number(e.target.value) })} />
        </div>
        <div className="field">
          <label>device</label>
          <select value={hyper.device} onChange={(e) => setHyper({ ...hyper, device: e.target.value })}>
            {["cpu", "cuda"].map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </div>
      </div>

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
