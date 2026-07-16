// Reusable editor for everything that defines a network + its training run:
// input size, the conv backbone, the head, and the training hyperparameters.
// Shared by the Train panel (fresh model) and the Runs panel (view / edit the
// frozen config of a trained run before retraining it).

export interface Block {
  filters: number;
  kernel: number;
  stride: number;
  pool: number;
  batchnorm: boolean;
  dropout: number;
  activation: string;
}

export const defaultBlock = (): Block => ({
  filters: 32, kernel: 3, stride: 1, pool: 2, batchnorm: true, dropout: 0, activation: "relu",
});

export interface ModelForm {
  input_size: number;
  borderFeatures: boolean; // feed per-patch edge flags into the head
  blocks: Block[];
  headHidden: string; // comma-separated hidden sizes
  headDropout: number;
  epochs: number;
  batch_size: number;
  lr: number;
  optimizer: string;
  lambda_pos: number;
  device: string;
}

export const defaultModelForm = (): ModelForm => ({
  input_size: 40,
  borderFeatures: true,
  blocks: [
    { ...defaultBlock(), filters: 32 },
    { ...defaultBlock(), filters: 64 },
    { ...defaultBlock(), filters: 128, dropout: 0.1 },
  ],
  headHidden: "128",
  headDropout: 0.3,
  epochs: 20, batch_size: 64, lr: 0.001, optimizer: "adam", lambda_pos: 1.0, device: "cpu",
});

/** Build the API `model` object (architecture) from the form. */
export function modelFormToModel(f: ModelForm) {
  return {
    input_size: Number(f.input_size),
    in_channels: 1,
    border_features: f.borderFeatures,
    backbone: f.blocks.map((b) => ({
      filters: Number(b.filters), kernel: Number(b.kernel), stride: Number(b.stride),
      batchnorm: b.batchnorm, activation: b.activation,
      ...(Number(b.pool) > 1 ? { pool: Number(b.pool) } : {}),
      ...(Number(b.dropout) > 0 ? { dropout: Number(b.dropout) } : {}),
    })),
    head: {
      hidden: f.headHidden.split(",").map((s) => Number(s.trim())).filter((n) => n > 0),
      dropout: Number(f.headDropout),
    },
  };
}

/** Training hyperparameters from the form (as the API expects them). */
export function modelFormHyper(f: ModelForm) {
  return {
    epochs: Number(f.epochs), batch_size: Number(f.batch_size), lr: Number(f.lr),
    optimizer: f.optimizer, lambda_pos: Number(f.lambda_pos), device: f.device,
  };
}

/** Parse a run's frozen config.json back into an editable form. */
export function configToModelForm(config: Record<string, any> | null | undefined): ModelForm {
  const d = defaultModelForm();
  if (!config) return d;
  const m = config.model ?? {};
  const bb = (m.backbone ?? []) as any[];
  return {
    input_size: m.input_size ?? d.input_size,
    borderFeatures: m.border_features ?? false,
    blocks: bb.length
      ? bb.map((b) => ({
          filters: b.filters ?? 32, kernel: b.kernel ?? 3, stride: b.stride ?? 1,
          pool: b.pool ?? 1, batchnorm: b.batchnorm ?? false,
          dropout: b.dropout ?? 0, activation: b.activation ?? "relu",
        }))
      : d.blocks,
    headHidden: (m.head?.hidden ?? []).join(","),
    headDropout: m.head?.dropout ?? 0,
    epochs: config.epochs ?? d.epochs,
    batch_size: config.batch_size ?? d.batch_size,
    lr: config.lr ?? d.lr,
    optimizer: config.optimizer ?? d.optimizer,
    lambda_pos: config.lambda_pos ?? d.lambda_pos,
    device: config.device ?? d.device,
  };
}

export default function ModelConfigForm({
  value,
  onChange,
  readOnly = false,
  lockArchitecture = false,
}: {
  value: ModelForm;
  onChange: (v: ModelForm) => void;
  readOnly?: boolean;
  /** Freeze the parts that define the network itself (input size, backbone,
   *  head) while leaving the training hyperparameters editable — used when
   *  retraining "the same network" on new data / with new hyperparameters. */
  lockArchitecture?: boolean;
}) {
  const set = (patch: Partial<ModelForm>) => onChange({ ...value, ...patch });
  const editBlock = (i: number, patch: Partial<Block>) =>
    set({ blocks: value.blocks.map((b, j) => (j === i ? { ...b, ...patch } : b)) });
  const ro = readOnly;                       // training hyperparameters
  const archRo = readOnly || lockArchitecture; // architecture (network identity)
  const lockedNote = lockArchitecture && !readOnly ? " — fixed (defines the network)" : "";

  return (
    <>
      <div className="row">
        <div className="field">
          <label>input_size (n){lockedNote}</label>
          <input type="number" value={value.input_size} disabled={archRo}
                 onChange={(e) => set({ input_size: Number(e.target.value) })} />
        </div>
        <div className="field">
          <label>border features{lockedNote}</label>
          <label className="checkbox">
            <input type="checkbox" checked={value.borderFeatures} disabled={archRo}
                   onChange={(e) => set({ borderFeatures: e.target.checked })} />
            <span>feed per-patch edge flags (top/right/bottom/left) into the head</span>
          </label>
        </div>
      </div>

      <h3>Backbone (conv blocks){lockedNote}</h3>
      {value.blocks.map((b, i) => (
        <div className="block-row" key={i}>
          <div>
            <label>filters</label>
            <input type="number" value={b.filters} disabled={archRo} onChange={(e) => editBlock(i, { filters: Number(e.target.value) })} />
          </div>
          <div>
            <label>kernel</label>
            <input type="number" value={b.kernel} disabled={archRo} onChange={(e) => editBlock(i, { kernel: Number(e.target.value) })} />
          </div>
          <div>
            <label>stride</label>
            <input type="number" value={b.stride} disabled={archRo} onChange={(e) => editBlock(i, { stride: Number(e.target.value) })} />
          </div>
          <div>
            <label>pool</label>
            <input type="number" value={b.pool} disabled={archRo} onChange={(e) => editBlock(i, { pool: Number(e.target.value) })} />
          </div>
          <div>
            <label>dropout</label>
            <input type="number" step="0.05" value={b.dropout} disabled={archRo} onChange={(e) => editBlock(i, { dropout: Number(e.target.value) })} />
          </div>
          <div>
            <label>act</label>
            <select value={b.activation} disabled={archRo} onChange={(e) => editBlock(i, { activation: e.target.value })}>
              {["relu", "leaky_relu", "gelu", "elu", "tanh"].map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </div>
          <button className="btn ghost" disabled={archRo}
                  onClick={() => set({ blocks: value.blocks.filter((_, j) => j !== i) })}>✕</button>
        </div>
      ))}
      {!archRo && (
        <button className="btn ghost" onClick={() => set({ blocks: [...value.blocks, defaultBlock()] })}>+ block</button>
      )}

      <h3>Head{lockedNote}</h3>
      <div className="row">
        <div className="field">
          <label>hidden (comma-sep)</label>
          <input value={value.headHidden} disabled={archRo} onChange={(e) => set({ headHidden: e.target.value })} />
        </div>
        <div className="field">
          <label>dropout</label>
          <input type="number" step="0.05" value={value.headDropout} disabled={archRo} onChange={(e) => set({ headDropout: Number(e.target.value) })} />
        </div>
      </div>

      <h3>Training</h3>
      <div className="row">
        <div className="field">
          <label>epochs</label>
          <input type="number" value={value.epochs} disabled={ro} onChange={(e) => set({ epochs: Number(e.target.value) })} />
        </div>
        <div className="field">
          <label>batch_size</label>
          <input type="number" value={value.batch_size} disabled={ro} onChange={(e) => set({ batch_size: Number(e.target.value) })} />
        </div>
        <div className="field">
          <label>lr</label>
          <input type="number" step="0.0001" value={value.lr} disabled={ro} onChange={(e) => set({ lr: Number(e.target.value) })} />
        </div>
        <div className="field">
          <label>optimizer</label>
          <select value={value.optimizer} disabled={ro} onChange={(e) => set({ optimizer: e.target.value })}>
            {["adam", "adamw", "sgd", "rmsprop"].map((o) => (
              <option key={o}>{o}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>λ position</label>
          <input type="number" step="0.1" value={value.lambda_pos} disabled={ro} onChange={(e) => set({ lambda_pos: Number(e.target.value) })} />
        </div>
        <div className="field">
          <label>device</label>
          <select value={value.device} disabled={ro} onChange={(e) => set({ device: e.target.value })}>
            {["cpu", "cuda"].map((dv) => (
              <option key={dv}>{dv}</option>
            ))}
          </select>
        </div>
      </div>
    </>
  );
}
