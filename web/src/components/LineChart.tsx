interface Series {
  label: string;
  color: string;
  points: (number | null | undefined)[];
}

interface Props {
  series: Series[];
  xs: number[];
  height?: number;
  yLabel?: string;
}

// Minimal dependency-free SVG line chart for training curves.
export default function LineChart({ series, xs, height = 200, yLabel }: Props) {
  const width = 560;
  const pad = { l: 44, r: 12, t: 12, b: 26 };
  const iw = width - pad.l - pad.r;
  const ih = height - pad.t - pad.b;

  const all = series.flatMap((s) => s.points.filter((v): v is number => v != null));
  const yMin = all.length ? Math.min(...all) : 0;
  const yMax = all.length ? Math.max(...all) : 1;
  const span = yMax - yMin || 1;
  const xMax = Math.max(xs.length - 1, 1);

  const px = (i: number) => pad.l + (i / xMax) * iw;
  const py = (v: number) => pad.t + ih - ((v - yMin) / span) * ih;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => yMin + f * span);

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img">
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={pad.l} x2={width - pad.r} y1={py(t)} y2={py(t)} stroke="#2b3542" strokeWidth={1} />
          <text x={pad.l - 6} y={py(t) + 4} textAnchor="end" fontSize={10} fill="#8a97a8">
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {series.map((s) => {
        const d = s.points
          .map((v, i) => (v == null ? null : `${px(i)},${py(v)}`))
          .filter(Boolean)
          .join(" ");
        return <polyline key={s.label} points={d} fill="none" stroke={s.color} strokeWidth={2} />;
      })}
      {xs.length > 0 && (
        <text x={pad.l} y={height - 6} fontSize={10} fill="#8a97a8">
          epoch 1 → {xs[xs.length - 1]}
        </text>
      )}
      {yLabel && (
        <text x={12} y={pad.t + 4} fontSize={10} fill="#8a97a8">
          {yLabel}
        </text>
      )}
    </svg>
  );
}
