/**
 * Validate the palette of src/theme/tokens.css. Run it; never eyeball it.
 *
 *   npm run validate:palette
 *
 * ui.md §4.0 R6 and plan-ui.md fase 1.2: the palette is passed through the
 * colourblindness validator, in light AND in dark, and corrected until it passes.
 * tests.md §6 puts it outside pytest on purpose -- the UI's shape and colour
 * rules are reviewed, not tested, "salvo la paleta, que sí tiene un validador
 * ejecutable y hay que correrlo". This is that script.
 *
 * It parses tokens.css rather than re-declaring the hexes, so what it checks is
 * what actually ships. A second copy of the palette here would be a copy that
 * drifts, and a validator validating a stale copy is worse than none.
 *
 * What it checks:
 *
 *   0. The two dark blocks agree      -- tokens.css declares dark twice (media
 *                                        query + [data-theme] toggle) and they
 *                                        must not drift.
 *   1. Categorical, per mode          -- lightness band, chroma floor, CVD
 *                                        separation (Machado-Oliveira-Fernandes
 *                                        2009, severity 1.0, protan+deutan),
 *                                        normal-vision floor, contrast vs surface.
 *                                        ALL PAIRS, not adjacent: see tokens.css.
 *   2. Sequential ramp, per mode      -- lightness strictly monotone. A ramp whose
 *                                        lightness is not monotone does not encode
 *                                        magnitude, whatever it looks like.
 *   3. Diverging ramp, per mode       -- the midpoint is NEUTRAL (chroma below the
 *                                        floor: a tint at 0 hides the sign
 *                                        structure, R2), the two poles are
 *                                        distinguishable under CVD, and the arms
 *                                        are balanced in lightness.
 *
 * Exit 0 if every hard gate passes; 1 otherwise. WARNs are legal only where the
 * design already commits to the mitigation -- and here it does: R1 makes direct
 * labelling mandatory ("no cortesía") and R5 makes the number table the accessible
 * twin of every heatmap. Both are exactly the secondary encoding a WARN requires.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const TOKENS = join(dirname(fileURLToPath(import.meta.url)), "..", "src", "theme", "tokens.css");

// ── thresholds (from the documented method; the sim model is part of the standard) ──
const BAND = { light: [0.43, 0.77], dark: [0.48, 0.67] }; // OKLCH L
const CHROMA_FLOOR = 0.1;
const CVD_TARGET = 8.0;
const CVD_FLOOR = 6.0; // 6-8 is legal ONLY with secondary encoding
const NORMAL_FLOOR = 15.0; // hard gate
const CONTRAST_MIN = 3.0;

// Machado, Oliveira & Fernandes (2009), severity 1.0, linear RGB.
const MACHADO = {
  protan: [
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281, 0.099216],
    [-0.003882, -0.048116, 1.051998],
  ],
  deutan: [
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501, 0.047413],
    [-0.01182, 0.04294, 0.968881],
  ],
};

// ── colour maths ──────────────────────────────────────────────────────────────
const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

function parseHex(hex) {
  const h = hex.trim().replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
}

function oklab(hex) {
  const [r, g, b] = parseHex(hex).map(srgbToLinear);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

const lightness = (hex) => oklab(hex)[0];
const chroma = (hex) => {
  const [, a, b] = oklab(hex);
  return Math.hypot(a, b);
};

function simulate(hex, kind) {
  const lin = parseHex(hex).map(srgbToLinear);
  const m = MACHADO[kind];
  const out = m.map((row) => row.reduce((acc, k, i) => acc + k * lin[i], 0));
  const toHex = (v) => {
    const c = Math.max(0, Math.min(1, v));
    const s = c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
    return Math.round(s * 255)
      .toString(16)
      .padStart(2, "0");
  };
  return "#" + out.map(toHex).join("");
}

function deltaE(a, b, kind) {
  const [x, y] = kind ? [simulate(a, kind), simulate(b, kind)] : [a, b];
  const [p, q] = [oklab(x), oklab(y)];
  return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) * 100;
}

function relLum(hex) {
  const [r, g, b] = parseHex(hex).map(srgbToLinear);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a, b) {
  const [hi, lo] = [relLum(a), relLum(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

// ── parsing tokens.css ────────────────────────────────────────────────────────
function block(css, name) {
  const marker = new RegExp(`/\\* @palette ${name} \\*/([\\s\\S]*?)/\\* @end \\*/`);
  const found = css.match(marker);
  if (!found) throw new Error(`tokens.css: no encuentro el bloque "@palette ${name}"`);
  const vars = {};
  for (const [, key, value] of found[1].matchAll(/(--itf-[\w-]+)\s*:\s*([^;]+);/g)) {
    vars[key] = value.trim();
  }
  return vars;
}

const hexes = (vars, prefix, count) =>
  Array.from({ length: count }, (_, i) => vars[`${prefix}${i}`]).filter(Boolean);

// ── the report ────────────────────────────────────────────────────────────────
let failed = false;
const results = [];

function record(scope, name, state, detail) {
  if (state === "fail") failed = true;
  results.push({ scope, name, state, detail });
}

function checkCategorical(mode, vars) {
  const surface = vars["--itf-surface"];
  const slots = [
    ["TL", vars["--itf-corner-tl"]],
    ["TR", vars["--itf-corner-tr"]],
    ["BR", vars["--itf-corner-br"]],
    ["BL", vars["--itf-corner-bl"]],
  ];
  const [lo, hi] = BAND[mode];

  const offband = slots.filter(([, c]) => lightness(c) < lo || lightness(c) > hi);
  record(
    mode,
    "banda de luminosidad",
    offband.length ? "fail" : "pass",
    offband.length
      ? offband.map(([n, c]) => `${n} ${c} L=${lightness(c).toFixed(3)}`).join(", ")
      : `los 4 dentro de L ${lo}–${hi}`
  );

  const gray = slots.filter(([, c]) => chroma(c) < CHROMA_FLOOR);
  record(
    mode,
    "suelo de croma",
    gray.length ? "fail" : "pass",
    gray.length ? gray.map(([n, c]) => `${n} ${c} C=${chroma(c).toFixed(3)}`).join(", ") : `los 4 >= ${CHROMA_FLOOR}`
  );

  // ALL pairs: any two corners can touch (V3, V9, V11).
  const pairs = [];
  for (let i = 0; i < slots.length; i++)
    for (let j = i + 1; j < slots.length; j++) pairs.push([slots[i], slots[j]]);

  let worst = null;
  for (const kind of ["protan", "deutan"]) {
    for (const [[na, ca], [nb, cb]] of pairs) {
      const d = deltaE(ca, cb, kind);
      if (!worst || d < worst.d) worst = { d, kind, pair: `${na}↔${nb}` };
    }
  }
  record(
    mode,
    "separación CVD (all-pairs)",
    worst.d >= CVD_TARGET ? "pass" : worst.d >= CVD_FLOOR ? "warn" : "fail",
    `peor ${worst.pair} ΔE ${worst.d.toFixed(1)} (${worst.kind})`
  );

  let nworst = null;
  for (const [[na, ca], [nb, cb]] of pairs) {
    const d = deltaE(ca, cb);
    if (!nworst || d < nworst.d) nworst = { d, pair: `${na}↔${nb}` };
  }
  record(
    mode,
    "suelo de visión normal",
    nworst.d >= NORMAL_FLOOR ? "pass" : "fail",
    `peor ${nworst.pair} ΔE ${nworst.d.toFixed(1)}`
  );

  const low = slots.filter(([, c]) => contrast(c, surface) < CONTRAST_MIN);
  record(
    mode,
    "contraste vs superficie",
    low.length ? "warn" : "pass",
    low.length
      ? low.map(([n, c]) => `${n} ${c} ${contrast(c, surface).toFixed(2)}:1`).join(", ")
      : `los 4 >= ${CONTRAST_MIN}:1`
  );
}

function checkSequential(mode, vars) {
  const ramp = hexes(vars, "--itf-seq-", 7);
  if (ramp.length !== 7) throw new Error(`${mode}: la rampa secuencial no tiene 7 pasos`);
  const ls = ramp.map(lightness);
  // Step 0 is "near zero" and recedes into the surface: light surface => ramp
  // gets darker; dark surface => ramp gets lighter. Either way, MONOTONE.
  const wantsDarker = mode === "light";
  const monotone = ls.every((l, i) => i === 0 || (wantsDarker ? l < ls[i - 1] : l > ls[i - 1]));
  record(
    mode,
    "rampa secuencial monótona",
    monotone ? "pass" : "fail",
    monotone
      ? `L ${ls[0].toFixed(2)} → ${ls[6].toFixed(2)} (${wantsDarker ? "clara→oscura" : "oscura→clara"})`
      : `L no monótona: ${ls.map((l) => l.toFixed(2)).join(" ")}`
  );
}

function checkDiverging(mode, vars) {
  const neg = vars["--itf-div-neg"];
  const mid = vars["--itf-div-mid"];
  const pos = vars["--itf-div-pos"];

  // R2: neutral grey at 0. A tint at the midpoint hides the sign structure.
  record(
    mode,
    "divergente: 0 neutro",
    chroma(mid) < CHROMA_FLOOR ? "pass" : "fail",
    `mid ${mid} C=${chroma(mid).toFixed(3)} (< ${CHROMA_FLOOR} = gris)`
  );

  let worst = null;
  for (const kind of ["protan", "deutan"]) {
    const d = deltaE(neg, pos, kind);
    if (!worst || d < worst.d) worst = { d, kind };
  }
  record(
    mode,
    "divergente: polos distinguibles",
    worst.d >= CVD_TARGET ? "pass" : worst.d >= CVD_FLOOR ? "warn" : "fail",
    `${neg}↔${pos} ΔE ${worst.d.toFixed(1)} (${worst.kind})`
  );

  // Symmetric range means neither arm may look "louder" than the other at the
  // same |w|, so their lightness must sit about the same distance from the mid.
  const dneg = Math.abs(lightness(neg) - lightness(mid));
  const dpos = Math.abs(lightness(pos) - lightness(mid));
  const skew = Math.abs(dneg - dpos);
  record(
    mode,
    "divergente: brazos equilibrados",
    skew <= 0.1 ? "pass" : "warn",
    `ΔL neg=${dneg.toFixed(2)} pos=${dpos.toFixed(2)} (sesgo ${skew.toFixed(2)})`
  );
}

// ── run ───────────────────────────────────────────────────────────────────────
const css = readFileSync(TOKENS, "utf8");
const light = block(css, "light");
const darkMedia = block(css, "dark-media");
const darkToggle = block(css, "dark-toggle");

// 0. The drift guard. tokens.css declares dark twice by necessity; nothing but
//    this check stops the two copies from diverging.
const drift = Object.keys({ ...darkMedia, ...darkToggle }).filter((k) => darkMedia[k] !== darkToggle[k]);
record(
  "estructura",
  "los dos bloques oscuros coinciden",
  drift.length ? "fail" : "pass",
  drift.length ? `divergen en: ${drift.join(", ")}` : `${Object.keys(darkMedia).length} tokens idénticos`
);

for (const [mode, vars] of [
  ["light", light],
  ["dark", darkMedia],
]) {
  checkCategorical(mode, vars);
  checkSequential(mode, vars);
  checkDiverging(mode, vars);
}

const ICON = { pass: "PASS", warn: "WARN", fail: "FAIL" };
const SURFACE = { light: light["--itf-surface"], dark: darkMedia["--itf-surface"] };
let scope = null;
for (const r of results) {
  if (r.scope !== scope) {
    scope = r.scope;
    console.log(`\n  ── ${scope}${SURFACE[scope] ? ` (superficie ${SURFACE[scope]})` : ""}`);
  }
  console.log(`  [${ICON[r.state]}] ${r.name.padEnd(30)} ${r.detail}`);
}

const warns = results.filter((r) => r.state === "warn");
console.log("");
if (failed) {
  console.log("  → FALLA. La paleta no se da por buena hasta que pase (ui.md §4.0 R6).\n");
  process.exit(1);
}
console.log("  → PASA en claro y en oscuro.");
if (warns.length) {
  console.log(
    "\n  Los WARN son legales SOLO con codificación secundaria, y el diseño ya la exige:\n" +
      "    · R1 — etiquetado directo obligatorio en los 4 tipos de esquina, «no cortesía».\n" +
      "    · R5 — la tabla de números es el equivalente accesible de todo mapa de calor.\n" +
      "  Si alguna vista se salta esas dos reglas, estos WARN pasan a ser fallos reales."
  );
}
console.log("");
