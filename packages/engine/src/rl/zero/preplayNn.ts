// The few dense-network operations the declaration and burial networks
// need, on plain arrays, accumulated in double precision. Weights are
// PyTorch's row-major layout: a Linear(in, out) weight is [out, in].

/** erf(x) to double precision: fdlibm's s_erf.c, as V8 does not expose one. */
export function erf(x: number): number {
  if (Number.isNaN(x)) return x;
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  if (a < 0.84375) {
    if (a < 3.7252902984e-9) return x + 1.2837916709551257e-1 * x;
    const z = x * x;
    const r = 1.28379167095512558561e-1 + z * (-3.25042107247001499370e-1 + z * (-2.84817495755985104766e-2 +
      z * (-5.77027029648944159157e-3 + z * -2.37630166566501626084e-5)));
    const s = 1 + z * (3.97917223959155352819e-1 + z * (6.50222499887672944485e-2 +
      z * (5.08130628187576562776e-3 + z * (1.32494738004321644526e-4 + z * -3.96022827877536812320e-6))));
    return x + x * (r / s);
  }
  if (a < 1.25) {
    const s = a - 1;
    const p = -2.36211856075265944077e-3 + s * (4.14856118683748331666e-1 + s * (-3.72207876035701323847e-1 +
      s * (3.18346619901161753674e-1 + s * (-1.10894694282396677476e-1 + s * (3.54783043256182359371e-2 +
      s * -2.16637559486879084300e-3)))));
    const q = 1 + s * (1.06420880400844228286e-1 + s * (5.40397917702171048937e-1 + s * (7.18286544141962662868e-2 +
      s * (1.26171219808761642112e-1 + s * (1.36370839120290507362e-2 + s * 1.19844998467991074170e-2)))));
    return sign * (8.45062911510467529297e-1 + p / q);
  }
  if (a >= 6) return sign * (1 - 1e-300);
  const s = 1 / (a * a);
  let r: number;
  let ss: number;
  if (a < 1 / 0.35) {
    r = -9.86494403484714822705e-3 + s * (-6.93858572707181764372e-1 + s * (-1.05586262253232909814e1 +
      s * (-6.23753324503260060396e1 + s * (-1.62396669462573470355e2 + s * (-1.84605092906711035994e2 +
      s * (-8.12874355063065934246e1 + s * -9.81432934416914548592))))));
    ss = 1 + s * (1.96512716674392571292e1 + s * (1.37657754143519042600e2 + s * (4.34565877475229228821e2 +
      s * (6.45387271733267880336e2 + s * (4.29008140027567833386e2 + s * (1.08635005541779435134e2 +
      s * (6.57024977031928170135 + s * -6.04244152148580987438e-2)))))));
  } else {
    r = -9.86494292470009928597e-3 + s * (-7.99283237680523006574e-1 + s * (-1.77579549177547519889e1 +
      s * (-1.60636384855821916062e2 + s * (-6.37566443368389627722e2 + s * (-1.02509513161107724954e3 +
      s * -4.83519191608651397019e2)))));
    ss = 1 + s * (3.03380607434824582924e1 + s * (3.25792512996573918826e2 + s * (1.53672958608443695994e3 +
      s * (3.19985821950859553908e3 + s * (2.55305040643316442583e3 + s * (4.74528541206955367215e2 +
      s * -2.24409524465858183362e1))))));
  }
  // z = a with the low 32 bits of its double cleared, as fdlibm does.
  const buf = new DataView(new ArrayBuffer(8));
  buf.setFloat64(0, a);
  buf.setUint32(4, 0);
  const z = buf.getFloat64(0);
  const e = Math.exp(-z * z - 0.5625) * Math.exp((z - a) * (z + a) + r / ss);
  return sign * (1 - e / a);
}

/** torch.nn.GELU() (approximate='none'). */
export function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

/** y[o] = b[o] + sum_i w[o, i] x[i]. */
export function linear(
  x: ArrayLike<number>,
  w: Float32Array,
  b: Float32Array | null,
  dIn: number,
  dOut: number,
): Float64Array {
  if (w.length !== dIn * dOut) throw new Error(`linear: weight ${w.length} != ${dOut}x${dIn}`);
  const y = new Float64Array(dOut);
  for (let o = 0; o < dOut; o++) {
    let acc = b === null ? 0 : b[o]!;
    const row = o * dIn;
    for (let i = 0; i < dIn; i++) acc += w[row + i]! * x[i]!;
    y[o] = acc;
  }
  return y;
}

/** The same Linear applied to each of `n` rows of `x` ([n, dIn] row-major). */
export function linearRows(
  x: Float64Array,
  n: number,
  w: Float32Array,
  b: Float32Array | null,
  dIn: number,
  dOut: number,
): Float64Array {
  if (w.length !== dIn * dOut) throw new Error(`linear: weight ${w.length} != ${dOut}x${dIn}`);
  const y = new Float64Array(n * dOut);
  for (let r = 0; r < n; r++) {
    const xr = r * dIn;
    const yr = r * dOut;
    for (let o = 0; o < dOut; o++) {
      let acc = b === null ? 0 : b[o]!;
      const row = o * dIn;
      for (let i = 0; i < dIn; i++) acc += w[row + i]! * x[xr + i]!;
      y[yr + o] = acc;
    }
  }
  return y;
}

/** torch.nn.LayerNorm over the last dimension of [n, d], in place. */
export function layerNormRows(
  x: Float64Array,
  n: number,
  d: number,
  gamma: Float32Array,
  beta: Float32Array,
  eps = 1e-5,
): Float64Array {
  const out = new Float64Array(n * d);
  for (let r = 0; r < n; r++) {
    const base = r * d;
    let mean = 0;
    for (let i = 0; i < d; i++) mean += x[base + i]!;
    mean /= d;
    let v = 0;
    for (let i = 0; i < d; i++) {
      const c = x[base + i]! - mean;
      v += c * c;
    }
    const inv = 1 / Math.sqrt(v / d + eps);
    for (let i = 0; i < d; i++) out[base + i] = (x[base + i]! - mean) * inv * gamma[i]! + beta[i]!;
  }
  return out;
}

/** torch.nn.MultiheadAttention self-attention over [n, d] with no mask:
 *  packed in_proj [3d, d], out_proj [d, d]. */
export function selfAttention(
  x: Float64Array,
  n: number,
  d: number,
  heads: number,
  inW: Float32Array,
  inB: Float32Array,
  outW: Float32Array,
  outB: Float32Array,
): Float64Array {
  const qkv = linearRows(x, n, inW, inB, d, 3 * d);
  const hd = d / heads;
  const scale = 1 / Math.sqrt(hd);
  const ctx = new Float64Array(n * d);
  const scores = new Float64Array(n);
  for (let h = 0; h < heads; h++) {
    const off = h * hd;
    for (let i = 0; i < n; i++) {
      const qi = i * 3 * d + off;
      let max = -Infinity;
      for (let j = 0; j < n; j++) {
        const kj = j * 3 * d + d + off;
        let s = 0;
        for (let c = 0; c < hd; c++) s += qkv[qi + c]! * qkv[kj + c]!;
        s *= scale;
        scores[j] = s;
        if (s > max) max = s;
      }
      let sum = 0;
      for (let j = 0; j < n; j++) {
        const e = Math.exp(scores[j]! - max);
        scores[j] = e;
        sum += e;
      }
      const oi = i * d + off;
      for (let j = 0; j < n; j++) {
        const p = scores[j]! / sum;
        const vj = j * 3 * d + 2 * d + off;
        for (let c = 0; c < hd; c++) ctx[oi + c] = ctx[oi + c]! + p * qkv[vj + c]!;
      }
    }
  }
  return linearRows(ctx, n, outW, outB, d, d);
}

/** Index of the largest `values[i]` among `allowed[i]`; the first on ties,
 *  as torch and numpy argmax. -1 when nothing is allowed. */
export function maskedArgmax(values: ArrayLike<number>, allowed: ArrayLike<boolean>): number {
  let best = -1;
  let bestV = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (allowed[i] !== true) continue;
    const v = values[i]!;
    if (best < 0 || v > bestV) {
      best = i;
      bestV = v;
    }
  }
  return best;
}
