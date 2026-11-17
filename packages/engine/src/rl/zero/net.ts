// The PPO play network's forward pass, in plain TypeScript: what
// `TractorNet.forward` in research/py/tractor/nn/model.py computes for
// one decision, down to the policy logits. The value, outcome and belief
// heads are left out; the bot reads only the logits.
//
// Plain float32 loops rather than a runtime binding for the reason the v2
// port gives (rl/v2/net.ts): the ws server is a 256 MB node:alpine
// machine, where torch does not fit and onnxruntime-node has no musl
// build.
//
// Three things make it fast enough for a live game on a shared CPU.
//
// * Linear layers run four tokens against two output rows at a time, so
//   each weight is read once per four tokens (~3 GFLOP/s here, against
//   ~1.3 for a plain dot product).
// * The history encoder is causal, so a position's encoding depends only
//   on the tokens up to it. Keys and values of every history layer, and
//   the query block's cross-attention keys and values, are cached per
//   token stream; a decision re-encodes only the tokens that are new or
//   changed since the stream was last seen (a token can change: the
//   "won" flag of an open trick's plays is written when it closes). The
//   cached and uncached paths do the same arithmetic in the same order,
//   so they agree bit for bit (test/zero-net.test.ts checks).
// * Masked query tokens -- card types the seat does not hold -- are never
//   computed. In the Python model they are zeroed after every block and
//   excluded as attention keys, so nothing they hold can reach the
//   summary; leaving them out is exact. In the last query block only the
//   context token's output is read, so the others contribute keys and
//   values and nothing more.

import type { ZeroWeights } from './weights';

// --- layout constants (research/py/tractor/nn/features.py, model.py) ---

export const HISTORY_DIM = 37;
export const CONTEXT_DIM = 37;
export const NUM_TYPES = 54;
export const D_CARD = 30;
export const NUM_PLANES = 10;
const PLANE_HAND = 0;
const PLANE_PLAYED = 1;
const PLANE_TRICK = 5;
const PLANE_KITTY = 9;
const NUM_REL_SEATS = 4;
// Context, unseen, kitty, then one token per relative seat.
const FIXED_QUERY_TOKENS = 3 + NUM_REL_SEATS;
// The static card block's one-hot offsets (features.py).
const STATIC_SUIT = 0;
const STATIC_STRENGTH = 5;
const STATIC_POINTS = 21;
const NUM_STRENGTH = 16;
const TRUMP = 4;
const NUM_CAND_SCALARS = 6;
// torch.nn.LayerNorm's default.
const LN_EPS = 1e-5;

export type ZeroObs = {
  /** historyLen x HISTORY_DIM, row-major, oldest first, unpadded. */
  readonly history: Float32Array;
  readonly historyLen: number;
  /** CONTEXT_DIM. */
  readonly context: Float32Array;
  /** 54 raw counts. */
  readonly unseen: Float32Array;
  /** NUM_PLANES x 54 raw counts. */
  readonly planes: Float32Array;
  /** 54 x D_CARD. */
  readonly cards: Float32Array;
  /** numCandidates x 54 raw counts. */
  readonly candidates: Float32Array;
  readonly numCandidates: number;
};

// --- weights, gathered once per ZeroWeights ------------------------------

type Lin = {
  readonly w: Float32Array;
  readonly b: Float32Array | null;
  readonly inDim: number;
  readonly outDim: number;
};
type Norm = { readonly w: Float32Array; readonly b: Float32Array };

type CausalBlock = { norm1: Norm; qkv: Lin; proj: Lin; norm2: Norm; fc1: Lin; fc2: Lin };
type QueryBlock = {
  normSa: Norm;
  saQkv: Lin;
  saProj: Lin;
  normCa: Norm;
  caQ: Lin;
  caKv: Lin;
  caProj: Lin;
  normFf: Norm;
  fc1: Lin;
  fc2: Lin;
};

type HistoryCache = {
  len: number;
  stamp: number;
  readonly tokens: Float32Array; // maxHistory x HISTORY_DIM
  readonly keys: Float32Array[]; // per history layer, maxHistory x d
  readonly values: Float32Array[];
  readonly memKeys: Float32Array[]; // per query layer, maxHistory x d
  readonly memValues: Float32Array[];
};

type Net = {
  readonly d: number;
  readonly heads: number;
  readonly dh: number;
  readonly dff: number;
  readonly maxHistory: number;
  readonly histIn: Lin;
  readonly histPos: Float32Array;
  readonly histBlocks: CausalBlock[];
  readonly histNorm: Norm;
  readonly ctxProj: Lin;
  readonly unseenProj: Lin;
  readonly seatProj: Lin;
  readonly kittyProj: Lin;
  readonly seatEmbed: Float32Array;
  readonly handProj: Lin;
  readonly cardEmbed: Float32Array;
  readonly memNorm: Norm;
  readonly queryBlocks: QueryBlock[];
  readonly outNorm: Norm;
  readonly cardProj: Lin;
  readonly scalar1: Lin;
  readonly scalar2: Lin;
  readonly scoreNorm: Norm;
  readonly scoreQ: Lin;
  readonly candBias: Lin;
  // The cross-attention sink, normed and projected, per query layer.
  readonly sinkKeys: Float32Array[];
  readonly sinkValues: Float32Array[];
  readonly caches: HistoryCache[];
  scratch: HistoryCache | null;
  clock: number;
  readonly s: Scratch;
};

type Scratch = {
  x: Float32Array;
  h: Float32Array;
  qkv: Float32Array;
  att: Float32Array;
  tmp: Float32Array;
  ff: Float32Array;
  scores: Float64Array;
  row: Float32Array;
  seatIn: Float32Array;
};

function lin(w: ZeroWeights, name: string, inDim: number, outDim: number, bias = true): Lin {
  return {
    w: w.get(`${name}.weight`, outDim, inDim),
    b: bias ? w.get(`${name}.bias`, outDim) : null,
    inDim,
    outDim,
  };
}

function norm(w: ZeroWeights, name: string, d: number): Norm {
  return { w: w.get(`${name}.weight`, d), b: w.get(`${name}.bias`, d) };
}

const NETS = new WeakMap<ZeroWeights, Net>();
let cacheLimit = 8;

/** How many history streams to keep encoded (default 8: a table's four
 *  seats, twice over). Each holds about 2.4 MB at the v5 widths. */
export function setZeroNetCacheSize(n: number): void {
  cacheLimit = Math.max(0, Math.floor(n));
}

function prepare(w: ZeroWeights): Net {
  const hit = NETS.get(w);
  if (hit !== undefined) return hit;
  const cfg = w.manifest.model_config;
  if (cfg.ffn !== 'gelu') throw new Error(`zero net: only GELU networks are ported, not ${cfg.ffn}`);
  const d = cfg.d_model;
  const heads = cfg.n_heads;
  const dff = cfg.d_ff;
  const maxHistory = cfg.max_history;
  if (d % heads !== 0) throw new Error('zero net: d_model must divide by n_heads');

  const histBlocks: CausalBlock[] = [];
  for (let l = 0; l < cfg.n_layers_history; l++) {
    const p = `actor.history.blocks.${l}`;
    histBlocks.push({
      norm1: norm(w, `${p}.norm1`, d),
      qkv: lin(w, `${p}.attn.qkv`, d, 3 * d),
      proj: lin(w, `${p}.attn.proj`, d, d),
      norm2: norm(w, `${p}.norm2`, d),
      fc1: lin(w, `${p}.ff.fc1`, d, dff),
      fc2: lin(w, `${p}.ff.fc2`, dff, d),
    });
  }
  const queryBlocks: QueryBlock[] = [];
  for (let l = 0; l < cfg.n_layers_query; l++) {
    const p = `actor.query.blocks.${l}`;
    queryBlocks.push({
      normSa: norm(w, `${p}.norm_sa`, d),
      saQkv: lin(w, `${p}.sa.qkv`, d, 3 * d),
      saProj: lin(w, `${p}.sa.proj`, d, d),
      normCa: norm(w, `${p}.norm_ca`, d),
      caQ: lin(w, `${p}.ca.q`, d, d),
      caKv: lin(w, `${p}.ca.kv`, d, 2 * d),
      caProj: lin(w, `${p}.ca.proj`, d, d),
      normFf: norm(w, `${p}.norm_ff`, d),
      fc1: lin(w, `${p}.ff.fc1`, d, dff),
      fc2: lin(w, `${p}.ff.fc2`, dff, d),
    });
  }
  const scalarD = cfg.scalar_d;
  const maxRows = Math.max(maxHistory, FIXED_QUERY_TOKENS + NUM_TYPES);
  const s: Scratch = {
    x: new Float32Array(maxRows * d),
    h: new Float32Array(maxRows * d),
    qkv: new Float32Array(maxRows * 3 * d),
    att: new Float32Array(maxRows * d),
    tmp: new Float32Array(maxRows * d),
    ff: new Float32Array(maxRows * Math.max(dff, scalarD)),
    scores: new Float64Array(maxHistory + FIXED_QUERY_TOKENS + NUM_TYPES + 1),
    row: new Float32Array(Math.max(d, 2 * d)),
    seatIn: new Float32Array(NUM_REL_SEATS * 2 * NUM_TYPES),
  };

  const memNorm = norm(w, 'actor.query.mem_norm', d);
  const sink = w.get('actor.query.mem_sink', 1, d);
  const sinkNormed = new Float32Array(d);
  layerNorm(sink, 0, 1, d, memNorm, sinkNormed, 0);
  const sinkKeys: Float32Array[] = [];
  const sinkValues: Float32Array[] = [];
  for (const blk of queryBlocks) {
    const kv = new Float32Array(2 * d);
    linear(sinkNormed, 0, 1, blk.caKv, kv, 0);
    sinkKeys.push(kv.slice(0, d));
    sinkValues.push(kv.slice(d, 2 * d));
  }

  const net: Net = {
    d,
    heads,
    dh: d / heads,
    dff,
    maxHistory,
    histIn: lin(w, 'actor.history.in_proj', HISTORY_DIM, d),
    histPos: w.get('actor.history.pos', maxHistory, d),
    histBlocks,
    histNorm: norm(w, 'actor.history.norm', d),
    ctxProj: lin(w, 'actor.query.ctx_proj', CONTEXT_DIM, d),
    unseenProj: lin(w, 'actor.query.unseen_proj', NUM_TYPES, d),
    seatProj: lin(w, 'actor.query.seat_proj', 2 * NUM_TYPES, d),
    kittyProj: lin(w, 'actor.query.kitty_proj', NUM_TYPES, d),
    seatEmbed: w.get('actor.query.seat_embed', NUM_REL_SEATS, d),
    handProj: lin(w, 'actor.query.hand_proj', D_CARD, d),
    cardEmbed: w.get('actor.query.card_embed', NUM_TYPES, d),
    memNorm,
    queryBlocks,
    outNorm: norm(w, 'actor.query.out_norm', d),
    cardProj: lin(w, 'actor.scorer.card_proj', D_CARD, d, false),
    scalar1: lin(w, 'actor.scorer.scalar_mlp.0', NUM_CAND_SCALARS, scalarD),
    scalar2: lin(w, 'actor.scorer.scalar_mlp.2', scalarD, d),
    scoreNorm: norm(w, 'actor.scorer.norm', d),
    scoreQ: lin(w, 'actor.scorer.score_q', d, d),
    candBias: lin(w, 'actor.scorer.cand_bias', d, 1),
    sinkKeys,
    sinkValues,
    caches: [],
    scratch: null,
    clock: 0,
    s,
  };
  NETS.set(w, net);
  return net;
}

// --- kernels -------------------------------------------------------------

/** y[r, :] = W x[r, :] + b for `rows` rows. Four rows against two outputs
 *  at a time; every (row, output) sums over the input in index order
 *  whichever block it falls in, so the result does not depend on how the
 *  rows are grouped. */
function linear(
  x: Float32Array,
  xOff: number,
  rows: number,
  L: Lin,
  y: Float32Array,
  yOff: number,
): void {
  const { w, b, inDim: n, outDim: m } = L;
  let r = 0;
  for (; r + 4 <= rows; r += 4) {
    const x0 = xOff + r * n;
    const x1 = x0 + n;
    const x2 = x1 + n;
    const x3 = x2 + n;
    const y0 = yOff + r * m;
    const y1 = y0 + m;
    const y2 = y1 + m;
    const y3 = y2 + m;
    let o = 0;
    for (; o + 2 <= m; o += 2) {
      const wa = o * n;
      const wb = wa + n;
      let a0 = 0;
      let a1 = 0;
      let a2 = 0;
      let a3 = 0;
      let b0 = 0;
      let b1 = 0;
      let b2 = 0;
      let b3 = 0;
      for (let i = 0; i < n; i++) {
        const u = w[wa + i]!;
        const v = w[wb + i]!;
        const p0 = x[x0 + i]!;
        const p1 = x[x1 + i]!;
        const p2 = x[x2 + i]!;
        const p3 = x[x3 + i]!;
        a0 += u * p0;
        a1 += u * p1;
        a2 += u * p2;
        a3 += u * p3;
        b0 += v * p0;
        b1 += v * p1;
        b2 += v * p2;
        b3 += v * p3;
      }
      const ba = b === null ? 0 : b[o]!;
      const bb = b === null ? 0 : b[o + 1]!;
      y[y0 + o] = a0 + ba;
      y[y1 + o] = a1 + ba;
      y[y2 + o] = a2 + ba;
      y[y3 + o] = a3 + ba;
      y[y0 + o + 1] = b0 + bb;
      y[y1 + o + 1] = b1 + bb;
      y[y2 + o + 1] = b2 + bb;
      y[y3 + o + 1] = b3 + bb;
    }
    for (; o < m; o++) {
      const wa = o * n;
      let a0 = 0;
      let a1 = 0;
      let a2 = 0;
      let a3 = 0;
      for (let i = 0; i < n; i++) {
        const u = w[wa + i]!;
        a0 += u * x[x0 + i]!;
        a1 += u * x[x1 + i]!;
        a2 += u * x[x2 + i]!;
        a3 += u * x[x3 + i]!;
      }
      const ba = b === null ? 0 : b[o]!;
      y[y0 + o] = a0 + ba;
      y[y1 + o] = a1 + ba;
      y[y2 + o] = a2 + ba;
      y[y3 + o] = a3 + ba;
    }
  }
  for (; r < rows; r++) {
    const x0 = xOff + r * n;
    const y0 = yOff + r * m;
    let o = 0;
    for (; o + 2 <= m; o += 2) {
      const wa = o * n;
      const wb = wa + n;
      let a0 = 0;
      let b0 = 0;
      for (let i = 0; i < n; i++) {
        const p0 = x[x0 + i]!;
        a0 += w[wa + i]! * p0;
        b0 += w[wb + i]! * p0;
      }
      y[y0 + o] = a0 + (b === null ? 0 : b[o]!);
      y[y0 + o + 1] = b0 + (b === null ? 0 : b[o + 1]!);
    }
    for (; o < m; o++) {
      const wa = o * n;
      let a0 = 0;
      for (let i = 0; i < n; i++) a0 += w[wa + i]! * x[x0 + i]!;
      y[y0 + o] = a0 + (b === null ? 0 : b[o]!);
    }
  }
}

function layerNorm(
  x: Float32Array,
  xOff: number,
  rows: number,
  d: number,
  N: Norm,
  y: Float32Array,
  yOff: number,
): void {
  for (let r = 0; r < rows; r++) {
    const xo = xOff + r * d;
    const yo = yOff + r * d;
    let mean = 0;
    for (let i = 0; i < d; i++) mean += x[xo + i]!;
    mean /= d;
    let varSum = 0;
    for (let i = 0; i < d; i++) {
      const c = x[xo + i]! - mean;
      varSum += c * c;
    }
    const inv = 1 / Math.sqrt(varSum / d + LN_EPS);
    for (let i = 0; i < d; i++) y[yo + i] = (x[xo + i]! - mean) * inv * N.w[i]! + N.b[i]!;
  }
}

// erf to ~1.2e-7 (Numerical Recipes' erfc, Chebyshev fit). torch's
// nn.GELU() default is the exact erf form, not the tanh approximation.
function erf(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 +
                            t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? 1 - r : r - 1;
}

function geluInPlace(x: Float32Array, n: number): void {
  for (let i = 0; i < n; i++) {
    const v = x[i]!;
    x[i] = 0.5 * v * (1 + erf(v * Math.SQRT1_2));
  }
}

function addInPlace(x: Float32Array, y: Float32Array, n: number): void {
  for (let i = 0; i < n; i++) x[i] = x[i]! + y[i]!;
}

/** One attention head row: out[0:dh] = softmax(q.k / sqrt(dh)) v over the
 *  key rows listed. Keys come from up to two sources: a single extra row
 *  (the cross-attention sink) and a contiguous block. */
function attendRow(
  q: Float32Array,
  qOff: number,
  dh: number,
  extraK: Float32Array | null,
  extraV: Float32Array | null,
  extraOff: number,
  K: Float32Array,
  V: Float32Array,
  kvOff: number,
  kvStride: number,
  nKeys: number,
  scores: Float64Array,
  out: Float32Array,
  outOff: number,
): void {
  const scale = 1 / Math.sqrt(dh);
  let total = 0;
  let max = -Infinity;
  if (extraK !== null) {
    let s = 0;
    for (let i = 0; i < dh; i++) s += q[qOff + i]! * extraK[extraOff + i]!;
    s *= scale;
    scores[0] = s;
    max = s;
    total = 1;
  }
  for (let k = 0; k < nKeys; k++) {
    const ko = kvOff + k * kvStride;
    let s = 0;
    for (let i = 0; i < dh; i++) s += q[qOff + i]! * K[ko + i]!;
    s *= scale;
    scores[total + k] = s;
    if (s > max) max = s;
  }
  const n = total + nKeys;
  let sum = 0;
  for (let k = 0; k < n; k++) {
    const e = Math.exp(scores[k]! - max);
    scores[k] = e;
    sum += e;
  }
  const inv = 1 / sum;
  for (let i = 0; i < dh; i++) {
    let acc = 0;
    if (extraV !== null) acc += scores[0]! * extraV[extraOff + i]!;
    for (let k = 0; k < nKeys; k++) acc += scores[total + k]! * V[kvOff + k * kvStride + i]!;
    out[outOff + i] = acc * inv;
  }
}

// --- the history encoder, incrementally ----------------------------------

function newCache(net: Net): HistoryCache {
  const { d, maxHistory } = net;
  const per = () => new Float32Array(maxHistory * d);
  return {
    len: 0,
    stamp: 0,
    tokens: new Float32Array(maxHistory * HISTORY_DIM),
    keys: net.histBlocks.map(per),
    values: net.histBlocks.map(per),
    memKeys: net.queryBlocks.map(per),
    memValues: net.queryBlocks.map(per),
  };
}

function prefixMatch(c: HistoryCache, hist: Float32Array, hOff: number, n: number): number {
  const upto = Math.min(c.len, n);
  for (let t = 0; t < upto; t++) {
    const a = t * HISTORY_DIM;
    const b = hOff + t * HISTORY_DIM;
    for (let i = 0; i < HISTORY_DIM; i++) {
      if (c.tokens[a + i] !== hist[b + i]) return t;
    }
  }
  return upto;
}

/** The cache entry to encode this stream into, and how much of it is
 *  already right. */
function pickCache(net: Net, hist: Float32Array, hOff: number, n: number, useCache: boolean): [HistoryCache, number] {
  net.clock += 1;
  if (!useCache || cacheLimit === 0) {
    // A private entry, encoded from scratch every time.
    if (net.scratch === null) net.scratch = newCache(net);
    net.scratch.len = 0;
    return [net.scratch, 0];
  }
  let best: HistoryCache | null = null;
  let bestMatch = 0;
  for (const c of net.caches) {
    const m = prefixMatch(c, hist, hOff, n);
    if (m > bestMatch || (m === bestMatch && m > 0 && best !== null && c.stamp > best.stamp)) {
      best = c;
      bestMatch = m;
    }
  }
  if (best === null || bestMatch === 0) {
    // A new stream: a fresh entry while there is room, else the least
    // recently used one.
    if (net.caches.length < cacheLimit) {
      best = newCache(net);
      net.caches.push(best);
    } else {
      best = net.caches[0]!;
      for (const c of net.caches) if (c.stamp < best.stamp) best = c;
    }
    bestMatch = 0;
  }
  best.stamp = net.clock;
  return [best, bestMatch];
}

/** Encode positions start..n-1 of the stream into `c`: per-layer history
 *  keys and values, and the query block's cross-attention keys and
 *  values of the normed output. */
function advanceHistory(net: Net, c: HistoryCache, hist: Float32Array, hOff: number, start: number, n: number): void {
  const rows = n - start;
  if (rows <= 0) {
    c.len = n;
    return;
  }
  const { d, dh, heads, dff, s } = net;
  const { x, h, qkv, att, tmp, ff } = s;

  linear(hist, hOff + start * HISTORY_DIM, rows, net.histIn, x, 0);
  for (let r = 0; r < rows; r++) {
    const po = (start + r) * d;
    const xo = r * d;
    for (let i = 0; i < d; i++) x[xo + i] = x[xo + i]! + net.histPos[po + i]!;
  }

  for (let l = 0; l < net.histBlocks.length; l++) {
    const blk = net.histBlocks[l]!;
    const K = c.keys[l]!;
    const V = c.values[l]!;
    layerNorm(x, 0, rows, d, blk.norm1, h, 0);
    linear(h, 0, rows, blk.qkv, qkv, 0);
    for (let r = 0; r < rows; r++) {
      const src = r * 3 * d;
      const dst = (start + r) * d;
      for (let i = 0; i < d; i++) {
        K[dst + i] = qkv[src + d + i]!;
        V[dst + i] = qkv[src + 2 * d + i]!;
      }
    }
    for (let r = 0; r < rows; r++) {
      const pos = start + r;
      for (let hd = 0; hd < heads; hd++) {
        attendRow(qkv, r * 3 * d + hd * dh, dh, null, null, 0, K, V, hd * dh, d, pos + 1, s.scores, att, r * d + hd * dh);
      }
    }
    linear(att, 0, rows, blk.proj, tmp, 0);
    addInPlace(x, tmp, rows * d);
    layerNorm(x, 0, rows, d, blk.norm2, h, 0);
    linear(h, 0, rows, blk.fc1, ff, 0);
    geluInPlace(ff, rows * dff);
    linear(ff, 0, rows, blk.fc2, tmp, 0);
    addInPlace(x, tmp, rows * d);
  }
  // Encoded output, then the memory norm the query block applies to it.
  layerNorm(x, 0, rows, d, net.histNorm, h, 0);
  layerNorm(h, 0, rows, d, net.memNorm, x, 0);
  for (let l = 0; l < net.queryBlocks.length; l++) {
    const blk = net.queryBlocks[l]!;
    linear(x, 0, rows, blk.caKv, qkv, 0);
    const MK = c.memKeys[l]!;
    const MV = c.memValues[l]!;
    for (let r = 0; r < rows; r++) {
      const src = r * 2 * d;
      const dst = (start + r) * d;
      for (let i = 0; i < d; i++) {
        MK[dst + i] = qkv[src + i]!;
        MV[dst + i] = qkv[src + d + i]!;
      }
    }
  }
  c.tokens.set(hist.subarray(hOff + start * HISTORY_DIM, hOff + n * HISTORY_DIM), start * HISTORY_DIM);
  c.len = n;
}

// --- the query block -----------------------------------------------------

function querySummary(net: Net, obs: ZeroObs, c: HistoryCache, n: number, summary: Float32Array): void {
  const { d, dh, heads, dff, s } = net;
  const { x, h, qkv, att, tmp, ff } = s;

  // Tokens: context, unseen, kitty, four seats, then each card type held.
  const held: number[] = [];
  for (let t = 0; t < NUM_TYPES; t++) if (obs.planes[PLANE_HAND * NUM_TYPES + t]! > 0.5) held.push(t);
  const T = FIXED_QUERY_TOKENS + held.length;

  linear(obs.context, 0, 1, net.ctxProj, x, 0);
  linear(obs.unseen, 0, 1, net.unseenProj, x, d);
  linear(obs.planes, PLANE_KITTY * NUM_TYPES, 1, net.kittyProj, x, 2 * d);
  const seatIn = s.seatIn;
  for (let r = 0; r < NUM_REL_SEATS; r++) {
    seatIn.set(obs.planes.subarray((PLANE_PLAYED + r) * NUM_TYPES, (PLANE_PLAYED + r + 1) * NUM_TYPES), r * 2 * NUM_TYPES);
    seatIn.set(
      obs.planes.subarray((PLANE_TRICK + r) * NUM_TYPES, (PLANE_TRICK + r + 1) * NUM_TYPES),
      r * 2 * NUM_TYPES + NUM_TYPES,
    );
  }
  linear(seatIn, 0, NUM_REL_SEATS, net.seatProj, x, 3 * d);
  for (let r = 0; r < NUM_REL_SEATS; r++) {
    const xo = (3 + r) * d;
    for (let i = 0; i < d; i++) x[xo + i] = x[xo + i]! + net.seatEmbed[r * d + i]!;
  }
  for (let j = 0; j < held.length; j++) {
    const t = held[j]!;
    const xo = (FIXED_QUERY_TOKENS + j) * d;
    linear(obs.cards, t * D_CARD, 1, net.handProj, x, xo);
    for (let i = 0; i < d; i++) x[xo + i] = x[xo + i]! + net.cardEmbed[t * d + i]!;
  }

  const L = net.queryBlocks.length;
  for (let l = 0; l < L; l++) {
    const blk = net.queryBlocks[l]!;
    // In the last block only the context token's output is read.
    const live = l === L - 1 ? 1 : T;
    layerNorm(x, 0, T, d, blk.normSa, h, 0);
    linear(h, 0, T, blk.saQkv, qkv, 0);
    for (let r = 0; r < live; r++) {
      for (let hd = 0; hd < heads; hd++) {
        selfAttendRow(qkv, r, hd, d, dh, T, s.scores, att);
      }
    }
    linear(att, 0, live, blk.saProj, tmp, 0);
    addInPlace(x, tmp, live * d);

    layerNorm(x, 0, live, d, blk.normCa, h, 0);
    linear(h, 0, live, blk.caQ, tmp, 0);
    const MK = c.memKeys[l]!;
    const MV = c.memValues[l]!;
    for (let r = 0; r < live; r++) {
      for (let hd = 0; hd < heads; hd++) {
        attendRow(tmp, r * d + hd * dh, dh, net.sinkKeys[l]!, net.sinkValues[l]!, hd * dh, MK, MV, hd * dh, d, n, s.scores, att, r * d + hd * dh);
      }
    }
    linear(att, 0, live, blk.caProj, tmp, 0);
    addInPlace(x, tmp, live * d);

    layerNorm(x, 0, live, d, blk.normFf, h, 0);
    linear(h, 0, live, blk.fc1, ff, 0);
    geluInPlace(ff, live * dff);
    linear(ff, 0, live, blk.fc2, tmp, 0);
    addInPlace(x, tmp, live * d);
  }
  layerNorm(x, 0, 1, d, net.outNorm, summary, 0);
}

/** Self-attention of query token `r`, head `hd`, over all T tokens, with
 *  q, k and v packed per row as [q | k | v] in `qkv`. */
function selfAttendRow(
  qkv: Float32Array,
  r: number,
  hd: number,
  d: number,
  dh: number,
  T: number,
  scores: Float64Array,
  out: Float32Array,
): void {
  const stride = 3 * d;
  const qOff = r * stride + hd * dh;
  const kOff = d + hd * dh;
  const vOff = 2 * d + hd * dh;
  const scale = 1 / Math.sqrt(dh);
  let max = -Infinity;
  for (let k = 0; k < T; k++) {
    const ko = k * stride + kOff;
    let sc = 0;
    for (let i = 0; i < dh; i++) sc += qkv[qOff + i]! * qkv[ko + i]!;
    sc *= scale;
    scores[k] = sc;
    if (sc > max) max = sc;
  }
  let sum = 0;
  for (let k = 0; k < T; k++) {
    const e = Math.exp(scores[k]! - max);
    scores[k] = e;
    sum += e;
  }
  const inv = 1 / sum;
  const oo = r * d + hd * dh;
  for (let i = 0; i < dh; i++) {
    let acc = 0;
    for (let k = 0; k < T; k++) acc += scores[k]! * qkv[k * stride + vOff + i]!;
    out[oo + i] = acc * inv;
  }
}

// --- the candidate scorer ------------------------------------------------

function scoreCandidates(net: Net, obs: ZeroObs, summary: Float32Array, out: Float32Array): void {
  const { d, s } = net;
  const N = obs.numCandidates;
  const cards = obs.cards;
  const cand = obs.candidates;

  // Per card type: points (over 10), trump-ness, strength (over 15),
  // read out of the static one-hot block as _card_scalar_basis does.
  const pts = new Float64Array(NUM_TYPES);
  const trump = new Float64Array(NUM_TYPES);
  const strength = new Float64Array(NUM_TYPES);
  for (let t = 0; t < NUM_TYPES; t++) {
    const o = t * D_CARD;
    pts[t] = 0.5 * cards[o + STATIC_POINTS + 1]! + 1.0 * cards[o + STATIC_POINTS + 2]!;
    trump[t] = cards[o + STATIC_SUIT + TRUMP]!;
    let st = 0;
    for (let k = 0; k < NUM_STRENGTH; k++) st += cards[o + STATIC_STRENGTH + k]! * (k / (NUM_STRENGTH - 1));
    strength[t] = st;
  }

  // card_proj rows for the types any candidate uses.
  const used = new Uint8Array(NUM_TYPES);
  for (let j = 0; j < N; j++) for (let t = 0; t < NUM_TYPES; t++) if (cand[j * NUM_TYPES + t]! !== 0) used[t] = 1;
  const basis = new Float32Array(NUM_TYPES * d);
  for (let t = 0; t < NUM_TYPES; t++) if (used[t] === 1) linear(cards, t * D_CARD, 1, net.cardProj, basis, t * d);

  const q = s.row;
  linear(summary, 0, 1, net.scoreQ, q, 0);
  const scale = 1 / Math.sqrt(d);
  const scal = new Float32Array(NUM_CAND_SCALARS);
  const hidden = new Float32Array(net.scalar1.outDim);
  const sc = new Float32Array(d);
  const emb = new Float32Array(d);
  const embN = new Float32Array(d);
  const bias = new Float32Array(1);
  for (let j = 0; j < N; j++) {
    const co = j * NUM_TYPES;
    emb.fill(0);
    let nCards = 0;
    let p = 0;
    let tr = 0;
    let st = 0;
    for (let t = 0; t < NUM_TYPES; t++) {
      const k = cand[co + t]!;
      if (k === 0) continue;
      nCards += k;
      p += k * pts[t]!;
      tr += k * trump[t]!;
      st += k * strength[t]!;
      const bo = t * d;
      for (let i = 0; i < d; i++) emb[i] = emb[i]! + k * basis[bo + i]!;
    }
    const nz = Math.max(nCards, 1);
    const nonempty = nCards > 0.5 ? 1 : 0;
    scal[0] = nCards / 8;
    scal[1] = p;
    scal[2] = (Math.abs(tr - nCards) < 0.5 ? 1 : 0) * nonempty;
    scal[3] = tr / nz;
    scal[4] = st / nz;
    scal[5] = Math.abs(nCards - 1) < 0.5 ? 1 : 0;
    linear(scal, 0, 1, net.scalar1, hidden, 0);
    geluInPlace(hidden, hidden.length);
    linear(hidden, 0, 1, net.scalar2, sc, 0);
    for (let i = 0; i < d; i++) sc[i] = emb[i]! + sc[i]!;
    layerNorm(sc, 0, 1, d, net.scoreNorm, embN, 0);
    let dot = 0;
    for (let i = 0; i < d; i++) dot += embN[i]! * q[i]!;
    linear(embN, 0, 1, net.candBias, bias, 0);
    out[j] = dot * scale + bias[0]!;
  }
}

// --- entry point ---------------------------------------------------------

export type ActorOptions = {
  /** Reuse encoded history across calls (default true). Off computes the
   *  whole history afresh; the logits are bit-identical either way. */
  readonly cache?: boolean;
};

/** The policy logits for each of `obs.numCandidates` candidates: what the
 *  Python actor's `out.logits` holds at the valid slots. */
export function actorLogits(w: ZeroWeights, obs: ZeroObs, opts: ActorOptions = {}): Float32Array {
  const net = prepare(w);
  if (obs.numCandidates < 1) throw new Error('zero net: no candidates');
  if (obs.candidates.length < obs.numCandidates * NUM_TYPES) throw new Error('zero net: candidates too short');
  if (obs.history.length < obs.historyLen * HISTORY_DIM) throw new Error('zero net: history too short');
  // The encoder holds max_history positions; the featuriser already keeps
  // only the most recent, so this only trims a caller that did not.
  const n = Math.min(obs.historyLen, net.maxHistory);
  const hOff = (obs.historyLen - n) * HISTORY_DIM;

  const [c, start] = pickCache(net, obs.history, hOff, n, opts.cache !== false);
  advanceHistory(net, c, obs.history, hOff, start, n);

  const summary = new Float32Array(net.d);
  querySummary(net, obs, c, n, summary);
  const out = new Float32Array(obs.numCandidates);
  scoreCandidates(net, obs, summary, out);
  return out;
}

/** Drop every cached history stream (tests; a weights reload). */
export function resetZeroNetCache(w: ZeroWeights): void {
  const net = NETS.get(w);
  if (net !== undefined) net.caches.length = 0;
}
