// The play network's privileged critic (research/py/tractor/nn/model.py
// `Critic`, the v5 "flat" kind), which search reads at its leaves.
//
// It reads the actor's encoded history (net.ts `encodeHistory`), the
// position's context vector, and the privileged view: all four hands in
// order from the seat to act, then the kitty (`privileged_view`). Its
// value is the seat's team's future point margin in units of
// 2 * 100 * num_decks (eval/search.py `_value_to_points`).
//
// Exported under `critic.*` by research/py/tractor/export_ts.py.

import {
  addInPlace,
  encodeHistory,
  geluInPlace,
  layerNorm,
  linear,
  NUM_TYPES,
  CONTEXT_DIM,
  type Lin,
  type Norm,
} from './net';
import type { ZeroWeights } from './weights';

const NUM_REL_SEATS = 4;
export const NUM_PRIVILEGED_ROWS = NUM_REL_SEATS + 1;

type Block = { norm1: Norm; qkv: Lin; proj: Lin; norm2: Norm; fc1: Lin; fc2: Lin };

type Critic = {
  readonly cd: number;
  readonly heads: number;
  readonly dh: number;
  readonly dff: number;
  readonly inProj: Lin;
  readonly cls: Float32Array;
  readonly pos: Float32Array;
  readonly blocks: Block[];
  readonly norm: Norm;
  readonly priv0: Lin;
  readonly priv2: Lin;
  readonly ctxProj: Lin;
  readonly head0: Lin;
  readonly head2: Lin;
};

const CRITICS = new WeakMap<ZeroWeights, Critic>();

function lin(w: ZeroWeights, name: string, inDim: number, outDim: number): Lin {
  return { w: w.get(`${name}.weight`, outDim, inDim), b: w.get(`${name}.bias`, outDim), inDim, outDim };
}
function nrm(w: ZeroWeights, name: string, d: number): Norm {
  return { w: w.get(`${name}.weight`, d), b: w.get(`${name}.bias`, d) };
}

function prepare(w: ZeroWeights): Critic {
  const hit = CRITICS.get(w);
  if (hit !== undefined) return hit;
  const cfg = w.manifest.model_config as Record<string, unknown> & typeof w.manifest.model_config;
  if ((cfg.critic_kind ?? 'flat') !== 'flat') throw new Error(`zero critic: only the flat critic is ported, not ${String(cfg.critic_kind)}`);
  if (!w.has('critic.head.2.weight')) {
    throw new Error('zero critic: these weights carry no critic; re-export with research/py/tractor/export_ts.py');
  }
  const cd = Number(cfg.critic_d);
  const heads = Number(cfg.critic_heads);
  const dff = Number(cfg.critic_d_ff);
  const layers = Number(cfg.critic_layers);
  const d = cfg.d_model;
  const blocks: Block[] = [];
  for (let l = 0; l < layers; l++) {
    const p = `critic.blocks.${l}`;
    blocks.push({
      norm1: nrm(w, `${p}.norm1`, cd),
      qkv: lin(w, `${p}.attn.qkv`, cd, 3 * cd),
      proj: lin(w, `${p}.attn.proj`, cd, cd),
      norm2: nrm(w, `${p}.norm2`, cd),
      fc1: lin(w, `${p}.ff.fc1`, cd, dff),
      fc2: lin(w, `${p}.ff.fc2`, dff, cd),
    });
  }
  const c: Critic = {
    cd,
    heads,
    dh: cd / heads,
    dff,
    inProj: lin(w, 'critic.in_proj', d, cd),
    cls: w.get('critic.cls', 1, cd),
    pos: w.get('critic.pos', cfg.max_history, cd),
    blocks,
    norm: nrm(w, 'critic.norm', cd),
    priv0: lin(w, 'critic.privileged.0', NUM_PRIVILEGED_ROWS * NUM_TYPES, 2 * cd),
    priv2: lin(w, 'critic.privileged.2', 2 * cd, cd),
    ctxProj: lin(w, 'critic.context_proj', CONTEXT_DIM, 2 * cd),
    head0: lin(w, 'critic.head.0', 2 * cd, cd),
    head2: lin(w, 'critic.head.2', cd, 1),
  };
  CRITICS.set(w, c);
  return c;
}

/** float32[5 x 54]: the four hands from `seat` round, then the kitty
 *  (model.py `privileged_view`). Counts, exact in float16. */
export function privilegedView(
  hands: ReadonlyArray<ArrayLike<number>>,
  kitty: ArrayLike<number>,
  seat: number,
): Float32Array {
  const out = new Float32Array(NUM_PRIVILEGED_ROWS * NUM_TYPES);
  for (let rel = 0; rel < NUM_REL_SEATS; rel++) {
    const h = hands[(seat + rel) % NUM_REL_SEATS]!;
    for (let t = 0; t < NUM_TYPES; t++) out[rel * NUM_TYPES + t] = h[t] ?? 0;
  }
  for (let t = 0; t < NUM_TYPES; t++) out[NUM_REL_SEATS * NUM_TYPES + t] = kitty[t] ?? 0;
  return out;
}

/** The critic's value given the encoded history (n x d_model rows), the
 *  context vector and the privileged view. */
export function criticFromEncoded(
  w: ZeroWeights,
  encoded: Float32Array,
  n: number,
  context: Float32Array,
  privileged: Float32Array,
): number {
  const c = prepare(w);
  const { cd, dh, heads, dff } = c;
  const T = n + 1;
  const x = new Float32Array(T * cd);
  const h = new Float32Array(T * cd);
  const qkv = new Float32Array(T * 3 * cd);
  const att = new Float32Array(T * cd);
  const tmp = new Float32Array(T * cd);
  const ff = new Float32Array(T * dff);
  const scores = new Float64Array(T);

  x.set(c.cls, 0);
  if (n > 0) {
    linear(encoded, 0, n, c.inProj, x, cd);
    for (let r = 0; r < n; r++) {
      const xo = (r + 1) * cd;
      const po = r * cd;
      for (let i = 0; i < cd; i++) x[xo + i] = x[xo + i]! + c.pos[po + i]!;
    }
  }
  // Non-causal self-attention over the CLS token and every real token;
  // in the last block only the CLS row's output is read.
  for (let l = 0; l < c.blocks.length; l++) {
    const blk = c.blocks[l]!;
    const live = l === c.blocks.length - 1 ? 1 : T;
    layerNorm(x, 0, T, cd, blk.norm1, h, 0);
    linear(h, 0, T, blk.qkv, qkv, 0);
    for (let r = 0; r < live; r++) {
      for (let hd = 0; hd < heads; hd++) {
        valueRow(qkv, r, hd, cd, dh, T, scores, att);
      }
    }
    linear(att, 0, live, blk.proj, tmp, 0);
    addInPlace(x, tmp, live * cd);
    layerNorm(x, 0, live, cd, blk.norm2, h, 0);
    linear(h, 0, live, blk.fc1, ff, 0);
    geluInPlace(ff, live * dff);
    linear(ff, 0, live, blk.fc2, tmp, 0);
    addInPlace(x, tmp, live * cd);
  }
  const both = new Float32Array(2 * cd);
  layerNorm(x, 0, 1, cd, c.norm, both, 0);

  const first = new Float32Array(2 * cd);
  linear(privileged, 0, 1, c.priv0, first, 0);
  const ctx = new Float32Array(2 * cd);
  linear(context, 0, 1, c.ctxProj, ctx, 0);
  addInPlace(first, ctx, 2 * cd);
  geluInPlace(first, 2 * cd);
  const priv = new Float32Array(cd);
  linear(first, 0, 1, c.priv2, priv, 0);
  geluInPlace(priv, cd);
  both.set(priv, cd);
  const hid = new Float32Array(cd);
  linear(both, 0, 1, c.head0, hid, 0);
  geluInPlace(hid, cd);
  const out = new Float32Array(1);
  linear(hid, 0, 1, c.head2, out, 0);
  return out[0]!;
}

// One query row and head of self-attention over all T rows, with q, k and
// v packed per row as [q | k | v].
function valueRow(
  qkv: Float32Array,
  r: number,
  hd: number,
  cd: number,
  dh: number,
  T: number,
  scores: Float64Array,
  out: Float32Array,
): void {
  const stride = 3 * cd;
  const qOff = r * stride + hd * dh;
  const kOff = cd + hd * dh;
  const vOff = 2 * cd + hd * dh;
  const scale = 1 / Math.sqrt(dh);
  let max = -Infinity;
  for (let k = 0; k < T; k++) {
    let s = 0;
    const ko = k * stride + kOff;
    for (let i = 0; i < dh; i++) s += qkv[qOff + i]! * qkv[ko + i]!;
    s *= scale;
    scores[k] = s;
    if (s > max) max = s;
  }
  let sum = 0;
  for (let k = 0; k < T; k++) {
    const e = Math.exp(scores[k]! - max);
    scores[k] = e;
    sum += e;
  }
  const inv = 1 / sum;
  const oo = r * cd + hd * dh;
  for (let i = 0; i < dh; i++) {
    let acc = 0;
    for (let k = 0; k < T; k++) acc += scores[k]! * qkv[k * stride + vOff + i]!;
    out[oo + i] = acc * inv;
  }
}

/** The critic's value of a position: history tokens as the featuriser
 *  makes them, its context vector, and the privileged view. */
export function criticValue(
  w: ZeroWeights,
  history: Float32Array,
  historyLen: number,
  context: Float32Array,
  privileged: Float32Array,
): number {
  const { encoded, n } = encodeHistory(w, history, historyLen);
  return criticFromEncoded(w, encoded, n, context, privileged);
}
