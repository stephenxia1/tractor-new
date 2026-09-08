// Forward pass for the v2 network, in plain TypeScript.
//
// The twin of `research/py/tractor_rl/model2.py`, restricted to what
// playing needs: state encoder, causal history attention, action encoder,
// the per-role points head, the value head that judges search leaves, and
// the belief head the determinized sampler weights its worlds by.
//
// Plain TS rather than a runtime binding because the ws server is a
// 256 MB `node:20-alpine` machine: onnxruntime-node ships glibc builds
// only, and torch does not fit. At 3M parameters and a handful of
// candidates per decision this costs a few milliseconds, which is far
// below what a human notices between plays.
//
// Conventions follow PyTorch: `Linear` weights are stored [out, in] and
// applied as y = x @ Wᵀ + b, and LayerNorm uses eps = 1e-5.

import {
  ACTION_DIM,
  HISTORY_TOKEN_DIM,
  NUM_ROLES,
  STATE_DIM,
} from './features';
import { NUM_ENC_SLOTS } from './encoding';
import { HISTORY_STEPS } from './record';
import { loadWeights, type LoadedWeights } from './weights';

const LAYER_NORM_EPS = 1e-5;

const WIDTH = 512;
const BLOCKS = 2;
const HIST_MODEL = 96;
const HIST_LAYERS = 2;
const HIST_HEADS = 4;
const HIST_HEAD_DIM = HIST_MODEL / HIST_HEADS; // 24
const HIST_FF = 2 * HIST_MODEL; // 192
const ENCODER_DIM = 128; // action encoder width
const HEAD_HIDDEN = 256;

function linear(
  x: Float32Array,
  w: Float32Array,
  b: Float32Array,
  inDim: number,
  outDim: number,
  out: Float32Array,
): Float32Array {
  for (let o = 0; o < outDim; o++) {
    let sum = b[o]!;
    const row = o * inDim;
    for (let i = 0; i < inDim; i++) sum += x[i]! * w[row + i]!;
    out[o] = sum;
  }
  return out;
}

function relu(x: Float32Array): Float32Array {
  for (let i = 0; i < x.length; i++) if (x[i]! < 0) x[i] = 0;
  return x;
}

function layerNorm(
  x: Float32Array,
  gamma: Float32Array,
  beta: Float32Array,
  dim: number,
  out: Float32Array,
  xOffset = 0,
  outOffset = 0,
): void {
  let mean = 0;
  for (let i = 0; i < dim; i++) mean += x[xOffset + i]!;
  mean /= dim;
  let variance = 0;
  for (let i = 0; i < dim; i++) {
    const d = x[xOffset + i]! - mean;
    variance += d * d;
  }
  variance /= dim;
  const scale = 1 / Math.sqrt(variance + LAYER_NORM_EPS);
  for (let i = 0; i < dim; i++) {
    out[outOffset + i] = (x[xOffset + i]! - mean) * scale * gamma[i]! + beta[i]!;
  }
}

// Causal self-attention over the recent move sequence. Causal because a
// move may only attend to what preceded it, which keeps the
// representation of a prefix stable as the trick fills in.
function historyEncoder(
  w: LoadedWeights,
  tokens: Float32Array, // [HISTORY_STEPS, HISTORY_TOKEN_DIM]
  mask: Float32Array, // [HISTORY_STEPS]
): Float32Array {
  let length = 0;
  for (let i = 0; i < HISTORY_STEPS; i++) if (mask[i]! >= 0.5) length++;
  // No history at all would leave every key masked, which makes attention
  // undefined. Python lets such a row attend to itself and zeroes the
  // result; returning zeros directly is the same thing, sooner.
  if (length === 0) return new Float32Array(HIST_MODEL);

  const projW = w.get('history.proj.weight', HIST_MODEL, HISTORY_TOKEN_DIM);
  const projB = w.get('history.proj.bias', HIST_MODEL);
  const pos = w.get('history.pos', HISTORY_STEPS, HIST_MODEL);

  const x = new Float32Array(HISTORY_STEPS * HIST_MODEL);
  const scratch = new Float32Array(HIST_MODEL);
  for (let t = 0; t < HISTORY_STEPS; t++) {
    linear(
      tokens.subarray(t * HISTORY_TOKEN_DIM, (t + 1) * HISTORY_TOKEN_DIM),
      projW, projB, HISTORY_TOKEN_DIM, HIST_MODEL, scratch,
    );
    for (let i = 0; i < HIST_MODEL; i++) {
      x[t * HIST_MODEL + i] = scratch[i]! + pos[t * HIST_MODEL + i]!;
    }
  }

  const h = new Float32Array(HISTORY_STEPS * HIST_MODEL);
  const q = new Float32Array(HISTORY_STEPS * HIST_MODEL);
  const k = new Float32Array(HISTORY_STEPS * HIST_MODEL);
  const v = new Float32Array(HISTORY_STEPS * HIST_MODEL);
  const attnOut = new Float32Array(HISTORY_STEPS * HIST_MODEL);
  const ff1 = new Float32Array(HIST_FF);
  const scale = 1 / Math.sqrt(HIST_HEAD_DIM);
  const scores = new Float32Array(HISTORY_STEPS);

  for (let layer = 0; layer < HIST_LAYERS; layer++) {
    const p = `history.layers.${layer}.`;
    const n1g = w.get(`${p}norm1.weight`, HIST_MODEL);
    const n1b = w.get(`${p}norm1.bias`, HIST_MODEL);
    for (let t = 0; t < HISTORY_STEPS; t++) {
      layerNorm(x, n1g, n1b, HIST_MODEL, h, t * HIST_MODEL, t * HIST_MODEL);
    }

    // PyTorch packs q, k and v into one [3*d, d] projection.
    const inW = w.get(`${p}attn.in_proj_weight`, 3 * HIST_MODEL, HIST_MODEL);
    const inB = w.get(`${p}attn.in_proj_bias`, 3 * HIST_MODEL);
    for (let t = 0; t < HISTORY_STEPS; t++) {
      const row = h.subarray(t * HIST_MODEL, (t + 1) * HIST_MODEL);
      for (let o = 0; o < HIST_MODEL; o++) {
        let sq = inB[o]!;
        let sk = inB[HIST_MODEL + o]!;
        let sv = inB[2 * HIST_MODEL + o]!;
        const rq = o * HIST_MODEL;
        const rk = (HIST_MODEL + o) * HIST_MODEL;
        const rv = (2 * HIST_MODEL + o) * HIST_MODEL;
        for (let i = 0; i < HIST_MODEL; i++) {
          const xi = row[i]!;
          sq += xi * inW[rq + i]!;
          sk += xi * inW[rk + i]!;
          sv += xi * inW[rv + i]!;
        }
        q[t * HIST_MODEL + o] = sq;
        k[t * HIST_MODEL + o] = sk;
        v[t * HIST_MODEL + o] = sv;
      }
    }

    attnOut.fill(0);
    for (let head = 0; head < HIST_HEADS; head++) {
      const base = head * HIST_HEAD_DIM;
      for (let t = 0; t < HISTORY_STEPS; t++) {
        // Causal: attend to 0..t. Padding: keys at or beyond `length` are
        // masked for every query, matching key_padding_mask.
        let best = -Infinity;
        const limit = Math.min(t, length - 1);
        for (let s = 0; s <= limit; s++) {
          let dot = 0;
          for (let i = 0; i < HIST_HEAD_DIM; i++) {
            dot += q[t * HIST_MODEL + base + i]! * k[s * HIST_MODEL + base + i]!;
          }
          dot *= scale;
          scores[s] = dot;
          if (dot > best) best = dot;
        }
        if (limit < 0) continue; // query before any valid key; stays zero
        let denom = 0;
        for (let s = 0; s <= limit; s++) {
          const e = Math.exp(scores[s]! - best);
          scores[s] = e;
          denom += e;
        }
        for (let s = 0; s <= limit; s++) {
          const weight = scores[s]! / denom;
          for (let i = 0; i < HIST_HEAD_DIM; i++) {
            attnOut[t * HIST_MODEL + base + i]! += weight * v[s * HIST_MODEL + base + i]!;
          }
        }
      }
    }

    const outW = w.get(`${p}attn.out_proj.weight`, HIST_MODEL, HIST_MODEL);
    const outB = w.get(`${p}attn.out_proj.bias`, HIST_MODEL);
    for (let t = 0; t < HISTORY_STEPS; t++) {
      linear(
        attnOut.subarray(t * HIST_MODEL, (t + 1) * HIST_MODEL),
        outW, outB, HIST_MODEL, HIST_MODEL, scratch,
      );
      for (let i = 0; i < HIST_MODEL; i++) x[t * HIST_MODEL + i]! += scratch[i]!;
    }

    const n2g = w.get(`${p}norm2.weight`, HIST_MODEL);
    const n2b = w.get(`${p}norm2.bias`, HIST_MODEL);
    const f0w = w.get(`${p}ff.0.weight`, HIST_FF, HIST_MODEL);
    const f0b = w.get(`${p}ff.0.bias`, HIST_FF);
    const f2w = w.get(`${p}ff.2.weight`, HIST_MODEL, HIST_FF);
    const f2b = w.get(`${p}ff.2.bias`, HIST_MODEL);
    for (let t = 0; t < HISTORY_STEPS; t++) {
      layerNorm(x, n2g, n2b, HIST_MODEL, h, t * HIST_MODEL, t * HIST_MODEL);
      relu(linear(h.subarray(t * HIST_MODEL, (t + 1) * HIST_MODEL),
        f0w, f0b, HIST_MODEL, HIST_FF, ff1));
      linear(ff1, f2w, f2b, HIST_FF, HIST_MODEL, scratch);
      for (let i = 0; i < HIST_MODEL; i++) x[t * HIST_MODEL + i]! += scratch[i]!;
    }
  }

  // Summarise with the most recent real step.
  const last = length - 1;
  return x.slice(last * HIST_MODEL, (last + 1) * HIST_MODEL);
}

/** The per-decision state embedding, reused across every candidate. */
export function encodeState(
  w: LoadedWeights,
  state: Float32Array,
  tokens: Float32Array,
  mask: Float32Array,
): Float32Array {
  let x = new Float32Array(WIDTH);
  linear(state, w.get('stem.weight', WIDTH, STATE_DIM), w.get('stem.bias', WIDTH),
    STATE_DIM, WIDTH, x);

  const normed = new Float32Array(WIDTH);
  const hidden = new Float32Array(WIDTH);
  const delta = new Float32Array(WIDTH);
  for (let b = 0; b < BLOCKS; b++) {
    const p = `blocks.${b}.`;
    layerNorm(x, w.get(`${p}norm.weight`, WIDTH), w.get(`${p}norm.bias`, WIDTH), WIDTH, normed);
    relu(linear(normed, w.get(`${p}fc1.weight`, WIDTH, WIDTH), w.get(`${p}fc1.bias`, WIDTH),
      WIDTH, WIDTH, hidden));
    linear(hidden, w.get(`${p}fc2.weight`, WIDTH, WIDTH), w.get(`${p}fc2.bias`, WIDTH),
      WIDTH, WIDTH, delta);
    for (let i = 0; i < WIDTH; i++) x[i]! += delta[i]!;
  }

  const summary = historyEncoder(w, tokens, mask);
  const fuseIn = WIDTH + summary.length;
  const cat = new Float32Array(fuseIn);
  cat.set(x, 0);
  cat.set(summary, WIDTH);

  const fnormed = new Float32Array(fuseIn);
  layerNorm(cat, w.get('fuse.0.weight', fuseIn), w.get('fuse.0.bias', fuseIn), fuseIn, fnormed);
  const fused = new Float32Array(WIDTH);
  relu(linear(fnormed, w.get('fuse.1.weight', WIDTH, fuseIn), w.get('fuse.1.bias', WIDTH),
    fuseIn, WIDTH, fused));
  x = fused;
  return x;
}

function encodeAction(w: LoadedWeights, action: Float32Array): Float32Array {
  const h1 = new Float32Array(ENCODER_DIM);
  relu(linear(action, w.get('action_encoder.0.weight', ENCODER_DIM, ACTION_DIM),
    w.get('action_encoder.0.bias', ENCODER_DIM), ACTION_DIM, ENCODER_DIM, h1));
  const h2 = new Float32Array(ENCODER_DIM);
  relu(linear(h1, w.get('action_encoder.2.weight', ENCODER_DIM, ENCODER_DIM),
    w.get('action_encoder.2.bias', ENCODER_DIM), ENCODER_DIM, ENCODER_DIM, h2));
  return h2;
}

function runHead(w: LoadedWeights, prefix: string, pair: Float32Array): number {
  const inDim = WIDTH + ENCODER_DIM;
  const h1 = new Float32Array(HEAD_HIDDEN);
  relu(linear(pair, w.get(`${prefix}.0.weight`, HEAD_HIDDEN, inDim),
    w.get(`${prefix}.0.bias`, HEAD_HIDDEN), inDim, HEAD_HIDDEN, h1));
  const h2 = new Float32Array(HEAD_HIDDEN / 2);
  relu(linear(h1, w.get(`${prefix}.2.weight`, HEAD_HIDDEN / 2, HEAD_HIDDEN),
    w.get(`${prefix}.2.bias`, HEAD_HIDDEN / 2), HEAD_HIDDEN, HEAD_HIDDEN / 2, h2));
  const out = new Float32Array(1);
  linear(h2, w.get(`${prefix}.4.weight`, 1, HEAD_HIDDEN / 2), w.get(`${prefix}.4.bias`, 1),
    HEAD_HIDDEN / 2, 1, out);
  return out[0]!;
}

/** Q(s, a) for each candidate, using the role's own points head. */
export function scoreCandidates(
  w: LoadedWeights,
  state: Float32Array,
  tokens: Float32Array,
  mask: Float32Array,
  actions: ReadonlyArray<Float32Array>,
  role: number,
): Float32Array {
  if (role < 0 || role >= NUM_ROLES) throw new Error(`role ${role} out of range`);
  const z = encodeState(w, state, tokens, mask);
  const pair = new Float32Array(WIDTH + ENCODER_DIM);
  pair.set(z, 0);
  const scores = new Float32Array(actions.length);
  for (let i = 0; i < actions.length; i++) {
    pair.set(encodeAction(w, actions[i]!), WIDTH);
    scores[i] = runHead(w, `q_points.${role}`, pair);
  }
  return scores;
}

/** P(opponent r holds a card at chain position p), for the 3 opponents by
 *  relative seat. Trained as an auxiliary head, and measured calibrated
 *  (ECE 0.015 on uncertain positions, no temperature scaling needed), which
 *  is what lets the determinized sampler weight its worlds by it. */
export function beliefs(w: LoadedWeights, z: Float32Array): Float32Array {
  const outDim = 3 * NUM_ENC_SLOTS;
  const h = new Float32Array(HEAD_HIDDEN);
  relu(linear(z, w.get('belief_head.0.weight', HEAD_HIDDEN, WIDTH),
    w.get('belief_head.0.bias', HEAD_HIDDEN), WIDTH, HEAD_HIDDEN, h));
  const logits = new Float32Array(outDim);
  linear(h, w.get('belief_head.2.weight', outDim, HEAD_HIDDEN),
    w.get('belief_head.2.bias', outDim), HEAD_HIDDEN, outDim, logits);
  for (let i = 0; i < outDim; i++) logits[i] = 1 / (1 + Math.exp(-logits[i]!));
  return logits;
}

/** V(s), the state value the search uses to judge a leaf. */
export function stateValue(w: LoadedWeights, z: Float32Array): number {
  const h = new Float32Array(HEAD_HIDDEN);
  relu(linear(z, w.get('v_head.0.weight', HEAD_HIDDEN, WIDTH), w.get('v_head.0.bias', HEAD_HIDDEN),
    WIDTH, HEAD_HIDDEN, h));
  const out = new Float32Array(1);
  linear(h, w.get('v_head.2.weight', 1, HEAD_HIDDEN), w.get('v_head.2.bias', 1),
    HEAD_HIDDEN, 1, out);
  return out[0]!;
}

export { loadWeights };
