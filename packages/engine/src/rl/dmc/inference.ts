// Pure-TS forward pass for the DMC Q-networks.
//
// Mirrors `python/dmc/network.py` exactly. Three role-specific nets, each
// with the structure:
//
//   static + scalars (772) → FC(512) → ReLU → FC(256) → ReLU = h_static (256)
//   history (T, 113)       → LSTM(128)                       = h_hist  (128)
//   action  (115)          → FC(64) → ReLU                   = h_act   (64)
//
//   concat(h_static, h_hist, h_act) (448)
//     → FC(256) → ReLU → FC(128) → ReLU → FC(1) = Q(s, a)
//
// PyTorch LSTM weight conventions (single layer, batch_first=True):
//   weight_ih_l0: (4H, IN)  — gates concatenated as [i, f, g, o]
//   weight_hh_l0: (4H, H)
//   bias_ih_l0:   (4H,)
//   bias_hh_l0:   (4H,)

import {
  ACTION_ENCODING_LEN,
} from './actions';
import {
  HISTORY_MAX_STEPS,
  PER_STEP_DIM,
  SCALAR_DIM,
  STATIC_CHANNELS,
  type Observation,
} from './observation';
import { ENCODING_LEN } from './cardEncoding';
import type { Role } from './roles';

// Layer sizes (must match network.py).
export const STATIC_DIM = STATIC_CHANNELS * ENCODING_LEN; // 756
const STATIC_FC_IN = STATIC_DIM + SCALAR_DIM;             // 772
const HIDDEN_STATIC_1 = 512;
const HIDDEN_STATIC = 256;
const HIDDEN_HIST = 128;
const HIDDEN_ACT = 64;
const HIDDEN_HEAD_1 = 256;
const HIDDEN_HEAD_2 = 128;

// One role's complete weight bundle. Sizes derived from network.py.
export type RoleWeights = {
  // static_encoder.0: Linear(STATIC_FC_IN → HIDDEN_STATIC_1)
  staticW1: Float32Array; // HIDDEN_STATIC_1 × STATIC_FC_IN
  staticB1: Float32Array; // HIDDEN_STATIC_1
  // static_encoder.2: Linear(HIDDEN_STATIC_1 → HIDDEN_STATIC)
  staticW2: Float32Array; // HIDDEN_STATIC × HIDDEN_STATIC_1
  staticB2: Float32Array; // HIDDEN_STATIC
  // history_lstm.weight_ih_l0 / weight_hh_l0 / bias_ih_l0 / bias_hh_l0
  lstmWi: Float32Array;   // 4*HIDDEN_HIST × PER_STEP_DIM
  lstmWh: Float32Array;   // 4*HIDDEN_HIST × HIDDEN_HIST
  lstmBi: Float32Array;   // 4*HIDDEN_HIST
  lstmBh: Float32Array;   // 4*HIDDEN_HIST
  // action_encoder.0: Linear(ACTION_ENCODING_LEN → HIDDEN_ACT)
  actionW: Float32Array;  // HIDDEN_ACT × ACTION_ENCODING_LEN
  actionB: Float32Array;  // HIDDEN_ACT
  // head.0: Linear(448 → 256)
  headW1: Float32Array;   // HIDDEN_HEAD_1 × (HIDDEN_STATIC+HIDDEN_HIST+HIDDEN_ACT)
  headB1: Float32Array;
  // head.2: Linear(256 → 128)
  headW2: Float32Array;
  headB2: Float32Array;
  // head.4: Linear(128 → 1)
  headW3: Float32Array;
  headB3: Float32Array;
};

export type DmcWeights = {
  readonly landlord: RoleWeights;
  readonly defender: RoleWeights;
  readonly attacker: RoleWeights;
};

// ---- helpers: dense matmul (row-major), ReLU, sigmoid, tanh -------------

// y = W @ x + b. W is (out × in), row-major. x and b are length in / out.
function linear(
  out: Float32Array,
  W: Float32Array,
  x: Float32Array,
  b: Float32Array,
  outDim: number,
  inDim: number,
): void {
  for (let i = 0; i < outDim; i++) {
    let acc = b[i]!;
    const row = i * inDim;
    for (let j = 0; j < inDim; j++) {
      acc += W[row + j]! * x[j]!;
    }
    out[i] = acc;
  }
}

function relu(v: Float32Array): void {
  for (let i = 0; i < v.length; i++) if (v[i]! < 0) v[i] = 0;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// ---- LSTM forward over a variable-length sequence -----------------------
//
// Input: history of shape (T, PER_STEP_DIM), with T <= HISTORY_MAX_STEPS.
// Returns the final hidden state (HIDDEN_HIST,). If T == 0, returns zeros.
function lstmForward(
  w: RoleWeights,
  history: Float32Array,    // HISTORY_MAX_STEPS × PER_STEP_DIM, oldest-first
  historyLen: number,
): Float32Array {
  const h = new Float32Array(HIDDEN_HIST);
  const c = new Float32Array(HIDDEN_HIST);
  if (historyLen === 0) return h;
  const H = HIDDEN_HIST;
  const gateBuf = new Float32Array(4 * H); // gates: [i, f, g, o]
  const xBuf = new Float32Array(PER_STEP_DIM);
  for (let t = 0; t < historyLen; t++) {
    // x = history[t]
    const off = t * PER_STEP_DIM;
    for (let k = 0; k < PER_STEP_DIM; k++) xBuf[k] = history[off + k]!;
    // gates = W_ih @ x + b_ih + W_hh @ h + b_hh
    // Compute W_ih @ x + b_ih
    for (let i = 0; i < 4 * H; i++) {
      let acc = w.lstmBi[i]! + w.lstmBh[i]!;
      const row = i * PER_STEP_DIM;
      for (let j = 0; j < PER_STEP_DIM; j++) {
        acc += w.lstmWi[row + j]! * xBuf[j]!;
      }
      gateBuf[i] = acc;
    }
    // Add W_hh @ h
    for (let i = 0; i < 4 * H; i++) {
      let acc = 0;
      const row = i * H;
      for (let j = 0; j < H; j++) {
        acc += w.lstmWh[row + j]! * h[j]!;
      }
      gateBuf[i] = gateBuf[i]! + acc;
    }
    // Gates: i_t, f_t, g_t, o_t at offsets [0, H, 2H, 3H].
    for (let k = 0; k < H; k++) {
      const i_t = sigmoid(gateBuf[k]!);
      const f_t = sigmoid(gateBuf[H + k]!);
      const g_t = Math.tanh(gateBuf[2 * H + k]!);
      const o_t = sigmoid(gateBuf[3 * H + k]!);
      const c_new = f_t * c[k]! + i_t * g_t;
      c[k] = c_new;
      h[k] = o_t * Math.tanh(c_new);
    }
  }
  return h;
}

// ---- Encode the state portion of (s, a) once per decision ---------------

export type EncodedState = {
  readonly hState: Float32Array; // HIDDEN_STATIC
  readonly hHist: Float32Array;  // HIDDEN_HIST
};

export function encodeState(
  w: RoleWeights,
  obs: Observation,
): EncodedState {
  // Concatenate static + scalars.
  const sIn = new Float32Array(STATIC_FC_IN);
  for (let i = 0; i < STATIC_DIM; i++) sIn[i] = obs.staticChannels[i] ?? 0;
  for (let i = 0; i < SCALAR_DIM; i++) sIn[STATIC_DIM + i] = obs.scalars[i] ?? 0;

  // Static FC stack: 772 → 512 → ReLU → 256 → ReLU.
  const h1 = new Float32Array(HIDDEN_STATIC_1);
  linear(h1, w.staticW1, sIn, w.staticB1, HIDDEN_STATIC_1, STATIC_FC_IN);
  relu(h1);
  const h2 = new Float32Array(HIDDEN_STATIC);
  linear(h2, w.staticW2, h1, w.staticB2, HIDDEN_STATIC, HIDDEN_STATIC_1);
  relu(h2);

  // History LSTM. Materialize history Float32Array if needed.
  const histArr = new Float32Array(HISTORY_MAX_STEPS * PER_STEP_DIM);
  for (let i = 0; i < histArr.length; i++) histArr[i] = obs.history[i] ?? 0;
  const hHist = lstmForward(w, histArr, obs.historyLen);

  return { hState: h2, hHist };
}

// ---- Score one candidate action given the encoded state -----------------

export function scoreAction(
  w: RoleWeights,
  enc: EncodedState,
  actionEncoding: ReadonlyArray<number>,
): number {
  // Action encoder.
  const aIn = new Float32Array(ACTION_ENCODING_LEN);
  for (let i = 0; i < ACTION_ENCODING_LEN; i++) aIn[i] = actionEncoding[i] ?? 0;
  const hAct = new Float32Array(HIDDEN_ACT);
  linear(hAct, w.actionW, aIn, w.actionB, HIDDEN_ACT, ACTION_ENCODING_LEN);
  relu(hAct);

  // Concat → head FCs.
  const headIn = new Float32Array(HIDDEN_STATIC + HIDDEN_HIST + HIDDEN_ACT);
  for (let i = 0; i < HIDDEN_STATIC; i++) headIn[i] = enc.hState[i]!;
  for (let i = 0; i < HIDDEN_HIST; i++) headIn[HIDDEN_STATIC + i] = enc.hHist[i]!;
  for (let i = 0; i < HIDDEN_ACT; i++) headIn[HIDDEN_STATIC + HIDDEN_HIST + i] = hAct[i]!;

  const h1 = new Float32Array(HIDDEN_HEAD_1);
  linear(h1, w.headW1, headIn, w.headB1, HIDDEN_HEAD_1, headIn.length);
  relu(h1);
  const h2 = new Float32Array(HIDDEN_HEAD_2);
  linear(h2, w.headW2, h1, w.headB2, HIDDEN_HEAD_2, HIDDEN_HEAD_1);
  relu(h2);
  const out = new Float32Array(1);
  linear(out, w.headW3, h2, w.headB3, 1, HIDDEN_HEAD_2);
  return out[0]!;
}

// Score K candidates in one call, returning a Float32Array of length K.
// Encodes the state once and reuses across all candidates.
export function scoreCandidates(
  w: RoleWeights,
  obs: Observation,
  candidateEncodings: ReadonlyArray<ReadonlyArray<number>>,
): Float32Array {
  const enc = encodeState(w, obs);
  const out = new Float32Array(candidateEncodings.length);
  for (let i = 0; i < candidateEncodings.length; i++) {
    out[i] = scoreAction(w, enc, candidateEncodings[i]!);
  }
  return out;
}

// Convenience: dispatch by role.
export function weightsFor(weights: DmcWeights, role: Role): RoleWeights {
  if (role === 'landlord') return weights.landlord;
  if (role === 'defender') return weights.defender;
  return weights.attacker;
}
