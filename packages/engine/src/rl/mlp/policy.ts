// Pure-TS MLP policy for the RL bot. Two hidden layers.
//
// Architecture: input FEATURE_DIM → hidden HIDDEN_1 (ReLU)
//                                  → hidden HIDDEN_2 (ReLU)
//                                  → output 1 scalar.
// Used as a per-candidate scorer: at each decision point we run the MLP
// over each candidate's feature vector, softmax the scores, and either
// sample (training) or argmax (inference).
//
// Implementation choices kept deliberately spare:
//   - Dense Float64Array storage in row-major order
//   - Hand-rolled forward and backward passes
//   - Adam optimizer
//   - JSON serialization for round-tripping weights to disk

import { FEATURE_DIM } from './features';

export const HIDDEN_1 = 128;
export const HIDDEN_2 = 128;

// Internal layout: W1 (HIDDEN_1 x FEATURE_DIM, row-major) | b1 (HIDDEN_1) |
//                  W2 (HIDDEN_2 x HIDDEN_1, row-major) | b2 (HIDDEN_2) |
//                  W3 (HIDDEN_2)                       | b3 (1)
const W1_SIZE = HIDDEN_1 * FEATURE_DIM;
const B1_SIZE = HIDDEN_1;
const W2_SIZE = HIDDEN_2 * HIDDEN_1;
const B2_SIZE = HIDDEN_2;
const W3_SIZE = HIDDEN_2;
const B3_SIZE = 1;
export const PARAM_COUNT =
  W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE + W3_SIZE + B3_SIZE;

// Slot offsets.
const W1_OFFSET = 0;
const B1_OFFSET = W1_OFFSET + W1_SIZE;
const W2_OFFSET = B1_OFFSET + B1_SIZE;
const B2_OFFSET = W2_OFFSET + W2_SIZE;
const W3_OFFSET = B2_OFFSET + B2_SIZE;
const B3_OFFSET = W3_OFFSET + W3_SIZE;

export type Policy = {
  // All trainable parameters in one flat buffer.
  params: Float64Array;
};

// Xavier-ish initialization: weights scaled by sqrt(1 / fan_in).
export function createPolicy(rng: () => number): Policy {
  const params = new Float64Array(PARAM_COUNT);
  const w1Scale = Math.sqrt(2 / FEATURE_DIM); // He init for ReLU
  for (let i = W1_OFFSET; i < W1_OFFSET + W1_SIZE; i++) {
    params[i] = (rng() * 2 - 1) * w1Scale;
  }
  // b1 starts at 0.
  const w2Scale = Math.sqrt(2 / HIDDEN_1);
  for (let i = W2_OFFSET; i < W2_OFFSET + W2_SIZE; i++) {
    params[i] = (rng() * 2 - 1) * w2Scale;
  }
  // b2 starts at 0.
  const w3Scale = Math.sqrt(1 / HIDDEN_2);
  for (let i = W3_OFFSET; i < W3_OFFSET + W3_SIZE; i++) {
    params[i] = (rng() * 2 - 1) * w3Scale;
  }
  // b3 starts at 0.
  return { params };
}

export function clonePolicy(p: Policy): Policy {
  return { params: new Float64Array(p.params) };
}

// Forward pass for one feature vector.
// Returns score, plus the intermediate activations needed for backprop.
export type ForwardCache = {
  readonly x: Float64Array;
  readonly h1: Float64Array;
  readonly z1Mask: Uint8Array;
  readonly h2: Float64Array;
  readonly z2Mask: Uint8Array;
  readonly score: number;
};

export function forward(p: Policy, x: Float64Array): ForwardCache {
  const params = p.params;
  // Layer 1.
  const h1 = new Float64Array(HIDDEN_1);
  const mask1 = new Uint8Array(HIDDEN_1);
  for (let j = 0; j < HIDDEN_1; j++) {
    let z = params[B1_OFFSET + j]!;
    const rowOff = W1_OFFSET + j * FEATURE_DIM;
    for (let i = 0; i < FEATURE_DIM; i++) {
      z += params[rowOff + i]! * x[i]!;
    }
    if (z > 0) {
      h1[j] = z;
      mask1[j] = 1;
    } else {
      h1[j] = 0;
      mask1[j] = 0;
    }
  }
  // Layer 2.
  const h2 = new Float64Array(HIDDEN_2);
  const mask2 = new Uint8Array(HIDDEN_2);
  for (let j = 0; j < HIDDEN_2; j++) {
    let z = params[B2_OFFSET + j]!;
    const rowOff = W2_OFFSET + j * HIDDEN_1;
    for (let i = 0; i < HIDDEN_1; i++) {
      z += params[rowOff + i]! * h1[i]!;
    }
    if (z > 0) {
      h2[j] = z;
      mask2[j] = 1;
    } else {
      h2[j] = 0;
      mask2[j] = 0;
    }
  }
  // Output.
  let score = params[B3_OFFSET]!;
  for (let j = 0; j < HIDDEN_2; j++) {
    score += params[W3_OFFSET + j]! * h2[j]!;
  }
  return { x, h1, z1Mask: mask1, h2, z2Mask: mask2, score };
}

// Numerically-stable softmax over scores.
export function softmax(
  scores: ReadonlyArray<number>,
  temperature: number = 1.0,
): { probs: Float64Array; logZ: number } {
  const n = scores.length;
  const probs = new Float64Array(n);
  if (n === 0) return { probs, logZ: 0 };
  let maxS = -Infinity;
  for (let i = 0; i < n; i++) {
    const s = scores[i]! / temperature;
    if (s > maxS) maxS = s;
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(scores[i]! / temperature - maxS);
    probs[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) probs[i] = probs[i]! / sum;
  const logZ = maxS + Math.log(sum);
  return { probs, logZ };
}

export function sampleFromProbs(
  probs: Float64Array,
  rng: () => number,
): number {
  const r = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i]!;
    if (r < acc) return i;
  }
  return probs.length - 1;
}

// Compute the gradient of `score = forward(p, x).score` with respect to
// every parameter, scaled by `coef`, and accumulate into `grad`.
//
// Backprop chain (output is scalar `s`):
//   ds/dW3[j] = h2[j];        ds/db3 = 1;        ds/dh2[j] = W3[j]
//   dh2[j]/dz2[j] = mask2[j]; dz2[j]/dW2[j,i] = h1[i]; dz2[j]/db2[j] = 1
//   dz2[j]/dh1[i] = W2[j,i];  dh1[i]/dz1[i] = mask1[i]
//   dz1[i]/dW1[i,k] = x[k];   dz1[i]/db1[i] = 1
export function accumulateScoreGradient(
  p: Policy,
  cache: ForwardCache,
  coef: number,
  grad: Float64Array,
): void {
  if (coef === 0) return;
  const params = p.params;

  // Output layer: ds/dW3, ds/db3.
  grad[B3_OFFSET] = grad[B3_OFFSET]! + coef;
  for (let j = 0; j < HIDDEN_2; j++) {
    grad[W3_OFFSET + j] = grad[W3_OFFSET + j]! + coef * cache.h2[j]!;
  }

  // dz2[j] = ds/dz2[j] = coef * W3[j] * mask2[j]
  // Pre-compute it once and reuse for W2/b2 and for back-propagating into h1.
  const dz2 = new Float64Array(HIDDEN_2);
  for (let j = 0; j < HIDDEN_2; j++) {
    if (cache.z2Mask[j] === 0) continue;
    dz2[j] = coef * params[W3_OFFSET + j]!;
  }

  // Layer 2: ds/dW2 and ds/db2 from dz2.
  for (let j = 0; j < HIDDEN_2; j++) {
    if (dz2[j]! === 0) continue;
    const v = dz2[j]!;
    grad[B2_OFFSET + j] = grad[B2_OFFSET + j]! + v;
    const rowOff = W2_OFFSET + j * HIDDEN_1;
    for (let i = 0; i < HIDDEN_1; i++) {
      grad[rowOff + i] = grad[rowOff + i]! + v * cache.h1[i]!;
    }
  }

  // Back-prop into h1: dh1[i] = sum_j dz2[j] * W2[j,i]
  // Then dz1[i] = dh1[i] * mask1[i]
  const dz1 = new Float64Array(HIDDEN_1);
  for (let i = 0; i < HIDDEN_1; i++) {
    if (cache.z1Mask[i] === 0) continue;
    let acc = 0;
    for (let j = 0; j < HIDDEN_2; j++) {
      if (dz2[j]! === 0) continue;
      acc += dz2[j]! * params[W2_OFFSET + j * HIDDEN_1 + i]!;
    }
    dz1[i] = acc;
  }

  // Layer 1: ds/dW1 and ds/db1 from dz1.
  for (let i = 0; i < HIDDEN_1; i++) {
    if (dz1[i]! === 0) continue;
    const v = dz1[i]!;
    grad[B1_OFFSET + i] = grad[B1_OFFSET + i]! + v;
    const rowOff = W1_OFFSET + i * FEATURE_DIM;
    for (let k = 0; k < FEATURE_DIM; k++) {
      grad[rowOff + k] = grad[rowOff + k]! + v * cache.x[k]!;
    }
  }
}

// REINFORCE gradient for a single decision:
//   ∇ log p[k*] = (1[k = k*] - p[k]) / temperature * ∇ s[k]   (summed over k)
export function accumulateReinforceGradient(
  p: Policy,
  caches: ReadonlyArray<ForwardCache>,
  probs: Float64Array,
  sampledIdx: number,
  temperature: number,
  coef: number,
  grad: Float64Array,
): void {
  if (coef === 0) return;
  const invT = 1 / temperature;
  for (let k = 0; k < caches.length; k++) {
    const indicator = k === sampledIdx ? 1 : 0;
    const w = (indicator - probs[k]!) * invT * coef;
    if (w === 0) continue;
    accumulateScoreGradient(p, caches[k]!, w, grad);
  }
}

// -- Adam optimizer --------------------------------------------------------

export type Adam = {
  readonly lr: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  m: Float64Array;
  v: Float64Array;
  t: number;
};

export function createAdam(opts?: {
  lr?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
}): Adam {
  return {
    lr: opts?.lr ?? 3e-4,
    beta1: opts?.beta1 ?? 0.9,
    beta2: opts?.beta2 ?? 0.999,
    eps: opts?.eps ?? 1e-8,
    m: new Float64Array(PARAM_COUNT),
    v: new Float64Array(PARAM_COUNT),
    t: 0,
  };
}

// Apply an Adam update. We do gradient ASCENT on the REINFORCE objective.
export function adamStep(
  p: Policy,
  grad: Float64Array,
  adam: Adam,
  gradClip?: number,
): void {
  if (gradClip !== undefined && gradClip > 0) {
    let sq = 0;
    for (let i = 0; i < grad.length; i++) sq += grad[i]! * grad[i]!;
    const norm = Math.sqrt(sq);
    if (norm > gradClip) {
      const scale = gradClip / norm;
      for (let i = 0; i < grad.length; i++) grad[i] = grad[i]! * scale;
    }
  }
  adam.t += 1;
  const t = adam.t;
  const b1 = adam.beta1;
  const b2 = adam.beta2;
  const lr = adam.lr;
  const eps = adam.eps;
  const bc1 = 1 - Math.pow(b1, t);
  const bc2 = 1 - Math.pow(b2, t);
  for (let i = 0; i < PARAM_COUNT; i++) {
    const g = grad[i]!;
    const m = b1 * adam.m[i]! + (1 - b1) * g;
    const v = b2 * adam.v[i]! + (1 - b2) * g * g;
    adam.m[i] = m;
    adam.v[i] = v;
    const mh = m / bc1;
    const vh = v / bc2;
    p.params[i] = p.params[i]! + (lr * mh) / (Math.sqrt(vh) + eps);
  }
}

// -- Serialization ---------------------------------------------------------

export type SerializedPolicy = {
  readonly featureDim: number;
  readonly hidden1: number;
  readonly hidden2: number;
  readonly params: number[];
};

export function serialize(p: Policy): SerializedPolicy {
  return {
    featureDim: FEATURE_DIM,
    hidden1: HIDDEN_1,
    hidden2: HIDDEN_2,
    params: Array.from(p.params),
  };
}

export function deserialize(s: SerializedPolicy): Policy {
  if (s.featureDim !== FEATURE_DIM) {
    throw new Error(
      `weight feature dim mismatch: file=${s.featureDim} code=${FEATURE_DIM}`,
    );
  }
  if (s.hidden1 !== HIDDEN_1) {
    throw new Error(
      `weight hidden1 dim mismatch: file=${s.hidden1} code=${HIDDEN_1}`,
    );
  }
  if (s.hidden2 !== HIDDEN_2) {
    throw new Error(
      `weight hidden2 dim mismatch: file=${s.hidden2} code=${HIDDEN_2}`,
    );
  }
  if (s.params.length !== PARAM_COUNT) {
    throw new Error(
      `weight count mismatch: file=${s.params.length} expected=${PARAM_COUNT}`,
    );
  }
  return { params: Float64Array.from(s.params) };
}
