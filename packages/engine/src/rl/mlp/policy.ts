// Tiny pure-TS MLP for the RL bot.
//
// Architecture: input FEATURE_DIM → hidden HIDDEN (ReLU) → output 1 scalar.
// Used as a per-candidate scorer: at each decision point we run the MLP
// over each candidate's feature vector, softmax the scores, and either
// sample (training) or argmax (inference).
//
// Implementation choices kept deliberately spare:
//   - Dense Float64Array storage in row-major order for the hidden weights
//   - Hand-rolled forward and backward passes
//   - Adam optimizer (good default; ~negligible overhead at this scale)
//   - JSON serialization for round-tripping weights to disk

import { FEATURE_DIM } from './features';

export const HIDDEN = 64;

// Internal layout: W1 (HIDDEN x FEATURE_DIM, row-major) | b1 (HIDDEN) |
//                  W2 (HIDDEN) | b2 (1)
const W1_SIZE = HIDDEN * FEATURE_DIM;
const B1_SIZE = HIDDEN;
const W2_SIZE = HIDDEN;
const B2_SIZE = 1;
export const PARAM_COUNT = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE;

// Slot offsets.
const W1_OFFSET = 0;
const B1_OFFSET = W1_OFFSET + W1_SIZE;
const W2_OFFSET = B1_OFFSET + B1_SIZE;
const B2_OFFSET = W2_OFFSET + W2_SIZE;

export type Policy = {
  // All trainable parameters in one flat buffer.
  params: Float64Array;
};

// Xavier-ish initialization: weights scaled by sqrt(1 / fan_in).
export function createPolicy(rng: () => number): Policy {
  const params = new Float64Array(PARAM_COUNT);
  const w1Scale = Math.sqrt(1 / FEATURE_DIM);
  for (let i = W1_OFFSET; i < W1_OFFSET + W1_SIZE; i++) {
    params[i] = (rng() * 2 - 1) * w1Scale;
  }
  // b1 starts at 0.
  const w2Scale = Math.sqrt(1 / HIDDEN);
  for (let i = W2_OFFSET; i < W2_OFFSET + W2_SIZE; i++) {
    params[i] = (rng() * 2 - 1) * w2Scale;
  }
  // b2 starts at 0.
  return { params };
}

export function clonePolicy(p: Policy): Policy {
  return { params: new Float64Array(p.params) };
}

// Forward pass for one feature vector.
// Returns score, plus the intermediate activations needed for backprop:
//   z1 = W1 * x + b1     (pre-activation, length HIDDEN)
//   h  = relu(z1)        (post-activation, length HIDDEN)
//   s  = W2 . h + b2     (scalar score)
export type ForwardCache = {
  readonly x: Float64Array;
  readonly h: Float64Array;
  readonly z1Mask: Uint8Array; // 1 where z1 > 0, 0 where z1 ≤ 0
  readonly score: number;
};

export function forward(p: Policy, x: Float64Array): ForwardCache {
  const params = p.params;
  const h = new Float64Array(HIDDEN);
  const mask = new Uint8Array(HIDDEN);
  for (let j = 0; j < HIDDEN; j++) {
    let z = params[B1_OFFSET + j]!;
    const rowOff = W1_OFFSET + j * FEATURE_DIM;
    for (let i = 0; i < FEATURE_DIM; i++) {
      z += params[rowOff + i]! * x[i]!;
    }
    if (z > 0) {
      h[j] = z;
      mask[j] = 1;
    } else {
      h[j] = 0;
      mask[j] = 0;
    }
  }
  let score = params[B2_OFFSET]!;
  for (let j = 0; j < HIDDEN; j++) {
    score += params[W2_OFFSET + j]! * h[j]!;
  }
  return { x, h, z1Mask: mask, score };
}

// Numerically-stable softmax over scores.
// Returns { probs, logZ } where logZ is logsumexp(scores) for log-prob computation.
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

// Sample an index from a discrete probability distribution.
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
// Uses the cached activations from the forward pass to avoid recomputing.
export function accumulateScoreGradient(
  p: Policy,
  cache: ForwardCache,
  coef: number,
  grad: Float64Array,
): void {
  if (coef === 0) return;
  const params = p.params;
  // d score / d b2 = 1
  grad[B2_OFFSET] = grad[B2_OFFSET]! + coef;
  // d score / d W2[j] = h[j]
  for (let j = 0; j < HIDDEN; j++) {
    grad[W2_OFFSET + j] = grad[W2_OFFSET + j]! + coef * cache.h[j]!;
  }
  // d score / d h[j] = W2[j]
  // d h[j] / d z1[j] = 1 if mask[j], else 0
  // d z1[j] / d W1[j,i] = x[i]
  // d z1[j] / d b1[j] = 1
  for (let j = 0; j < HIDDEN; j++) {
    if (cache.z1Mask[j] === 0) continue;
    const w2j = params[W2_OFFSET + j]!;
    const dz = coef * w2j;
    grad[B1_OFFSET + j] = grad[B1_OFFSET + j]! + dz;
    const rowOff = W1_OFFSET + j * FEATURE_DIM;
    for (let i = 0; i < FEATURE_DIM; i++) {
      grad[rowOff + i] = grad[rowOff + i]! + dz * cache.x[i]!;
    }
  }
}

// REINFORCE gradient for a single decision:
//   ∇ log p[k*] = (1[k = k*] - p[k]) / temperature * ∇ s[k]   (summed over k)
//
// Accumulates that gradient into `grad`, scaled by an additional `coef`
// (typically the per-decision (reward - baseline)).
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

// Apply an Adam update. We do gradient ASCENT on the REINFORCE objective,
// so we ADD `lr * m_hat / (sqrt(v_hat) + eps)`.
export function adamStep(
  p: Policy,
  grad: Float64Array,
  adam: Adam,
  gradClip?: number,
): void {
  // Optional global L2-norm gradient clipping for stability.
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
  readonly hidden: number;
  readonly params: number[];
};

export function serialize(p: Policy): SerializedPolicy {
  return {
    featureDim: FEATURE_DIM,
    hidden: HIDDEN,
    params: Array.from(p.params),
  };
}

export function deserialize(s: SerializedPolicy): Policy {
  if (s.featureDim !== FEATURE_DIM) {
    throw new Error(
      `weight feature dim mismatch: file=${s.featureDim} code=${FEATURE_DIM}`,
    );
  }
  if (s.hidden !== HIDDEN) {
    throw new Error(`weight hidden dim mismatch: file=${s.hidden} code=${HIDDEN}`);
  }
  if (s.params.length !== PARAM_COUNT) {
    throw new Error(
      `weight count mismatch: file=${s.params.length} expected=${PARAM_COUNT}`,
    );
  }
  return { params: Float64Array.from(s.params) };
}
