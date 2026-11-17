// Checks the TypeScript play network (packages/engine/src/rl/zero/net.ts)
// against the Python actor on the positions `py/zero_net_parity.py` wrote:
//
//   python py/zero_net_parity.py --checkpoint CKPT --out-dir DIR
//   npx tsx research/src/zeroNetParity.ts DIR [--file positions.jsonl] [--tol 1e-3]
//
// Reports the largest logit error, argmax agreement, and the time a
// decision takes with and without the history cache. Exits non-zero when
// the error passes --tol or any argmax disagrees.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { actorLogits, resetZeroNetCache, type ZeroObs } from '@tractor/engine/rl/zero/net';
import { loadZeroWeights } from '@tractor/engine/rl/zero/weights';

const dir = process.argv[2];
if (dir === undefined) throw new Error('usage: zeroNetParity.ts DIR [--file NAME] [--tol X]');
const argOf = (name: string, fallback: string): string => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? (process.argv[i + 1] ?? fallback) : fallback;
};
const file = argOf('file', 'positions.jsonl');
const tol = Number(argOf('tol', '1e-3'));

type Rec = {
  hand: number;
  step: number;
  history_len: number;
  history: string;
  context: string;
  unseen: string;
  planes: string;
  cards: string;
  num_candidates: number;
  candidates: string;
  logits: number[];
};

function f32(b64: string): Float32Array {
  const buf = Buffer.from(b64, 'base64');
  return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

function toObs(r: Rec): ZeroObs {
  return {
    history: f32(r.history),
    historyLen: r.history_len,
    context: f32(r.context),
    unseen: f32(r.unseen),
    planes: f32(r.planes),
    cards: f32(r.cards),
    candidates: f32(r.candidates),
    numCandidates: r.num_candidates,
  };
}

const argmax = (a: ArrayLike<number>): number => {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i]! > a[best]!) best = i;
  return best;
};

const w = loadZeroWeights(dir);
const recs = readFileSync(join(dir, file), 'utf-8')
  .split('\n')
  .filter((l) => l.trim() !== '')
  .map((l) => JSON.parse(l) as Rec);

let maxErr = 0;
let maxRel = 0;
let agree = 0;
let multi = 0;
let cachedMismatch = 0;
const byLen: Record<string, { n: number; err: number }> = {};
let spread = 0;
for (const r of recs) {
  const obs = toObs(r);
  const got = actorLogits(w, obs, { cache: false });
  // The cached path, fed this hand's positions in order, must give the
  // same numbers bit for bit.
  const cached = actorLogits(w, obs);
  for (let i = 0; i < got.length; i++) if (got[i] !== cached[i]) cachedMismatch += 1;
  let err = 0;
  let scale = 0;
  for (let i = 0; i < got.length; i++) {
    err = Math.max(err, Math.abs(got[i]! - r.logits[i]!));
    scale = Math.max(scale, Math.abs(r.logits[i]!));
  }
  maxErr = Math.max(maxErr, err);
  maxRel = Math.max(maxRel, err / Math.max(scale, 1e-6));
  if (r.num_candidates > 1) {
    multi += 1;
    if (argmax(got) === argmax(r.logits)) agree += 1;
    spread += Math.max(...r.logits) - Math.min(...r.logits);
  }
  const bucket = r.history_len < 30 ? '0-29' : r.history_len < 70 ? '30-69' : '70+';
  const b = (byLen[bucket] ??= { n: 0, err: 0 });
  b.n += 1;
  b.err = Math.max(b.err, err);
}

// Timing: a late-hand position, ~100 history tokens, uncached (a stream
// seen for the first time) and cached (one new trick since last time).
const late = recs
  .filter((r) => r.history_len >= 80)
  .sort((a, b) => b.num_candidates - a.num_candidates || b.history_len - a.history_len)[0];
const early = recs.filter((r) => r.history_len <= 8).sort((a, b) => b.num_candidates - a.num_candidates)[0];
let timing = '';
if (late !== undefined) {
  const obs = toObs(late);
  const reps = 10;
  for (let i = 0; i < 2; i++) actorLogits(w, obs, { cache: false });
  let t0 = performance.now();
  for (let i = 0; i < reps; i++) actorLogits(w, obs, { cache: false });
  const fresh = (performance.now() - t0) / reps;
  // Cached: the same stream four tokens shorter, then the full one.
  const shorter: ZeroObs = { ...obs, historyLen: obs.historyLen - 4 };
  let total = 0;
  for (let i = 0; i < reps; i++) {
    resetZeroNetCache(w);
    actorLogits(w, shorter);
    t0 = performance.now();
    actorLogits(w, obs);
    total += performance.now() - t0;
  }
  timing =
    `timing at history ${obs.historyLen}, ${obs.numCandidates} candidates: ` +
    `${fresh.toFixed(1)} ms uncached, ${(total / reps).toFixed(1)} ms with 4 new tokens`;
}
if (early !== undefined) {
  const obs = toObs(early);
  const reps = 10;
  for (let i = 0; i < 2; i++) actorLogits(w, obs, { cache: false });
  const t0 = performance.now();
  for (let i = 0; i < reps; i++) actorLogits(w, obs, { cache: false });
  timing += `\ntiming at history ${obs.historyLen}, ${obs.numCandidates} candidates (early, full hand): ${((performance.now() - t0) / reps).toFixed(1)} ms uncached`;
}

console.log(`${recs.length} positions from ${dir}/${file}`);
console.log(`max |logit error| ${maxErr.toExponential(2)} (relative ${maxRel.toExponential(2)})`);
for (const [k, v] of Object.entries(byLen)) console.log(`  history ${k}: ${v.n} positions, max error ${v.err.toExponential(2)}`);
console.log(`argmax agreement ${agree}/${multi} (mean logit spread ${(spread / Math.max(multi, 1)).toFixed(3)})`);
console.log(`cached vs uncached: ${cachedMismatch} logits differ`);
if (timing) console.log(timing);
if (maxErr > tol || agree !== multi || cachedMismatch > 0) {
  console.log('FAIL');
  process.exit(1);
}
console.log('OK');
