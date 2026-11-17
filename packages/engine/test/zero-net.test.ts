// The TS port of the PPO play network against the Python actor, on a
// small random network (fixtures/zero-tiny, written by
// research/py/zero_net_parity.py --tiny): eight positions from one hand,
// the opening lead through the last trick, with the Python logits for each.
//
// The full-size parity check, on the v5 widths and a few hundred
// positions, is research/src/zeroNetParity.ts.

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { actorLogits, resetZeroNetCache, type ZeroObs } from '../src/rl/zero/net';
import { loadZeroWeights } from '../src/rl/zero/weights';

const DIR = join(dirname(fileURLToPath(import.meta.url)), 'fixtures', 'zero-tiny');

type Case = {
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

function obsOf(c: Case): ZeroObs {
  return {
    history: f32(c.history),
    historyLen: c.history_len,
    context: f32(c.context),
    unseen: f32(c.unseen),
    planes: f32(c.planes),
    cards: f32(c.cards),
    candidates: f32(c.candidates),
    numCandidates: c.num_candidates,
  };
}

const cases: Case[] = readFileSync(join(DIR, 'cases.jsonl'), 'utf-8')
  .split('\n')
  .filter((l) => l.trim() !== '')
  .map((l) => JSON.parse(l) as Case);

const argmax = (a: ArrayLike<number>): number => {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i]! > a[best]!) best = i;
  return best;
};

describe('zero play network', () => {
  const w = loadZeroWeights(DIR);

  it('matches the Python actor', () => {
    expect(cases.length).toBeGreaterThanOrEqual(6);
    for (const c of cases) {
      const got = actorLogits(w, obsOf(c), { cache: false });
      expect(got.length).toBe(c.num_candidates);
      for (let i = 0; i < got.length; i++) {
        expect(Math.abs(got[i]! - c.logits[i]!)).toBeLessThan(1e-4);
      }
      expect(argmax(got)).toBe(argmax(c.logits));
    }
  });

  it('gives the same logits through the history cache, in any order', () => {
    const fresh = cases.map((c) => actorLogits(w, obsOf(c), { cache: false }));
    for (const order of [cases.map((_, i) => i), cases.map((_, i) => cases.length - 1 - i)]) {
      resetZeroNetCache(w);
      for (const i of order) {
        const got = actorLogits(w, obsOf(cases[i]!));
        expect(Array.from(got)).toEqual(Array.from(fresh[i]!));
      }
    }
  });

  it('re-encodes a token that changed in place', () => {
    // A trick closing rewrites the "won" flag of its plays; a cached
    // stream must notice rather than serve stale keys.
    const c = cases.find((x) => x.history_len >= 20)!;
    const obs = obsOf(c);
    resetZeroNetCache(w);
    actorLogits(w, obs);
    const edited = new Float32Array(obs.history);
    edited[5 * 37 + 35] = edited[5 * 37 + 35] === 0 ? 1 : 0;
    const changed: ZeroObs = { ...obs, history: edited };
    const viaCache = actorLogits(w, changed);
    const direct = actorLogits(w, changed, { cache: false });
    expect(Array.from(viaCache)).toEqual(Array.from(direct));
  });

  it('refuses a malformed observation', () => {
    const obs = obsOf(cases[0]!);
    expect(() => actorLogits(w, { ...obs, numCandidates: 0 })).toThrow();
    expect(() => actorLogits(w, { ...obs, numCandidates: obs.numCandidates + 1 })).toThrow();
  });
});
