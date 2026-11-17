// rl-zero-search against the Python it ports, on a tiny random network
// (fixtures/zero-tiny-search, from research/py/test_zero_search_parity.py
// --fixture): the privileged critic's values, and whole searches given the
// same worlds -- chosen where both sides' simulations play the same moves,
// so the per-candidate means must agree. Also: the world sampler deals
// exactly the unseen cards, sized and void-respecting, every time. The
// full check over hundreds of engine positions is research/src/
// zeroSearchParity.ts with research/py/test_zero_search_parity.py.

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { criticValue, privilegedView } from '../src/rl/zero/critic';
import { encodePosition, makeCtx, packCandidates } from '../src/rl/zero/features';
import { actorLogits } from '../src/rl/zero/net';
import { plane, positionFor } from '../src/rl/zero/position';
import { knownVoids, mulberry32, sampleWorld, searchMove, unseenFrom, type World } from '../src/rl/zero/search';
import { loadZeroWeights } from '../src/rl/zero/weights';
import { enumerateCandidates } from '../src/rl/v2/candidates';
import type { GameState, Seat } from '../src/state';

const here = dirname(fileURLToPath(import.meta.url));
const dir = join(here, 'fixtures/zero-tiny-search');
const w = loadZeroWeights(dir);
type WorldJson = { hands: number[][]; kitty: number[]; relaxed: boolean };
const fixture = JSON.parse(readFileSync(join(dir, 'cases.json'), 'utf-8')) as {
  critic: Array<{ state: GameState; seat: Seat; value: number }>;
  search: Array<{ state: GameState; seat: Seat; worlds: WorldJson[]; top: number[]; means: number[]; choice: number }>;
};

const kittyOf = (state: GameState, seat: Seat) =>
  seat === state.currentHand!.landlord ? state.currentHand!.kitty : null;

describe('rl-zero critic', () => {
  for (const [i, c] of fixture.critic.entries()) {
    it(`matches the PyTorch critic on case ${i}`, () => {
      const hand = c.state.currentHand!;
      const f = encodePosition(positionFor(c.state, c.seat, kittyOf(c.state, c.seat)));
      const hands = [0, 1, 2, 3].map((s) => plane(hand.hands[s as Seat] ?? []));
      const v = criticValue(w, f.history, f.historyLen, f.context, privilegedView(hands, plane(hand.kitty), c.seat));
      expect(Math.abs(v - c.value)).toBeLessThan(1e-5 * Math.max(1, Math.abs(c.value)));
    });
  }
});

describe('rl-zero search', () => {
  for (const [i, c] of fixture.search.entries()) {
    it(`reproduces the Python search on the same worlds, case ${i}`, () => {
      const candidates = enumerateCandidates(c.state, c.seat, {
        maxCandidates: w.manifest.model_config.max_candidates,
        myKitty: kittyOf(c.state, c.seat) ?? [],
      });
      const moves = candidates.map((m) => plane(m.cards));
      const logits = actorLogits(w, {
        ...encodePosition(positionFor(c.state, c.seat, kittyOf(c.state, c.seat))),
        candidates: packCandidates(moves),
        numCandidates: moves.length,
      });
      const worlds: World[] = c.worlds.map((wd) => ({
        hands: wd.hands.map((h) => Int8Array.from(h)),
        kitty: Int8Array.from(wd.kitty),
        relaxed: wd.relaxed,
      }));
      const trace: { top?: number[]; means?: number[] } = {};
      const res = searchMove(w, c.state, c.seat, candidates.map((m) => m.cards), logits, {
        config: { worlds: worlds.length, topK: c.top.length, extraTricks: 0, confident: 1.01 },
        worlds,
        trace,
      });
      expect(res.searched).toBe(true);
      expect(trace.top).toEqual(c.top);
      trace.means!.forEach((m, k) => expect(Math.abs(m - c.means[k]!)).toBeLessThan(1e-3));
      expect(res.index).toBe(c.choice);
    });
  }

  it('deals exactly the unseen cards, sized and void-respecting', () => {
    const states = [...fixture.critic, ...fixture.search];
    for (const c of states) {
      const hand = c.state.currentHand!;
      const pos = positionFor(c.state, c.seat, kittyOf(c.state, c.seat));
      const handSizes = [0, 1, 2, 3].map((s) => (hand.hands[s as Seat] ?? []).length);
      const unseen = unseenFrom(pos);
      const voids = knownVoids(pos);
      const eff = makeCtx(pos.levelRank, pos.trumpSuit).eff;
      const landlord = c.seat === hand.landlord;
      const rng = mulberry32(12345);
      for (let k = 0; k < 100; k++) {
        const world = sampleWorld({ pos, handSizes, kittySize: hand.kitty.length }, rng);
        const placed = new Array<number>(54).fill(0);
        for (let s = 0; s < 4; s++) {
          let size = 0;
          for (let t = 0; t < 54; t++) size += world.hands[s]![t]!;
          expect(size).toBe(handSizes[s]);
          if (s === c.seat) {
            expect(Array.from(world.hands[s]!)).toEqual(Array.from(pos.hand));
            continue;
          }
          for (let t = 0; t < 54; t++) {
            placed[t] = placed[t]! + world.hands[s]![t]!;
            if (!world.relaxed && world.hands[s]![t]! > 0) expect(voids[s]![eff[t]!]).toBe(false);
          }
        }
        if (landlord) expect(Array.from(world.kitty)).toEqual(Array.from(pos.kitty!));
        else for (let t = 0; t < 54; t++) placed[t] = placed[t]! + world.kitty[t]!;
        expect(placed).toEqual(Array.from(unseen));
      }
    }
  });
});
