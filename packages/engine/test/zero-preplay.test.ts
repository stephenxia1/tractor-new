// The zero bot's declaration and burial networks against the Python ones,
// on a small random-weight export (test/fixtures/zero-tiny-preplay, made by
// research/py/test_zero_preplay_parity.py). The full-size check over real
// deals is that script's `check` on a research/src/zeroPrePlayParity.ts
// dump; this one keeps the port honest without the 22 MB weights.

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { applyEvent, createGame, defaultV1Config, type GameEvent, type GameState } from '../src';
import { buryChoice, buryCardTokens, buryContext, buryLogits, zeroBury } from '../src/rl/zero/bury';
import {
  declareChoice,
  declareFeatures,
  declareLogits,
  declareView,
  zeroDeclare,
  type DeclareView,
} from '../src/rl/zero/declare';
import { erf } from '../src/rl/zero/preplayNn';
import { loadZeroWeights } from '../src/rl/zero/weights';

const dir = join(dirname(fileURLToPath(import.meta.url)), 'fixtures', 'zero-tiny-preplay');
const w = loadZeroWeights(dir);
type DeclareCase = {
  seat: number; level_rank: number; landlord: number; dealt: number; total: number;
  closing: boolean; standing_tier: number; standing_suit: number; standing_seat: number;
  hand_counts: number[]; mask: boolean[]; features: string; logits: number[]; choice: number;
};
type BuryCase = {
  level_rank: number; trump_suit: number; standing_tier: number; declarer_rel: number;
  pool: number[]; counts: number[]; step0_logits: number[];
};
const cases = JSON.parse(readFileSync(join(dir, 'cases.json'), 'utf-8')) as {
  declare: DeclareCase[];
  bury: BuryCase[];
};

const view = (c: DeclareCase): DeclareView => ({
  seat: c.seat, levelRank: c.level_rank, landlord: c.landlord, dealt: c.dealt, total: c.total,
  closing: c.closing, standingTier: c.standing_tier, standingSuit: c.standing_suit,
  standingSeat: c.standing_seat, hand: c.hand_counts, mask: c.mask, numDecks: 2,
});

describe('zero pre-play networks', () => {
  it('erf matches double precision', () => {
    // math.erf at a spread of points, from Python.
    const want: Array<[number, number]> = [
      [0, 0], [1e-10, 1.1283791670955126e-10], [0.3, 0.3286267594591274],
      [0.84375, 0.7672256612323416], [1, 0.8427007929497149], [-1.2, -0.9103139782296353],
      [2, 0.9953222650189527], [3, 0.9999779095030014], [-4.5, -0.9999999998033839], [7, 1],
    ];
    for (const [x, y] of want) expect(erf(x)).toBeCloseTo(y, 14);
  });

  it('declaration features are bit-identical to Python', () => {
    expect(cases.declare.length).toBeGreaterThan(10);
    for (const c of cases.declare) {
      const got = declareFeatures(view(c));
      const want = new Float32Array(Uint8Array.from(Buffer.from(c.features, 'base64')).buffer);
      expect(Array.from(got)).toEqual(Array.from(want));
    }
  });

  it('declaration logits and greedy choices match Python', () => {
    for (const c of cases.declare) {
      const logits = declareLogits(w, declareFeatures(view(c)));
      for (let a = 0; a < 11; a++) expect(Math.abs(logits[a]! - c.logits[a]!)).toBeLessThan(1e-4);
      expect(declareChoice(w, view(c))).toBe(c.choice);
    }
  });

  it('burial logits and the eight greedy picks match Python', () => {
    expect(cases.bury.length).toBeGreaterThan(10);
    for (const c of cases.bury) {
      const v = {
        levelRank: c.level_rank, trumpSuit: c.trump_suit, standingTier: c.standing_tier,
        declarerRel: c.declarer_rel, pool: c.pool, numDecks: 2,
      };
      const { cards, mask } = buryCardTokens(v, new Array<number>(54).fill(0));
      const logits = buryLogits(w, cards, buryContext(v, 0));
      for (let t = 0; t < 54; t++) {
        if (mask[t]) expect(Math.abs(logits[t]! - c.step0_logits[t]!)).toBeLessThan(1e-4);
      }
      expect(buryChoice(w, v).counts).toEqual(c.counts);
    }
  });

  it('plays a live deal: every declaration and the burial are accepted', () => {
    const apply = (s: GameState, e: GameEvent): GameState => {
      const r = applyEvent(s, e);
      if (!r.ok) throw new Error(`${e.kind}: ${r.error}`);
      return r.state;
    };
    for (const landlord of [null, 2] as const) {
      let s = createGame({ ...defaultV1Config(), incrementalDeal: true });
      s = apply({ ...s, landlord }, { kind: 'start_hand', seed: 12345 });
      let declared = 0;
      for (let i = 0; i < 100; i++) {
        s = apply(s, { kind: 'reveal_card', seat: i % 4 });
        for (let seat = 0; seat < 4; seat++) {
          const a = zeroDeclare(w, s, seat);
          if (a === null) {
            expect(declareView(s, seat) === null || declareChoice(w, declareView(s, seat)!) === 0).toBe(true);
            continue;
          }
          expect(a.kind).toBe('declare');
          if (a.kind === 'declare') s = apply(s, { kind: 'declare', seat, cards: a.cards });
          declared += 1;
        }
      }
      s = apply(s, { kind: 'finish_dealing' });
      s = apply(s, { kind: 'release_lockout', seed: 7 });
      if (s.phase === 'declare_open') {
        s = apply(s, { kind: 'pick_up_kitty', seat: s.currentHand!.landlord! });
      }
      expect(s.phase).toBe('kitty_replace');
      const ll = s.currentHand!.landlord!;
      if (landlord !== null) expect(ll).toBe(landlord);
      expect(declared).toBeGreaterThanOrEqual(0);
      const kitty = zeroBury(w, s, ll);
      expect(kitty.length).toBe(8);
      s = apply(s, { kind: 'replace_kitty', seat: ll, cards: kitty });
      expect(s.phase).toBe('play');
    }
  });
});
