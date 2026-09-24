// Dumps declaration scenarios decided by the TypeScript engine, so the
// Python deal environment (py/tractor/sim/deal.py) can be checked against
// the rules the game actually enforces.
//
// Each scenario is a landlord-set hand in the declaration window: a random
// shuffle, a random landlord and level, and zero to three declarations
// already made by random seats. For one more random seat it records which
// of the 11 fixed actions the engine accepts -- pass, a single or a pair
// of the level rank in each suit, the small- and big-joker pairs -- by
// applying each and keeping the ones that come back ok.
//
//   npx tsx src/dumpDeclare.ts 600 20260924 py/tractor/tests/fixtures/declare.json

import { writeFileSync } from 'node:fs';

import {
  NATURAL_RANKS,
  SUITS,
  applyEvent,
  createGame,
  defaultV1Config,
  mulberry32,
  type Card,
  type GameEvent,
  type GameState,
  type Seat,
} from '@tractor/engine';

import { cardSlot } from './cards';

const NUM_ACTIONS = 11;

// The cards action `a` declares with, from `hand`, or null if not held.
function cardsFor(hand: ReadonlyArray<Card>, level: string, a: number): Card[] | null {
  const pick = (pred: (c: Card) => boolean, n: number): Card[] | null => {
    const got = hand.filter(pred);
    return got.length >= n ? got.slice(0, n) : null;
  };
  if (a >= 1 && a <= 4) {
    const suit = SUITS[a - 1];
    return pick((c) => c.kind === 'standard' && c.suit === suit && c.rank === level, 1);
  }
  if (a >= 5 && a <= 8) {
    const suit = SUITS[a - 5];
    return pick((c) => c.kind === 'standard' && c.suit === suit && c.rank === level, 2);
  }
  if (a === 9) return pick((c) => c.kind === 'joker' && c.rank === 'small', 2);
  if (a === 10) return pick((c) => c.kind === 'joker' && c.rank === 'big', 2);
  return null;
}

function accepted(state: GameState, seat: Seat, level: string): boolean[] {
  const hand = state.currentHand!.hands[seat] ?? [];
  const out: boolean[] = [true];
  for (let a = 1; a < NUM_ACTIONS; a++) {
    const cards = cardsFor(hand, level, a);
    if (cards === null) {
      out.push(false);
      continue;
    }
    const ev: GameEvent = { kind: 'declare', seat, cards };
    out.push(applyEvent(state, ev).ok);
  }
  return out;
}

function main(): void {
  const n = Number(process.argv[2] ?? 600);
  const seed = Number(process.argv[3] ?? 20260924);
  const path = process.argv[4];
  const rng = mulberry32(seed);
  const config = defaultV1Config();
  const scenarios: unknown[] = [];

  for (let i = 0; i < n; i++) {
    const landlord = Math.floor(rng() * 4) as Seat;
    const levelIndex = Math.floor(rng() * NATURAL_RANKS.length);
    const level = NATURAL_RANKS[levelIndex]!;
    let state: GameState = {
      ...createGame(config),
      landlord,
      teamLevels: { A: level, B: level },
    };
    const started = applyEvent(state, { kind: 'start_hand', seed: Math.floor(rng() * 0x7fffffff) });
    if (!started.ok) throw new Error(`start_hand: ${started.error}`);
    state = started.state;

    const tierOf = [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4];
    let standingTier = 0;
    const prior = Math.floor(rng() * 4);
    for (let k = 0; k < prior; k++) {
      const seat = Math.floor(rng() * 4) as Seat;
      const ok = accepted(state, seat, level);
      const choices = ok.map((x, a) => (x && a > 0 ? a : -1)).filter((a) => a > 0);
      if (choices.length === 0) continue;
      const a = choices[Math.floor(rng() * choices.length)]!;
      const r = applyEvent(state, {
        kind: 'declare',
        seat,
        cards: cardsFor(state.currentHand!.hands[seat] ?? [], level, a)!,
      });
      if (!r.ok) throw new Error(`declare refused after being accepted: ${r.error}`);
      state = r.state;
      standingTier = tierOf[a]!;
    }

    const target = Math.floor(rng() * 4) as Seat;
    const hand = state.currentHand!.hands[target] ?? [];
    const counts = new Array<number>(54).fill(0);
    for (const c of hand) counts[cardSlot(c)]! += 1;
    scenarios.push({
      levelRank: levelIndex,
      landlord,
      seat: target,
      standingTier,
      hand: counts,
      mask: accepted(state, target, level),
    });
  }
  const text = JSON.stringify({ seed, scenarios });
  if (path) writeFileSync(path, text);
  else process.stdout.write(text);
}

main();
