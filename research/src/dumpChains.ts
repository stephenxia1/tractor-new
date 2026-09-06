// Dumps the engine's own view of card ordering for every trump context, so
// the Python featuriser's chain-position table can be checked against the
// rules implementation instead of against my reading of it.
//
//   npx tsx src/dumpChains.ts > chains.json
//
// For each (trump rank, trump suit) it reports, per card slot: the
// effective suit, whether it is trump, and the chain position -- the three
// facts the encoding depends on.

import {
  NATURAL_RANKS,
  SUITS,
  chainPosition,
  effectiveSuit,
  isTrump,
  type Card,
  type NaturalRank,
  type Suit,
  type TrumpContext,
} from '@tractor/engine';

import { NUM_SLOTS } from './cards';

function cardForSlot(slot: number): Card {
  if (slot === 52) return { kind: 'joker', rank: 'small', deckId: 0 };
  if (slot === 53) return { kind: 'joker', rank: 'big', deckId: 0 };
  return {
    kind: 'standard',
    suit: SUITS[Math.floor(slot / 13)]!,
    rank: NATURAL_RANKS[slot % 13]!,
    deckId: 0,
  };
}

const out: Record<string, unknown> = {};
for (let r = 0; r < NATURAL_RANKS.length; r++) {
  for (let t = 0; t <= SUITS.length; t++) {
    const ctx: TrumpContext = {
      trumpRank: NATURAL_RANKS[r] as NaturalRank,
      trumpSuit: t < SUITS.length ? (SUITS[t] as Suit) : null,
    };
    const rows: Array<[string, boolean, number]> = [];
    for (let slot = 0; slot < NUM_SLOTS; slot++) {
      const card = cardForSlot(slot);
      rows.push([
        effectiveSuit(card, ctx),
        isTrump(card, ctx),
        chainPosition(card, ctx),
      ]);
    }
    out[`${r}:${t}`] = rows;
  }
}
process.stdout.write(JSON.stringify(out));
