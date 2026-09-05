// The 54-slot card index shared by the TS environment and the Python
// featurizer. A "slot" identifies a card by (suit, rank) only — the two
// physical copies in a 2-deck game share a slot, so a plane of slot
// counts in {0,1,2} is a complete description of a card multiset.
//
// Layout (must match research/spec.json, consumed by py/tractor_rl):
//   slot = suitIndex * 13 + rankIndex   for standard cards
//   52 = small joker, 53 = big joker
//
// Suit order is the engine's SUITS (spades, hearts, diamonds, clubs);
// rank order is NATURAL_RANKS (2..A). The featurizer permutes slots into
// a trump-anchored order at training time — the wire format stays raw so
// TypeScript never has to know about the network's view of the board.

import {
  NATURAL_RANKS,
  SUITS,
  cardKey,
  type Card,
  type NaturalRank,
  type Suit,
} from '@tractor/engine';

export const NUM_SLOTS = 54;
export const NUM_RANKS = NATURAL_RANKS.length; // 13
export const NUM_SUITS = SUITS.length; // 4
export const SMALL_JOKER_SLOT = 52;
export const BIG_JOKER_SLOT = 53;

const SLOT_BY_KEY = new Map<string, number>();
const KEY_BY_SLOT: string[] = new Array<string>(NUM_SLOTS).fill('');

for (let s = 0; s < SUITS.length; s++) {
  for (let r = 0; r < NATURAL_RANKS.length; r++) {
    const slot = s * NUM_RANKS + r;
    const key = `${SUITS[s]!}:${NATURAL_RANKS[r]!}`;
    SLOT_BY_KEY.set(key, slot);
    KEY_BY_SLOT[slot] = key;
  }
}
SLOT_BY_KEY.set('joker:small', SMALL_JOKER_SLOT);
SLOT_BY_KEY.set('joker:big', BIG_JOKER_SLOT);
KEY_BY_SLOT[SMALL_JOKER_SLOT] = 'joker:small';
KEY_BY_SLOT[BIG_JOKER_SLOT] = 'joker:big';

export function cardSlot(card: Card): number {
  const slot = SLOT_BY_KEY.get(cardKey(card));
  if (slot === undefined) throw new Error(`no slot for card ${cardKey(card)}`);
  return slot;
}

export function slotKey(slot: number): string {
  const key = KEY_BY_SLOT[slot];
  if (key === undefined || key === '') throw new Error(`bad slot ${slot}`);
  return key;
}

export function suitIndex(suit: Suit): number {
  return SUITS.indexOf(suit);
}

export function rankIndex(rank: NaturalRank): number {
  return NATURAL_RANKS.indexOf(rank);
}

// Accumulate a card multiset into a 54-slot count plane at `offset`.
export function addToPlane(
  cards: ReadonlyArray<Card>,
  out: Uint8Array,
  offset: number,
): void {
  for (const c of cards) {
    const idx = offset + cardSlot(c);
    out[idx] = out[idx]! + 1;
  }
}
