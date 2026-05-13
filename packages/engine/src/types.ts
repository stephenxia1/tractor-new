export const SUITS = ['spades', 'hearts', 'diamonds', 'clubs'] as const;
export type Suit = (typeof SUITS)[number];

export const NATURAL_RANKS = [
  '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A',
] as const;
export type NaturalRank = (typeof NATURAL_RANKS)[number];

export const JOKER_RANKS = ['small', 'big'] as const;
export type JokerRank = (typeof JOKER_RANKS)[number];

export type StandardCard = {
  readonly kind: 'standard';
  readonly suit: Suit;
  readonly rank: NaturalRank;
  readonly deckId: number;
};

export type JokerCard = {
  readonly kind: 'joker';
  readonly rank: JokerRank;
  readonly deckId: number;
};

export type Card = StandardCard | JokerCard;

// Partitions cards by trick-taking suit: either a standard suit or "trump".
export type EffectiveSuit = Suit | 'trump';

// `trumpSuit = null` represents a no-trump (joker) declaration —
// only jokers and rank-R cards are trump (rules.md §3.3).
export type TrumpContext = {
  readonly trumpRank: NaturalRank;
  readonly trumpSuit: Suit | null;
};

// Identity ignoring `deckId`. Two cards with equal `cardKey` are the
// "same physical card" — pairs (n-of-a-kind with more decks) require
// multiple cards sharing a key. Used in combo detection.
export function cardKey(c: Card): string {
  if (c.kind === 'joker') return `joker:${c.rank}`;
  return `${c.suit}:${c.rank}`;
}

// Full-identity equality, including `deckId`. Two cards are `cardEquals`
// only if they refer to the exact same physical card (same key AND same
// deckId).
export function cardEquals(a: Card, b: Card): boolean {
  if (a.kind !== b.kind) return false;
  if (a.deckId !== b.deckId) return false;
  if (a.kind === 'joker' && b.kind === 'joker') return a.rank === b.rank;
  if (a.kind === 'standard' && b.kind === 'standard') {
    return a.suit === b.suit && a.rank === b.rank;
  }
  return false;
}

// Returns `true` iff every card in `subset` appears in `multiset`, treating
// both as multisets (one occurrence in `subset` matches one occurrence in
// `multiset`). Compares by `cardEquals` (full identity).
export function containsAll(
  multiset: ReadonlyArray<Card>,
  subset: ReadonlyArray<Card>,
): boolean {
  const remaining = [...multiset];
  for (const c of subset) {
    const idx = remaining.findIndex((m) => cardEquals(m, c));
    if (idx === -1) return false;
    remaining.splice(idx, 1);
  }
  return true;
}

// Removes one occurrence of each card in `toRemove` from `multiset` and
// returns the result. Throws if any card isn't present.
export function removeCards(
  multiset: ReadonlyArray<Card>,
  toRemove: ReadonlyArray<Card>,
): Card[] {
  const out = [...multiset];
  for (const c of toRemove) {
    const idx = out.findIndex((m) => cardEquals(m, c));
    if (idx === -1) {
      throw new Error(`card not in multiset: ${JSON.stringify(c)}`);
    }
    out.splice(idx, 1);
  }
  return out;
}
