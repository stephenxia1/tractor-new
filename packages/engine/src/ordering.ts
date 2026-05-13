import {
  NATURAL_RANKS,
  SUITS,
  type Card,
  type EffectiveSuit,
  type NaturalRank,
  type Suit,
  type TrumpContext,
} from './types';

const NATURAL_RANK_INDEX: Readonly<Record<NaturalRank, number>> =
  Object.fromEntries(
    NATURAL_RANKS.map((r, i) => [r, i] as const),
  ) as Record<NaturalRank, number>;

const SUIT_INDEX: Readonly<Record<Suit, number>> = Object.fromEntries(
  SUITS.map((s, i) => [s, i] as const),
) as Record<Suit, number>;

export function naturalRankIndex(rank: NaturalRank): number {
  return NATURAL_RANK_INDEX[rank];
}

export function isTrump(card: Card, ctx: TrumpContext): boolean {
  if (card.kind === 'joker') return true;
  if (card.rank === ctx.trumpRank) return true;
  if (ctx.trumpSuit !== null && card.suit === ctx.trumpSuit) return true;
  return false;
}

export function effectiveSuit(card: Card, ctx: TrumpContext): EffectiveSuit {
  if (isTrump(card, ctx)) return 'trump';
  if (card.kind === 'standard') return card.suit;
  return 'trump'; // unreachable: jokers are always trump
}

// Tier within the trump category (rules.md §3.2):
//   1 = natural trump-suit ranks (not R)
//   2 = off-suit-R (rank R, non-trump suit)
//   3 = trump-suit-R
//   4 = small joker
//   5 = big joker
// Returns 0 for non-trump (callers should gate on `isTrump`).
export function trumpTier(card: Card, ctx: TrumpContext): number {
  if (!isTrump(card, ctx)) return 0;
  if (card.kind === 'joker') return card.rank === 'big' ? 5 : 4;
  if (card.rank === ctx.trumpRank) {
    if (ctx.trumpSuit !== null && card.suit === ctx.trumpSuit) return 3;
    return 2;
  }
  return 1;
}

// Total order over all cards under a trump context. Returns negative,
// zero, or positive for a < b, equal, a > b respectively.
//
// Equality (return 0) covers:
//   - Two cards with the same key and same tier (e.g., two K♥).
//   - Two distinct off-suit-R cards (tier 2 is flat across suits).
//
// Non-trump cards of different suits get a deterministic ordering by
// `SUITS` index — useful for sorting hands but NOT meaningful for
// trick-winning (cross-suit comparisons never decide tricks).
export function compareCards(a: Card, b: Card, ctx: TrumpContext): number {
  const aTrump = isTrump(a, ctx);
  const bTrump = isTrump(b, ctx);

  if (aTrump !== bTrump) return aTrump ? 1 : -1;

  if (aTrump) {
    const aTier = trumpTier(a, ctx);
    const bTier = trumpTier(b, ctx);
    if (aTier !== bTier) return aTier - bTier;
    if (aTier === 1 && a.kind === 'standard' && b.kind === 'standard') {
      return naturalRankIndex(a.rank) - naturalRankIndex(b.rank);
    }
    return 0;
  }

  // Both non-trump → both must be standard (jokers are always trump).
  if (a.kind !== 'standard' || b.kind !== 'standard') return 0;
  if (a.suit !== b.suit) return SUIT_INDEX[a.suit] - SUIT_INDEX[b.suit];
  return naturalRankIndex(a.rank) - naturalRankIndex(b.rank);
}

// Like `compareCards`, but breaks ties (e.g., off-suit-R cards within
// tier 2) by suit order. This is a *display* convenience — game-rules
// resolution should keep using `compareCards`, since rules.md §3.2 treats
// tier 2 cards as equal in rank (ties broken by play order, not by suit).
export function compareCardsForDisplay(
  a: Card,
  b: Card,
  ctx: TrumpContext,
): number {
  const primary = compareCards(a, b, ctx);
  if (primary !== 0) return primary;
  if (a.kind === 'standard' && b.kind === 'standard' && a.suit !== b.suit) {
    return SUIT_INDEX[a.suit] - SUIT_INDEX[b.suit];
  }
  return 0;
}
