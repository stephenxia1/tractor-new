// Enumerates every declaration a player can make from a given hand under
// trump rank R and the currently-standing declaration.
//
// Tier-filtered: returns only declarations that would *override* the
// standing one (per rules.md §5.1). Pre-filtered so bots / UIs don't
// have to recompute tier comparisons.

import type { Declaration } from './state';
import type { Card, JokerRank, NaturalRank, Suit } from './types';

export type AvailableDeclaration =
  | {
      readonly kind: 'single';
      readonly suit: Suit;
      readonly cards: readonly [Card];
      readonly tier: 1;
    }
  | {
      readonly kind: 'pair';
      readonly suit: Suit;
      readonly cards: readonly [Card, Card];
      readonly tier: 2;
    }
  | {
      readonly kind: 'joker-pair';
      readonly jokerRank: JokerRank;
      readonly cards: readonly [Card, Card];
      readonly tier: 3 | 4;
    };

function standingTier(d: Declaration | null): number {
  if (d === null) return 0;
  if (d.kind === 'single') return 1;
  if (d.kind === 'pair') return 2;
  return d.jokerRank === 'small' ? 3 : 4;
}

export function availableDeclarationsFor(
  hand: ReadonlyArray<Card>,
  R: NaturalRank,
  standing: Declaration | null,
): ReadonlyArray<AvailableDeclaration> {
  const out: AvailableDeclaration[] = [];

  const bySuit = new Map<Suit, Card[]>();
  for (const c of hand) {
    if (c.kind === 'standard' && c.rank === R) {
      const list = bySuit.get(c.suit) ?? [];
      list.push(c);
      bySuit.set(c.suit, list);
    }
  }

  // Singles — one per suit you hold rank-R in.
  for (const [suit, cards] of bySuit) {
    out.push({ kind: 'single', suit, cards: [cards[0]!], tier: 1 });
  }
  // Pairs — both copies in the same suit.
  for (const [suit, cards] of bySuit) {
    if (cards.length >= 2 && cards[0]!.deckId !== cards[1]!.deckId) {
      out.push({
        kind: 'pair',
        suit,
        cards: [cards[0]!, cards[1]!],
        tier: 2,
      });
    }
  }
  // Joker pairs (small then big).
  const small = hand.filter((c) => c.kind === 'joker' && c.rank === 'small');
  if (small.length >= 2 && small[0]!.deckId !== small[1]!.deckId) {
    out.push({
      kind: 'joker-pair',
      jokerRank: 'small',
      cards: [small[0]!, small[1]!],
      tier: 3,
    });
  }
  const big = hand.filter((c) => c.kind === 'joker' && c.rank === 'big');
  if (big.length >= 2 && big[0]!.deckId !== big[1]!.deckId) {
    out.push({
      kind: 'joker-pair',
      jokerRank: 'big',
      cards: [big[0]!, big[1]!],
      tier: 4,
    });
  }

  const minTier = standingTier(standing);
  return out.filter((d) => d.tier > minTier);
}
