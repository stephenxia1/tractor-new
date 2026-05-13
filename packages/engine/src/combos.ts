import {
  cardKey,
  type Card,
  type EffectiveSuit,
  type NaturalRank,
  type TrumpContext,
} from './types';
import {
  effectiveSuit,
  isTrump,
  naturalRankIndex,
  trumpTier,
} from './ordering';

// A "single" is one card.
export type Single = {
  readonly kind: 'single';
  readonly card: Card;
};

// An "n-of-a-kind": n cards sharing the same physical identity (same
// suit + rank, or same joker rank). Generalizes "pair" (n=2) to multi-deck
// variants where n can be 3, 4, etc.
export type NKind = {
  readonly kind: 'nkind';
  readonly n: number;
  readonly cards: ReadonlyArray<Card>;
};

// A "tractor": `length` consecutive n-of-a-kinds in the same effective-suit
// chain. Total cards = n × length. `units` lists the n-of-a-kinds in chain
// order (low → high position).
export type Tractor = {
  readonly kind: 'tractor';
  readonly n: number;
  readonly length: number;
  readonly cards: ReadonlyArray<Card>;
  readonly units: ReadonlyArray<NKind>;
};

export type Combo = Single | NKind | Tractor;

// Position of a card within its effective-suit "chain" (rules.md §7.2).
// Two cards in the same chain at positions p and p+1 are tractor-consecutive.
// Cards in different chains (different non-trump suits, or non-trump vs
// trump) have positions that are NOT directly comparable.
//
// Regular trump (trumpSuit != null): tier 1 occupies positions 0..11
// (natural trump-suit ranks with R skipped), tier 2 = 12 (off-suit-R, flat),
// tier 3 = 13 (trump-suit-R), tier 4 = 14 (small joker), tier 5 = 15 (big).
//
// No-trump (trumpSuit = null): tier 2 = 0, tier 4 = 1, tier 5 = 2. Tiers
// 1 and 3 are empty in no-trump mode.
//
// Non-trump suits: position = natural rank index with R skipped (0..11).
export function chainPosition(card: Card, ctx: TrumpContext): number {
  if (!isTrump(card, ctx)) {
    if (card.kind === 'standard') {
      return rankPositionSkippingR(card.rank, ctx.trumpRank);
    }
    return -1; // unreachable: jokers are always trump
  }
  const tier = trumpTier(card, ctx);
  if (ctx.trumpSuit === null) {
    if (tier === 2) return 0;
    if (tier === 4) return 1;
    return 2; // tier 5
  }
  if (tier === 1 && card.kind === 'standard') {
    return rankPositionSkippingR(card.rank, ctx.trumpRank);
  }
  if (tier === 2) return 12;
  if (tier === 3) return 13;
  if (tier === 4) return 14;
  return 15; // tier 5
}

function rankPositionSkippingR(rank: NaturalRank, R: NaturalRank): number {
  const rIdx = naturalRankIndex(R);
  const cardIdx = naturalRankIndex(rank);
  return cardIdx < rIdx ? cardIdx : cardIdx - 1;
}

// All n-of-a-kinds present in `cards`. Returns one canonical n-of-a-kind
// per `cardKey` group of size ≥ n (using the first n matching cards).
//
// With v1 (numDecks = 2, n = 2), each cardKey appears at most 2 times in a
// hand, so each pair is uniquely determined. With more decks, multiple
// choices exist; this function returns the canonical pick.
export function findNKinds(
  cards: ReadonlyArray<Card>,
  n: number,
): NKind[] {
  if (n < 2 || !Number.isInteger(n)) {
    throw new Error(`n must be an integer >= 2, got ${n}`);
  }
  const groups = new Map<string, Card[]>();
  for (const c of cards) {
    const key = cardKey(c);
    const g = groups.get(key) ?? [];
    g.push(c);
    groups.set(key, g);
  }
  const result: NKind[] = [];
  for (const g of groups.values()) {
    if (g.length >= n) {
      result.push({ kind: 'nkind', n, cards: g.slice(0, n) });
    }
  }
  return result;
}

// All tractors of length ≥ `minLength` (default 2) formable from `cards`.
// A tractor is `k` consecutive n-of-a-kinds (k ≥ minLength) in the same
// effective-suit chain (rules.md §7.2).
//
// Returns one tractor per (chain, sub-run of consecutive chain positions).
// A maximal run of length L contributes L-1 length-2 tractors, L-2
// length-3 tractors, ..., 1 length-L tractor — i.e., `triangular(L-1)`
// total. Callers wanting only maximal tractors can filter afterward.
//
// Caveat: when multiple n-of-a-kinds exist at the same chain position
// (e.g., 5♠5♠ and 5♦5♦ at the off-suit-R tier with R=5), a canonical
// pick is used. Higher-level play logic should choose among alternatives
// where it matters.
export function findTractors(
  cards: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
  minLength = 2,
): Tractor[] {
  if (minLength < 2 || !Number.isInteger(minLength)) {
    throw new Error(`minLength must be an integer >= 2, got ${minLength}`);
  }
  const nkinds = findNKinds(cards, n);

  // Group n-of-a-kinds by effective suit.
  const byChain = new Map<EffectiveSuit, NKind[]>();
  for (const nk of nkinds) {
    const c0 = nk.cards[0]!;
    const es = effectiveSuit(c0, ctx);
    const list = byChain.get(es) ?? [];
    list.push(nk);
    byChain.set(es, list);
  }

  const result: Tractor[] = [];

  for (const group of byChain.values()) {
    // Bucket by chain position.
    const byPos = new Map<number, NKind[]>();
    for (const nk of group) {
      const pos = chainPosition(nk.cards[0]!, ctx);
      const list = byPos.get(pos) ?? [];
      list.push(nk);
      byPos.set(pos, list);
    }
    const positions = [...byPos.keys()].sort((a, b) => a - b);

    // Walk to find maximal runs of consecutive positions.
    let runStart = 0;
    while (runStart < positions.length) {
      let runEnd = runStart;
      while (
        runEnd + 1 < positions.length &&
        positions[runEnd + 1]! - positions[runEnd]! === 1
      ) {
        runEnd++;
      }
      const runLen = runEnd - runStart + 1;

      // Emit every sub-run of length >= minLength.
      for (let len = minLength; len <= runLen; len++) {
        for (let start = runStart; start + len - 1 <= runEnd; start++) {
          const units: NKind[] = [];
          for (let i = 0; i < len; i++) {
            const pos = positions[start + i]!;
            const candidates = byPos.get(pos)!;
            units.push(candidates[0]!);
          }
          result.push({
            kind: 'tractor',
            n,
            length: len,
            cards: units.flatMap((u) => [...u.cards]),
            units,
          });
        }
      }

      runStart = runEnd + 1;
    }
  }

  return result;
}
