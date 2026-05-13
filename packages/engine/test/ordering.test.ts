import { describe, it, expect } from 'vitest';
import {
  SUITS,
  NATURAL_RANKS,
  buildDeck,
  isTrump,
  trumpTier,
  effectiveSuit,
  compareCards,
  type Card,
  type TrumpContext,
} from '../src';

describe('buildDeck', () => {
  it('produces 54 cards per deck (52 standard + 2 jokers)', () => {
    expect(buildDeck(1)).toHaveLength(54);
    expect(buildDeck(2)).toHaveLength(108);
    expect(buildDeck(3)).toHaveLength(162);
  });

  it('contains every (suit, rank) pair once per deck', () => {
    const deck = buildDeck(2);
    for (const suit of SUITS) {
      for (const rank of NATURAL_RANKS) {
        const found = deck.filter(
          (c) => c.kind === 'standard' && c.suit === suit && c.rank === rank,
        );
        expect(found).toHaveLength(2);
      }
    }
  });

  it('contains two of each joker per deck', () => {
    const deck = buildDeck(2);
    expect(deck.filter((c) => c.kind === 'joker' && c.rank === 'small')).toHaveLength(2);
    expect(deck.filter((c) => c.kind === 'joker' && c.rank === 'big')).toHaveLength(2);
  });

  it('rejects non-positive or non-integer deck counts', () => {
    expect(() => buildDeck(0)).toThrow();
    expect(() => buildDeck(-1)).toThrow();
    expect(() => buildDeck(1.5)).toThrow();
  });
});

describe('isTrump (R=5, S=hearts)', () => {
  const ctx: TrumpContext = { trumpRank: '5', trumpSuit: 'hearts' };

  it('counts the trump set: 26 hearts + 6 off-suit 5s + 4 jokers = 36', () => {
    const deck = buildDeck(2);
    expect(deck.filter((c) => isTrump(c, ctx))).toHaveLength(36);
  });

  it('identifies all hearts as trump', () => {
    for (const rank of NATURAL_RANKS) {
      expect(isTrump({ kind: 'standard', suit: 'hearts', rank, deckId: 0 }, ctx)).toBe(true);
    }
  });

  it('identifies rank-R cards in every suit as trump', () => {
    for (const suit of SUITS) {
      expect(isTrump({ kind: 'standard', suit, rank: '5', deckId: 0 }, ctx)).toBe(true);
    }
  });

  it('identifies jokers as trump', () => {
    expect(isTrump({ kind: 'joker', rank: 'small', deckId: 0 }, ctx)).toBe(true);
    expect(isTrump({ kind: 'joker', rank: 'big', deckId: 0 }, ctx)).toBe(true);
  });

  it('rejects non-trump cards', () => {
    expect(isTrump({ kind: 'standard', suit: 'spades', rank: 'A', deckId: 0 }, ctx)).toBe(false);
    expect(isTrump({ kind: 'standard', suit: 'clubs', rank: '2', deckId: 0 }, ctx)).toBe(false);
  });
});

describe('isTrump (no-trump, R=5)', () => {
  const ctx: TrumpContext = { trumpRank: '5', trumpSuit: null };

  it('only rank-R cards (any suit) and jokers are trump — 12 total', () => {
    const deck = buildDeck(2);
    expect(deck.filter((c) => isTrump(c, ctx))).toHaveLength(12);
  });

  it('no non-R standard cards are trump', () => {
    for (const suit of SUITS) {
      for (const rank of NATURAL_RANKS) {
        if (rank === '5') continue;
        expect(isTrump({ kind: 'standard', suit, rank, deckId: 0 }, ctx)).toBe(false);
      }
    }
  });
});

describe('trumpTier (R=5, S=hearts)', () => {
  const ctx: TrumpContext = { trumpRank: '5', trumpSuit: 'hearts' };

  it('non-trump → tier 0', () => {
    expect(trumpTier({ kind: 'standard', suit: 'spades', rank: 'A', deckId: 0 }, ctx)).toBe(0);
  });

  it('natural trump suit (non-R) → tier 1', () => {
    for (const rank of NATURAL_RANKS) {
      if (rank === '5') continue;
      expect(trumpTier({ kind: 'standard', suit: 'hearts', rank, deckId: 0 }, ctx)).toBe(1);
    }
  });

  it('off-suit 5s → tier 2', () => {
    for (const suit of SUITS) {
      if (suit === 'hearts') continue;
      expect(trumpTier({ kind: 'standard', suit, rank: '5', deckId: 0 }, ctx)).toBe(2);
    }
  });

  it('trump-suit 5 → tier 3', () => {
    expect(trumpTier({ kind: 'standard', suit: 'hearts', rank: '5', deckId: 0 }, ctx)).toBe(3);
  });

  it('small joker → tier 4, big joker → tier 5', () => {
    expect(trumpTier({ kind: 'joker', rank: 'small', deckId: 0 }, ctx)).toBe(4);
    expect(trumpTier({ kind: 'joker', rank: 'big', deckId: 0 }, ctx)).toBe(5);
  });
});

describe('trumpTier (no-trump, R=5)', () => {
  const ctx: TrumpContext = { trumpRank: '5', trumpSuit: null };

  it('all rank-5 cards (any suit) → tier 2', () => {
    for (const suit of SUITS) {
      expect(trumpTier({ kind: 'standard', suit, rank: '5', deckId: 0 }, ctx)).toBe(2);
    }
  });

  it('no card lands in tier 1 or 3', () => {
    const deck = buildDeck(2);
    const tiers = deck.filter((c) => isTrump(c, ctx)).map((c) => trumpTier(c, ctx));
    expect(tiers).not.toContain(1);
    expect(tiers).not.toContain(3);
  });
});

describe('compareCards (R=5, S=hearts)', () => {
  const ctx: TrumpContext = { trumpRank: '5', trumpSuit: 'hearts' };
  const deck = buildDeck(2);
  const trumps = deck.filter((c) => isTrump(c, ctx));
  const sorted = [...trumps].sort((a, b) => compareCards(a, b, ctx));

  it('tier sequence is monotonic non-decreasing', () => {
    const tierSeq = sorted.map((c) => trumpTier(c, ctx));
    for (let i = 1; i < tierSeq.length; i++) {
      expect(tierSeq[i]!).toBeGreaterThanOrEqual(tierSeq[i - 1]!);
    }
  });

  it('tier counts match §3.2 (24/6/2/2/2)', () => {
    const tierSeq = sorted.map((c) => trumpTier(c, ctx));
    expect(tierSeq.filter((t) => t === 1)).toHaveLength(24);
    expect(tierSeq.filter((t) => t === 2)).toHaveLength(6);
    expect(tierSeq.filter((t) => t === 3)).toHaveLength(2);
    expect(tierSeq.filter((t) => t === 4)).toHaveLength(2);
    expect(tierSeq.filter((t) => t === 5)).toHaveLength(2);
  });

  it('tier 1 ranks appear in natural order (skipping R=5)', () => {
    const tier1 = sorted.filter((c) => trumpTier(c, ctx) === 1);
    const ranks = tier1
      .filter((c): c is Card & { kind: 'standard' } => c.kind === 'standard')
      .map((c) => c.rank);
    // Collapse runs of identical ranks (each appears twice — two decks).
    const unique: string[] = [];
    for (const r of ranks) {
      if (unique[unique.length - 1] !== r) unique.push(r);
    }
    expect(unique).toEqual(['2', '3', '4', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']);
  });

  it('any non-trump < any trump', () => {
    const offTrump: Card = { kind: 'standard', suit: 'spades', rank: 'A', deckId: 0 };
    const bigJoker: Card = { kind: 'joker', rank: 'big', deckId: 0 };
    const lowestTrumpHeart: Card = { kind: 'standard', suit: 'hearts', rank: '2', deckId: 0 };
    expect(compareCards(offTrump, bigJoker, ctx)).toBeLessThan(0);
    expect(compareCards(offTrump, lowestTrumpHeart, ctx)).toBeLessThan(0);
  });

  it('two copies of the same card (different deckIds) compare equal', () => {
    const a: Card = { kind: 'standard', suit: 'hearts', rank: 'K', deckId: 0 };
    const b: Card = { kind: 'standard', suit: 'hearts', rank: 'K', deckId: 1 };
    expect(compareCards(a, b, ctx)).toBe(0);
  });

  it('two off-suit-R cards of different suits compare equal (tier 2 is flat)', () => {
    const a: Card = { kind: 'standard', suit: 'spades', rank: '5', deckId: 0 };
    const b: Card = { kind: 'standard', suit: 'diamonds', rank: '5', deckId: 0 };
    expect(compareCards(a, b, ctx)).toBe(0);
  });

  it('A♥ < 5♠ (tier 1 → tier 2)', () => {
    const a: Card = { kind: 'standard', suit: 'hearts', rank: 'A', deckId: 0 };
    const b: Card = { kind: 'standard', suit: 'spades', rank: '5', deckId: 0 };
    expect(compareCards(a, b, ctx)).toBeLessThan(0);
  });

  it('5♠ < 5♥ (tier 2 → tier 3)', () => {
    const a: Card = { kind: 'standard', suit: 'spades', rank: '5', deckId: 0 };
    const b: Card = { kind: 'standard', suit: 'hearts', rank: '5', deckId: 0 };
    expect(compareCards(a, b, ctx)).toBeLessThan(0);
  });

  it('5♥ < small joker < big joker', () => {
    const five: Card = { kind: 'standard', suit: 'hearts', rank: '5', deckId: 0 };
    const small: Card = { kind: 'joker', rank: 'small', deckId: 0 };
    const big: Card = { kind: 'joker', rank: 'big', deckId: 0 };
    expect(compareCards(five, small, ctx)).toBeLessThan(0);
    expect(compareCards(small, big, ctx)).toBeLessThan(0);
  });
});

describe('compareCards (R=A, S=spades)', () => {
  const ctx: TrumpContext = { trumpRank: 'A', trumpSuit: 'spades' };

  it('tier 1 contains 2♠ through K♠ (A♠ is tier 3)', () => {
    const deck = buildDeck(1);
    const tier1 = deck.filter((c) => trumpTier(c, ctx) === 1);
    expect(tier1).toHaveLength(12);
    expect(
      tier1.every(
        (c) => c.kind === 'standard' && c.suit === 'spades' && c.rank !== 'A',
      ),
    ).toBe(true);
  });
});

describe('compareCards (R=2, S=diamonds)', () => {
  const ctx: TrumpContext = { trumpRank: '2', trumpSuit: 'diamonds' };

  it('tier 1 contains 3♦ through A♦ (2♦ is tier 3)', () => {
    const deck = buildDeck(1);
    const tier1 = deck.filter((c) => trumpTier(c, ctx) === 1);
    expect(tier1).toHaveLength(12);
    expect(
      tier1.every(
        (c) => c.kind === 'standard' && c.suit === 'diamonds' && c.rank !== '2',
      ),
    ).toBe(true);
  });
});

describe('effectiveSuit', () => {
  const ctx: TrumpContext = { trumpRank: '5', trumpSuit: 'hearts' };

  it('non-trump cards return their natural suit', () => {
    expect(
      effectiveSuit({ kind: 'standard', suit: 'spades', rank: 'A', deckId: 0 }, ctx),
    ).toBe('spades');
  });

  it('trump cards return "trump"', () => {
    expect(
      effectiveSuit({ kind: 'standard', suit: 'hearts', rank: '7', deckId: 0 }, ctx),
    ).toBe('trump');
    expect(
      effectiveSuit({ kind: 'standard', suit: 'spades', rank: '5', deckId: 0 }, ctx),
    ).toBe('trump');
    expect(effectiveSuit({ kind: 'joker', rank: 'big', deckId: 0 }, ctx)).toBe('trump');
  });
});
