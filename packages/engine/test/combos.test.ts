import { describe, it, expect } from 'vitest';
import {
  chainPosition,
  findNKinds,
  findTractors,
  type Card,
  type JokerRank,
  type NaturalRank,
  type Suit,
  type TrumpContext,
} from '../src';

// Test fixture helpers — keep card construction terse.
const std = (suit: Suit, rank: NaturalRank, deckId: number): Card => ({
  kind: 'standard',
  suit,
  rank,
  deckId,
});
const jok = (rank: JokerRank, deckId: number): Card => ({
  kind: 'joker',
  rank,
  deckId,
});
const pair = (suit: Suit, rank: NaturalRank): Card[] => [std(suit, rank, 0), std(suit, rank, 1)];
const jPair = (rank: JokerRank): Card[] => [jok(rank, 0), jok(rank, 1)];

describe('chainPosition (R=5, S=hearts)', () => {
  const ctx: TrumpContext = { trumpRank: '5', trumpSuit: 'hearts' };

  it('trump-suit naturals occupy positions 0..11 (skipping 5♥)', () => {
    expect(chainPosition(std('hearts', '2', 0), ctx)).toBe(0);
    expect(chainPosition(std('hearts', '3', 0), ctx)).toBe(1);
    expect(chainPosition(std('hearts', '4', 0), ctx)).toBe(2);
    expect(chainPosition(std('hearts', '6', 0), ctx)).toBe(3);
    expect(chainPosition(std('hearts', '7', 0), ctx)).toBe(4);
    expect(chainPosition(std('hearts', 'K', 0), ctx)).toBe(10);
    expect(chainPosition(std('hearts', 'A', 0), ctx)).toBe(11);
  });

  it('off-suit-R tier → position 12 (all suits)', () => {
    expect(chainPosition(std('spades', '5', 0), ctx)).toBe(12);
    expect(chainPosition(std('diamonds', '5', 0), ctx)).toBe(12);
    expect(chainPosition(std('clubs', '5', 0), ctx)).toBe(12);
  });

  it('trump-suit-R → 13, small joker → 14, big joker → 15', () => {
    expect(chainPosition(std('hearts', '5', 0), ctx)).toBe(13);
    expect(chainPosition(jok('small', 0), ctx)).toBe(14);
    expect(chainPosition(jok('big', 0), ctx)).toBe(15);
  });

  it('non-trump suit positions match rank order with R skipped', () => {
    expect(chainPosition(std('clubs', '2', 0), ctx)).toBe(0);
    expect(chainPosition(std('clubs', '4', 0), ctx)).toBe(2);
    expect(chainPosition(std('clubs', '6', 0), ctx)).toBe(3); // R=5 skipped
    expect(chainPosition(std('clubs', 'A', 0), ctx)).toBe(11);
  });
});

describe('chainPosition (R=A, S=spades) — R at the top', () => {
  const ctx: TrumpContext = { trumpRank: 'A', trumpSuit: 'spades' };

  it('trump-suit-K → 11 (highest tier 1), A♠ → 13 (tier 3)', () => {
    expect(chainPosition(std('spades', 'K', 0), ctx)).toBe(11);
    expect(chainPosition(std('spades', 'A', 0), ctx)).toBe(13);
  });
});

describe('chainPosition (R=2, S=diamonds) — R at the bottom', () => {
  const ctx: TrumpContext = { trumpRank: '2', trumpSuit: 'diamonds' };

  it('trump-suit-3 → 0 (lowest tier 1), 2♦ → 13 (tier 3)', () => {
    expect(chainPosition(std('diamonds', '3', 0), ctx)).toBe(0);
    expect(chainPosition(std('diamonds', '2', 0), ctx)).toBe(13);
  });
});

describe('chainPosition (no-trump, R=5)', () => {
  const ctx: TrumpContext = { trumpRank: '5', trumpSuit: null };

  it('off-suit-R → 0, small joker → 1, big joker → 2', () => {
    expect(chainPosition(std('spades', '5', 0), ctx)).toBe(0);
    expect(chainPosition(std('hearts', '5', 0), ctx)).toBe(0);
    expect(chainPosition(jok('small', 0), ctx)).toBe(1);
    expect(chainPosition(jok('big', 0), ctx)).toBe(2);
  });
});

describe('findNKinds', () => {
  it('finds one pair per cardKey', () => {
    const hand: Card[] = [
      ...pair('hearts', 'K'),
      ...pair('clubs', '3'),
      std('spades', '8', 0), // singleton — no pair
    ];
    expect(findNKinds(hand, 2)).toHaveLength(2);
  });

  it('empty hand → no n-of-a-kinds', () => {
    expect(findNKinds([], 2)).toEqual([]);
  });

  it('rejects n < 2 or non-integer n', () => {
    expect(() => findNKinds([], 1)).toThrow();
    expect(() => findNKinds([], 1.5)).toThrow();
    expect(() => findNKinds([], 0)).toThrow();
  });

  it('joker pairs detected', () => {
    expect(findNKinds(jPair('small'), 2)).toHaveLength(1);
    expect(findNKinds(jPair('big'), 2)).toHaveLength(1);
  });

  it('with 3 decks and n=3, finds triplets', () => {
    const hand: Card[] = [
      std('hearts', 'K', 0),
      std('hearts', 'K', 1),
      std('hearts', 'K', 2),
    ];
    expect(findNKinds(hand, 3)).toHaveLength(1);
    expect(findNKinds(hand, 3)[0]!.cards).toHaveLength(3);
  });
});

describe('findTractors — non-trump', () => {
  const ctx: TrumpContext = { trumpRank: '5', trumpSuit: 'hearts' };

  it('8♣8♣ + 9♣9♣ → length-2 tractor', () => {
    const tractors = findTractors([...pair('clubs', '8'), ...pair('clubs', '9')], ctx, 2);
    expect(tractors).toHaveLength(1);
    expect(tractors[0]!.length).toBe(2);
  });

  it('4♣4♣ + 6♣6♣ → length-2 tractor (R=5 is skipped)', () => {
    const tractors = findTractors([...pair('clubs', '4'), ...pair('clubs', '6')], ctx, 2);
    expect(tractors).toHaveLength(1);
  });

  it('8♣8♣ + 9♠9♠ → no tractor (different chains)', () => {
    const tractors = findTractors([...pair('clubs', '8'), ...pair('spades', '9')], ctx, 2);
    expect(tractors).toHaveLength(0);
  });

  it('3♣3♣ + 4♣4♣ + 6♣6♣ → length-2 + length-3 sub-runs (3 tractors)', () => {
    // Maximal run length 3 → triangular(2) = 3 tractors
    // (length-2 starting at 0, length-2 starting at 1, length-3 starting at 0)
    const tractors = findTractors(
      [...pair('clubs', '3'), ...pair('clubs', '4'), ...pair('clubs', '6')],
      ctx,
      2,
    );
    expect(tractors).toHaveLength(3);
    expect(tractors.filter((t) => t.length === 2)).toHaveLength(2);
    expect(tractors.filter((t) => t.length === 3)).toHaveLength(1);
  });

  it('gap breaks the run: 3♣3♣ + 4♣4♣ + 7♣7♣ → one length-2 tractor', () => {
    const tractors = findTractors(
      [...pair('clubs', '3'), ...pair('clubs', '4'), ...pair('clubs', '7')],
      ctx,
      2,
    );
    expect(tractors).toHaveLength(1);
  });
});

describe('findTractors — trump tier bridging (R=5, S=hearts)', () => {
  const ctx: TrumpContext = { trumpRank: '5', trumpSuit: 'hearts' };

  it('A♥A♥ + 5♠5♠ → length-2 tractor (bridges tier 1↔2)', () => {
    const tractors = findTractors([...pair('hearts', 'A'), ...pair('spades', '5')], ctx, 2);
    expect(tractors).toHaveLength(1);
    expect(tractors[0]!.length).toBe(2);
  });

  it('5♦5♦ + 5♥5♥ → length-2 tractor (bridges tier 2↔3)', () => {
    const tractors = findTractors([...pair('diamonds', '5'), ...pair('hearts', '5')], ctx, 2);
    expect(tractors).toHaveLength(1);
  });

  it('5♥5♥ + small-joker pair → length-2 tractor (bridges tier 3↔4)', () => {
    const tractors = findTractors([...pair('hearts', '5'), ...jPair('small')], ctx, 2);
    expect(tractors).toHaveLength(1);
  });

  it('small-joker pair + big-joker pair → length-2 tractor (bridges tier 4↔5)', () => {
    const tractors = findTractors([...jPair('small'), ...jPair('big')], ctx, 2);
    expect(tractors).toHaveLength(1);
  });

  it('5♠5♠ + 5♦5♦ alone is NOT a tractor (both tier 2, no internal consecutivity)', () => {
    const tractors = findTractors([...pair('spades', '5'), ...pair('diamonds', '5')], ctx, 2);
    expect(tractors).toHaveLength(0);
  });

  it('A♥A♥ + 5♠5♠ + 5♦5♦ → only the A♥A♥+5♠5♠ length-2 tractor (5♦ same tier as 5♠)', () => {
    const tractors = findTractors(
      [...pair('hearts', 'A'), ...pair('spades', '5'), ...pair('diamonds', '5')],
      ctx,
      2,
    );
    expect(tractors).toHaveLength(1);
    expect(tractors[0]!.length).toBe(2);
  });

  it('full trump tractor (K♥, A♥, off-suit-R, 5♥, small, big) → 15 sub-tractors of length ≥ 2', () => {
    // Run length 6 over positions 10..15. Sub-runs ≥ 2: triangular(5) = 15.
    const hand = [
      ...pair('hearts', 'K'),
      ...pair('hearts', 'A'),
      ...pair('clubs', '5'),
      ...pair('hearts', '5'),
      ...jPair('small'),
      ...jPair('big'),
    ];
    const tractors = findTractors(hand, ctx, 2);
    expect(tractors).toHaveLength(15);
    expect(Math.max(...tractors.map((t) => t.length))).toBe(6);
  });

  it('Q♥Q♥ + K♥K♥ + A♥A♥ → 3 tractors (run length 3 in tier 1)', () => {
    const hand = [...pair('hearts', 'Q'), ...pair('hearts', 'K'), ...pair('hearts', 'A')];
    const tractors = findTractors(hand, ctx, 2);
    expect(tractors).toHaveLength(3);
  });
});

describe('findTractors — no-trump', () => {
  const ctx: TrumpContext = { trumpRank: '5', trumpSuit: null };

  it('small-joker pair + big-joker pair → length-2 tractor', () => {
    expect(findTractors([...jPair('small'), ...jPair('big')], ctx, 2)).toHaveLength(1);
  });

  it('off-suit-R pair + small-joker pair → length-2 tractor (no-trump tier bridging)', () => {
    const tractors = findTractors([...pair('spades', '5'), ...jPair('small')], ctx, 2);
    expect(tractors).toHaveLength(1);
  });

  it('off-suit-R pair + big-joker pair → NOT a tractor (small-joker tier separates them)', () => {
    const tractors = findTractors([...pair('spades', '5'), ...jPair('big')], ctx, 2);
    expect(tractors).toHaveLength(0);
  });

  it('5♠5♠ + 5♦5♦ → NOT a tractor (both at no-trump position 0)', () => {
    expect(findTractors([...pair('spades', '5'), ...pair('diamonds', '5')], ctx, 2)).toHaveLength(0);
  });
});

describe('findTractors — minLength', () => {
  const ctx: TrumpContext = { trumpRank: '5', trumpSuit: 'hearts' };

  it('minLength=3 filters out length-2 tractors', () => {
    const hand = [...pair('clubs', '3'), ...pair('clubs', '4'), ...pair('clubs', '6')];
    const all = findTractors(hand, ctx, 2);
    const lengthThreePlus = findTractors(hand, ctx, 2, 3);
    expect(all).toHaveLength(3);
    expect(lengthThreePlus).toHaveLength(1);
    expect(lengthThreePlus[0]!.length).toBe(3);
  });

  it('rejects minLength < 2', () => {
    expect(() => findTractors([], ctx, 2, 1)).toThrow();
  });
});
