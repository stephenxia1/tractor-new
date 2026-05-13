import { describe, it, expect } from 'vitest';
import {
  classifyCombo,
  validateFollow,
  type Card,
  type JokerRank,
  type NaturalRank,
  type Suit,
  type TrumpContext,
} from '../src';

const std = (suit: Suit, rank: NaturalRank, deckId: number): Card => ({
  kind: 'standard',
  suit,
  rank,
  deckId,
});
const jok = (rank: JokerRank, deckId: number): Card => ({ kind: 'joker', rank, deckId });
const pair = (suit: Suit, rank: NaturalRank): Card[] => [std(suit, rank, 0), std(suit, rank, 1)];

const CTX: TrumpContext = { trumpRank: '5', trumpSuit: 'hearts' };

function classify(cards: Card[]) {
  const r = classifyCombo(cards, CTX, 2);
  if (!r.ok) throw new Error(r.reason);
  return r.combo;
}

describe('validateFollow — single lead', () => {
  const lead = classify([std('clubs', 'K', 0)]);

  it('must play a club if you have any', () => {
    const hand = [std('clubs', '3', 0), std('spades', 'A', 0)];
    const bad = validateFollow(lead, [std('spades', 'A', 0)], hand, CTX, 2);
    expect(bad.ok).toBe(false);
    const good = validateFollow(lead, [std('clubs', '3', 0)], hand, CTX, 2);
    expect(good.ok).toBe(true);
  });

  it('void in led suit → any card OK', () => {
    const hand = [std('spades', 'A', 0), std('diamonds', '7', 0)];
    const r = validateFollow(lead, [std('spades', 'A', 0)], hand, CTX, 2);
    expect(r.ok).toBe(true);
  });

  it('rejects wrong card count', () => {
    const hand = [std('clubs', '3', 0), std('clubs', '4', 0)];
    const r = validateFollow(lead, [std('clubs', '3', 0), std('clubs', '4', 0)], hand, CTX, 2);
    expect(r.ok).toBe(false);
  });
});

describe('validateFollow — pair lead', () => {
  const lead = classify(pair('clubs', '9'));

  it('must play pair of led suit when available', () => {
    const hand = [...pair('clubs', '8'), std('spades', 'A', 0), std('spades', 'A', 1)];
    const bad = validateFollow(
      lead,
      [std('spades', 'A', 0), std('spades', 'A', 1)],
      hand,
      CTX,
      2,
    );
    expect(bad.ok).toBe(false);
    const good = validateFollow(lead, pair('clubs', '8'), hand, CTX, 2);
    expect(good.ok).toBe(true);
  });

  it('no pair of led suit but ≥2 led-suit singles → must play 2 led-suit cards', () => {
    const hand = [std('clubs', '8', 0), std('clubs', 'J', 0), std('spades', 'A', 0)];
    const bad = validateFollow(
      lead,
      [std('clubs', '8', 0), std('spades', 'A', 0)],
      hand,
      CTX,
      2,
    );
    expect(bad.ok).toBe(false);
    const good = validateFollow(
      lead,
      [std('clubs', '8', 0), std('clubs', 'J', 0)],
      hand,
      CTX,
      2,
    );
    expect(good.ok).toBe(true);
  });

  it('one led-suit card only → must include it', () => {
    const hand = [std('clubs', '8', 0), std('spades', 'A', 0), std('diamonds', '3', 0)];
    const bad = validateFollow(
      lead,
      [std('spades', 'A', 0), std('diamonds', '3', 0)],
      hand,
      CTX,
      2,
    );
    expect(bad.ok).toBe(false);
    const good = validateFollow(
      lead,
      [std('clubs', '8', 0), std('spades', 'A', 0)],
      hand,
      CTX,
      2,
    );
    expect(good.ok).toBe(true);
  });

  it('void in led suit → any 2 cards', () => {
    const hand = [std('spades', 'A', 0), std('diamonds', '3', 0)];
    const r = validateFollow(
      lead,
      [std('spades', 'A', 0), std('diamonds', '3', 0)],
      hand,
      CTX,
      2,
    );
    expect(r.ok).toBe(true);
  });
});

describe('validateFollow — tractor lead', () => {
  const lead = classify([...pair('clubs', '8'), ...pair('clubs', '9')]);

  it('must include a same-length tractor in led suit if available', () => {
    const hand = [
      ...pair('clubs', 'J'),
      ...pair('clubs', 'Q'),
      std('spades', 'A', 0),
      std('spades', 'A', 1),
    ];
    // Two pairs of clubs forming a tractor available; must play them.
    const bad = validateFollow(
      lead,
      [...pair('clubs', 'J'), std('spades', 'A', 0), std('spades', 'A', 1)],
      hand,
      CTX,
      2,
    );
    expect(bad.ok).toBe(false);
    const good = validateFollow(
      lead,
      [...pair('clubs', 'J'), ...pair('clubs', 'Q')],
      hand,
      CTX,
      2,
    );
    expect(good.ok).toBe(true);
  });

  it('pairs but no tractor → must include the pairs from led suit', () => {
    const hand = [
      ...pair('clubs', '3'),
      ...pair('clubs', '7'),
      std('spades', 'A', 0),
      std('spades', 'A', 1),
    ];
    // No tractor (gap), but 2 pairs of clubs available → must include both pairs.
    const good = validateFollow(
      lead,
      [...pair('clubs', '3'), ...pair('clubs', '7')],
      hand,
      CTX,
      2,
    );
    expect(good.ok).toBe(true);
  });
});

describe('validateFollow — trump-effective-suit', () => {
  it('trump lead: cards in trump suit count as same chain (no follow violation across natural suits)', () => {
    const lead = classify([std('hearts', '7', 0)]);
    // Hand has a trump-suit-R off-suit (5♠ → trump) but no hearts.
    const hand = [std('spades', '5', 0), std('spades', 'A', 0)];
    // 5♠ is trump → playing it follows.
    const good = validateFollow(lead, [std('spades', '5', 0)], hand, CTX, 2);
    expect(good.ok).toBe(true);
  });

  it('non-trump lead, player has only trumps: can play any trump', () => {
    const lead = classify([std('clubs', 'K', 0)]);
    const hand = [std('hearts', '3', 0), jok('big', 0)];
    const r = validateFollow(lead, [jok('big', 0)], hand, CTX, 2);
    expect(r.ok).toBe(true);
  });
});
