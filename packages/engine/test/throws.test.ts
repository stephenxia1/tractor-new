import { describe, it, expect } from 'vitest';
import {
  classifyCombo,
  decomposeThrow,
  isThrowSafe,
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

describe('decomposeThrow', () => {
  it('pair + single in same suit → 2 components', () => {
    const cards = [...pair('clubs', 'J'), std('clubs', '7', 0)];
    const r = decomposeThrow(cards, CTX, 2);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.throw.components).toHaveLength(2);
    expect(r.throw.effectiveSuit).toBe('clubs');
  });

  it('two non-adjacent pairs in same suit → 2 components (not a tractor)', () => {
    const cards = [...pair('clubs', '3'), ...pair('clubs', '7')];
    const r = decomposeThrow(cards, CTX, 2);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.throw.components).toHaveLength(2);
    expect(r.throw.components.every((c) => c.kind === 'nkind')).toBe(true);
  });

  it('cards forming a tractor + extra single → 2 components (tractor + single)', () => {
    const cards = [...pair('clubs', '8'), ...pair('clubs', '9'), std('clubs', '3', 0)];
    const r = decomposeThrow(cards, CTX, 2);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.throw.components).toHaveLength(2);
    expect(r.throw.components.map((c) => c.kind).sort()).toEqual(['single', 'tractor']);
  });

  it('rejects single combo (would classify as nkind/tractor)', () => {
    const r = decomposeThrow(pair('clubs', 'J'), CTX, 2);
    expect(r.ok).toBe(false);
  });

  it('rejects mixed effective suits', () => {
    const cards = [std('clubs', 'J', 0), std('spades', 'J', 0)];
    const r = decomposeThrow(cards, CTX, 2);
    expect(r.ok).toBe(false);
  });
});

describe('isThrowSafe', () => {
  const ctx = CTX;

  it('safe when no opponent has any beating combo', () => {
    const cards = [...pair('clubs', 'A'), std('clubs', 'K', 0)];
    const decomp = decomposeThrow(cards, ctx, 2);
    expect(decomp.ok).toBe(true);
    if (!decomp.ok) return;
    // Opponents only have low non-trump cards in other suits.
    const opp1: Card[] = [std('spades', '3', 0), std('diamonds', '4', 0)];
    const opp2: Card[] = [std('spades', '7', 0), std('diamonds', '8', 0)];
    const r = isThrowSafe(decomp.throw, [opp1, opp2], ctx, 2);
    expect(r.safe).toBe(true);
  });

  it('trump combos cannot block a non-trump throw', () => {
    const cards = [...pair('clubs', '7'), std('clubs', '3', 0)];
    const decomp = decomposeThrow(cards, ctx, 2);
    expect(decomp.ok).toBe(true);
    if (!decomp.ok) return;
    // Opponent has a pair of hearts (trump) — but per the safety rule,
    // a trump-only response can never be forced over a non-trump throw,
    // so this throw is still safe.
    const opp1: Card[] = [...pair('hearts', '3')];
    const opp2: Card[] = [std('diamonds', '4', 0)];
    const r = isThrowSafe(decomp.throw, [opp1, opp2], ctx, 2);
    expect(r.safe).toBe(true);
  });

  it('safety check is per-opponent — teammates do not count', () => {
    const cards = [...pair('clubs', '7'), std('clubs', '3', 0)];
    const decomp = decomposeThrow(cards, ctx, 2);
    expect(decomp.ok).toBe(true);
    if (!decomp.ok) return;
    // Both "opponent" hands are low — no beats.
    const opp1: Card[] = [std('spades', '6', 0)];
    const opp2: Card[] = [std('diamonds', '6', 0)];
    const r = isThrowSafe(decomp.throw, [opp1, opp2], ctx, 2);
    expect(r.safe).toBe(true);
  });

  it('reports smallest beatable component on failure', () => {
    // Throw is a pair of 7♣ plus a single 3♣. An opponent holding the
    // 8♣ singleton (same-suit higher) beats the SINGLE 3♣ but not the
    // pair of 7♣. The single is the smallest beatable component.
    const cards = [...pair('clubs', '7'), std('clubs', '3', 0)];
    const decomp = decomposeThrow(cards, ctx, 2);
    if (!decomp.ok) throw new Error();
    const opp1: Card[] = [std('clubs', '8', 0)];
    const opp2: Card[] = [];
    const r = isThrowSafe(decomp.throw, [opp1, opp2], ctx, 2);
    expect(r.safe).toBe(false);
    if (r.safe) return;
    expect(r.smallestBeatable.kind).toBe('single');
  });

  it('trump throw with all top trumps is always safe', () => {
    const cards = [...pair('hearts', '5'), jok('big', 0)];
    const decomp = decomposeThrow(cards, ctx, 2);
    expect(decomp.ok).toBe(true);
    if (!decomp.ok) return;
    const opp1: Card[] = [std('clubs', 'A', 0)]; // can't beat trump
    const opp2: Card[] = [std('diamonds', '4', 0)];
    const r = isThrowSafe(decomp.throw, [opp1, opp2], ctx, 2);
    expect(r.safe).toBe(true);
  });
});
