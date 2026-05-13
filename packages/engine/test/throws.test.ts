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
import { decomposePlay, decomposedBeats } from '../src/throws';

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

  it('trump combos do NOT block a non-trump throw at lead time', () => {
    // Block is led-suit-only (rules.md §7.6). A trump pair the opponent
    // holds may BEAT the throw at trick time, but the throw is still
    // allowed to be led — followers can't be forced to trump.
    const cards = [...pair('clubs', '7'), std('clubs', '3', 0)];
    const decomp = decomposeThrow(cards, ctx, 2);
    expect(decomp.ok).toBe(true);
    if (!decomp.ok) return;
    const opp1: Card[] = [...pair('hearts', '3')];
    const opp2: Card[] = [std('diamonds', '4', 0)];
    const r = isThrowSafe(decomp.throw, [opp1, opp2], ctx, 2);
    expect(r.safe).toBe(true);
  });

  it('big joker does not block a throw but can beat it at trick time', () => {
    // User-reported case: throw of three non-trump singles. A big joker
    // in an opponent's hand does NOT make the throw "unsafe" at lead
    // time (block is led-suit-only), but at trick resolution the
    // follower may play the joker (plus other cards) and structurally
    // beat a component. The block check should return safe here.
    const cards = [std('clubs', '3', 0), std('clubs', '7', 0), std('clubs', '9', 0)];
    const decomp = decomposeThrow(cards, ctx, 2);
    expect(decomp.ok).toBe(true);
    if (!decomp.ok) return;
    const opp1: Card[] = [jok('big', 0)];
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

describe('decomposedBeats', () => {
  const ctx = CTX; // trump = hearts, rank 5

  it('two trump singles beat a throw of two off-suit singles', () => {
    const lead = [std('clubs', 'A', 0), std('clubs', 'K', 0)];
    const resp = [std('hearts', '3', 0), std('hearts', '4', 0)];
    const leadD = decomposePlay(lead, ctx, 2);
    const respD = decomposePlay(resp, ctx, 2);
    expect(leadD).not.toBeNull();
    expect(respD).not.toBeNull();
    expect(decomposedBeats(respD!, leadD!)).toBe(true);
  });

  it('lower throw in same suit does not beat the lead', () => {
    const lead = [std('clubs', 'A', 0), std('clubs', 'K', 0)];
    const resp = [std('clubs', 'Q', 0), std('clubs', 'J', 0)];
    const leadD = decomposePlay(lead, ctx, 2);
    const respD = decomposePlay(resp, ctx, 2);
    expect(decomposedBeats(respD!, leadD!)).toBe(false);
  });

  it('a pair does not beat a throw of two singles (shape mismatch)', () => {
    const lead = [std('clubs', 'A', 0), std('clubs', 'K', 0)];
    const resp = pair('hearts', '3'); // pair of trump 3
    const leadD = decomposePlay(lead, ctx, 2);
    const respD = decomposePlay(resp, ctx, 2);
    expect(decomposedBeats(respD!, leadD!)).toBe(false);
  });

  it('off-suit non-trump response cannot beat throw', () => {
    const lead = [std('clubs', 'A', 0), std('clubs', 'K', 0)];
    const resp = [std('spades', '2', 0), std('spades', '3', 0)];
    const leadD = decomposePlay(lead, ctx, 2);
    const respD = decomposePlay(resp, ctx, 2);
    expect(decomposedBeats(respD!, leadD!)).toBe(false);
  });
});
