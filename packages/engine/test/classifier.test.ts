import { describe, it, expect } from 'vitest';
import {
  classifyCombo,
  comboBeats,
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

const CTX: TrumpContext = { trumpRank: '5', trumpSuit: 'hearts' };

describe('classifyCombo', () => {
  it('single card → single', () => {
    const r = classifyCombo([std('clubs', '8', 0)], CTX, 2);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.combo.kind).toBe('single');
  });

  it('matching pair → nkind n=2', () => {
    const r = classifyCombo([std('clubs', '8', 0), std('clubs', '8', 1)], CTX, 2);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.combo.kind).toBe('nkind');
    if (r.combo.kind !== 'nkind') return;
    expect(r.combo.n).toBe(2);
  });

  it('mismatched "pair" → invalid (different ranks)', () => {
    const r = classifyCombo([std('clubs', '8', 0), std('clubs', '9', 0)], CTX, 2);
    expect(r.ok).toBe(false);
  });

  it('consecutive pairs in same suit → tractor', () => {
    const r = classifyCombo(
      [std('clubs', '8', 0), std('clubs', '8', 1), std('clubs', '9', 0), std('clubs', '9', 1)],
      CTX,
      2,
    );
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.combo.kind).toBe('tractor');
    if (r.combo.kind !== 'tractor') return;
    expect(r.combo.length).toBe(2);
  });

  it('non-consecutive pairs → invalid', () => {
    const r = classifyCombo(
      [std('clubs', '8', 0), std('clubs', '8', 1), std('clubs', 'J', 0), std('clubs', 'J', 1)],
      CTX,
      2,
    );
    expect(r.ok).toBe(false);
  });

  it('mixed effective suits → invalid', () => {
    const r = classifyCombo([std('clubs', '8', 0), std('spades', '8', 0)], CTX, 2);
    expect(r.ok).toBe(false);
  });

  it('trump tractor across tier boundary (A♥A♥ + 5♠5♠) → tractor', () => {
    const r = classifyCombo(
      [std('hearts', 'A', 0), std('hearts', 'A', 1), std('spades', '5', 0), std('spades', '5', 1)],
      CTX,
      2,
    );
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.combo.kind).toBe('tractor');
  });
});

describe('comboBeats', () => {
  it('higher single beats lower single in same suit', () => {
    const a = classifyCombo([std('clubs', 'K', 0)], CTX, 2);
    const b = classifyCombo([std('clubs', '3', 0)], CTX, 2);
    if (!a.ok || !b.ok) throw new Error();
    expect(comboBeats(a.combo, b.combo)).toBe(1);
    expect(comboBeats(b.combo, a.combo)).toBe(-1);
  });

  it('trump single beats non-trump single', () => {
    const trump = classifyCombo([std('hearts', '2', 0)], CTX, 2);
    const nonTrump = classifyCombo([std('clubs', 'A', 0)], CTX, 2);
    if (!trump.ok || !nonTrump.ok) throw new Error();
    expect(comboBeats(trump.combo, nonTrump.combo)).toBe(1);
  });

  it('pair vs single → no competition (different structure)', () => {
    const single = classifyCombo([std('clubs', 'K', 0)], CTX, 2);
    const pair = classifyCombo([std('clubs', '8', 0), std('clubs', '8', 1)], CTX, 2);
    if (!single.ok || !pair.ok) throw new Error();
    expect(comboBeats(pair.combo, single.combo)).toBe(0);
  });

  it('higher trump pair beats lower trump pair', () => {
    const high = classifyCombo([jok('big', 0), jok('big', 1)], CTX, 2);
    const low = classifyCombo([std('hearts', '7', 0), std('hearts', '7', 1)], CTX, 2);
    if (!high.ok || !low.ok) throw new Error();
    expect(comboBeats(high.combo, low.combo)).toBe(1);
  });

  it('cross-suit non-trump comparison → 0 (incomparable)', () => {
    const a = classifyCombo([std('clubs', 'K', 0)], CTX, 2);
    const b = classifyCombo([std('spades', 'A', 0)], CTX, 2);
    if (!a.ok || !b.ok) throw new Error();
    expect(comboBeats(a.combo, b.combo)).toBe(0);
  });

  it('tied combos → 0', () => {
    const a = classifyCombo([std('clubs', 'K', 0)], CTX, 2);
    const b = classifyCombo([std('clubs', 'K', 1)], CTX, 2);
    if (!a.ok || !b.ok) throw new Error();
    expect(comboBeats(a.combo, b.combo)).toBe(0);
  });
});
