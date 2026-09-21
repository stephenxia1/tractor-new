// Trick resolution against a throw (rules.md §7.6 step 3, §7.4).

import { describe, expect, it } from 'vitest';
import { decomposePlay, decomposedBeats } from '../src/throws';
import type {
  Card,
  JokerRank,
  NaturalRank,
  Suit,
  TrumpContext,
} from '../src/types';

const DECKS = 2;

function c(suit: Suit, rank: NaturalRank, deckId = 0): Card {
  return { kind: 'standard', suit, rank, deckId };
}
function j(rank: JokerRank, deckId = 0): Card {
  return { kind: 'joker', rank, deckId };
}

describe('decomposedBeats', () => {
  // Trump hearts, level 5. Chain: 2H=0 .. AH=11, off-suit fives = 12,
  // 5H = 13, small joker = 14, big joker = 15.
  const CTX: TrumpContext = { trumpRank: '5', trumpSuit: 'hearts' };

  const HEARTS: NaturalRank[] = [
    '3', '4', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A',
  ];

  // Incumbent: a pair of the lowest trump plus eleven rising singles and
  // the small joker. Fourteen cards, one non-single component.
  const incumbentCards: Card[] = [
    c('hearts', '2', 0), c('hearts', '2', 1),
    ...HEARTS.map((r) => c('hearts', r, 0)),
    j('small', 0),
  ];

  // Response: a pair of off-suit fives outranks the pair of twos, each
  // heart outranks the one below it, an off-suit five outranks the ace,
  // and the big joker outranks the small. A complete cover.
  const responseCards: Card[] = [
    c('diamonds', '5', 0), c('diamonds', '5', 1),
    ...HEARTS.slice(1).map((r) => c('hearts', r, 1)),
    c('spades', '5', 0),
    j('big', 0),
  ];

  it('resolves a fourteen-card cover without a search blow-up', () => {
    expect(incumbentCards).toHaveLength(14);
    expect(responseCards).toHaveLength(14);
    const inc = decomposePlay(incumbentCards, CTX, DECKS);
    const cand = decomposePlay(responseCards, CTX, DECKS);
    expect(inc).not.toBeNull();
    expect(cand).not.toBeNull();
    if (inc === null || cand === null) return;
    expect(inc.components.filter((x) => x.kind === 'single')).toHaveLength(12);

    const t0 = Date.now();
    expect(decomposedBeats(cand, inc, CTX, DECKS)).toBe(true);
    expect(Date.now() - t0).toBeLessThan(200);
  });

  it('gives the same verdict whatever order the response is held in', () => {
    const inc = decomposePlay(incumbentCards, CTX, DECKS);
    expect(inc).not.toBeNull();
    if (inc === null) return;

    // The joker at the front was the expensive ordering; at the back the
    // old search finished in thirteen nodes. Same multiset either way.
    const rotations: Card[][] = [];
    for (let k = 0; k < responseCards.length; k++) {
      rotations.push([...responseCards.slice(k), ...responseCards.slice(0, k)]);
    }
    for (const order of rotations) {
      const cand = decomposePlay(order, CTX, DECKS);
      expect(cand).not.toBeNull();
      if (cand === null) continue;
      expect(decomposedBeats(cand, inc, CTX, DECKS)).toBe(true);
    }
  });

  it('matches components optimally rather than first-fit', () => {
    // K beats the 9 and the 4 beats the 3, but a greedy pass spends the
    // king on the three and strands the four.
    const inc = decomposePlay([c('diamonds', '3'), c('diamonds', '9')], CTX, DECKS);
    const cand = decomposePlay([c('diamonds', 'K'), c('diamonds', '4')], CTX, DECKS);
    expect(inc).not.toBeNull();
    expect(cand).not.toBeNull();
    if (inc === null || cand === null) return;
    // The two-argument form takes path (a) alone, so it must be exact.
    expect(decomposedBeats(cand, inc)).toBe(true);
    expect(decomposedBeats(cand, inc, CTX, DECKS)).toBe(true);
  });

  it('still refuses to merge singles into a pair', () => {
    // A pair lead is not beatable by two stronger singletons: the
    // incumbent's structure is fixed and only the candidate is split.
    const inc = decomposePlay(
      [c('diamonds', '3'), c('diamonds', '3', 1), c('diamonds', '4')],
      CTX,
      DECKS,
    );
    const cand = decomposePlay(
      [c('diamonds', 'K'), c('diamonds', 'Q'), c('diamonds', 'J')],
      CTX,
      DECKS,
    );
    expect(inc).not.toBeNull();
    expect(cand).not.toBeNull();
    if (inc === null || cand === null) return;
    expect(decomposedBeats(cand, inc, CTX, DECKS)).toBe(false);
  });
});
