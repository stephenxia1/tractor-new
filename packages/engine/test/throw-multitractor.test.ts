// A throw may carry more than one tractor component. rules.md §7.6 step 3
// routes the follower's obligation through §7.3, whose closing principle
// is that a player "may not avoid following with structure they hold" —
// so a throw of two tractors demands two tractors from a follower who has
// them, not one tractor plus loose pairs.

import { describe, expect, it } from 'vitest';
import { classifyCombo } from '../src/classifier';
import { canonicalFollow, validateFollowThrow } from '../src/follow';
import { decomposeThrow } from '../src/throws';
import type { Card, NaturalRank, Suit, TrumpContext } from '../src/types';

const DECKS = 2;
const CTX: TrumpContext = { trumpRank: '2', trumpSuit: 'hearts' };

function pair(suit: Suit, rank: NaturalRank): Card[] {
  return [
    { kind: 'standard', suit, rank, deckId: 0 },
    { kind: 'standard', suit, rank, deckId: 1 },
  ];
}

describe('throw leads with several tractor components', () => {
  it('rejects a follow that withholds the second tractor (T2 + T2)', () => {
    // Lead: 3-3 4-4 | 7-7 8-8 — two length-2 tractors, 8 cards.
    const leadCards = [
      ...pair('clubs', '3'), ...pair('clubs', '4'),
      ...pair('clubs', '7'), ...pair('clubs', '8'),
    ];
    expect(classifyCombo(leadCards, CTX, DECKS).ok).toBe(false);
    const d = decomposeThrow(leadCards, CTX, DECKS);
    expect(d.ok).toBe(true);
    if (!d.ok) return;
    expect(d.throw.components.filter((c) => c.kind === 'tractor')).toHaveLength(2);

    // Hand: 9-10-J-Q-A all paired. 9-10-J-Q is a run of four, so two
    // disjoint length-2 tractors are available.
    const hand = [
      ...pair('clubs', '9'), ...pair('clubs', '10'),
      ...pair('clubs', 'J'), ...pair('clubs', 'Q'),
      ...pair('clubs', 'A'),
    ];

    // One tractor (9-10) plus two isolated pairs.
    const withholding = [
      ...pair('clubs', '9'), ...pair('clubs', '10'),
      ...pair('clubs', 'Q'), ...pair('clubs', 'A'),
    ];
    // Two tractors.
    const complying = [
      ...pair('clubs', '9'), ...pair('clubs', '10'),
      ...pair('clubs', 'J'), ...pair('clubs', 'Q'),
    ];

    expect(
      validateFollowThrow(d.throw, complying, hand, CTX, DECKS).ok,
    ).toBe(true);
    expect(
      validateFollowThrow(d.throw, withholding, hand, CTX, DECKS).ok,
    ).toBe(false);
  });

  it('rejects a follow that withholds the shorter tractor (T3 + T2)', () => {
    // Lead: 3-3 4-4 5-5 | 7-7 8-8 — a length-3 and a length-2, 10 cards.
    const leadCards = [
      ...pair('clubs', '3'), ...pair('clubs', '4'), ...pair('clubs', '5'),
      ...pair('clubs', '7'), ...pair('clubs', '8'),
    ];
    const d = decomposeThrow(leadCards, CTX, DECKS);
    expect(d.ok).toBe(true);
    if (!d.ok) return;

    // 9-10-J is a run of three; K-A is a run of two; 3 and 6 are isolated.
    const hand = [
      ...pair('clubs', '9'), ...pair('clubs', '10'), ...pair('clubs', 'J'),
      ...pair('clubs', 'K'), ...pair('clubs', 'A'),
      ...pair('clubs', '3'), ...pair('clubs', '6'),
    ];
    const withholding = [
      ...pair('clubs', '9'), ...pair('clubs', '10'), ...pair('clubs', 'J'),
      ...pair('clubs', '3'), ...pair('clubs', '6'),
    ];
    const complying = [
      ...pair('clubs', '9'), ...pair('clubs', '10'), ...pair('clubs', 'J'),
      ...pair('clubs', 'K'), ...pair('clubs', 'A'),
    ];

    expect(validateFollowThrow(d.throw, complying, hand, CTX, DECKS).ok).toBe(true);
    expect(
      validateFollowThrow(d.throw, withholding, hand, CTX, DECKS).ok,
    ).toBe(false);
  });

  it('canonicalFollow supplies every tractor the profile demands', () => {
    const leadCards = [
      ...pair('clubs', '3'), ...pair('clubs', '4'),
      ...pair('clubs', '7'), ...pair('clubs', '8'),
    ];
    const d = decomposeThrow(leadCards, CTX, DECKS);
    expect(d.ok).toBe(true);
    if (!d.ok) return;
    const hand = [
      ...pair('clubs', '9'), ...pair('clubs', '10'),
      ...pair('clubs', 'J'), ...pair('clubs', 'Q'),
      ...pair('clubs', 'A'),
    ];
    const built = canonicalFollow(
      d.throw.components,
      leadCards.length,
      'clubs',
      hand,
      CTX,
      DECKS,
    );
    expect(built).toHaveLength(leadCards.length);
    expect(validateFollowThrow(d.throw, built, hand, CTX, DECKS).ok).toBe(true);
  });
});
