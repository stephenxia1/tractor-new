// Trick resolution must not depend on the order responses arrive in.
//
// Once a response may be re-decomposed to the lead's shape (rules.md
// §7.6 step 3), "covers" stops being a total order: two responses can
// each cover the throw while neither covers the other. Comparing each
// response against the running winner then awards the trick to whoever
// played first. Comparing each against the lead, and ranking the ones
// that cover it by rules.md §7.4, does not.

import { describe, expect, it } from 'vitest';
import { applyEvent, createGame, defaultV1Config } from '../src';
import type { Card, GameState, NaturalRank, Seat, Suit } from '../src';

function c(suit: Suit, rank: NaturalRank, deckId = 0): Card {
  return { kind: 'standard', suit, rank, deckId };
}
function pair(suit: Suit, rank: NaturalRank): Card[] {
  return [c(suit, rank, 0), c(suit, rank, 1)];
}

// Trump hearts, level 2. Diamond chain: 3D=0 .. AD=11, so 3D and 6D are
// two non-consecutive pairs and the lead is a throw of two pairs.
const LEAD = [...pair('diamonds', '3'), ...pair('diamonds', '6')];
// A trump tractor. Covers the lead by splitting into two pairs. Top 8H.
const TRACTOR = [...pair('hearts', '7'), ...pair('hearts', '8')];
// Two non-consecutive trump pairs. Covers the lead directly. Top QH,
// which outranks 8H, so this play wins wherever it is played.
const PAIRS = [...pair('hearts', '10'), ...pair('hearts', 'Q')];
// No diamonds, so this seat is free to discard.
const JUNK = [c('clubs', '3'), c('clubs', '4'), c('clubs', '5'), c('clubs', '7')];

function stateWith(seat1: Card[], seat2: Card[]): GameState {
  const base = createGame(defaultV1Config());
  return {
    ...base,
    landlord: 0,
    phase: 'play',
    currentHand: {
      handNumber: 1,
      trumpRank: '2',
      trumpSuit: 'hearts',
      trumpFinalized: true,
      landlord: 0,
      hands: { 0: [...LEAD], 1: seat1, 2: seat2, 3: [...JUNK] },
      cardsDealtBySeat: { 0: 4, 1: 4, 2: 4, 3: 4 },
      kitty: [],
      declarationLog: [],
      standingDeclaration: null,
      tricks: [],
      currentTrick: null,
      attackerPoints: 0,
    },
  };
}

function playTrick(seat1: Card[], seat2: Card[]): Seat {
  let s = stateWith(seat1, seat2);
  const order: Array<[Seat, Card[]]> = [
    [0, LEAD],
    [1, seat1],
    [2, seat2],
    [3, JUNK],
  ];
  for (const [seat, cards] of order) {
    const r = applyEvent(s, { kind: 'play_cards', seat, cards });
    if (!r.ok) throw new Error(`seat ${seat}: ${r.error}`);
    s = r.state;
  }
  // The fourth play leaves the hand in trick_review; advancing commits
  // the resolved trick to the history.
  if (s.phase === 'trick_review') {
    const r = applyEvent(s, { kind: 'advance_trick' });
    if (!r.ok) throw new Error(`advance: ${r.error}`);
    s = r.state;
  }
  const trick = s.currentHand!.tricks[0];
  if (trick === undefined) throw new Error(`trick did not resolve (phase ${s.phase})`);
  return trick.winner;
}

describe('trick resolution against a throw', () => {
  it('awards the trick to the higher play, not the earlier one', () => {
    // Tractor first, then the higher pairs.
    expect(playTrick(TRACTOR, PAIRS)).toBe(2);
    // Higher pairs first, then the tractor. Same winner.
    expect(playTrick(PAIRS, TRACTOR)).toBe(1);
  });
});
