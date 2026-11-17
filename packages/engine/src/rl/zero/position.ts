// A live engine position as the play featuriser reads it: the same fields
// research/src/evalhost.ts sends to the Python evaluator (`pending`
// messages), restricted to what the seat to act may see.

import { NATURAL_RANKS, type Card } from '../../types';
import { resolveTrick } from '../../reducer';
import type { GameState, Seat } from '../../state';
import { cardSlot, NUM_SLOTS } from '../v2/cards';
import { trumpSuitIndex } from '../v2/encode';

import type { ZeroPosition, ZeroTrick } from './features';

export function plane(cards: ReadonlyArray<Card>): Int8Array {
  const out = new Int8Array(NUM_SLOTS);
  for (const c of cards) {
    const slot = cardSlot(c);
    out[slot] = out[slot]! + 1;
  }
  return out;
}

/** `seat`'s view of the play-phase position in `state`. */
export function positionFor(state: GameState, seat: Seat, myKitty: ReadonlyArray<Card> | null): ZeroPosition {
  const hand = state.currentHand;
  if (hand === null || hand.trumpRank === null || hand.landlord === null) {
    throw new Error('zero: no live hand to encode');
  }
  const numSeats = state.config.numPlayers;
  if (numSeats !== 4) throw new Error(`zero: the network plays four seats, not ${numSeats}`);
  const tricks: ZeroTrick[] = hand.tricks.map((t) => {
    t.plays.forEach((p, i) => {
      if (p.seat !== (t.leader + i) % numSeats) throw new Error('zero: trick plays out of seat order');
    });
    return { leader: t.leader, winner: t.winner, points: t.points, plays: t.plays.map((p) => plane(p.cards)) };
  });
  const open = hand.currentTrick;
  const trickPlays: Int8Array[] = [];
  let currentWinner: number | null = null;
  if (open !== null && open.plays.length > 0) {
    open.plays.forEach((p, i) => {
      if (p.seat !== (open.leader + i) % numSeats) throw new Error('zero: open trick out of seat order');
      trickPlays.push(plane(p.cards));
    });
    currentWinner = resolveTrick(
      open,
      { trumpRank: hand.trumpRank, trumpSuit: hand.trumpSuit },
      state.config.numDecks,
    ).winner;
  }
  const isLandlord = seat === hand.landlord;
  return {
    seat,
    levelRank: NATURAL_RANKS.indexOf(hand.trumpRank),
    trumpSuit: trumpSuitIndex(hand),
    numDecks: state.config.numDecks,
    landlord: hand.landlord,
    hand: plane(hand.hands[seat] ?? []),
    kitty: isLandlord && myKitty !== null ? plane(myKitty) : null,
    tricks,
    trickPlays,
    trickLeader: open === null || open.plays.length === 0 ? seat : open.leader,
    currentWinner,
  };
}
