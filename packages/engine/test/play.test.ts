import { describe, it, expect } from 'vitest';
import {
  applyEvent,
  createGame,
  defaultV1Config,
  effectiveSuit,
  handFor,
  nextToPlay,
  type Card,
  type GameState,
  type TrumpContext,
} from '../src';

// Drives a fresh game through declaration + kitty so the state is in 'play'
// with seat 0 as landlord. Doesn't try to control which cards each seat
// holds — just gets us to the play phase quickly.
function gameInPlayPhase(seed = 1): GameState {
  const game = createGame(defaultV1Config());
  const a = applyEvent(game, { kind: 'start_hand', seed });
  if (!a.ok) throw new Error(a.error);
  // Force a kitty-flip path so we don't need to declare specific cards.
  const b = applyEvent(a.state, { kind: 'release_lockout', seed: 1 });
  if (!b.ok) throw new Error(b.error);
  // Now phase = 'kitty_replace'. Landlord has 33 cards.
  const landlord = b.state.currentHand!.landlord!;
  const toBury = handFor(b.state, landlord).slice(0, 8);
  const c = applyEvent(b.state, {
    kind: 'replace_kitty',
    seat: landlord,
    cards: toBury,
  });
  if (!c.ok) throw new Error(c.error);
  return c.state;
}

// Plays the full hand: leader plays a single (their first card), each
// follower plays a card of the led effective suit if they have one
// (else their first card). This satisfies follow-suit rules for the
// simplest case (single leads).
function playFullHand(initial: GameState): GameState {
  let s = initial;
  while (s.phase === 'play' || s.phase === 'trick_review') {
    if (s.phase === 'trick_review') {
      const r = applyEvent(s, { kind: 'advance_trick' });
      if (!r.ok) throw new Error(`advance failed: ${r.error}`);
      s = r.state;
      continue;
    }
    const seat = nextToPlay(s)!;
    const hand = handFor(s, seat);
    const ch = s.currentHand!;
    const ctx: TrumpContext = { trumpRank: ch.trumpRank!, trumpSuit: ch.trumpSuit };
    let cards: Card[];
    const trick = ch.currentTrick;
    if (trick === null) {
      cards = [hand[0]!];
    } else {
      const ledSuit = effectiveSuit(trick.plays[0]!.cards[0]!, ctx);
      const followCard =
        hand.find((c) => effectiveSuit(c, ctx) === ledSuit) ?? hand[0]!;
      cards = [followCard];
    }
    const res = applyEvent(s, { kind: 'play_cards', seat, cards });
    if (!res.ok) throw new Error(`play failed for seat ${seat}: ${res.error}`);
    s = res.state;
  }
  return s;
}

describe('play_cards — basic mechanics', () => {
  it('plays a full hand of 25 tricks and transitions to between_hands', () => {
    const initial = gameInPlayPhase();
    const ended = playFullHand(initial);
    expect(ended.phase).toBe('between_hands');
    expect(ended.handsCompleted).toBe(1);
    expect(ended.currentHand!.tricks).toHaveLength(25);
    for (let s = 0; s < 4; s++) {
      expect(handFor(ended, s)).toHaveLength(0);
    }
  });

  it('rejects play out of turn', () => {
    const s = gameInPlayPhase();
    const wrongSeat = (nextToPlay(s)! + 1) % 4;
    const res = applyEvent(s, {
      kind: 'play_cards',
      seat: wrongSeat,
      cards: [handFor(s, wrongSeat)[0]!],
    });
    expect(res.ok).toBe(false);
  });

  it('rejects play of card not in hand', () => {
    const s = gameInPlayPhase();
    const seat = nextToPlay(s)!;
    // Pick a card from another seat.
    const otherSeat = (seat + 1) % 4;
    const stolen = handFor(s, otherSeat)[0]!;
    if (handFor(s, seat).some((c) => c.kind === stolen.kind &&
      (c.kind === 'standard' && stolen.kind === 'standard' &&
        c.suit === stolen.suit && c.rank === stolen.rank && c.deckId === stolen.deckId))) {
      // by chance, seat has the same card — skip
      return;
    }
    const res = applyEvent(s, { kind: 'play_cards', seat, cards: [stolen] });
    expect(res.ok).toBe(false);
  });

  it('rejects follow with wrong card count', () => {
    const s = gameInPlayPhase();
    const leader = nextToPlay(s)!;
    const lead = applyEvent(s, {
      kind: 'play_cards',
      seat: leader,
      cards: [handFor(s, leader)[0]!],
    });
    if (!lead.ok) throw new Error(lead.error);
    const follower = nextToPlay(lead.state)!;
    const follow = applyEvent(lead.state, {
      kind: 'play_cards',
      seat: follower,
      cards: handFor(lead.state, follower).slice(0, 2), // 2 cards when 1 expected
    });
    expect(follow.ok).toBe(false);
  });

  it('trick winner leads the next trick', () => {
    let s = gameInPlayPhase();
    const firstLeader = nextToPlay(s)!;
    expect(firstLeader).toBe(s.currentHand!.landlord);

    // Play 4 cards, picking led-suit when possible to satisfy follow rules.
    for (let i = 0; i < 4; i++) {
      const seat = nextToPlay(s)!;
      const ch = s.currentHand!;
      const ctx: TrumpContext = { trumpRank: ch.trumpRank!, trumpSuit: ch.trumpSuit };
      const hand = handFor(s, seat);
      let cards: Card[];
      const trick = ch.currentTrick;
      if (trick === null) {
        cards = [hand[0]!];
      } else {
        const ledSuit = effectiveSuit(trick.plays[0]!.cards[0]!, ctx);
        const followCard =
          hand.find((c) => effectiveSuit(c, ctx) === ledSuit) ?? hand[0]!;
        cards = [followCard];
      }
      const r = applyEvent(s, { kind: 'play_cards', seat, cards });
      if (!r.ok) throw new Error(r.error);
      s = r.state;
    }

    // Trick is complete but pending advance.
    expect(s.phase).toBe('trick_review');
    const adv = applyEvent(s, { kind: 'advance_trick' });
    expect(adv.ok).toBe(true);
    if (!adv.ok) throw new Error(adv.error);
    s = adv.state;
    const completedFirst = s.currentHand!.tricks[0]!;
    // After trick advances, next leader should match trick winner.
    expect(nextToPlay(s)).toBe(completedFirst.winner);
  });
});

describe('hand finalization', () => {
  it('a completed hand updates team levels and landlord state', () => {
    const initial = gameInPlayPhase();
    const ended = playFullHand(initial);
    expect(ended.handsCompleted).toBe(1);
    // Either landlord was rotated to partner (defenders won) or unset (attackers won).
    const landlordTeam = initial.config.seatToTeam[initial.currentHand!.landlord!]!;
    const startingLevel = '2';
    const newDef = ended.teamLevels[landlordTeam]!;
    const newAtt = ended.teamLevels[landlordTeam === 'A' ? 'B' : 'A']!;
    // Exactly one team should have gained levels (or stayed if the bracket is 80-119).
    expect(newDef !== startingLevel || newAtt !== startingLevel).toBe(true);
  });

  it('landlord rotates to the next seat on the winning team', () => {
    // The landlord always moves to the next clockwise seat on the team that
    // won the hand. Defenders win → that's the partner (skipping the next
    // attacker). Attackers win → that's the next attacker seat.
    const initial = gameInPlayPhase();
    const ended = playFullHand(initial);
    const initialLandlord = initial.currentHand!.landlord!;
    const winnerIsAttackers = ended.currentHand!.attackerPoints >= 80;
    const landlordTeam = initial.config.seatToTeam[initialLandlord]!;
    const winningTeam = winnerIsAttackers
      ? initial.config.teamIds.find((t) => t !== landlordTeam)!
      : landlordTeam;
    expect(ended.landlord).not.toBeNull();
    expect(initial.config.seatToTeam[ended.landlord!]).toBe(winningTeam);
    expect(ended.landlord).not.toBe(initialLandlord);
  });
});
