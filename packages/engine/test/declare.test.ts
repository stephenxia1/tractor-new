import { describe, it, expect } from 'vitest';
import {
  applyEvent,
  cardEquals,
  createGame,
  defaultV1Config,
  handFor,
  type Card,
  type GameState,
  type JokerRank,
  type NaturalRank,
  type Seat,
  type Suit,
} from '../src';

// Forces a specific card into a player's hand for testing — swaps with the
// first card in the recipient's hand that isn't equal to the injected card.
// Returns a state where seat `seat` is guaranteed to hold all `cards`.
function planHand(state: GameState, seat: Seat, cards: ReadonlyArray<Card>): GameState {
  if (state.currentHand === null) throw new Error('no current hand');
  const hand = state.currentHand;
  const hands: Record<number, Card[]> = Object.fromEntries(
    Object.entries(hand.hands).map(([k, v]) => [Number(k), [...v]]),
  );

  for (const target of cards) {
    if (hands[seat]!.some((c) => cardEquals(c, target))) continue;
    // Locate `target` somewhere and swap into `seat`'s hand.
    let found = false;
    for (let s = 0; s < state.config.numPlayers && !found; s++) {
      if (s === seat) continue;
      const idx = hands[s]!.findIndex((c) => cardEquals(c, target));
      if (idx !== -1) {
        // Move target into seat's hand; move a "spare" card out of seat's hand
        // to take its place.
        const spareIdx = hands[seat]!.findIndex(
          (c) => !cards.some((want) => cardEquals(want, c)),
        );
        if (spareIdx === -1) throw new Error('no spare card to swap');
        const spare = hands[seat]![spareIdx]!;
        hands[seat]![spareIdx] = target;
        hands[s]![idx] = spare;
        found = true;
      }
    }
    if (!found) {
      // Maybe it's in the kitty.
      const kIdx = hand.kitty.findIndex((c) => cardEquals(c, target));
      if (kIdx === -1) throw new Error(`card not in deck: ${JSON.stringify(target)}`);
      const newKitty = [...hand.kitty];
      const spareIdx = hands[seat]!.findIndex(
        (c) => !cards.some((want) => cardEquals(want, c)),
      );
      const spare = hands[seat]![spareIdx]!;
      hands[seat]![spareIdx] = target;
      newKitty[kIdx] = spare;
      return planHand(
        {
          ...state,
          currentHand: { ...hand, hands, kitty: newKitty },
        },
        seat,
        cards,
      );
    }
  }
  return { ...state, currentHand: { ...hand, hands } };
}

const std = (suit: Suit, rank: NaturalRank, deckId: number): Card => ({
  kind: 'standard',
  suit,
  rank,
  deckId,
});
const jok = (rank: JokerRank, deckId: number): Card => ({ kind: 'joker', rank, deckId });

function startedGame(seed = 1): GameState {
  const res = applyEvent(createGame(defaultV1Config()), { kind: 'start_hand', seed });
  if (!res.ok) throw new Error(res.error);
  return res.state;
}

describe('declare — landlord unset (first hand)', () => {
  it('single declaration of rank-2 by seat 0 sets trump suit + tentative landlord', () => {
    let s = startedGame();
    s = planHand(s, 0, [std('hearts', '2', 0)]);
    const res = applyEvent(s, { kind: 'declare', seat: 0, cards: [std('hearts', '2', 0)] });
    expect(res.ok).toBe(true);
    if (!res.ok) return;
    const h = res.state.currentHand!;
    expect(h.standingDeclaration?.kind).toBe('single');
    expect(h.trumpSuit).toBe('hearts');
    expect(h.trumpRank).toBe('2'); // declarer's team's level (default 2)
    expect(h.landlord).toBe(0);
    // Game-level landlord NOT yet committed.
    expect(res.state.landlord).toBeNull();
  });

  it('rejects declaration of cards not in hand', () => {
    let s = startedGame();
    // Give seat 0 the 2♥ in deckId 0; try to declare deckId 1 card they may not hold.
    s = planHand(s, 1, [std('hearts', '2', 1)]); // ensure deckId 1 is at seat 1, not 0
    const inHand0 = handFor(s, 0).some((c) => cardEquals(c, std('hearts', '2', 1)));
    if (inHand0) return; // can happen if the planning swapped — skip
    const res = applyEvent(s, { kind: 'declare', seat: 0, cards: [std('hearts', '2', 1)] });
    expect(res.ok).toBe(false);
  });

  it('rejects declaration with wrong rank for declarer', () => {
    let s = startedGame();
    s = planHand(s, 0, [std('hearts', '7', 0)]);
    const res = applyEvent(s, { kind: 'declare', seat: 0, cards: [std('hearts', '7', 0)] });
    // Declarer's team is A at level 2; rank-7 card not valid for declaration.
    expect(res.ok).toBe(false);
  });

  it('pair beats single (tier 2 > tier 1)', () => {
    let s = startedGame();
    s = planHand(s, 0, [std('hearts', '2', 0)]);
    s = planHand(s, 1, [std('clubs', '2', 0), std('clubs', '2', 1)]);
    const r1 = applyEvent(s, { kind: 'declare', seat: 0, cards: [std('hearts', '2', 0)] });
    expect(r1.ok).toBe(true);
    if (!r1.ok) return;
    const r2 = applyEvent(r1.state, {
      kind: 'declare',
      seat: 1,
      cards: [std('clubs', '2', 0), std('clubs', '2', 1)],
    });
    expect(r2.ok).toBe(true);
    if (!r2.ok) return;
    expect(r2.state.currentHand!.standingDeclaration?.kind).toBe('pair');
    expect(r2.state.currentHand!.trumpSuit).toBe('clubs');
    expect(r2.state.currentHand!.landlord).toBe(1);
  });

  it('small-joker pair beats rank pair (tier 3 > tier 2)', () => {
    let s = startedGame();
    s = planHand(s, 0, [std('hearts', '2', 0), std('hearts', '2', 1)]);
    s = planHand(s, 1, [jok('small', 0), jok('small', 1)]);
    const r1 = applyEvent(s, {
      kind: 'declare',
      seat: 0,
      cards: [std('hearts', '2', 0), std('hearts', '2', 1)],
    });
    expect(r1.ok).toBe(true);
    if (!r1.ok) return;
    const r2 = applyEvent(r1.state, {
      kind: 'declare',
      seat: 1,
      cards: [jok('small', 0), jok('small', 1)],
    });
    expect(r2.ok).toBe(true);
    if (!r2.ok) return;
    expect(r2.state.currentHand!.trumpSuit).toBeNull(); // no-trump
    expect(r2.state.currentHand!.landlord).toBe(1);
  });

  it('big-joker pair beats small-joker pair (tier 4 > tier 3)', () => {
    let s = startedGame();
    s = planHand(s, 0, [jok('small', 0), jok('small', 1)]);
    s = planHand(s, 1, [jok('big', 0), jok('big', 1)]);
    const r1 = applyEvent(s, { kind: 'declare', seat: 0, cards: [jok('small', 0), jok('small', 1)] });
    expect(r1.ok).toBe(true);
    if (!r1.ok) return;
    const r2 = applyEvent(r1.state, { kind: 'declare', seat: 1, cards: [jok('big', 0), jok('big', 1)] });
    expect(r2.ok).toBe(true);
    if (!r2.ok) return;
    expect(r2.state.currentHand!.standingDeclaration?.kind).toBe('joker-pair');
  });

  it('rejects same-tier override (single in different suit)', () => {
    let s = startedGame();
    s = planHand(s, 0, [std('hearts', '2', 0)]);
    s = planHand(s, 1, [std('spades', '2', 0)]);
    const r1 = applyEvent(s, { kind: 'declare', seat: 0, cards: [std('hearts', '2', 0)] });
    expect(r1.ok).toBe(true);
    if (!r1.ok) return;
    const r2 = applyEvent(r1.state, { kind: 'declare', seat: 1, cards: [std('spades', '2', 0)] });
    expect(r2.ok).toBe(false);
  });

  it('rejects same-tier override (pair vs pair)', () => {
    let s = startedGame();
    s = planHand(s, 0, [std('hearts', '2', 0), std('hearts', '2', 1)]);
    s = planHand(s, 1, [std('spades', '2', 0), std('spades', '2', 1)]);
    const r1 = applyEvent(s, {
      kind: 'declare',
      seat: 0,
      cards: [std('hearts', '2', 0), std('hearts', '2', 1)],
    });
    expect(r1.ok).toBe(true);
    if (!r1.ok) return;
    const r2 = applyEvent(r1.state, {
      kind: 'declare',
      seat: 1,
      cards: [std('spades', '2', 0), std('spades', '2', 1)],
    });
    expect(r2.ok).toBe(false);
  });
});

describe('release_lockout', () => {
  it('with standing declaration → declare_open', () => {
    let s = startedGame();
    s = planHand(s, 0, [std('hearts', '2', 0)]);
    const r = applyEvent(s, { kind: 'declare', seat: 0, cards: [std('hearts', '2', 0)] });
    if (!r.ok) throw new Error(r.error);
    const r2 = applyEvent(r.state, { kind: 'release_lockout' });
    expect(r2.ok).toBe(true);
    if (!r2.ok) return;
    expect(r2.state.phase).toBe('declare_open');
  });

  it('with no declaration → kitty flip, assigns trump suit + random landlord', () => {
    const s = startedGame(99);
    const r = applyEvent(s, { kind: 'release_lockout', seed: 5 });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    const h = r.state.currentHand!;
    expect(r.state.phase).toBe('kitty_replace');
    expect(h.kitty).toHaveLength(0); // moved into landlord hand
    expect(h.landlord).not.toBeNull();
    expect(h.trumpRank).toBe('2');
    expect(h.trumpFinalized).toBe(true);
    // Landlord now holds 25 + 8 = 33 cards.
    expect(handFor(r.state, h.landlord!)).toHaveLength(33);
    expect(r.state.landlord).toBe(h.landlord);
  });
});

describe('pick_up_kitty + replace_kitty', () => {
  it('landlord picks up kitty → 33 cards in hand, phase = kitty_replace', () => {
    let s = startedGame();
    s = planHand(s, 0, [std('hearts', '2', 0)]);
    const a = applyEvent(s, { kind: 'declare', seat: 0, cards: [std('hearts', '2', 0)] });
    if (!a.ok) throw new Error(a.error);
    const b = applyEvent(a.state, { kind: 'release_lockout' });
    if (!b.ok) throw new Error(b.error);
    const c = applyEvent(b.state, { kind: 'pick_up_kitty', seat: 0 });
    expect(c.ok).toBe(true);
    if (!c.ok) return;
    expect(c.state.phase).toBe('kitty_replace');
    expect(handFor(c.state, 0)).toHaveLength(33);
    expect(c.state.currentHand!.kitty).toHaveLength(0);
    expect(c.state.landlord).toBe(0);
  });

  it('non-landlord cannot pick up kitty', () => {
    let s = startedGame();
    s = planHand(s, 0, [std('hearts', '2', 0)]);
    const a = applyEvent(s, { kind: 'declare', seat: 0, cards: [std('hearts', '2', 0)] });
    if (!a.ok) throw new Error(a.error);
    const b = applyEvent(a.state, { kind: 'release_lockout' });
    if (!b.ok) throw new Error(b.error);
    const c = applyEvent(b.state, { kind: 'pick_up_kitty', seat: 1 });
    expect(c.ok).toBe(false);
  });

  it('replace_kitty: landlord submits 8 cards from hand → phase = play, hand = 25', () => {
    let s = startedGame();
    s = planHand(s, 0, [std('hearts', '2', 0)]);
    const a = applyEvent(s, { kind: 'declare', seat: 0, cards: [std('hearts', '2', 0)] });
    if (!a.ok) throw new Error(a.error);
    const b = applyEvent(a.state, { kind: 'release_lockout' });
    if (!b.ok) throw new Error(b.error);
    const c = applyEvent(b.state, { kind: 'pick_up_kitty', seat: 0 });
    if (!c.ok) throw new Error(c.error);
    const fullHand = handFor(c.state, 0);
    const toBury = fullHand.slice(25); // last 8 cards
    const d = applyEvent(c.state, { kind: 'replace_kitty', seat: 0, cards: toBury });
    expect(d.ok).toBe(true);
    if (!d.ok) return;
    expect(d.state.phase).toBe('play');
    expect(handFor(d.state, 0)).toHaveLength(25);
    expect(d.state.currentHand!.kitty).toHaveLength(8);
  });

  it('replace_kitty rejects wrong card count', () => {
    let s = startedGame();
    s = planHand(s, 0, [std('hearts', '2', 0)]);
    const a = applyEvent(s, { kind: 'declare', seat: 0, cards: [std('hearts', '2', 0)] });
    if (!a.ok) throw new Error(a.error);
    const b = applyEvent(a.state, { kind: 'release_lockout' });
    if (!b.ok) throw new Error(b.error);
    const c = applyEvent(b.state, { kind: 'pick_up_kitty', seat: 0 });
    if (!c.ok) throw new Error(c.error);
    const d = applyEvent(c.state, {
      kind: 'replace_kitty',
      seat: 0,
      cards: handFor(c.state, 0).slice(0, 7),
    });
    expect(d.ok).toBe(false);
  });

  it('replace_kitty rejects card not in landlord hand', () => {
    let s = startedGame();
    s = planHand(s, 0, [std('hearts', '2', 0)]);
    const a = applyEvent(s, { kind: 'declare', seat: 0, cards: [std('hearts', '2', 0)] });
    if (!a.ok) throw new Error(a.error);
    const b = applyEvent(a.state, { kind: 'release_lockout' });
    if (!b.ok) throw new Error(b.error);
    const c = applyEvent(b.state, { kind: 'pick_up_kitty', seat: 0 });
    if (!c.ok) throw new Error(c.error);
    // Take cards from seat 1, not seat 0.
    const wrong = handFor(c.state, 1).slice(0, 8);
    const d = applyEvent(c.state, { kind: 'replace_kitty', seat: 0, cards: wrong });
    expect(d.ok).toBe(false);
  });
});
