// Sampling opponent hands consistent with what the acting seat knows.
//
// Search over hidden information needs concrete worlds to search in. A
// world here is an assignment of every unseen card to the other seats (and
// to the kitty, when we are not the landlord and so do not know it),
// subject to the two things public play tells us: how many cards each seat
// still holds, and which suits they have shown themselves out of.
//
// Hand *sizes* are public -- everyone plays the same number of cards to a
// trick -- so using them is not cheating. Void inference comes from the
// follow rules: a seat that answered a heart lead with anything else has
// no hearts left (rules.md §7.3).

import { effectiveSuit } from '../../ordering';
import { mulberry32 } from '../../prng';
import { type GameState, type Seat } from '../../state';
import { cardKey, type Card, type EffectiveSuit, type TrumpContext } from '../../types';

import { cardSlot, NUM_SLOTS } from './cards';

export type World = {
  // Sampled hand per seat. The acting seat's entry is its real hand.
  readonly hands: ReadonlyArray<ReadonlyArray<Card>>;
  readonly kitty: ReadonlyArray<Card>;
  // True when a void constraint had to be broken to place every card,
  // which the caller may want to count rather than ignore.
  readonly relaxed: boolean;
};

function cardFromSlot(slot: number): Card {
  if (slot === 52) return { kind: 'joker', rank: 'small', deckId: 0 };
  if (slot === 53) return { kind: 'joker', rank: 'big', deckId: 0 };
  const suits = ['spades', 'hearts', 'diamonds', 'clubs'] as const;
  const ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] as const;
  return {
    kind: 'standard',
    suit: suits[Math.floor(slot / 13)]!,
    rank: ranks[slot % 13]!,
    deckId: 0,
  };
}

// Suits each seat has shown itself out of, as a set of effective suits.
export function knownVoids(
  state: GameState,
  ctx: TrumpContext,
): Array<Set<EffectiveSuit>> {
  const out: Array<Set<EffectiveSuit>> = [];
  for (let s = 0; s < state.config.numPlayers; s++) out.push(new Set());
  const hand = state.currentHand;
  if (hand === null) return out;
  const scan = (plays: ReadonlyArray<{ seat: Seat; cards: ReadonlyArray<Card> }>): void => {
    const lead = plays[0];
    if (lead === undefined) return;
    const led = effectiveSuit(lead.cards[0]!, ctx);
    for (let i = 1; i < plays.length; i++) {
      const play = plays[i]!;
      if (play.cards.some((c) => effectiveSuit(c, ctx) !== led)) out[play.seat]!.add(led);
    }
  };
  for (const t of hand.tricks) scan(t.plays);
  if (hand.currentTrick !== null) scan(hand.currentTrick.plays);
  return out;
}

// Every card the acting seat has not seen: the deck minus its own hand,
// minus everything played, minus its own buried kitty.
export function unseenCards(
  state: GameState,
  seat: Seat,
  myKitty: ReadonlyArray<Card>,
): Card[] {
  const counts = new Uint8Array(NUM_SLOTS).fill(state.config.numDecks);
  const hand = state.currentHand;
  if (hand === null) return [];
  const drop = (c: Card): void => {
    const s = cardSlot(c);
    if (counts[s]! > 0) counts[s] = counts[s]! - 1;
  };
  for (const c of hand.hands[seat] ?? []) drop(c);
  for (const t of hand.tricks) for (const p of t.plays) for (const c of p.cards) drop(c);
  if (hand.currentTrick !== null) {
    for (const p of hand.currentTrick.plays) for (const c of p.cards) drop(c);
  }
  for (const c of myKitty) drop(c);
  const out: Card[] = [];
  for (let slot = 0; slot < NUM_SLOTS; slot++) {
    for (let i = 0; i < counts[slot]!; i++) out.push(cardFromSlot(slot));
  }
  return out;
}

// One consistent world. Cards are placed hardest-first -- into whichever
// seat has the fewest remaining legal homes -- because filling greedily in
// card order paints itself into corners when several seats are void.
// Optional belief weighting. `weightOf(card, seat)` is a relative
// preference for putting that card in that seat's hand; feasibility is
// still enforced by the caller-independent machinery below, so a weight
// can shift a world but never make it contradict public play. Omitted,
// the sampler behaves exactly as it always has.
export type PlacementWeight = (card: Card, seat: number) => number;

export function sampleWorld(
  state: GameState,
  seat: Seat,
  myKitty: ReadonlyArray<Card>,
  ctx: TrumpContext,
  rand: () => number,
  weightOf?: PlacementWeight,
): World {
  const hand = state.currentHand!;
  const numPlayers = state.config.numPlayers;
  const voids = knownVoids(state, ctx);
  const pool = unseenCards(state, seat, myKitty);

  // Shuffle the pool so repeated samples differ.
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = pool[i]!;
    pool[i] = pool[j]!;
    pool[j] = tmp;
  }

  const need: number[] = [];
  for (let s = 0; s < numPlayers; s++) {
    need.push(s === seat ? 0 : (hand.hands[s] ?? []).length);
  }
  // Whatever is left over belongs to the kitty, which we cannot see unless
  // we buried it ourselves.
  const assignedTotal = need.reduce((a, b) => a + b, 0);
  const kittyNeed = Math.max(0, pool.length - assignedTotal);

  const hands: Card[][] = [];
  for (let s = 0; s < numPlayers; s++) hands.push(s === seat ? [...(hand.hands[s] ?? [])] : []);
  const kitty: Card[] = [];
  let relaxed = false;
  let kittyLeft = kittyNeed;

  for (const card of pool) {
    const es = effectiveSuit(card, ctx);
    const legal: number[] = [];
    for (let s = 0; s < numPlayers; s++) {
      if (s === seat || need[s]! <= 0) continue;
      if (voids[s]!.has(es)) continue;
      legal.push(s);
    }
    if (legal.length === 0) {
      if (kittyLeft > 0) {
        kitty.push(card);
        kittyLeft -= 1;
        continue;
      }
      // No legal home: place it where there is room and record that the
      // world is not fully consistent, rather than dropping the card.
      const any: number[] = [];
      for (let s = 0; s < numPlayers; s++) if (s !== seat && need[s]! > 0) any.push(s);
      if (any.length === 0) {
        kitty.push(card);
        continue;
      }
      const pick = any[Math.floor(rand() * any.length)]!;
      hands[pick]!.push(card);
      need[pick] = need[pick]! - 1;
      relaxed = true;
      continue;
    }
    // Without beliefs: prefer the seat with the greatest remaining need,
    // so tight seats do not run out of legal cards at the end.
    //
    // With beliefs: sample among the legal seats in proportion to
    // belief x remaining need. The need factor is what keeps the greedy
    // property that made the uniform version work -- a seat close to full
    // stops attracting cards -- while the belief decides among seats that
    // are equally able to take one.
    let best = legal[0]!;
    if (weightOf === undefined) {
      for (const s of legal) if (need[s]! > need[best]!) best = s;
    } else {
      let total = 0;
      for (const s of legal) total += weightOf(card, s) * need[s]!;
      if (total <= 0) {
        for (const s of legal) if (need[s]! > need[best]!) best = s;
      } else {
        let draw = rand() * total;
        for (const s of legal) {
          draw -= weightOf(card, s) * need[s]!;
          if (draw <= 0) {
            best = s;
            break;
          }
          best = s;
        }
      }
    }
    // With the kitty still hungry, sometimes feed it instead, in
    // proportion to how much of the pool it still claims.
    if (kittyLeft > 0 && rand() < kittyLeft / (kittyLeft + need[best]!)) {
      kitty.push(card);
      kittyLeft -= 1;
      continue;
    }
    hands[best]!.push(card);
    need[best] = need[best]! - 1;
  }

  return { hands, kitty, relaxed };
}

// A state whose hidden cards are replaced by a sampled world, for
// simulating forward. Only the other seats' hands and the kitty change.
export function stateWithWorld(state: GameState, world: World): GameState {
  const hand = state.currentHand!;
  const hands: Record<Seat, Card[]> = {};
  for (let s = 0; s < state.config.numPlayers; s++) hands[s] = [...world.hands[s]!];
  return {
    ...state,
    currentHand: { ...hand, hands, kitty: [...world.kitty] },
  };
}

export function makeRng(seed: number): () => number {
  return mulberry32(seed >>> 0);
}

export function worldSignature(world: World): string {
  return world.hands.map((h) => h.map(cardKey).sort().join(',')).join('|');
}
