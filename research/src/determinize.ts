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

import {
  cardKey,
  effectiveSuit,
  mulberry32,
  type Card,
  type EffectiveSuit,
  type GameState,
  type Seat,
  type TrumpContext,
} from '@tractor/engine';

import { cardSlot, NUM_SLOTS } from './cards';

export type World = {
  // Sampled hand per seat. The acting seat's entry is its real hand.
  readonly hands: ReadonlyArray<ReadonlyArray<Card>>;
  readonly kitty: ReadonlyArray<Card>;
  // True when a void constraint had to be broken to place every card,
  // which the caller may want to count rather than ignore.
  readonly relaxed: boolean;
};

// One bit per deck copy in the seen-mask below, which is a Uint32Array.
const MAX_TRACKED_DECKS = 32;

function cardFromSlot(slot: number, deckId: number): Card {
  if (slot === 52) return { kind: 'joker', rank: 'small', deckId };
  if (slot === 53) return { kind: 'joker', rank: 'big', deckId };
  const suits = ['spades', 'hearts', 'diamonds', 'clubs'] as const;
  const ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] as const;
  return {
    kind: 'standard',
    suit: suits[Math.floor(slot / 13)]!,
    rank: ranks[slot % 13]!,
    deckId,
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
//
// *Which* copy is unseen matters, not merely how many. `cardEquals` is
// full identity including `deckId`, and it is what `containsAll`,
// `removeCards` and the legality check in `legalActions.ts` compare with,
// while combo detection (`validateFollow`, `findNKinds`) compares by
// `cardKey` and ignores `deckId` entirely. Deal a rollout bot two copies
// of a slot that both say `deckId: 0` and the two views disagree about
// the same hand: the bot sees a pair, builds a follow out of it, and the
// reducer then rejects the move it was just asked for ("invalid follow:
// must include 1 pair(s) of hearts: 0 given"), which kills the
// simulation. So track which deck copies of a slot have been seen and
// emit exactly the ones that have not -- that gives every world card a
// distinct identity, and keeps it from colliding with a card the acting
// seat can already see.
export function unseenCards(
  state: GameState,
  seat: Seat,
  myKitty: ReadonlyArray<Card>,
): Card[] {
  const numDecks = state.config.numDecks;
  if (numDecks > MAX_TRACKED_DECKS) {
    // Silently emitting only the first eight copies of every slot would
    // hand the sampler a pool smaller than the hands it has to fill.
    throw new Error(`unseenCards tracks at most ${MAX_TRACKED_DECKS} decks, got ${numDecks}`);
  }
  // Bit d of `seen[slot]` marks deck copy d of that slot as accounted for.
  const seen = new Uint32Array(NUM_SLOTS);
  const hand = state.currentHand;
  if (hand === null) return [];
  const drop = (c: Card): void => {
    // Every card reaching here came out of a deck this config built, so a
    // deckId outside 0..numDecks-1 is a programming error rather than an
    // input. Absorbing it quietly goes wrong in both directions depending
    // on the value: a shift still inside the mask marks a copy nobody
    // holds, leaving the real card in the pool as a phantom, while a
    // shift past the mask's width wraps onto bit 0 and deletes a copy
    // that is genuinely unseen. Either way the sampler gets a pool that
    // cannot fill the hands it has to fill, and nothing says so until the
    // worlds come out quietly wrong.
    if (!Number.isInteger(c.deckId) || c.deckId < 0 || c.deckId >= numDecks) {
      throw new Error(`card ${cardKey(c)} has deckId ${c.deckId}, outside 0..${numDecks - 1}`);
    }
    const s = cardSlot(c);
    seen[s] = seen[s]! | (1 << c.deckId);
  };
  for (const c of hand.hands[seat] ?? []) drop(c);
  for (const t of hand.tricks) for (const p of t.plays) for (const c of p.cards) drop(c);
  if (hand.currentTrick !== null) {
    for (const p of hand.currentTrick.plays) for (const c of p.cards) drop(c);
  }
  for (const c of myKitty) drop(c);
  const out: Card[] = [];
  for (let slot = 0; slot < NUM_SLOTS; slot++) {
    const mask = seen[slot]!;
    for (let d = 0; d < numDecks; d++) {
      if ((mask & (1 << d)) === 0) out.push(cardFromSlot(slot, d));
    }
  }
  return out;
}

// One consistent world. Cards are placed hardest-first -- into whichever
// seat has the fewest remaining legal homes -- because filling greedily in
// card order paints itself into corners when several seats are void.
export function sampleWorld(
  state: GameState,
  seat: Seat,
  myKitty: ReadonlyArray<Card>,
  ctx: TrumpContext,
  rand: () => number,
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
  // Seed the kitty with the cards we buried ourselves. `unseenCards`
  // already dropped them from the pool, so nothing left to place can ever
  // land here: starting empty leaves every simulated hand with a kitty of
  // zero points, which deletes the last-trick kitty bonus (rules.md §8.2)
  // -- the largest single swing in the game -- from the whole search.
  const kitty: Card[] = [...myKitty];
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
    // Prefer the seat with the greatest remaining need, so tight seats do
    // not run out of legal cards at the end.
    let best = legal[0]!;
    for (const s of legal) if (need[s]! > need[best]!) best = s;
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
