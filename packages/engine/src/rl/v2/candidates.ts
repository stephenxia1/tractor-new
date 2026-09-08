// Candidate move generation for the RL actor.
//
// Legality comes from the engine (validateFollow / applyEvent); *which*
// legal moves we offer the network is a modelling choice, so it lives
// here rather than in the engine.
//
// Two differences from the engine's allLegalPlayActions:
//
//   1. Fair play. The engine's lead-throw enumerator runs the rules.md
//      §7.6 block check against the other seats' actual hands, which
//      would leak hidden information into the *action set* itself. We
//      pass includeThrows: false and generate throws from public
//      information only: a component is offered only when no unseen card
//      can beat it, which is a guarantee rather than a guess (and is
//      conservative, since unseen cards may sit in the kitty).
//
//   2. Deduplication by card *key* rather than identity. The engine
//      treats the two physical copies of 5-of-spades as distinct moves;
//      for scoring purposes they are the same move, and offering both
//      wastes a forward pass and splits probability mass.

import { classifyCombo } from '../../classifier';
import { findNKinds } from '../../combos';
import { compareCards, effectiveSuit, isTrump } from '../../ordering';
import { allLegalPlayActions } from '../../rl/mlp/actions';
import { sumPoints } from '../../scoring';
import { type ActiveTrick, type GameState, type Seat } from '../../state';
import { decomposePlay, decomposeThrow, decomposedBeats } from '../../throws';
import { type Card, type EffectiveSuit, type TrumpContext } from '../../types';

import { cardSlot, NUM_SLOTS } from './cards';
import {
  FLAG_ALL_LED_SUIT,
  FLAG_ALL_TRUMP,
  FLAG_BEATS_WINNER,
  FLAG_HAS_POINTS,
  SHAPE_BROKEN,
  SHAPE_PAIR,
  SHAPE_SINGLE,
  SHAPE_THROW,
  SHAPE_TRACTOR2,
  SHAPE_TRACTOR3PLUS,
  MAX_CANDIDATE_CARDS,
} from './record';

export type Candidate = {
  readonly cards: ReadonlyArray<Card>;
  readonly shape: number;
  readonly flags: number;
  readonly points: number;
};

export type CandidateOptions = {
  // Hard cap on candidates per decision. Measured distribution is mean 16
  // for leads and 5 for follows, so 48 almost never binds; when it does,
  // `enumerateCandidates` keeps the lowest-index (structurally simplest)
  // moves and reports the truncation via `lastTruncated`.
  readonly maxCandidates?: number;
  // Offer public-information throws on leads. Default true.
  readonly includeThrows?: boolean;
  // Cards the landlord buried, if this seat is the landlord. Used to
  // sharpen the unseen-card pool for throw safety.
  readonly myKitty?: ReadonlyArray<Card>;
};

const DEFAULTS = {
  maxCandidates: 48,
  includeThrows: true,
};

// Set when the last enumerateCandidates call hit maxCandidates. Read by
// the worker so silent truncation shows up in training logs.
export let lastTruncated = false;
// Count of candidates dropped for exceeding MAX_CANDIDATE_CARDS.
export let lastOversized = 0;

export function enumerateCandidates(
  state: GameState,
  seat: Seat,
  opts: CandidateOptions = {},
): Candidate[] {
  lastTruncated = false;
  lastOversized = 0;
  const hand = state.currentHand;
  if (hand === null || hand.trumpRank === null) return [];
  if (state.phase !== 'play') return [];

  const maxCandidates = opts.maxCandidates ?? DEFAULTS.maxCandidates;
  const includeThrows = opts.includeThrows ?? DEFAULTS.includeThrows;
  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const myCards = hand.hands[seat] ?? [];
  const trick = hand.currentTrick;
  const isLead = trick === null || trick.plays.length === 0;

  // Engine enumeration minus the hidden-information throw branch.
  const base = allLegalPlayActions(state, seat, { includeThrows: false });
  const cardSets: Array<ReadonlyArray<Card>> = base.map((a) => a.cards);

  if (isLead && includeThrows) {
    for (const t of publicInfoThrows(myCards, state, seat, ctx, n, opts.myKitty)) {
      cardSets.push(t);
    }
  }

  // The led suit and the play currently winning the trick, for flags.
  let ledSuit: EffectiveSuit | null = null;
  let winning: ReturnType<typeof decomposePlay> = null;
  if (!isLead && trick !== null) {
    const lead = trick.plays[0]!;
    ledSuit = effectiveSuit(lead.cards[0]!, ctx);
    winning = winningPlay(trick, ctx, n);
  }

  const seen = new Set<string>();
  const out: Candidate[] = [];
  for (const cards of cardSets) {
    // The wire format carries a fixed-width card list; a play longer than
    // that (a huge tractor or throw lead we must follow) is dropped, and
    // `lastOversized` lets the worker count how often that happens.
    if (cards.length > MAX_CANDIDATE_CARDS) {
      lastOversized += 1;
      continue;
    }
    const sig = keySignature(cards);
    if (seen.has(sig)) continue;
    seen.add(sig);
    out.push({
      cards,
      shape: shapeOf(cards, ctx, n),
      flags: flagsOf(cards, ctx, n, ledSuit, winning),
      points: Math.min(sumPoints(cards), 255),
    });
    if (out.length >= maxCandidates) {
      lastTruncated = cardSets.length > out.length;
      break;
    }
  }
  return out;
}

// --- throw generation from public information ------------------------------

// Counts of each card slot that this seat has not seen: the full deck
// minus my hand, minus everything played, minus my own buried kitty.
// Cards still in the kitty count as unseen, which only makes the throw
// test stricter.
export function unseenCounts(
  state: GameState,
  seat: Seat,
  myKitty?: ReadonlyArray<Card>,
): Uint8Array {
  const counts = new Uint8Array(NUM_SLOTS).fill(state.config.numDecks);
  const hand = state.currentHand;
  if (hand === null) return counts;
  const drop = (c: Card): void => {
    const s = cardSlot(c);
    if (counts[s]! > 0) counts[s] = counts[s]! - 1;
  };
  for (const c of hand.hands[seat] ?? []) drop(c);
  for (const t of hand.tricks) for (const p of t.plays) for (const c of p.cards) drop(c);
  if (hand.currentTrick !== null) {
    for (const p of hand.currentTrick.plays) for (const c of p.cards) drop(c);
  }
  for (const c of myKitty ?? []) drop(c);
  return counts;
}

// A representative Card for a slot, for ordering comparisons only.
function slotCard(slot: number): Card {
  if (slot === 52) return { kind: 'joker', rank: 'small', deckId: 0 };
  if (slot === 53) return { kind: 'joker', rank: 'big', deckId: 0 };
  const suitIdx = Math.floor(slot / 13);
  const rankIdx = slot % 13;
  const suits = ['spades', 'hearts', 'diamonds', 'clubs'] as const;
  const ranks = [
    '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A',
  ] as const;
  return {
    kind: 'standard',
    suit: suits[suitIdx]!,
    rank: ranks[rankIdx]!,
    deckId: 0,
  };
}

// Throw candidates that are guaranteed to pass the §7.6 block check no
// matter how the unseen cards are distributed: every component is a pair
// no unseen pair outranks, or a single no unseen card outranks.
function publicInfoThrows(
  myCards: ReadonlyArray<Card>,
  state: GameState,
  seat: Seat,
  ctx: TrumpContext,
  n: number,
  myKitty: ReadonlyArray<Card> | undefined,
): Array<ReadonlyArray<Card>> {
  const out: Array<ReadonlyArray<Card>> = [];
  if (myCards.length < 2) return out;
  const unseen = unseenCounts(state, seat, myKitty);

  // Bucket unseen cards by effective suit once.
  const unseenBySuit = new Map<EffectiveSuit, Array<{ card: Card; count: number }>>();
  for (let slot = 0; slot < NUM_SLOTS; slot++) {
    const count = unseen[slot]!;
    if (count === 0) continue;
    const card = slotCard(slot);
    const es = effectiveSuit(card, ctx);
    const list = unseenBySuit.get(es);
    if (list === undefined) unseenBySuit.set(es, [{ card, count }]);
    else list.push({ card, count });
  }

  const mineBySuit = new Map<EffectiveSuit, Card[]>();
  for (const c of myCards) {
    const es = effectiveSuit(c, ctx);
    const list = mineBySuit.get(es);
    if (list === undefined) mineBySuit.set(es, [c]);
    else list.push(c);
  }

  for (const [suit, suitCards] of mineBySuit) {
    if (suitCards.length < 2) continue;
    const outstanding = unseenBySuit.get(suit) ?? [];

    // Unbeatable pairs: no unseen *pair* of this suit outranks them.
    const pairs = findNKinds(suitCards, n);
    const usedInPairs = new Set<Card>();
    const safeComponents: Card[][] = [];
    for (const p of pairs) {
      const probe = p.cards[0]!;
      const beatable = outstanding.some(
        (u) => u.count >= n && compareCards(u.card, probe, ctx) > 0,
      );
      if (beatable) continue;
      safeComponents.push([...p.cards]);
      for (const c of p.cards) usedInPairs.add(c);
    }

    // Unbeatable singles among the cards not committed to a safe pair.
    for (const c of suitCards) {
      if (usedInPairs.has(c)) continue;
      const beatable = outstanding.some((u) => compareCards(u.card, c, ctx) > 0);
      if (beatable) continue;
      safeComponents.push([c]);
    }

    if (safeComponents.length < 2) continue;
    // Strongest first, so prefixes are the "cash my winners" throws.
    safeComponents.sort((a, b) => compareCards(b[0]!, a[0]!, ctx));

    const emit = (components: Card[][]): void => {
      const cards = components.flat();
      if (cards.length < 2) return;
      // A set that classifies as one combo is already enumerated as a
      // pair/tractor; the engine would reject it as a throw.
      if (classifyCombo(cards, ctx, n).ok) return;
      if (!decomposeThrow(cards, ctx, n).ok) return;
      out.push(cards);
    };

    emit(safeComponents);
    if (safeComponents.length > 2) emit(safeComponents.slice(0, 2));
    const pairsOnly = safeComponents.filter((c) => c.length >= n);
    if (pairsOnly.length >= 2 && pairsOnly.length < safeComponents.length) {
      emit(pairsOnly);
    }
  }
  return out;
}

// --- structure tagging ------------------------------------------------------

function shapeOf(cards: ReadonlyArray<Card>, ctx: TrumpContext, n: number): number {
  const cls = classifyCombo(cards, ctx, n);
  if (cls.ok) {
    if (cls.combo.kind === 'single') return SHAPE_SINGLE;
    if (cls.combo.kind === 'nkind') return SHAPE_PAIR;
    return cls.combo.length === 2 ? SHAPE_TRACTOR2 : SHAPE_TRACTOR3PLUS;
  }
  if (decomposeThrow(cards, ctx, n).ok) return SHAPE_THROW;
  return SHAPE_BROKEN;
}

function flagsOf(
  cards: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
  ledSuit: EffectiveSuit | null,
  winning: ReturnType<typeof decomposePlay>,
): number {
  let flags = 0;
  if (cards.every((c) => isTrump(c, ctx))) flags |= FLAG_ALL_TRUMP;
  if (ledSuit !== null && cards.every((c) => effectiveSuit(c, ctx) === ledSuit)) {
    flags |= FLAG_ALL_LED_SUIT;
  }
  if (sumPoints(cards) > 0) flags |= FLAG_HAS_POINTS;
  if (winning !== null) {
    const mine = decomposePlay(cards, ctx, n);
    if (mine !== null && decomposedBeats(mine, winning, ctx, n)) {
      flags |= FLAG_BEATS_WINNER;
    }
  }
  return flags;
}

// The decomposition of the play currently winning the in-flight trick,
// mirroring the reducer's resolveTrick.
export function winningPlay(
  trick: ActiveTrick,
  ctx: TrumpContext,
  n: number,
): ReturnType<typeof decomposePlay> {
  const lead = trick.plays[0];
  if (lead === undefined) return null;
  let best = decomposePlay(lead.cards, ctx, n);
  if (best === null) return null;
  for (let i = 1; i < trick.plays.length; i++) {
    const play = trick.plays[i]!;
    const d = decomposePlay(play.cards, ctx, n);
    if (d === null) continue;
    if (decomposedBeats(d, best, ctx, n)) best = d;
  }
  return best;
}

// Which seat is currently winning the in-flight trick.
export function winningSeat(
  trick: ActiveTrick,
  ctx: TrumpContext,
  n: number,
): Seat | null {
  const lead = trick.plays[0];
  if (lead === undefined) return null;
  let best = decomposePlay(lead.cards, ctx, n);
  if (best === null) return lead.seat;
  let seat = lead.seat;
  for (let i = 1; i < trick.plays.length; i++) {
    const play = trick.plays[i]!;
    const d = decomposePlay(play.cards, ctx, n);
    if (d === null) continue;
    if (decomposedBeats(d, best, ctx, n)) {
      best = d;
      seat = play.seat;
    }
  }
  return seat;
}

function keySignature(cards: ReadonlyArray<Card>): string {
  const slots = cards.map(cardSlot);
  slots.sort((a, b) => a - b);
  return slots.join(',');
}
