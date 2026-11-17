// The PPO bot's trump declaration: research/py/tractor/nn/declare.py's
// network on sim/deal.py's `declare_features`, asked as the research
// evaluation host asks it (research/src/env.ts, research/src/evalhost.ts,
// research/py/tractor/eval/bots.py::_declare_choice).
//
// Eleven fixed actions (sim/deal.py): 0 pass, 1..4 a single of the level
// rank in suit 0..3, 5..8 the pair, 9 the small-joker pair, 10 the
// big-joker pair. The legal set is whatever the engine would accept from
// the seat right now, so the choice can never be refused.
//
// Two ways a live game differs from the one the network trained in:
//
// * Training dealt only landlord-set hands. In the app the first hand,
//   and every hand after the attackers win, starts with no landlord, and
//   a declarer must show their own team's level. Then the level is the
//   seat's own team's, and the landlord is the tentative one the engine
//   holds (the standing declarer), or the seat itself when nobody has
//   declared -- a declaration would make it the landlord.
// * Training asked a seat at each card it received, then once per turn of
//   a closing round after the last card. The ws server asks every bot at
//   every broadcast it is not already acting on. The decision is a pure
//   function of the position, so asking more often only shifts the deal's
//   progress feature by a card or two; nothing here gates on it.

import type { BotAction } from '../../bot';
import { applyEvent } from '../../reducer';
import { teamOf, type GameState, type Seat } from '../../state';
import { NATURAL_RANKS, SUITS, type Card, type NaturalRank } from '../../types';
import { cardSlot } from '../v2/cards';

import { BIG_JOKER, NUM_TYPES, SMALL_JOKER } from './ctx';
import { gelu, linear, maskedArgmax } from './preplayNn';
import type { ZeroWeights } from './weights';

export const DECLARE_ACTIONS = 11;
export const DECLARE_OBS_DIM = 54 + 20 + 2 + 2 + 1 + 13 + 4 + 5 + 5 + 4 + DECLARE_ACTIONS; // 121
const NUM_SEATS = 4;

/** What the evaluation host's `declare` message carries. */
export type DeclareView = {
  readonly seat: number;
  readonly levelRank: number;
  readonly landlord: number;
  /** Cards dealt so far in the hand, all seats. */
  readonly dealt: number;
  /** Cards dealt in a whole hand: numDecks * 54 - kittySize. */
  readonly total: number;
  readonly closing: boolean;
  readonly standingTier: number;
  readonly standingSuit: number;
  readonly standingSeat: number;
  /** The seat's revealed cards as 54 slot counts. */
  readonly hand: ReadonlyArray<number>;
  readonly mask: ReadonlyArray<boolean>;
  readonly numDecks: number;
};

/** sim/deal.py::declare_features, float32 bit for bit. */
export function declareFeatures(v: DeclareView): Float32Array {
  const f = Math.fround;
  const n = v.numDecks;
  const hand = v.hand;
  const out = new Float32Array(DECLARE_OBS_DIM);
  let at = 0;
  for (let t = 0; t < NUM_TYPES; t++) out[at + t] = f(hand[t]! / n);
  at += NUM_TYPES;
  for (let s = 0; s < 4; s++) {
    const c = (r: number): number => hand[s * 13 + r]!;
    let sum = 0;
    let pairs = 0;
    let high = 0;
    for (let r = 0; r < 13; r++) {
      sum += c(r);
      if (c(r) >= n) pairs += 1;
      if (r >= 8) high += c(r);
    }
    out[at + 0] = f(sum / 13);
    out[at + 1] = f(c(v.levelRank) / n);
    out[at + 2] = f(pairs / 6);
    out[at + 3] = f(high / 10);
    out[at + 4] = f((c(3) * 5 + c(8) * 10 + c(11) * 10) / 50);
    at += 5;
  }
  out[at] = f(hand[SMALL_JOKER]! / n);
  out[at + 1] = f(hand[BIG_JOKER]! / n);
  at += 2;
  let held = 0;
  for (let t = 0; t < NUM_TYPES; t++) held += hand[t]!;
  out[at] = f(v.dealt / v.total);
  out[at + 1] = f(held / (v.total / NUM_SEATS));
  at += 2;
  out[at] = v.closing ? 1 : 0;
  at += 1;
  out[at + v.levelRank] = 1;
  at += 13;
  out[at + mod4(v.landlord - v.seat)] = 1;
  at += 4;
  out[at + v.standingTier] = 1;
  at += 5;
  if (v.standingTier > 0) out[at + v.standingSuit] = 1;
  at += 5;
  if (v.standingTier > 0) out[at + mod4(v.standingSeat - v.seat)] = 1;
  at += 4;
  for (let a = 0; a < DECLARE_ACTIONS; a++) out[at + a] = v.mask[a] === true ? 1 : 0;
  at += DECLARE_ACTIONS;
  if (at !== DECLARE_OBS_DIM) throw new Error('declareFeatures: layout drifted');
  return out;
}

function mod4(x: number): number {
  return ((x % NUM_SEATS) + NUM_SEATS) % NUM_SEATS;
}

/** The declaration actor's raw logits (illegal actions not yet masked). */
export function declareLogits(w: ZeroWeights, obs: Float32Array): Float64Array {
  const cfg = w.manifest.declare_config;
  if (cfg.residual || cfg.ffn !== 'gelu') {
    throw new Error('zero declare: only the plain GELU declaration MLP is ported');
  }
  let x: ArrayLike<number> = obs;
  let dIn = DECLARE_OBS_DIM;
  for (let layer = 0; layer < cfg.depth; layer++) {
    const i = 2 * layer;
    const h = linear(
      x,
      w.get(`declare.actor.${i}.weight`, cfg.width, dIn),
      w.get(`declare.actor.${i}.bias`, cfg.width),
      dIn,
      cfg.width,
    );
    for (let k = 0; k < h.length; k++) h[k] = gelu(h[k]!);
    x = h;
    dIn = cfg.width;
  }
  const last = 2 * cfg.depth;
  return linear(
    x,
    w.get(`declare.actor.${last}.weight`, DECLARE_ACTIONS, dIn),
    w.get(`declare.actor.${last}.bias`, DECLARE_ACTIONS),
    dIn,
    DECLARE_ACTIONS,
  );
}

/** The greedy action over the legal mask (`_declare_choice`). */
export function declareChoice(w: ZeroWeights, v: DeclareView): number {
  const logits = declareLogits(w, declareFeatures(v));
  const a = maskedArgmax(logits, v.mask);
  return a < 0 ? 0 : a;
}

// --- the live engine position -----------------------------------------------

const DECLARE_PHASES = new Set(['dealing', 'declare_lockout', 'declare_open']);

/** The level rank `seat` must declare with: the hand's when a landlord is
 *  set, else the seat's own team's (the engine's `declare`). */
function declareLevel(state: GameState, seat: Seat): NaturalRank | null {
  const hand = state.currentHand;
  if (hand === null) return null;
  if (state.landlord !== null) return hand.trumpRank;
  return state.teamLevels[teamOf(state, seat)] ?? null;
}

/** The cards the seat may declare with now: its revealed prefix while
 *  dealing, its whole hand after (env.ts `revealedFor`). */
export function revealedCards(state: GameState, seat: Seat): ReadonlyArray<Card> {
  const hand = state.currentHand;
  if (hand === null) return [];
  const all = hand.hands[seat] ?? [];
  if (state.phase !== 'dealing') return all;
  return all.slice(0, hand.cardsDealtBySeat[seat] ?? 0);
}

/** The cards declaration action `a` shows, or null if not held
 *  (env.ts `declareCards`). */
export function declareCards(state: GameState, seat: Seat, a: number): Card[] | null {
  const level = declareLevel(state, seat);
  if (level === null) return null;
  const held = revealedCards(state, seat);
  const take = (pred: (c: Card) => boolean, n: number): Card[] | null => {
    const got = held.filter(pred);
    return got.length >= n ? got.slice(0, n) : null;
  };
  if (a >= 1 && a <= 4) {
    const suit = SUITS[a - 1];
    return take((c) => c.kind === 'standard' && c.suit === suit && c.rank === level, 1);
  }
  if (a >= 5 && a <= 8) {
    const suit = SUITS[a - 5];
    return take((c) => c.kind === 'standard' && c.suit === suit && c.rank === level, 2);
  }
  if (a === 9) return take((c) => c.kind === 'joker' && c.rank === 'small', 2);
  if (a === 10) return take((c) => c.kind === 'joker' && c.rank === 'big', 2);
  return null;
}

/** Which of the 11 actions the engine would accept from `seat` now. */
export function declarationMask(state: GameState, seat: Seat): boolean[] {
  const mask = [true];
  for (let a = 1; a < DECLARE_ACTIONS; a++) {
    const cards = declareCards(state, seat, a);
    mask.push(cards !== null && applyEvent(state, { kind: 'declare', seat, cards }).ok);
  }
  return mask;
}

/** Tier (0 none, 1 single, 2 pair, 3/4 joker pairs), suit (0..3, 4 for
 *  no-trump, -1 none) and seat (-1 none) of the standing declaration. */
export function standing(state: GameState): { tier: number; suit: number; seat: number } {
  const d = state.currentHand?.standingDeclaration ?? null;
  if (d === null) return { tier: 0, suit: -1, seat: -1 };
  if (d.kind === 'joker-pair') return { tier: d.jokerRank === 'big' ? 4 : 3, suit: 4, seat: d.seat };
  return { tier: d.kind === 'pair' ? 2 : 1, suit: SUITS.indexOf(d.suit), seat: d.seat };
}

/** The network's view of a declaration decision for `seat`, or null when
 *  it has nothing to decide: wrong phase, or no declaration legal. */
export function declareView(state: GameState, seat: Seat): DeclareView | null {
  const hand = state.currentHand;
  if (hand === null || !DECLARE_PHASES.has(state.phase)) return null;
  const level = declareLevel(state, seat);
  if (level === null) return null;
  const mask = declarationMask(state, seat);
  if (!mask.slice(1).some(Boolean)) return null;
  const { numDecks, kittySize, numPlayers } = state.config;
  const total = numDecks * 54 - kittySize;
  let dealt = total;
  if (state.phase === 'dealing') {
    dealt = 0;
    for (let s = 0; s < numPlayers; s++) dealt += hand.cardsDealtBySeat[s] ?? 0;
  }
  const counts = new Array<number>(NUM_TYPES).fill(0);
  for (const c of revealedCards(state, seat)) {
    const s = cardSlot(c);
    counts[s] = (counts[s] ?? 0) + 1;
  }
  const st = standing(state);
  return {
    seat,
    levelRank: NATURAL_RANKS.indexOf(level),
    landlord: hand.landlord ?? seat,
    dealt,
    total,
    closing: state.phase !== 'dealing',
    standingTier: st.tier,
    standingSuit: st.suit,
    standingSeat: st.seat,
    hand: counts,
    mask,
    numDecks,
  };
}

/** The network's declaration for `seat` now, or null to pass (or when it
 *  has nothing to decide). Called in 'dealing', 'declare_lockout' and
 *  'declare_open'; in 'declare_open' the caller picks up the kitty for a
 *  landlord this passes for. */
export function zeroDeclare(w: ZeroWeights, state: GameState, seat: Seat): BotAction | null {
  if (state.config.numPlayers !== NUM_SEATS) return null;
  const view = declareView(state, seat);
  if (view === null) return null;
  const a = declareChoice(w, view);
  if (a === 0) return null;
  const cards = declareCards(state, seat, a);
  if (cards === null) return null;
  return { kind: 'declare', cards };
}
