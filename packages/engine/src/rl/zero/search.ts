// Determinized search at inference time around the zero agent's policy:
// research/py/tractor/eval/search.py `search_move`, ported.
//
// 1. Worlds: deal every card the seat has not seen to the other three
//    hands (sized as they stand) and to the kitty unless the seat buried
//    it, respecting the suits each seat has shown itself out of
//    (`sampleWorld`, an exact port of `sample_world`).
// 2. Candidates: the policy's top `topK` moves by prior; a decision with
//    one move, or with one move holding `confident` of the policy, is not
//    searched.
// 3. Simulation: in each world each candidate is played, then the policy
//    plays every seat greedily, each from its own view of that world,
//    until `extraTricks` tricks have closed after the current one.
// 4. Leaf value: the points those tricks settled, signed for the
//    searching team, plus the privileged critic's estimate of the points
//    to come from the seat to act, signed likewise; a hand that ends
//    inside the simulation is scored on what it settled, kitty bonus
//    included.
//
// One deliberate difference from the Python: the simulation runs on the
// engine, so a simulated seat chooses among the engine's candidate list
// (`enumerateCandidates`, what the root is offered too) where the Python
// simulator offers its own `legal_moves`. The two lists agree on leads and
// single-card follows; for multi-card follows the simulator's is more
// complete (research/README.md, "Which follows the network is shown"), and
// the engine offers only provably safe throws. research/py/
// test_zero_search_parity.py measures how often that changes a simulated
// move.
//
// Nothing here reads a hidden card. Worlds are drawn from the seat's own
// hand, the public record and the hand sizes; the physical card objects
// the engine needs are the unseen ones regrouped by those counts.

import { applyEvent, nextToPlay, resolveTrick } from '../../reducer';
import type { GameState, Seat } from '../../state';
import type { Card } from '../../types';
import { enumerateCandidates } from '../v2/candidates';
import { cardSlot } from '../v2/cards';

import { criticValue, privilegedView } from './critic';
import { encodePosition, makeCtx, packCandidates, POINTS, type Ctx, type ZeroPosition } from './features';
import { actorLogits } from './net';
import { plane, positionFor } from './position';
import type { ZeroWeights } from './weights';

const NUM_SEATS = 4;
const NUM_TYPES = 54;
const NUM_EFF_SUITS = 5;
const POINTS_PER_DECK = 100;
const MAX_SAMPLE_ATTEMPTS = 32;

export type SearchConfig = {
  readonly worlds: number;
  readonly topK: number;
  readonly extraTricks: number;
  readonly confident: number;
};

/** The setting the research evaluation measured best (0.607 / +10.0). */
export const DEFAULT_SEARCH: SearchConfig = { worlds: 8, topK: 3, extraTricks: 0, confident: 0.99 };

// --- random numbers --------------------------------------------------------

export type Rng = () => number; // uniform [0, 1)

export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A seed from the position, so the same decision searches the same worlds
 *  whenever it is asked. */
export function positionSeed(base: number, pos: ZeroPosition): number {
  let h = (2166136261 ^ base) >>> 0;
  const mix = (v: number) => {
    h ^= v & 0xff;
    h = Math.imul(h, 16777619) >>> 0;
  };
  mix(pos.seat);
  mix(pos.tricks.length);
  mix(pos.trickPlays.length);
  for (let t = 0; t < NUM_TYPES; t++) mix(pos.hand[t] ?? 0);
  for (const play of pos.trickPlays) for (let t = 0; t < NUM_TYPES; t++) mix(play[t] ?? 0);
  return h;
}

// --- worlds ------------------------------------------------------------------

export type World = {
  readonly hands: Int8Array[]; // [4][54]; the searching seat's row is its real hand
  readonly kitty: Int8Array; // 54
  readonly relaxed: boolean; // a void constraint had to be broken to place every card
};

/** What `sampleWorld` may read: the seat's view plus the public sizes. */
export type SearchView = {
  readonly pos: ZeroPosition;
  readonly handSizes: ReadonlyArray<number>; // every seat's card count, public
  readonly kittySize: number;
};

function playSuit(play: ArrayLike<number>, ctx: Ctx): number | null {
  let suit = -1;
  for (let t = 0; t < NUM_TYPES; t++) {
    if ((play[t] ?? 0) === 0) continue;
    const s = ctx.eff[t]!;
    if (suit === -1) suit = s;
    else if (s !== suit) return null;
  }
  return suit === -1 ? null : suit;
}

/** bool[4][5]: seats shown out of an effective suit (search.py `known_voids`). */
export function knownVoids(pos: ZeroPosition): boolean[][] {
  const ctx = makeCtx(pos.levelRank, pos.trumpSuit);
  const voids = Array.from({ length: NUM_SEATS }, () => new Array<boolean>(NUM_EFF_SUITS).fill(false));
  const tricks: Array<[number, ReadonlyArray<ArrayLike<number>>]> = pos.tricks.map((t) => [t.leader, t.plays]);
  if (pos.trickPlays.length > 0) tricks.push([pos.trickLeader, pos.trickPlays]);
  for (const [leader, plays] of tricks) {
    if (plays.length === 0) continue;
    const lead = plays[0]!;
    const suit = playSuit(lead, ctx);
    if (suit === null) continue;
    let need = 0;
    for (let t = 0; t < NUM_TYPES; t++) need += lead[t] ?? 0;
    for (let i = 1; i < plays.length; i++) {
      let inSuit = 0;
      const p = plays[i]!;
      for (let t = 0; t < NUM_TYPES; t++) if (ctx.eff[t] === suit) inSuit += p[t] ?? 0;
      if (inSuit < need) voids[(leader + i) % NUM_SEATS]![suit] = true;
    }
  }
  return voids;
}

/** Cards `seat` has not seen (PlayState.unseen_from). */
export function unseenFrom(pos: ZeroPosition): Int16Array {
  const seen = new Int16Array(NUM_TYPES);
  for (let t = 0; t < NUM_TYPES; t++) seen[t] = pos.hand[t] ?? 0;
  const add = (p: ArrayLike<number>) => {
    for (let t = 0; t < NUM_TYPES; t++) seen[t] = seen[t]! + (p[t] ?? 0);
  };
  for (const tr of pos.tricks) for (const p of tr.plays) add(p);
  for (const p of pos.trickPlays) add(p);
  if (pos.seat === pos.landlord && pos.kitty !== null) add(pos.kitty);
  const out = new Int16Array(NUM_TYPES);
  for (let t = 0; t < NUM_TYPES; t++) out[t] = Math.max(pos.numDecks - seen[t]!, 0);
  return out;
}

/** Deal every unseen card consistently with public play (sample_world). */
export function sampleWorld(view: SearchView, rng: Rng): World {
  const { pos } = view;
  const seat = pos.seat;
  const ctx = makeCtx(pos.levelRank, pos.trumpSuit);
  const unseen = unseenFrom(pos);
  const others = [0, 1, 2, 3].filter((s) => s !== seat);
  const sizes = others.map((s) => view.handSizes[s]!);
  const knowsKitty = seat === pos.landlord && pos.kitty !== null;
  if (!knowsKitty) sizes.push(view.kittySize);
  const cards: number[] = [];
  for (let t = 0; t < NUM_TYPES; t++) for (let k = 0; k < unseen[t]!; k++) cards.push(t);
  const total = sizes.reduce((a, b) => a + b, 0);
  if (cards.length !== total) {
    throw new Error(`zero search: seat ${seat} has ${cards.length} unseen cards for ${total} hidden places`);
  }
  const voids = knownVoids(pos);
  const L = sizes.length;
  const allowed: boolean[][] = cards.map((t) => {
    const row = new Array<boolean>(L).fill(true);
    others.forEach((s, j) => {
      row[j] = !voids[s]![ctx.eff[t]!];
    });
    return row;
  });

  let relaxed = false;
  let where = new Int32Array(cards.length);
  for (let attempt = 0; attempt <= MAX_SAMPLE_ATTEMPTS; attempt++) {
    if (attempt === MAX_SAMPLE_ATTEMPTS) {
      for (const row of allowed) row.fill(true);
      relaxed = true;
    }
    const room = sizes.slice();
    // Most constrained first; random within a tie (np.lexsort, last key primary).
    const tie = cards.map(() => rng());
    const count = allowed.map((row) => row.reduce((a, b) => a + (b ? 1 : 0), 0));
    const order = cards.map((_, i) => i).sort((a, b) => count[a]! - count[b]! || tie[a]! - tie[b]! || a - b);
    where = new Int32Array(cards.length);
    let ok = true;
    for (const i of order) {
      let sum = 0;
      for (let j = 0; j < L; j++) sum += allowed[i]![j] ? room[j]! : 0;
      if (sum === 0) {
        ok = false;
        break;
      }
      // np.searchsorted(cumsum(weight), u * total, side='right').
      const target = rng() * sum;
      let acc = 0;
      let loc = L - 1;
      for (let j = 0; j < L; j++) {
        acc += allowed[i]![j] ? room[j]! : 0;
        if (acc > target) {
          loc = j;
          break;
        }
      }
      where[i] = loc;
      room[loc] = room[loc]! - 1;
    }
    if (ok) break;
  }

  const hands = Array.from({ length: NUM_SEATS }, () => new Int8Array(NUM_TYPES));
  for (let t = 0; t < NUM_TYPES; t++) hands[seat]![t] = pos.hand[t] ?? 0;
  const kitty = new Int8Array(NUM_TYPES);
  if (knowsKitty) for (let t = 0; t < NUM_TYPES; t++) kitty[t] = pos.kitty![t] ?? 0;
  cards.forEach((t, i) => {
    const loc = where[i]!;
    if (loc < others.length) {
      const h = hands[others[loc]!]!;
      h[t] = h[t]! + 1;
    } else {
      kitty[t] = kitty[t]! + 1;
    }
  });
  return { hands, kitty, relaxed };
}

// --- simulating a world on the engine ----------------------------------------

/** `state` with the hidden cards placed as `world` says. The physical
 *  cards are the unseen ones, regrouped by type. */
export function stateInWorld(state: GameState, seat: Seat, world: World): GameState {
  const hand = state.currentHand!;
  const knowsKitty = seat === hand.landlord;
  const pool = new Map<number, Card[]>();
  const put = (c: Card) => {
    const s = cardSlot(c);
    const list = pool.get(s);
    if (list === undefined) pool.set(s, [c]);
    else list.push(c);
  };
  for (let s = 0; s < NUM_SEATS; s++) if (s !== seat) for (const c of hand.hands[s as Seat] ?? []) put(c);
  if (!knowsKitty) for (const c of hand.kitty) put(c);
  const take = (counts: Int8Array): Card[] => {
    const out: Card[] = [];
    for (let t = 0; t < NUM_TYPES; t++) {
      for (let k = 0; k < counts[t]!; k++) {
        const c = pool.get(t)?.pop();
        if (c === undefined) throw new Error('zero search: world does not match the unseen cards');
        out.push(c);
      }
    }
    return out;
  };
  const hands: Record<Seat, ReadonlyArray<Card>> = { ...hand.hands };
  for (let s = 0; s < NUM_SEATS; s++) if (s !== seat) hands[s as Seat] = take(world.hands[s]!);
  const kitty = knowsKitty ? hand.kitty : take(world.kitty);
  return { ...state, currentHand: { ...hand, hands, kitty } };
}

function apply(state: GameState, ev: Parameters<typeof applyEvent>[1]): GameState {
  const r = applyEvent(state, ev);
  if (!r.ok) throw new Error(`zero search: the engine refused ${ev.kind} in a simulation: ${r.error}`);
  return r.state;
}

function points(cards: ReadonlyArray<Card>): number {
  let p = 0;
  for (const c of cards) p += POINTS[cardSlot(c)]!;
  return p;
}

/** Who takes the full trick in progress, and its points. */
export function settleOpenTrick(state: GameState): { winner: number; points: number } {
  const h = state.currentHand!;
  const trick = h.currentTrick!;
  const res = resolveTrick(trick, { trumpRank: h.trumpRank!, trumpSuit: h.trumpSuit }, state.config.numDecks);
  return { winner: res.winner, points: points(trick.plays.flatMap((pl) => pl.cards)) };
}

type Trace = {
  worlds?: World[];
  top?: number[];
  scores?: number[]; // worlds x top, row-major
  means?: number[];
  leaves?: Array<{ world: number; cand: number; settled: number; future: number; value: number | null; state: GameState; seat: number | null; done: boolean }>;
  playouts?: Array<{ state: GameState; seat: number; candidates: Int8Array[]; chosen: number }>;
  searched?: boolean;
};

export type SearchOptions = {
  readonly config?: SearchConfig;
  readonly rng?: Rng;
  /** Worlds to use instead of sampling (parity tests). */
  readonly worlds?: ReadonlyArray<World>;
  /** Filled with the search's workings when given. */
  readonly trace?: Trace;
};

/** The policy's greedy move for the seat to act in `state`, from the
 *  engine's candidate list. */
function policyMove(w: ZeroWeights, state: GameState, seat: Seat, trace?: Trace): ReadonlyArray<Card> {
  const hand = state.currentHand!;
  const myKitty = seat === hand.landlord ? hand.kitty : null;
  const candidates = enumerateCandidates(state, seat, {
    maxCandidates: w.manifest.model_config.max_candidates,
    myKitty: myKitty ?? [],
  });
  if (candidates.length === 0) throw new Error(`zero search: seat ${seat} has no move in a simulation`);
  if (candidates.length === 1) return candidates[0]!.cards;
  const moves = candidates.map((c) => plane(c.cards));
  const logits = actorLogits(w, {
    ...encodePosition(positionFor(state, seat, myKitty)),
    candidates: packCandidates(moves),
    numCandidates: moves.length,
  });
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i]! > logits[best]!) best = i;
  trace?.playouts?.push({ state, seat, candidates: moves, chosen: best });
  return candidates[best]!.cards;
}

/**
 * The move to play: an index into `moves` (the candidate list the policy
 * scored, as `logits`) and whether a search decided it.
 */
export function searchMove(
  w: ZeroWeights,
  state: GameState,
  seat: Seat,
  moves: ReadonlyArray<ReadonlyArray<Card>>,
  logits: Float32Array,
  opts: SearchOptions = {},
): { index: number; searched: boolean } {
  const cfg = opts.config ?? DEFAULT_SEARCH;
  const n = moves.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => logits[b]! - logits[a]! || a - b);
  const best = order[0]!;
  let z = 0;
  for (let i = 0; i < n; i++) z += Math.exp(logits[i]! - logits[best]!);
  const pBest = 1 / z;
  if (n <= 1 || pBest >= cfg.confident) return { index: best, searched: false };
  const top = order.slice(0, cfg.topK);
  if (top.length === 1) return { index: best, searched: false };

  const hand = state.currentHand!;
  const myKitty = seat === hand.landlord ? hand.kitty : null;
  const pos = positionFor(state, seat, myKitty);
  const handSizes = [0, 1, 2, 3].map((s) => (hand.hands[s as Seat] ?? []).length);
  const rng = opts.rng ?? mulberry32(positionSeed(0, pos));
  const worlds =
    opts.worlds !== undefined
      ? [...opts.worlds]
      : Array.from({ length: cfg.worlds }, () => sampleWorld({ pos, handSizes, kittySize: hand.kitty.length }, rng));
  const trace = opts.trace;
  if (trace !== undefined) {
    trace.worlds = worlds;
    trace.top = top;
    trace.leaves = [];
    trace.playouts = [];
    trace.searched = true;
  }

  const team = seat % 2;
  const defenders = hand.landlord! % 2;
  const valueToPoints = 2 * POINTS_PER_DECK * state.config.numDecks;
  const scores: number[] = [];
  worlds.forEach((world, wi) => {
    for (const i of top) {
      let sim = apply(stateInWorld(state, seat, world), { kind: 'play_cards', seat, cards: moves[i]! });
      let closed = 0;
      let settled = 0;
      let done = false;
      for (;;) {
        if (sim.phase === 'trick_review') {
          const h = sim.currentHand!;
          const trick = h.currentTrick!;
          const res = settleOpenTrick(sim);
          const p = res.points;
          settled += res.winner % 2 === team ? p : -p;
          closed += 1;
          let left = 0;
          for (let s = 0; s < NUM_SEATS; s++) left += (h.hands[s as Seat] ?? []).length;
          if (left === 0) {
            // rules.md §8.2: attackers taking the last trick win the
            // kitty's points times twice the cards in the winning play.
            if (res.winner % 2 !== defenders) {
              const won = trick.plays.find((pl) => pl.seat === res.winner)!;
              const bonus = points(h.kitty) * 2 * won.cards.length;
              settled += team !== defenders ? bonus : -bonus;
            }
            done = true;
            break;
          }
          sim = apply(sim, { kind: 'advance_trick' });
          if (closed >= 1 + cfg.extraTricks) break;
          continue;
        }
        const s = nextToPlay(sim);
        if (s === null) throw new Error(`zero search: nobody to act in ${sim.phase}`);
        sim = apply(sim, { kind: 'play_cards', seat: s, cards: policyMove(w, sim, s, trace) });
      }
      let future = 0;
      let value: number | null = null;
      let leafSeat: number | null = null;
      if (!done) {
        leafSeat = nextToPlay(sim)!;
        const h = sim.currentHand!;
        const leafKitty = leafSeat === h.landlord ? h.kitty : null;
        const feats = encodePosition(positionFor(sim, leafSeat as Seat, leafKitty));
        const handsNow = [0, 1, 2, 3].map((s) => plane(h.hands[s as Seat] ?? []));
        value = criticValue(w, feats.history, feats.historyLen, feats.context, privilegedView(handsNow, plane(h.kitty), leafSeat));
        future = (leafSeat % 2 === team ? 1 : -1) * value * valueToPoints;
      }
      scores.push(settled + future);
      trace?.leaves?.push({ world: wi, cand: i, settled, future, value, state: sim, seat: leafSeat, done });
    }
  });

  const means = top.map((_, k) => {
    let sum = 0;
    for (let wi = 0; wi < worlds.length; wi++) sum += scores[wi * top.length + k]!;
    return sum / worlds.length;
  });
  let pick = 0;
  for (let k = 1; k < means.length; k++) if (means[k]! > means[pick]!) pick = k;
  if (trace !== undefined) {
    trace.scores = scores;
    trace.means = means;
  }
  return { index: top[pick]!, searched: true };
}
