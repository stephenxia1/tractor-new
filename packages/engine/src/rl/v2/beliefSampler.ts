// Belief-weighted determinization.
//
// `sampleWorld` already respects the two hard public facts — each seat's
// hand size, and the suits it has shown itself void in — but inside that
// feasible set it deals uniformly. That is the classic *non-locality*
// weakness of perfect-information Monte Carlo: the real distribution over
// hidden hands depends on how the opponents have played, and a uniform
// deal throws that away. A seat that declined to trump a twenty-point
// trick is telling you something no void mask records.
//
// The network's belief head already estimates exactly that: P(relative
// opponent r holds a card at chain position p). Measured on self-play it
// is well calibrated — ECE 0.015 on the uncertain positions that matter,
// with temperature scaling unable to improve on T=1 — which is the
// property this needs. An *accurate* but overconfident head would be
// worse than uniform here, because the search would spend all its worlds
// on positions that never come up.
//
// The problem is that the head gives per-cell marginals while a world has
// to satisfy three constraints at once: every unseen card is placed
// exactly once, every opponent gets exactly its hand size, and voids are
// forbidden. Independent draws from the marginals violate all three.
//
// So the marginals are projected onto the feasible set first, by Sinkhorn
// scaling: alternately normalise rows to the card counts and columns to
// the hand sizes until both hold. That yields the matrix closest to the
// belief (in KL) which is actually realisable, and sampling proceeds from
// it card by card, renormalising as capacity is used up. Voids stay hard
// zeros throughout, so no sampled world can contradict public play.

import { effectiveSuit } from '../../ordering';
import type { GameState, Seat } from '../../state';
import { cardKey, type Card, type TrumpContext } from '../../types';

import { cardSlot } from './cards';
import { knownVoids, unseenCards, type PlacementWeight } from './determinize';
import { encMap, NUM_ENC_SLOTS } from './encoding';

export type BeliefWeightOptions = {
  // 0 recovers uniform sampling, 1 uses the belief as-is. Anything in
  // between is `p^temperature`, renormalised: insurance against the head
  // being sharper than it deserves in some region.
  readonly temperature: number;
  // Sinkhorn iterations. The matrix is 4 columns wide, so this converges
  // in a handful; the cap is a guard, not a tuning knob.
  readonly iterations?: number;
};

export const DEFAULT_BELIEF_WEIGHTS: BeliefWeightOptions = {
  temperature: 1,
  iterations: 24,
};

// Never let a cell reach exactly zero except where a void forbids it:
// a belief of 0 would make a legal world unreachable, and the head is a
// model, not a rule.
const FLOOR = 1e-4;

export type PlacementWeights = {
  /** weight[cardIndex * numTargets + target], already feasible. */
  readonly weight: Float64Array;
  readonly numTargets: number;
};

/**
 * Projects belief marginals onto the set of realisable placements.
 *
 * `pool` is the unseen cards, `targets` the seats that can receive them
 * (plus, optionally, a kitty column), `need` how many each must end with,
 * and `allowed` a hard feasibility mask from the void inference.
 * `beliefs` is the 3x66 head output indexed by *relative* opponent.
 */
export function buildPlacementWeights(
  pool: ReadonlyArray<Card>,
  need: ReadonlyArray<number>,
  allowed: (cardIndex: number, target: number) => boolean,
  beliefOf: (cardIndex: number, target: number) => number | null,
  opts: BeliefWeightOptions = DEFAULT_BELIEF_WEIGHTS,
): PlacementWeights {
  const n = pool.length;
  const t = need.length;
  const weight = new Float64Array(n * t);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < t; j++) {
      if (!allowed(i, j) || need[j]! <= 0) continue;
      const p = beliefOf(i, j);
      // A target with no belief attached (the kitty) stays neutral rather
      // than being suppressed: we know nothing about it, which is not the
      // same as believing it empty.
      const base = p === null ? 0.5 : Math.max(p, FLOOR);
      weight[i * t + j] = Math.pow(base, opts.temperature);
    }
  }

  // Sinkhorn: rows sum to 1 (each card is placed exactly once), columns
  // sum to `need` (each seat ends with its hand size).
  const iterations = opts.iterations ?? DEFAULT_BELIEF_WEIGHTS.iterations!;
  for (let iter = 0; iter < iterations; iter++) {
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < t; j++) sum += weight[i * t + j]!;
      if (sum <= 0) continue; // infeasible row; the caller's fallback handles it
      const inv = 1 / sum;
      for (let j = 0; j < t; j++) weight[i * t + j]! *= inv;
    }
    for (let j = 0; j < t; j++) {
      if (need[j]! <= 0) continue;
      let sum = 0;
      for (let i = 0; i < n; i++) sum += weight[i * t + j]!;
      if (sum <= 0) continue;
      const scale = need[j]! / sum;
      for (let i = 0; i < n; i++) weight[i * t + j]! *= scale;
    }
  }
  return { weight, numTargets: t };
}

/**
 * Belief lookup for one (card, relative opponent) pair.
 *
 * The head is indexed by chain position, so several distinct cards can
 * share a cell — which is correct, since they are interchangeable in
 * exactly the way the encoding says.
 */
export function makeBeliefLookup(
  beliefs: Float32Array,
  ctx: TrumpContext,
  trumpRank: number,
  trumpSuit: number,
): (card: Card, relativeSeat: number) => number {
  const map = encMap(trumpRank, trumpSuit);
  return (card, relativeSeat) => {
    // Relative seat 0 is the acting seat, whose hand is known; the head
    // covers the other three, so index r-1 into it.
    const r = relativeSeat - 1;
    if (r < 0 || r >= 3) return 0.5;
    const enc = map[cardSlot(card)]!;
    return beliefs[r * NUM_ENC_SLOTS + enc]!;
  };
}

/**
 * Everything above, assembled once per decision.
 *
 * Sinkhorn runs on the whole unseen pool a single time; every world
 * sampled afterwards reuses the projected matrix, so the cost is paid once
 * rather than per world. Returns null when there is nothing to weight (no
 * unseen cards, or no opponent with cards left), which lets the caller
 * fall back to the uniform path without a special case.
 */
export function prepareBeliefWeights(
  state: GameState,
  seat: Seat,
  myKitty: ReadonlyArray<Card>,
  ctx: TrumpContext,
  beliefs: Float32Array,
  trumpRank: number,
  trumpSuit: number,
  opts: BeliefWeightOptions = DEFAULT_BELIEF_WEIGHTS,
): PlacementWeight | null {
  const hand = state.currentHand;
  if (hand === null) return null;
  const numPlayers = state.config.numPlayers;
  const pool = unseenCards(state, seat, myKitty);
  if (pool.length === 0) return null;

  const voids = knownVoids(state, ctx);
  const need: number[] = [];
  for (let s = 0; s < numPlayers; s++) {
    need.push(s === seat ? 0 : (hand.hands[s] ?? []).length);
  }
  if (need.reduce((a, b) => a + b, 0) === 0) return null;

  const lookup = makeBeliefLookup(beliefs, ctx, trumpRank, trumpSuit);
  const allowed = (i: number, j: number): boolean => {
    if (j === seat || need[j]! <= 0) return false;
    return !voids[j]!.has(effectiveSuit(pool[i]!, ctx));
  };
  const beliefOf = (i: number, j: number): number | null => {
    // Relative seat, the frame the belief head was trained in.
    const rel = (j - seat + numPlayers) % numPlayers;
    return lookup(pool[i]!, rel);
  };

  const { weight, numTargets } = buildPlacementWeights(pool, need, allowed, beliefOf, opts);

  // Index by card key: the sampler shuffles the pool, so position is not
  // a stable handle.
  const byKey = new Map<string, Float64Array>();
  for (let i = 0; i < pool.length; i++) {
    byKey.set(cardKey(pool[i]!), weight.slice(i * numTargets, (i + 1) * numTargets));
  }
  return (card, target) => {
    const row = byKey.get(cardKey(card));
    if (row === undefined) return 1; // not in the pool we projected; stay neutral
    return row[target] ?? 0;
  };
}
