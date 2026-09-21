// The v2 agent with determinized value-guided search, optionally weighted
// by the belief head.
//
// Shape: sample worlds consistent with public play, play each candidate in
// each world, roll forward to this seat's *own* next decision, and score
// the resulting position with the value head. A candidate is worth
//
//     mean over leaves of ( settled points / 100 + gamma * V(leaf) )
//
// which is the same quantity V was trained to predict, so the immediate
// reward and the continuation are in the same units.
//
// Two hard-won constraints are baked in, both from search being *worse*
// than no search when they were violated:
//
//   * leaves are taken at this seat's next turn, never at trick end. V is
//     trained only on "my turn" states; asking it about someone else's
//     turn is out of distribution and cost 23 points against `smart-v2`.
//   * the network's own ranking is only overridden when search covered
//     every candidate. Searching an arbitrary subset and forcing a pick
//     inside it discarded the policy's preference and cost 31 points.

import { pickBotAction, type BotAction, type BotType } from '../../bot';
import { nextToPlay } from '../../reducer';
import type { GameState, Seat } from '../../state';
import { NATURAL_RANKS, SUITS, type TrumpContext } from '../../types';

import {
  DEFAULT_BELIEF_WEIGHTS,
  prepareBeliefWeights,
  type BeliefWeightOptions,
} from './beliefSampler';
import { enumerateCandidates, type Candidate } from './candidates';
import { cardSlot } from './cards';
import { makeRng, sampleWorld, stateWithWorld, type PlacementWeight } from './determinize';
import { writeDecision } from './encode';
import { actionFeatures, historyFeatures, stateFeatures } from './features';
import { beliefs as beliefProbs, encodeState, loadWeights, scoreCandidates, stateValue } from './net';
import {
  ByteWriter,
  CANDIDATE_BYTES,
  H_ROLE,
  H_TRUMP_RANK,
  H_TRUMP_SUIT,
  STATE_BYTES,
} from './record';
import { simulateCandidate } from './search';
import type { LoadedWeights } from './weights';

const MAX_CANDIDATES = 48;
const GAMMA = 0.995;
const POINT_SCALE = 100;

export type SearchAgentOptions = {
  /** Worlds sampled per decision. Cost is linear in worlds x candidates. */
  readonly worlds: number;
  /** Cap on candidates searched. Beyond it, the network's ranking stands. */
  readonly maxSearched: number;
  /** Rollout policy for the other seats inside a simulation. */
  readonly opponentModel: BotType;
  /** null disables belief weighting, recovering uniform determinization. */
  readonly belief: BeliefWeightOptions | null;
  readonly weightsDir?: string;
  readonly fallback?: (state: GameState, seat: Seat) => BotAction;
};

export const DEFAULT_SEARCH_AGENT: SearchAgentOptions = {
  worlds: 2,
  maxSearched: 8,
  opponentModel: 'smart-v2',
  belief: DEFAULT_BELIEF_WEIGHTS,
};

export type SearchStats = {
  decisions: number;
  searched: number;
  leaves: number;
  overrides: number;
  relaxedWorlds: number;
};

export function makeV2SearchAgent(
  opts: Partial<SearchAgentOptions> = {},
): ((state: GameState, seat: Seat) => BotAction) & { stats: SearchStats } {
  const cfg: SearchAgentOptions = { ...DEFAULT_SEARCH_AGENT, ...opts };
  let weights: LoadedWeights | null = null;
  const stats: SearchStats = {
    decisions: 0, searched: 0, leaves: 0, overrides: 0, relaxedWorlds: 0,
  };

  const agent = (state: GameState, seat: Seat): BotAction => {
    const hand = state.currentHand;
    const usable =
      state.phase === 'play' &&
      hand !== null &&
      hand.trumpRank !== null &&
      hand.landlord !== null &&
      nextToPlay(state) === seat;
    if (!usable) {
      if (cfg.fallback === undefined) throw new Error('v2 search: no fallback for non-play phases');
      return cfg.fallback(state, seat);
    }

    const myKitty = seat === hand!.landlord ? hand!.kitty : [];
    const candidates = enumerateCandidates(state, seat, { maxCandidates: MAX_CANDIDATES, myKitty });
    if (candidates.length === 0) {
      if (cfg.fallback === undefined) throw new Error('v2 search: no candidates');
      return cfg.fallback(state, seat);
    }
    if (candidates.length === 1) return { kind: 'play_cards', cards: candidates[0]!.cards };

    if (weights === null) weights = loadWeights(cfg.weightsDir);
    stats.decisions += 1;

    const root = encodeDecision(state, seat, myKitty, candidates);
    const z = encodeState(weights, root.state, root.tokens, root.mask);
    const q = scoreCandidates(
      weights, root.state, root.tokens, root.mask, root.actions, root.role,
    );

    // Search everything, or nothing: a partial search cannot be allowed to
    // override a ranking it did not evaluate.
    if (candidates.length > cfg.maxSearched || cfg.worlds <= 0) {
      return { kind: 'play_cards', cards: candidates[argmax(q)]!.cards };
    }

    const ctx: TrumpContext = { trumpRank: hand!.trumpRank!, trumpSuit: hand!.trumpSuit };
    let weightOf: PlacementWeight | undefined;
    if (cfg.belief !== null) {
      const probs = beliefProbs(weights, z);
      weightOf =
        prepareBeliefWeights(
          state, seat, myKitty, ctx, probs, root.trumpRank, root.trumpSuit, cfg.belief,
        ) ?? undefined;
    }

    const totals = new Float64Array(candidates.length);
    const counts = new Int32Array(candidates.length);
    // Seed from the whole position, not just its shape. The old key --
    // seat, candidate count and trick index -- collides across deals: any
    // two hands that reached trick 3 as seat 1 with 5 candidates drew the
    // identical world sequence, so their determinization error was the
    // same error twice over instead of two draws that average out.
    const rand = makeRng(positionSeed(state, seat));

    for (let w = 0; w < cfg.worlds; w++) {
      const world = sampleWorld(state, seat, myKitty, ctx, rand, weightOf);
      if (world.relaxed) stats.relaxedWorlds += 1;
      const hypothetical = stateWithWorld(state, world);
      for (let c = 0; c < candidates.length; c++) {
        const sim = simulateCandidate(hypothetical, seat, candidates[c]!.cards, cfg.opponentModel);
        if (sim === null) continue;
        // A simulation that ran the hand out has no continuation to value,
        // and V does not know that: it was trained only on "my turn"
        // positions and answers a finished hand with a small positive
        // number, which lands on top of the points the last trick settled.
        // Score those on the settled points alone. Dropping them instead
        // would cost more than the bug does -- the final decision of a hand
        // is terminal for *every* candidate, so `complete` would fail and
        // search would never reach the trick the kitty bonus rides on.
        let continuation = 0;
        if (sim.state.phase === 'play') {
          const leaf = encodeLeaf(sim.state, seat, myKitty);
          if (leaf === null) continue;
          const lz = encodeState(weights, leaf.state, leaf.tokens, leaf.mask);
          continuation = stateValue(weights, lz);
        }
        totals[c]! += sim.points / POINT_SCALE + GAMMA * continuation;
        counts[c]! += 1;
        stats.leaves += 1;
      }
    }

    // Every candidate must have been reached in at least one world, or the
    // comparison is between different sample sizes.
    let complete = true;
    for (let c = 0; c < candidates.length; c++) if (counts[c] === 0) complete = false;
    if (!complete) return { kind: 'play_cards', cards: candidates[argmax(q)]!.cards };

    stats.searched += 1;
    const scores = new Float64Array(candidates.length);
    for (let c = 0; c < candidates.length; c++) scores[c] = totals[c]! / counts[c]!;
    const pick = argmax(scores);
    if (pick !== argmax(q)) stats.overrides += 1;
    return { kind: 'play_cards', cards: candidates[pick]!.cards };
  };

  agent.stats = stats;
  return agent as ((state: GameState, seat: Seat) => BotAction) & { stats: SearchStats };
}

function argmax(xs: Float32Array | Float64Array): number {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i]! > xs[best]!) best = i;
  return best;
}

// FNV-1a over what identifies this deal and the point reached inside it,
// in the style of the deal digest in `research/src/env.ts`. This agent is
// handed a state and nothing else -- no game or hand id to key on -- so
// the acting seat's own cards stand in for the deal, being the cheapest
// thing on hand that two deals reliably differ in, and the trick index
// and ply separate the decisions within one hand. Still a pure function
// of the position, so replaying a game samples the same worlds.
function positionSeed(state: GameState, seat: Seat): number {
  const hand = state.currentHand!;
  let h = 0x811c9dc5;
  const push = (v: number): void => {
    h = Math.imul(h ^ (v & 0xff), 0x01000193) >>> 0;
  };
  push(seat);
  push(hand.trumpRank === null ? 255 : NATURAL_RANKS.indexOf(hand.trumpRank));
  push(hand.trumpSuit === null ? 255 : SUITS.indexOf(hand.trumpSuit));
  push(hand.landlord ?? 255);
  push(hand.tricks.length);
  push(hand.currentTrick?.plays.length ?? 0);
  const slots = (hand.hands[seat] ?? []).map(cardSlot).sort((a, b) => a - b);
  for (const slot of slots) push(slot);
  return h >>> 0;
}

type Encoded = {
  state: Float32Array;
  tokens: Float32Array;
  mask: Float32Array;
  actions: Float32Array[];
  role: number;
  trumpRank: number;
  trumpSuit: number;
};

function encodeDecision(
  state: GameState,
  seat: Seat,
  myKitty: ReadonlyArray<import('../../types').Card>,
  candidates: ReadonlyArray<Candidate>,
): Encoded {
  const writer = new ByteWriter(1 << 16);
  writeDecision(writer, state, { gameId: 0, handId: 0, seat, myKitty, privileged: false }, candidates);
  const framed = writer.taken();
  const recordBytes = framed[0]! + framed[1]! * 256;
  const record = framed.subarray(2, 2 + recordBytes);
  const block = record.subarray(0, STATE_BYTES);
  const trumpRank = block[H_TRUMP_RANK]!;
  const trumpSuit = block[H_TRUMP_SUIT]!;
  const actions = candidates.map((_, i) => {
    const at = STATE_BYTES + i * CANDIDATE_BYTES;
    return actionFeatures(record.subarray(at, at + CANDIDATE_BYTES), trumpRank, trumpSuit);
  });
  const history = historyFeatures(block);
  return {
    state: stateFeatures(block),
    tokens: history.tokens,
    mask: history.mask,
    actions,
    role: block[H_ROLE]!,
    trumpRank,
    trumpSuit,
  };
}

// A leaf carries no candidates; only the state embedding is wanted. The
// landlord's buried kitty still has to be passed even though there is
// nothing here to score: `stateFeatures` derives the unseen pool as
// capacity - myHand - played - kitty, so an empty kitty plane on a
// landlord leaf reports the eight buried cards as still outstanding. That
// is a state the training data never contains, and because the distortion
// depends on what each leaf holds it tilts the comparison between
// candidates rather than cancelling out of it. Null when the position
// cannot be encoded at all -- no trump or no landlord, which
// `writeDecision` refuses. Positions where the hand has already ended
// never get here; the caller scores those on points alone.
function encodeLeaf(
  state: GameState,
  seat: Seat,
  myKitty: ReadonlyArray<import('../../types').Card>,
): { state: Float32Array; tokens: Float32Array; mask: Float32Array } | null {
  const hand = state.currentHand;
  if (hand === null || hand.trumpRank === null || hand.landlord === null) return null;
  try {
    const writer = new ByteWriter(1 << 16);
    writeDecision(writer, state, { gameId: 0, handId: 0, seat, myKitty, privileged: false }, []);
    const framed = writer.taken();
    const recordBytes = framed[0]! + framed[1]! * 256;
    const block = framed.subarray(2, 2 + recordBytes).subarray(0, STATE_BYTES);
    const history = historyFeatures(block);
    return { state: stateFeatures(block), tokens: history.tokens, mask: history.mask };
  } catch {
    return null;
  }
}

export { pickBotAction };
