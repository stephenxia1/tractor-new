// Glue between the policy and the engine's `Agent` contract.
//
// `makePolicyAgent` returns an Agent function that:
//   - Defers to the baseline `pickBotAction('lowball')` for declare /
//     pick-up-kitty / kitty-replace / wait phases.
//   - During the play phase (when it's our turn), enumerates candidates
//     via `legalActions`, runs the MLP over each, and either samples
//     (training) or picks argmax (inference).
//
// In training mode it records each decision into a per-seat trajectory so
// the trainer can compute the REINFORCE update later.

import { pickBotAction, type BotAction } from '../../bot';
import { nextToPlay } from '../../reducer';
import { type GameState, type Seat } from '../../state';
import { type Card } from '../../types';
import { allLegalPlayActions } from './actions';
import {
  buildDecisionContext,
  extractFeatures,
  FEATURE_DIM,
} from './features';
import {
  forward,
  sampleFromProbs,
  softmax,
  type ForwardCache,
  type Policy,
} from './policy';

export type Decision = {
  // Per-candidate forward caches (needed to backprop without re-forwarding).
  readonly caches: ReadonlyArray<ForwardCache>;
  readonly probs: Float64Array;
  readonly sampledIdx: number;
  readonly temperature: number;
};

export type Trajectory = Decision[];

export type PolicyAgentOptions = {
  readonly policy: Policy;
  readonly mode: 'train' | 'eval';
  readonly temperature?: number;
  readonly rng?: () => number;
  // Optional: in training mode, append each decision into this array.
  readonly trajectory?: Trajectory;
};

export function makePolicyAgent(
  opts: PolicyAgentOptions,
): (state: GameState, seat: Seat) => BotAction {
  const policy = opts.policy;
  const mode = opts.mode;
  const temperature = opts.temperature ?? (mode === 'eval' ? 0.01 : 1.0);
  const rng = opts.rng ?? Math.random;
  const trajectory = opts.trajectory ?? null;

  return (state, seat) => {
    // Only override behavior during play when it's our turn. Everything
    // else (declare, kitty, wait) defers to the baseline lowball logic.
    if (
      state.phase !== 'play' ||
      nextToPlay(state) !== seat ||
      state.currentHand === null ||
      state.currentHand.trumpRank === null
    ) {
      return pickBotAction(state, seat, 'lowball');
    }

    const moves = allLegalPlayActions(state, seat) as Array<{
      readonly kind: 'play_cards';
      readonly seat: Seat;
      readonly cards: ReadonlyArray<Card>;
      readonly tags: ReadonlyArray<string>;
    }>;

    if (moves.length === 0) {
      // Fallback: shouldn't happen in `play` if we're the next-to-play
      // and have cards. Defer to lowball.
      return pickBotAction(state, seat, 'lowball');
    }

    // Single legal move: skip the network entirely. Don't record this in
    // the trajectory (no learnable signal).
    if (moves.length === 1) {
      return { kind: 'play_cards', cards: moves[0]!.cards };
    }

    const dctx = buildDecisionContext(state, seat);
    const caches: ForwardCache[] = new Array(moves.length);
    const scores: number[] = new Array(moves.length);
    for (let k = 0; k < moves.length; k++) {
      const fx = extractFeatures(moves[k]!.cards, moves[k]!.tags, dctx);
      // Copy out of the shared `out` buffer (extractFeatures allocates
      // fresh when `out` is omitted, so we're safe).
      // (We pass no `out`, so each fx is its own array.)
      const cache = forward(policy, fx);
      caches[k] = cache;
      scores[k] = cache.score;
    }

    const { probs } = softmax(scores, temperature);
    const idx =
      mode === 'eval' ? argmax(probs) : sampleFromProbs(probs, rng);

    if (trajectory !== null && mode === 'train') {
      trajectory.push({
        caches,
        probs,
        sampledIdx: idx,
        temperature,
      });
    }

    return { kind: 'play_cards', cards: moves[idx]!.cards };
  };
}

function argmax(arr: Float64Array): number {
  let best = 0;
  let bestVal = arr[0]!;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i]! > bestVal) {
      bestVal = arr[i]!;
      best = i;
    }
  }
  return best;
}

// Touch FEATURE_DIM to silence unused-import.
const _ = FEATURE_DIM;
void _;
