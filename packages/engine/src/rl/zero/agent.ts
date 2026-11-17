// The `rl-zero` bot: the PPO research track's agent trained from zero,
// behind the ordinary `(state, seat) => BotAction` contract.
//
// Every decision is a network's, as in training:
//   * declaring, while the cards come in and in the lockout after the
//     last one: the declaration network (declare.ts);
//   * the burial, as landlord: the burial network's eight greedy picks
//     (bury.ts);
//   * play: the play network's greedy choice among the engine's candidate
//     moves, the list it was evaluated on (`enumerateCandidates`, public
//     information only, capped at the network's 128 slots).
// Picking up the kitty is not a decision.
//
// Greedy rather than sampled, as the research evaluation measured it.

import type { BotAction } from '../../bot';
import { nextToPlay } from '../../reducer';
import type { GameState, Seat } from '../../state';
import { enumerateCandidates } from '../v2/candidates';

import { zeroBury } from './bury';
import { zeroDeclare } from './declare';
import { encodePosition, packCandidates } from './features';
import { actorLogits } from './net';
import { plane, positionFor } from './position';
import { DEFAULT_SEARCH, mulberry32, positionSeed, searchMove, type SearchConfig } from './search';
import { loadZeroWeights, type ZeroWeights } from './weights';

export type ZeroAgentOptions = {
  /** Directory holding weights-zero.{bin,json}. Defaults to this module's. */
  readonly weightsDir?: string;
  /** Search play decisions (search.ts) rather than play the policy's
   *  greedy move. Pre-play is the networks' either way. */
  readonly search?: SearchConfig | null;
  /** Mixed into each decision's world-sampling seed. */
  readonly seed?: number;
};

const WAIT: BotAction = { kind: 'wait' };

export function makeZeroAgent(opts: ZeroAgentOptions = {}): (state: GameState, seat: Seat) => BotAction {
  let weights: ZeroWeights | null = null;
  const load = (): ZeroWeights => (weights ??= loadZeroWeights(opts.weightsDir));

  // The ws server asks every bot at each broadcast and again when its
  // timer fires, with the same state object. States are immutable, so a
  // decision can be remembered by identity and the network runs once.
  let lastState: GameState | null = null;
  const lastBySeat = new Map<Seat, BotAction>();

  const decide = (state: GameState, seat: Seat): BotAction => {
    const hand = state.currentHand;
    if (hand === null) return WAIT;
    switch (state.phase) {
      case 'dealing':
      case 'declare_lockout':
        return zeroDeclare(load(), state, seat) ?? WAIT;
      case 'declare_open':
        return hand.landlord === seat ? { kind: 'pick_up_kitty' } : WAIT;
      case 'kitty_replace':
        if (hand.landlord !== seat || hand.trumpRank === null) return WAIT;
        return { kind: 'replace_kitty', cards: zeroBury(load(), state, seat) };
      case 'play': {
        if (hand.trumpRank === null || hand.landlord === null) return WAIT;
        if (nextToPlay(state) !== seat) return WAIT;
        const w = load();
        const myKitty = seat === hand.landlord ? hand.kitty : null;
        const candidates = enumerateCandidates(state, seat, {
          maxCandidates: w.manifest.model_config.max_candidates,
          myKitty: myKitty ?? [],
        });
        if (candidates.length === 0) throw new Error(`rl-zero: seat ${seat} has no candidate move`);
        if (candidates.length === 1) return { kind: 'play_cards', cards: candidates[0]!.cards };
        const moves = candidates.map((c) => plane(c.cards));
        const pos = positionFor(state, seat, myKitty);
        const logits = actorLogits(w, {
          ...encodePosition(pos),
          candidates: packCandidates(moves),
          numCandidates: moves.length,
        });
        if (opts.search) {
          const { index } = searchMove(w, state, seat, candidates.map((c) => c.cards), logits, {
            config: opts.search,
            rng: mulberry32(positionSeed(opts.seed ?? 0, pos)),
          });
          return { kind: 'play_cards', cards: candidates[index]!.cards };
        }
        let best = 0;
        for (let i = 1; i < logits.length; i++) if (logits[i]! > logits[best]!) best = i;
        return { kind: 'play_cards', cards: candidates[best]!.cards };
      }
      default:
        return WAIT;
    }
  };

  return (state, seat) => {
    if (state !== lastState) {
      lastState = state;
      lastBySeat.clear();
    }
    const hit = lastBySeat.get(seat);
    if (hit !== undefined) return hit;
    const action = decide(state, seat);
    lastBySeat.set(seat, action);
    return action;
  };
}

/** `rl-zero-search`: the same agent, its play decisions searched. */
export function makeZeroSearchAgent(
  opts: Omit<ZeroAgentOptions, 'search'> & { readonly search?: SearchConfig } = {},
): (state: GameState, seat: Seat) => BotAction {
  return makeZeroAgent({ ...opts, search: opts.search ?? DEFAULT_SEARCH });
}
