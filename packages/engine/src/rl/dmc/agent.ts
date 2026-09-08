// Agent factory for the trained DMC policy.
//
// Wraps the role-dispatched Q-networks behind the same `(state, seat) =>
// BotAction` contract as the other smart bots. Non-play phases delegate
// to `pickSmartV34` (random declare hasn't shipped yet — v34's declare
// is well-tuned for now and matches what training assumed).
//
// At each play decision: enumerate candidates with the sub-action
// protocol, encode the observation once, run the role's Q-net over each
// candidate, argmax. For multi-pattern leads, this means looping
// internally over sub-actions until the chosen candidate is `lead-stop`
// or only one candidate exists.

import { pickBotAction, type BotAction } from '../../bot';
import { nextToPlay } from '../../reducer';
import type { GameState, Seat } from '../../state';

import {
  applyLeadSubAction,
  enumerateCandidates,
  EMPTY_LEAD,
  type LeadInProgress,
} from './actions';
import { buildObservation } from './observation';
import { roleFor } from './roles';
import {
  scoreCandidates,
  weightsFor,
  type DmcWeights,
} from './inference';

export type DmcAgentOptions = {
  readonly weights: DmcWeights;
};

export function makeDmcAgent(opts: DmcAgentOptions): (state: GameState, seat: Seat) => BotAction {
  const weights = opts.weights;
  return (state, seat) => {
    // Non-play phases (declare, kitty, dealing) → v34 handles them.
    if (
      state.phase !== 'play' ||
      nextToPlay(state) !== seat ||
      state.currentHand === null ||
      state.currentHand.trumpRank === null ||
      state.currentHand.trumpSuit === null // no-trump unsupported
    ) {
      return pickBotAction(state, seat, 'smart-v34');
    }

    let lead: LeadInProgress = EMPTY_LEAD;
    // Sub-action loop: at most a few iterations (capped at 3 lead patterns
    // + a possible stop), so the bounded `while` is safe.
    while (true) {
      const candidates = enumerateCandidates(state, seat, lead);
      if (candidates.length === 0) {
        // Shouldn't happen during play with a valid hand; defer to v34.
        return pickBotAction(state, seat, 'smart-v34');
      }
      // Single-candidate shortcut.
      if (candidates.length === 1) {
        const c = candidates[0]!;
        if (c.kind === 'lead-stop') {
          return { kind: 'play_cards', cards: lead.chosen };
        }
        if (c.kind === 'follow') {
          return { kind: 'play_cards', cards: c.cards };
        }
        // Lead-pattern, single option: commit it; loop will then offer stop.
        const next = applyLeadSubAction(lead, c);
        if (next === null) return { kind: 'play_cards', cards: lead.chosen };
        lead = next;
        continue;
      }

      const role = roleFor(state, seat);
      const obs = buildObservation(state, seat, lead.chosen);
      const w = weightsFor(weights, role.role);
      const scores = scoreCandidates(
        w,
        obs,
        candidates.map((c) => c.encoding),
      );
      let bestIdx = 0;
      let bestScore = scores[0]!;
      for (let i = 1; i < scores.length; i++) {
        if (scores[i]! > bestScore) {
          bestScore = scores[i]!;
          bestIdx = i;
        }
      }
      const chosen = candidates[bestIdx]!;

      if (chosen.kind === 'follow') {
        return { kind: 'play_cards', cards: chosen.cards };
      }
      if (chosen.kind === 'lead-stop') {
        return { kind: 'play_cards', cards: lead.chosen };
      }
      // Sub-action commit. Loop again with the updated lead state.
      const next = applyLeadSubAction(lead, chosen);
      if (next === null) {
        return { kind: 'play_cards', cards: lead.chosen };
      }
      lead = next;
    }
  };
}
