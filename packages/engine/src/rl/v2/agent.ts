// The trained v2 agent, behind the ordinary `(state, seat) => BotAction`
// contract.
//
// Play decisions go to the network. Everything else — declaring, burying
// the kitty, the incremental deal — delegates to `smart-v34`, because
// that is exactly what training assumed: every self-play worker ran with
// `preplay: 'smart-v34'`, so the network has only ever seen positions that
// a v34 declaration and a v34 kitty produced. Handing those phases to a
// different heuristic would put it out of distribution before the first
// card is led.
//
// Candidate enumeration matches training too, including
// `publicInfoThrows`: the engine's own throw enumeration consults hidden
// cards through `isThrowSafe`, which is fine for a heuristic that is
// allowed to and useless for an agent that is not.

import type { BotAction } from '../../bot';
import { nextToPlay } from '../../reducer';
import type { GameState, Seat } from '../../state';

import { enumerateCandidates } from './candidates';
import { writeDecision } from './encode';
import { actionFeatures, historyFeatures, stateFeatures } from './features';
import { loadWeights, scoreCandidates } from './net';
import type { LoadedWeights } from './weights';
import { ByteWriter, CANDIDATE_BYTES, H_ROLE, H_TRUMP_RANK, H_TRUMP_SUIT, STATE_BYTES } from './record';

// Matches the training workers' `--max-cands`.
const MAX_CANDIDATES = 48;

export type V2AgentOptions = {
  /** Directory holding weights-v2.{bin,json}. Defaults to this module's. */
  readonly weightsDir?: string;
  /** Phases the network does not cover. */
  readonly fallback?: (state: GameState, seat: Seat) => BotAction;
};

export function makeV2Agent(opts: V2AgentOptions = {}): (state: GameState, seat: Seat) => BotAction {
  let weights: LoadedWeights | null = null;
  const fallback = opts.fallback;

  return (state, seat) => {
    const hand = state.currentHand;
    const usable =
      state.phase === 'play' &&
      hand !== null &&
      hand.trumpRank !== null &&
      hand.landlord !== null &&
      nextToPlay(state) === seat;
    if (!usable) {
      if (fallback === undefined) throw new Error('v2 agent: no fallback for non-play phases');
      return fallback(state, seat);
    }

    const myKitty = seat === hand!.landlord ? hand!.kitty : [];
    const candidates = enumerateCandidates(state, seat, {
      maxCandidates: MAX_CANDIDATES,
      myKitty,
    });
    if (candidates.length === 0) {
      if (fallback === undefined) throw new Error('v2 agent: no candidates and no fallback');
      return fallback(state, seat);
    }
    if (candidates.length === 1) {
      return { kind: 'play_cards', cards: candidates[0]!.cards };
    }

    // Build the same record the trainer would have written, then read the
    // features straight back out of it. Going through the wire format
    // rather than reading `state` directly is deliberate: it is the one
    // encoding that has been checked against the Python side card for
    // card (`research/py/test_ts_parity.py`).
    const writer = new ByteWriter(1 << 16);
    writeDecision(
      writer,
      state,
      { gameId: 0, handId: 0, seat, myKitty, privileged: false },
      candidates,
    );
    const framed = writer.taken();
    const recordBytes = framed[0]! + framed[1]! * 256;
    const record = framed.subarray(2, 2 + recordBytes);
    const stateBlock = record.subarray(0, STATE_BYTES);

    if (weights === null) weights = loadWeights(opts.weightsDir);
    const trumpRank = stateBlock[H_TRUMP_RANK]!;
    const trumpSuit = stateBlock[H_TRUMP_SUIT]!;
    const actions = candidates.map((_, i) => {
      const at = STATE_BYTES + i * CANDIDATE_BYTES;
      return actionFeatures(record.subarray(at, at + CANDIDATE_BYTES), trumpRank, trumpSuit);
    });

    const history = historyFeatures(stateBlock);
    const scores = scoreCandidates(
      weights,
      stateFeatures(stateBlock),
      history.tokens,
      history.mask,
      actions,
      stateBlock[H_ROLE]!,
    );

    let best = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i]! > scores[best]!) best = i;
    return { kind: 'play_cards', cards: candidates[best]!.cards };
  };
}
