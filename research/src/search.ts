// Determinized one-trick lookahead, evaluated by the learned value head.
//
// The shape is AlphaGo's, adapted for hidden information: sample worlds,
// play a move in each, and let a value network judge the position that
// results. It is also, deliberately, `smart-v18`'s shape -- sample
// opponent hands, simulate, average -- with the hand-tuned rollout policy
// and cost heuristics replaced by a learned rollout and a learned
// evaluator. That is the comparison the whole thing is set up to make.
//
// Mechanically: for each candidate and each sampled world, play the
// candidate, let the remaining seats answer with a cheap heuristic, resolve
// the trick, and emit the resulting position as a *leaf*. Leaves ride the
// existing protocol as decision records carrying zero candidates, tagged
// with the candidate they belong to and the points that trick just
// settled, so no new message type and no nested round trips are needed.
//
// The opponent model inside the simulation is a heuristic rather than the
// network, which keeps one real decision to one network round trip.
// `smart-v18` does the same thing (it rolls out with `smart-v3`), so the
// comparison stays honest.

import {
  applyEvent,
  nextToPlay,
  pickBotAction,
  sumPoints,
  teamOf,
  type BotType,
  type Card,
  type GameState,
  type Seat,
  type TrumpContext,
} from '@tractor/engine';

import type { Candidate } from './candidates';
import { makeRng, sampleWorld, stateWithWorld } from './determinize';

export type SearchOptions = {
  // Worlds sampled per decision. More worlds cut sampling noise; cost is
  // linear in worlds x candidates.
  readonly worlds: number;
  // Candidates to search. Anything beyond this is left to the network's
  // own ranking -- and note the worker has no access to that ranking, so
  // this is a cap on *all* candidates rather than a top-k by prior. An
  // earlier version searched the first N in enumeration order and forced
  // a pick among them, which threw away whatever the network preferred
  // and cost ~30 points against smart-v2.
  readonly topCandidates: number;
  // Rollout policy for the other seats inside a simulation.
  readonly opponentModel: BotType;
  readonly seed: number;
};

export const DEFAULT_SEARCH: SearchOptions = {
  worlds: 4,
  topCandidates: 16,
  opponentModel: 'smart-v2',
  seed: 1,
};

export type Leaf = {
  // Which candidate this leaf is evidence about.
  readonly candidateIndex: number;
  // Points the resolved trick awarded, signed for the searching seat:
  // positive when its side gained, negative when the other side did.
  readonly points: number;
  // Position after the trick resolved, to be scored by the value head.
  readonly state: GameState;
};

export type SearchResult = {
  readonly leaves: ReadonlyArray<Leaf>;
  readonly searched: number;
  readonly relaxedWorlds: number;
};

// Plays one candidate in one world and runs on until the searching seat
// is next to act.
//
// Stopping at trick end would be the obvious thing and is wrong: the value
// head only ever saw states where the recorded seat was about to move,
// because those are the only states that become training records. A
// position where someone else is on lead is out of its training
// distribution, and asking anyway made search play *worse* than no search
// at all (-6.1 against smart-v2, versus +17.4 without). Running on to the
// seat's own next decision puts the leaf back in distribution.
function simulate(
  base: GameState,
  seat: Seat,
  cards: ReadonlyArray<Card>,
  opponentModel: BotType,
): { state: GameState; points: number } | null {
  const result = applyEvent(base, { kind: 'play_cards', seat, cards });
  if (!result.ok) return null;
  let state = result.state;

  // Finish the trick this move belongs to.
  let guard = 0;
  while (state.phase === 'play' && guard++ < 8) {
    const hand = state.currentHand;
    if (hand === null || hand.currentTrick === null) break;
    if (hand.currentTrick.plays.length >= state.config.numPlayers) break;
    const next = nextToPlay(state);
    if (next === null) break;
    const action = pickBotAction(state, next, opponentModel);
    if (action.kind !== 'play_cards') return null;
    const r = applyEvent(state, { kind: 'play_cards', seat: next, cards: action.cards });
    if (!r.ok) return null;
    state = r.state;
  }

  // Accumulate the attacker points settled between here and the seat's
  // next turn, signed the way the network's targets are signed.
  const landlordSeat = state.currentHand?.landlord ?? base.currentHand!.landlord!;
  const iAmAttacker = teamOf(state, seat) !== teamOf(state, landlordSeat);
  const sign = iAmAttacker ? 1 : -1;
  let points = 0;

  let outer = 0;
  while (outer++ < 12) {
    if (state.phase === 'trick_review') {
      const before = state.currentHand?.attackerPoints ?? 0;
      const advanced = applyEvent(state, { kind: 'advance_trick' });
      if (!advanced.ok) return null;
      state = advanced.state;
      points += ((state.currentHand?.attackerPoints ?? before) - before) * sign;
      continue;
    }
    if (state.phase !== 'play') break; // hand finished
    if (nextToPlay(state) === seat) break; // back to a state V understands
    const next = nextToPlay(state);
    if (next === null) break;
    const action = pickBotAction(state, next, opponentModel);
    if (action.kind !== 'play_cards') break;
    const r = applyEvent(state, { kind: 'play_cards', seat: next, cards: action.cards });
    if (!r.ok) break;
    state = r.state;
  }
  return { state, points };
}

export function buildLeaves(
  state: GameState,
  seat: Seat,
  candidates: ReadonlyArray<Candidate>,
  myKitty: ReadonlyArray<Card>,
  ctx: TrumpContext,
  opts: SearchOptions = DEFAULT_SEARCH,
): SearchResult {
  const leaves: Leaf[] = [];
  const searched = Math.min(candidates.length, opts.topCandidates);
  if (searched <= 1) {
    // Nothing to choose between; skip the work entirely.
    return { leaves, searched, relaxedWorlds: 0 };
  }
  const rand = makeRng(opts.seed);
  let relaxedWorlds = 0;

  for (let w = 0; w < opts.worlds; w++) {
    const world = sampleWorld(state, seat, myKitty, ctx, rand);
    if (world.relaxed) relaxedWorlds += 1;
    const hypothetical = stateWithWorld(state, world);
    for (let c = 0; c < searched; c++) {
      const sim = simulate(hypothetical, seat, candidates[c]!.cards, opts.opponentModel);
      if (sim === null) continue;
      leaves.push({ candidateIndex: c, points: sim.points, state: sim.state });
    }
  }
  return { leaves, searched, relaxedWorlds };
}
