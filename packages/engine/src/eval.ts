// Headless evaluation environment for agents.
//
// `runHand` plays a single deal end-to-end with four bots; `runGame`
// chains hands until a team wins; `runMatch` runs many games and
// returns aggregate stats. No timers, no network, no UI — purely
// synchronous reducer transitions, so 1000s of games take milliseconds.

import { pickBotAction, type BotAction, type BotType } from './bot';
import { applyEvent, nextToPlay } from './reducer';
import { mulberry32 } from './prng';
import { computeHandResult } from './scoring';
import {
  createGame,
  defaultV1Config,
  type GameConfig,
  type GameState,
  type Seat,
  type TeamId,
} from './state';
import type { GameEvent } from './events';
import type { NaturalRank } from './types';

// One agent per seat. Either a built-in type-string or a custom
// `(state, seat) => BotAction` function. Custom functions get full state
// access — they're appropriate for self-play training rigs.
export type Agent = BotType | ((state: GameState, seat: Seat) => BotAction);

export type SeatAgents = ReadonlyArray<Agent>;

export type HandSummary = {
  readonly handNumber: number;
  readonly landlord: Seat;
  readonly trumpRank: NaturalRank;
  readonly attackerPoints: number;
  readonly winner: 'attackers' | 'defenders';
  readonly levelDelta: number;
  readonly tricksPlayed: number;
  readonly teamLevels: Readonly<Record<TeamId, NaturalRank>>;
};

export type GameSummary = {
  readonly winner: TeamId | null;     // null = match capped before game_over
  readonly handsPlayed: number;
  readonly finalLevels: Readonly<Record<TeamId, NaturalRank>>;
  readonly hands: ReadonlyArray<HandSummary>;
  readonly stoppedReason: 'game_over' | 'max_hands_reached';
};

export type MatchSummary = {
  readonly numGames: number;
  readonly winsByTeam: Readonly<Record<TeamId, number>>;
  // null counts games that hit `maxHandsPerGame` without resolving.
  readonly stalemates: number;
  readonly avgHandsPerGame: number;
  readonly avgAttackerPoints: number;
  readonly games: ReadonlyArray<GameSummary>;
};

const PLACEHOLDER_HAND_CAP = 250;

// -- Helpers ---------------------------------------------------------------

function callAgent(agents: SeatAgents, state: GameState, seat: Seat): BotAction {
  const a = agents[seat];
  if (a === undefined) return { kind: 'wait' };
  if (typeof a === 'string') return pickBotAction(state, seat, a);
  return a(state, seat);
}

function actionToEvent(action: BotAction, seat: Seat): GameEvent | null {
  switch (action.kind) {
    case 'declare':
      return { kind: 'declare', seat, cards: action.cards };
    case 'pick_up_kitty':
      return { kind: 'pick_up_kitty', seat };
    case 'replace_kitty':
      return { kind: 'replace_kitty', seat, cards: action.cards };
    case 'play_cards':
      return { kind: 'play_cards', seat, cards: action.cards };
    case 'wait':
      return null;
  }
}

// -- Public: a single hand -------------------------------------------------

// Runs the current hand to completion (phase ∈ {between_hands, game_over}).
// Returns the final state and a per-hand summary.
//
// `state` must be in `between_hands` (start of a fresh hand) — call
// `runHand` directly after `createGame` or after the previous hand
// finishes. Pass the same `seed` to reproduce a deal.
export function runHand(opts: {
  agents: SeatAgents;
  seed: number;
  state?: GameState;
  config?: GameConfig;
}): { state: GameState; summary: HandSummary } {
  let state =
    opts.state ?? createGame(opts.config ?? defaultV1Config());
  if (state.phase !== 'between_hands') {
    throw new Error(
      `runHand: expected phase between_hands, got ${state.phase}`,
    );
  }
  const rng = mulberry32(opts.seed);
  const nextSeed = () => Math.floor(rng() * 0x7fffffff);

  // Start the hand.
  const start = applyEvent(state, { kind: 'start_hand', seed: opts.seed });
  if (!start.ok) throw new Error(start.error);
  state = start.state;

  const initialLandlord = state.currentHand?.landlord ?? null;
  const numPlayers = state.config.numPlayers;
  let safety = 0;

  while (
    state.phase !== 'between_hands' &&
    state.phase !== 'game_over' &&
    safety++ < 5000
  ) {
    // Declarations: poll each seat once. If none declare and we're still
    // in lockout, advance via release_lockout. In declare_open the
    // landlord's `pick_up_kitty` will fire from the bot itself.
    if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
      let progressed = false;
      for (let s = 0; s < numPlayers; s++) {
        const action = callAgent(opts.agents, state, s);
        const ev = actionToEvent(action, s);
        if (ev === null) continue;
        const r = applyEvent(state, ev);
        if (!r.ok) continue; // skip illegal claim
        state = r.state;
        progressed = true;
        break;
      }
      if (!progressed) {
        if (state.phase === 'declare_lockout') {
          const r = applyEvent(state, {
            kind: 'release_lockout',
            seed: nextSeed(),
          });
          if (!r.ok) throw new Error(r.error);
          state = r.state;
        } else {
          // declare_open + no one wants to act ⇒ landlord should always
          // take the kitty per the baseline bot; bail out defensively.
          throw new Error(
            'runHand: stuck in declare_open with no bot action',
          );
        }
      }
      continue;
    }

    if (state.phase === 'kitty_replace') {
      const seat = state.currentHand!.landlord!;
      const action = callAgent(opts.agents, state, seat);
      const ev = actionToEvent(action, seat);
      if (ev === null) throw new Error('runHand: landlord refused to bury');
      const r = applyEvent(state, ev);
      if (!r.ok) throw new Error(r.error);
      state = r.state;
      continue;
    }

    if (state.phase === 'play') {
      const seat = nextToPlay(state);
      if (seat === null) {
        throw new Error('runHand: play phase but nextToPlay is null');
      }
      const action = callAgent(opts.agents, state, seat);
      const ev = actionToEvent(action, seat);
      if (ev === null) throw new Error(`runHand: seat ${seat} refused to play`);
      const r = applyEvent(state, ev);
      if (!r.ok) throw new Error(`seat ${seat}: ${r.error}`);
      state = r.state;
      continue;
    }

    if (state.phase === 'trick_review') {
      const r = applyEvent(state, { kind: 'advance_trick' });
      if (!r.ok) throw new Error(r.error);
      state = r.state;
      continue;
    }
  }

  if (safety >= 5000) {
    throw new Error('runHand: exceeded 5000 iterations (stuck loop)');
  }

  const ch = state.currentHand;
  if (ch === null) throw new Error('runHand: currentHand is null at end');
  const attackerPoints = ch.attackerPoints;
  // Re-derive winner / levelDelta from the engine's scoring helpers so
  // the summary matches what the reducer applied internally.
  const r = computeHandResult(attackerPoints);

  const summary: HandSummary = {
    handNumber: ch.handNumber,
    landlord: initialLandlord ?? ch.landlord!,
    trumpRank: ch.trumpRank!,
    attackerPoints,
    winner: r.winner,
    levelDelta: r.levelDelta,
    tricksPlayed: ch.tricks.length,
    teamLevels: { ...state.teamLevels },
  };

  return { state, summary };
}

// -- Public: a full game ---------------------------------------------------

// Plays hands until one team wins (`phase = game_over`) or
// `maxHands` is reached. Uses a sub-seed derived from `seed` so different
// hands within a game have different deals.
export function runGame(opts: {
  agents: SeatAgents;
  seed: number;
  maxHands?: number;
  config?: GameConfig;
}): { state: GameState; summary: GameSummary } {
  const maxHands = opts.maxHands ?? PLACEHOLDER_HAND_CAP;
  let state = createGame(opts.config ?? defaultV1Config());
  const rng = mulberry32(opts.seed);
  const summaries: HandSummary[] = [];

  while (state.phase !== 'game_over' && summaries.length < maxHands) {
    const handSeed = Math.floor(rng() * 0x7fffffff);
    const { state: next, summary } = runHand({
      agents: opts.agents,
      seed: handSeed,
      state,
    });
    state = next;
    summaries.push(summary);
  }

  return {
    state,
    summary: {
      winner: state.winner,
      handsPlayed: summaries.length,
      finalLevels: { ...state.teamLevels },
      hands: summaries,
      stoppedReason: state.phase === 'game_over' ? 'game_over' : 'max_hands_reached',
    },
  };
}

// -- Public: a match (many games) ------------------------------------------

// Runs `numGames` games with the given seat-to-agent assignment and
// returns aggregate stats. Use this to compare a candidate agent against
// `'random'` (or any baseline) over many trials.
export function runMatch(opts: {
  agents: SeatAgents;
  numGames: number;
  seed?: number;
  maxHandsPerGame?: number;
  config?: GameConfig;
}): MatchSummary {
  const seed = opts.seed ?? 1;
  const rng = mulberry32(seed);
  const games: GameSummary[] = [];
  const winsByTeam: Record<TeamId, number> = {};
  let stalemates = 0;
  let totalAttackerPoints = 0;
  let totalHandsCounted = 0;

  for (let i = 0; i < opts.numGames; i++) {
    const gameSeed = Math.floor(rng() * 0x7fffffff);
    const { summary } = runGame({
      agents: opts.agents,
      seed: gameSeed,
      maxHands: opts.maxHandsPerGame,
      config: opts.config,
    });
    games.push(summary);
    if (summary.winner !== null) {
      winsByTeam[summary.winner] = (winsByTeam[summary.winner] ?? 0) + 1;
    } else {
      stalemates += 1;
    }
    for (const h of summary.hands) {
      totalAttackerPoints += h.attackerPoints;
      totalHandsCounted += 1;
    }
  }

  return {
    numGames: opts.numGames,
    winsByTeam,
    stalemates,
    avgHandsPerGame:
      games.reduce((acc, g) => acc + g.handsPlayed, 0) / Math.max(games.length, 1),
    avgAttackerPoints:
      totalHandsCounted === 0 ? 0 : totalAttackerPoints / totalHandsCounted,
    games,
  };
}

