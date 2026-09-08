// Subprocess driver for the DMC training pipeline.
//
// Reads newline-delimited JSON commands from stdin, runs Tractor hands
// using v34 for all non-play phases, and exposes the play phase as a
// stream of decision events for the Python actor on the other side.
//
// Protocol (stateful, transactional):
//
//   Python → TS: {"cmd":"new_episode","seed":<int>}
//   TS → Python: {"event":"decision", ...} | {"event":"done", ...}
//
//   Python → TS: {"cmd":"choose","index":<int>}
//   TS → Python: {"event":"decision", ...} | {"event":"done", ...}
//
//   Python → TS: {"cmd":"exit"}
//   TS exits.
//
// A "decision" event is emitted for every Q-network query — both
// follow plays and each sub-action step of a multi-pattern lead.
// A "done" event carries per-trick attacker captures so Python can
// compute Monte-Carlo returns.

import * as readline from 'node:readline';

import { pickBotAction } from '../../bot';
import { applyEvent, nextToPlay } from '../../reducer';
import { mulberry32 } from '../../prng';
import {
  createGame,
  defaultV1Config,
  teamOf,
  type GameState,
  type Seat,
} from '../../state';
import { type GameEvent } from '../../events';
import { sumPoints } from '../../scoring';
import { isTrump } from '../../ordering';

import {
  applyLeadSubAction,
  enumerateCandidates,
  EMPTY_LEAD,
  type Candidate,
  type LeadInProgress,
} from './actions';
import { buildObservation, type Observation } from './observation';
import { roleFor } from './roles';

// ---- types --------------------------------------------------------------

type DecisionEvent = {
  readonly event: 'decision';
  readonly trickIdx: number;
  readonly seat: number;
  readonly role: string;
  readonly side: string;
  readonly leadInProgress: { readonly chosen: number; readonly patternCount: number };
  readonly observation: Observation;
  readonly candidates: ReadonlyArray<Candidate>;
};

type DoneEvent = {
  readonly event: 'done';
  // per-trick attacker captures (no kitty bonus); index = trick number.
  readonly perTrickAttackerCaptures: ReadonlyArray<number>;
  // attacker indicator per trick (1 if attackers won that trick, else 0).
  readonly perTrickAttackerWon: ReadonlyArray<number>;
  // kitty multiplier bonus added to last trick if attackers won it.
  readonly lastTrickKittyBonus: number;
  // final attackerPoints (matches engine summary; sanity check).
  readonly attackerPoints: number;
  readonly landlord: number;
};

// ---- driver state -------------------------------------------------------

let state: GameState = createGame(defaultV1Config());
let lead: LeadInProgress = EMPTY_LEAD;
let pendingCandidates: Candidate[] = [];
let pendingSeat: Seat | null = null;
// Track per-trick attacker captures during the hand. Indexed by trick number.
const trickCaptures: number[] = [];
const trickAttackerWon: number[] = [];
let attackerPointsBefore = 0;
let kittyBonusAccumulated = 0;
// Per-episode RNG used for `release_lockout` seeds when no seat declares.
let episodeRng: () => number = mulberry32(0);

// ---- helpers ------------------------------------------------------------

function emit(payload: DecisionEvent | DoneEvent | Record<string, unknown>): void {
  process.stdout.write(JSON.stringify(payload) + '\n');
}

function err(message: string): void {
  emit({ event: 'error', message });
}

// Run any engine event and, if it caused a trick to resolve, record
// per-trick attacker capture + winner.
function applyAndTrack(ev: GameEvent): void {
  const tricksBefore = state.currentHand?.tricks.length ?? 0;
  const attackerBefore = state.currentHand?.attackerPoints ?? 0;
  const r = applyEvent(state, ev);
  if (!r.ok) throw new Error(`apply failed (${ev.kind}): ${r.error}`);
  state = r.state;
  const tricksAfter = state.currentHand?.tricks.length ?? 0;
  if (process.env.DMC_TRACE === '1') {
    process.stderr.write(`event=${ev.kind} phase=${state.phase} tricks=${tricksBefore}->${tricksAfter} atkPts=${state.currentHand?.attackerPoints ?? 'null'}\n`);
  }
  if (tricksAfter > tricksBefore) {
    const hand = state.currentHand;
    if (hand !== null) {
      const lastTrick = hand.tricks[hand.tricks.length - 1]!;
      const winnerTeam = teamOf(state, lastTrick.winner);
      const landlordTeam = teamOf(state, hand.landlord!);
      const attackersWon = winnerTeam !== landlordTeam;
      const captured = hand.attackerPoints - attackerBefore;
      trickCaptures.push(captured);
      trickAttackerWon.push(attackersWon ? 1 : 0);
    }
  }
}

// Drive the engine through non-play phases using v34. Returns when phase
// is 'play' AND a seat needs to make a decision, OR when the hand is
// over.
function advanceUntilPlayDecisionOrDone(): 'play' | 'done' {
  let safety = 0;
  while (state.phase !== 'between_hands' && state.phase !== 'game_over') {
    if (++safety > 10000) {
      throw new Error('advance loop safety exceeded');
    }
    if (state.phase === 'play') {
      // If a play decision is needed (someone's turn) → pause for Python.
      const seat = nextToPlay(state);
      if (seat !== null) return 'play';
      // Otherwise, advance the trick (engine auto-progress).
      applyAndTrack({ kind: 'advance_trick' });
      continue;
    }
    if (state.phase === 'trick_review') {
      applyAndTrack({ kind: 'advance_trick' });
      continue;
    }
    // Non-play phase: delegate to v34 for each seat in turn.
    let stepped = false;
    for (let s = 0; s < state.config.numPlayers; s++) {
      const action = pickBotAction(state, s as Seat, 'smart-v34');
      if (action.kind === 'wait') continue;
      let ev: GameEvent | null = null;
      switch (action.kind) {
        case 'declare':
          ev = { kind: 'declare', seat: s as Seat, cards: action.cards };
          break;
        case 'pick_up_kitty':
          ev = { kind: 'pick_up_kitty', seat: s as Seat };
          break;
        case 'replace_kitty':
          ev = { kind: 'replace_kitty', seat: s as Seat, cards: action.cards };
          break;
        case 'play_cards':
          ev = { kind: 'play_cards', seat: s as Seat, cards: action.cards };
          break;
      }
      if (ev !== null) {
        applyAndTrack(ev);
        stepped = true;
        break;
      }
    }
    if (!stepped) {
      // No bot wanted to act this round. If we're still in declare_lockout,
      // emit a `release_lockout` event (engine fallback: no-declare kitty
      // flip, random landlord). In declare_open the landlord must claim
      // the kitty; if no agent did, that's an upstream bug.
      if (state.phase === 'declare_lockout') {
        const seed = Math.floor(episodeRng() * 0x7fffffff);
        applyAndTrack({ kind: 'release_lockout', seed });
        continue;
      }
      throw new Error(`non-play stall at phase=${state.phase}`);
    }
  }
  return 'done';
}

function emitDecisionForSeat(seat: Seat): void {
  pendingSeat = seat;
  const obs = buildObservation(state, seat, lead.chosen);
  const cands = enumerateCandidates(state, seat, lead);
  if (cands.length === 0) {
    throw new Error(`no candidates at phase=${state.phase} seat=${seat}`);
  }
  pendingCandidates = [...cands];
  const role = roleFor(state, seat);
  const event: DecisionEvent = {
    event: 'decision',
    trickIdx: state.currentHand?.tricks.length ?? 0,
    seat,
    role: role.role,
    side: role.side,
    leadInProgress: {
      chosen: lead.chosen.length,
      patternCount: lead.patternCount,
    },
    observation: obs,
    candidates: cands,
  };
  emit(event);
}

function emitDoneEvent(): void {
  const hand = state.currentHand;
  // `state.landlord` is the persistent landlord; `currentHand.landlord` is
  // the per-hand assignment. The hand-level value is what was actually in
  // play; after the hand resolves, currentHand may be cleared but
  // state.landlord still reflects the rotated outcome. Fall back through.
  const landlord = hand?.landlord ?? state.landlord ?? 0;
  emit({
    event: 'done',
    perTrickAttackerCaptures: trickCaptures,
    perTrickAttackerWon: trickAttackerWon,
    lastTrickKittyBonus: kittyBonusAccumulated,
    attackerPoints: hand?.attackerPoints ?? attackerPointsBefore,
    landlord,
  });
}

// ---- command handlers ---------------------------------------------------

function handleNewEpisode(seed: number): void {
  // No-trump (joker) declarations aren't supported by the DMC encoding
  // yet; if a hand ends up in no-trump mode (joker declaration or kitty
  // flip lands on a joker), retry with a perturbed seed. Bounded to
  // avoid infinite loops on broken configs.
  let attempt = 0;
  let s = seed;
  while (true) {
    state = createGame(defaultV1Config());
    episodeRng = mulberry32(s);
    lead = EMPTY_LEAD;
    pendingCandidates = [];
    pendingSeat = null;
    trickCaptures.length = 0;
    trickAttackerWon.length = 0;
    kittyBonusAccumulated = 0;
    attackerPointsBefore = 0;

    const start = applyEvent(state, { kind: 'start_hand', seed: s });
    if (!start.ok) throw new Error(`start_hand failed: ${start.error}`);
    state = start.state;

    const status = advanceUntilPlayDecisionOrDone();
    if (status === 'done') {
      emitDoneEvent();
      return;
    }
    // Reached play. Confirm trump is set (non-no-trump).
    if (state.currentHand?.trumpSuit !== null) break;
    attempt++;
    if (attempt > 20) {
      err('failed to find a non-no-trump hand within 20 attempts');
      return;
    }
    s = (s * 1664525 + 1013904223) >>> 0; // simple LCG perturbation
  }

  const seat = nextToPlay(state);
  if (seat === null) throw new Error('expected play decision but got null seat');
  attackerPointsBefore = state.currentHand?.attackerPoints ?? 0;
  emitDecisionForSeat(seat);
}

function handleChoose(index: number): void {
  if (pendingSeat === null) {
    err('choose without pending decision');
    return;
  }
  if (index < 0 || index >= pendingCandidates.length) {
    err(`choose index out of range: ${index}/${pendingCandidates.length}`);
    return;
  }
  const choice = pendingCandidates[index]!;
  const seat = pendingSeat;

  if (choice.kind === 'lead-stop' || choice.kind === 'follow') {
    // Commit a full play. For lead-stop, the cards are the accumulated
    // lead.chosen; for follow, they're directly in choice.cards.
    const cardsToPlay = choice.kind === 'lead-stop' ? lead.chosen : choice.cards;
    try {
      applyAndTrack({ kind: 'play_cards', seat, cards: cardsToPlay });
    } catch (e) {
      err(`play_cards failed: ${(e as Error).message}`);
      return;
    }
    // Reset lead progress.
    lead = EMPTY_LEAD;
  } else {
    // Lead sub-action — commit pattern, stay in same seat.
    const next = applyLeadSubAction(lead, choice);
    if (next === null) {
      throw new Error('unexpected null from applyLeadSubAction (non-stop)');
    }
    lead = next;
  }

  // After applying, advance the engine if necessary (trick_review etc.).
  const status = advanceUntilPlayDecisionOrDone();
  if (status === 'done') {
    emitDoneEvent();
    return;
  }
  const next = nextToPlay(state);
  if (next === null) throw new Error('no next seat after advance');
  emitDecisionForSeat(next);
}

// ---- main loop ----------------------------------------------------------

function main(): void {
  // Silence the trumpFraction lint — touch isTrump/sumPoints exports so
  // the imports aren't reported as unused after we tighten the file.
  void isTrump;
  void sumPoints;
  void mulberry32;

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false,
  });
  emit({ event: 'ready' });

  rl.on('line', (line: string) => {
    const trimmed = line.trim();
    if (trimmed === '') return;
    let msg: { cmd: string;[k: string]: unknown };
    try {
      msg = JSON.parse(trimmed) as { cmd: string;[k: string]: unknown };
    } catch (e) {
      err(`bad JSON: ${(e as Error).message}`);
      return;
    }
    try {
      switch (msg.cmd) {
        case 'new_episode':
          handleNewEpisode((msg.seed as number) | 0);
          break;
        case 'choose':
          handleChoose((msg.index as number) | 0);
          break;
        case 'exit':
          process.exit(0);
          break;
        default:
          err(`unknown command: ${msg.cmd}`);
      }
    } catch (e) {
      err((e as Error).message);
    }
  });
  rl.on('close', () => process.exit(0));
}

main();
