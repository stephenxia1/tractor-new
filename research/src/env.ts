// One self-play game slot: drives hands end to end, pausing whenever a
// learner-controlled seat has to make a play-phase decision.
//
// Episodes are single hands. Landlord seat and team levels are randomised
// per hand so training sees all 13 trump ranks and all four landlord
// positions rather than the narrow slice a single game sequence visits.
//
// Declaration and kitty burial are delegated to a heuristic bot for v1;
// only in-trick play is learned.

import {
  NATURAL_RANKS,
  applyEvent,
  createGame,
  defaultV1Config,
  mulberry32,
  nextToPlay,
  pickBotAction,
  type BotType,
  type Card,
  type GameEvent,
  type GameState,
  type NaturalRank,
  type Seat,
} from '@tractor/engine';

import { enumerateCandidates, type Candidate } from './candidates';
import { summarizeHand, type EpisodeSummary } from './encode';

export type SlotOptions = {
  readonly gameId: number;
  readonly seed: number;
  // Seats whose play-phase decisions come from the learner. All four for
  // self-play; a subset when benchmarking against a built-in bot.
  readonly learnerSeats: ReadonlyArray<boolean>;
  // Bot used for non-learner seats during play.
  readonly baselineBot?: BotType;
  // Bot used for declaration and kitty burial at every seat.
  readonly prePlayBot?: BotType;
  readonly maxCandidates?: number;
  // Randomise landlord + levels per hand (on by default; off for
  // duplicate-deal evaluation, where the deal must be reproducible).
  readonly randomize?: boolean;
  // Fixed landlord/level when randomize is false.
  readonly fixedLandlord?: Seat;
  readonly fixedLevel?: NaturalRank;
};

export type Pending = {
  readonly seat: Seat;
  readonly candidates: ReadonlyArray<Candidate>;
};

export type AdvanceResult = {
  readonly episodes: ReadonlyArray<EpisodeSummary>;
  readonly pending: Pending | null;
};

const MAX_STEPS_PER_ADVANCE = 20000;

export class GameSlot {
  readonly gameId: number;
  private readonly opts: SlotOptions;
  private readonly rng: () => number;
  private readonly baselineBot: BotType;
  private readonly prePlayBot: BotType;
  private state: GameState;
  private handCounter = 0;
  private currentHandId = 0;
  private buriedKitty: Card[] = [];
  private pending: Pending | null = null;
  private forcedLeads = 0;
  private assertNoForcedLead = false;

  constructor(opts: SlotOptions) {
    this.opts = opts;
    this.gameId = opts.gameId;
    this.rng = mulberry32(opts.seed);
    this.baselineBot = opts.baselineBot ?? 'smart-v34';
    this.prePlayBot = opts.prePlayBot ?? 'smart-v34';
    this.state = createGame(defaultV1Config());
  }

  get gameState(): GameState {
    return this.state;
  }

  get pendingDecision(): Pending | null {
    return this.pending;
  }

  get handId(): number {
    return this.currentHandId;
  }

  get kitty(): ReadonlyArray<Card> {
    return this.buriedKitty;
  }

  // Runs until a learner seat must act (pending set) or the step budget
  // is exhausted. Any hands finished along the way come back as episodes.
  advance(): AdvanceResult {
    const episodes: EpisodeSummary[] = [];
    if (this.pending !== null) return { episodes, pending: this.pending };

    for (let step = 0; step < MAX_STEPS_PER_ADVANCE; step++) {
      const phase = this.state.phase;

      if (phase === 'game_over' || phase === 'between_hands') {
        this.startHand();
        continue;
      }

      if (phase === 'declare_lockout' || phase === 'declare_open') {
        this.runDeclarePhase();
        continue;
      }

      if (phase === 'kitty_replace') {
        this.runKittyReplace();
        continue;
      }

      if (phase === 'trick_review') {
        const before = this.state;
        this.apply({ kind: 'advance_trick' });
        const ended =
          this.state.phase === 'between_hands' || this.state.phase === 'game_over';
        if (ended) {
          const finished = this.state.currentHand ?? before.currentHand;
          if (finished !== null && finished !== undefined) {
            episodes.push(
              summarizeHand(this.state, finished, this.gameId, this.currentHandId),
            );
          }
        }
        continue;
      }

      if (phase === 'play') {
        const seat = nextToPlay(this.state);
        if (seat === null) throw new Error('play phase with no seat to act');
        if (this.opts.learnerSeats[seat] === true) {
          const candidates = enumerateCandidates(this.state, seat, {
            maxCandidates: this.opts.maxCandidates,
            myKitty: this.buriedKitty,
          });
          if (candidates.length > 0) {
            this.pending = { seat, candidates };
            return { episodes, pending: this.pending };
          }
        }
        this.playWithBot(seat, this.baselineBot);
        continue;
      }

      throw new Error(`unexpected phase ${phase}`);
    }
    throw new Error('GameSlot.advance exceeded step budget');
  }

  // Commits the learner's choice and clears the pending decision.
  choose(index: number): void {
    const pending = this.pending;
    if (pending === null) throw new Error('no pending decision');
    const candidate = pending.candidates[index];
    if (candidate === undefined) {
      throw new Error(
        `candidate ${index} out of range (have ${pending.candidates.length})`,
      );
    }
    this.pending = null;
    this.assertNoForcedLead = true;
    try {
      this.apply({
        kind: 'play_cards',
        seat: pending.seat,
        cards: candidate.cards,
      });
    } finally {
      this.assertNoForcedLead = false;
    }
  }

  // Forced-lead penalties seen on *bot* moves (learner moves throw).
  get forcedLeadCount(): number {
    return this.forcedLeads;
  }

  // --- internals -----------------------------------------------------------

  private nextSeed(): number {
    return Math.floor(this.rng() * 0x7fffffff);
  }

  private startHand(): void {
    const config = defaultV1Config();
    let base = createGame(config);
    const randomize = this.opts.randomize ?? true;
    const landlord: Seat = randomize
      ? Math.floor(this.rng() * config.numPlayers)
      : this.opts.fixedLandlord ?? 0;
    const levelFor = (): NaturalRank => {
      if (!randomize) return this.opts.fixedLevel ?? '2';
      return NATURAL_RANKS[Math.floor(this.rng() * NATURAL_RANKS.length)]!;
    };
    base = {
      ...base,
      landlord,
      teamLevels: { A: levelFor(), B: levelFor() },
    };
    this.state = base;
    this.buriedKitty = [];
    this.handCounter += 1;
    this.currentHandId = this.handCounter;
    this.apply({ kind: 'start_hand', seed: this.nextSeed() });
  }

  private runDeclarePhase(): void {
    const numPlayers = this.state.config.numPlayers;
    for (let seat = 0; seat < numPlayers; seat++) {
      const action = pickBotAction(this.state, seat, this.prePlayBot);
      if (action.kind === 'wait') continue;
      const event = this.toEvent(action, seat);
      if (event === null) continue;
      const result = applyEvent(this.state, event);
      if (!result.ok) continue; // bot proposed something illegal; ignore it
      this.state = result.state;
      return;
    }
    // Nobody acted.
    if (this.state.phase === 'declare_lockout') {
      this.apply({ kind: 'release_lockout', seed: this.nextSeed() });
      return;
    }
    const landlord = this.state.currentHand?.landlord;
    if (landlord === null || landlord === undefined) {
      throw new Error('declare_open with no landlord');
    }
    this.apply({ kind: 'pick_up_kitty', seat: landlord });
  }

  private runKittyReplace(): void {
    const hand = this.state.currentHand;
    if (hand === null || hand.landlord === null) {
      throw new Error('kitty_replace with no landlord');
    }
    const seat = hand.landlord;
    const action = pickBotAction(this.state, seat, this.prePlayBot);
    if (action.kind !== 'replace_kitty') {
      throw new Error(`expected replace_kitty from bot, got ${action.kind}`);
    }
    this.buriedKitty = [...action.cards];
    this.apply({ kind: 'replace_kitty', seat, cards: action.cards });
  }

  private playWithBot(seat: Seat, bot: BotType): void {
    const action = pickBotAction(this.state, seat, bot);
    if (action.kind !== 'play_cards') {
      throw new Error(`expected play_cards from ${bot}, got ${action.kind}`);
    }
    this.apply({ kind: 'play_cards', seat, cards: action.cards });
  }

  private toEvent(
    action: ReturnType<typeof pickBotAction>,
    seat: Seat,
  ): GameEvent | null {
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

  private apply(event: GameEvent): void {
    const result = applyEvent(this.state, event);
    if (!result.ok) {
      throw new Error(`${event.kind} rejected: ${result.error}`);
    }
    // A forced-lead notice means the engine judged a throw unsafe and
    // played only part of it (rules.md §7.6). Candidates from
    // publicInfoThrows are provably safe, so this firing on a learner
    // move means the safety derivation is wrong -- fail loudly rather
    // than train on an action that was never actually played.
    if (result.notice !== undefined && result.notice.kind === 'forced_lead') {
      this.forcedLeads += 1;
      if (this.assertNoForcedLead) {
        throw new Error(
          `forced-lead penalty fired on a learner throw: played ` +
            `${result.notice.playedCards.length} of ` +
            `${result.notice.playedCards.length + result.notice.returnedCards.length} cards`,
        );
      }
    }
    this.state = result.state;
  }
}
