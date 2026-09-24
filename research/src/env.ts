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
  SUITS,
  applyEvent,
  cardKey,
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

import { cardSlot } from './cards';
// Makes `rl-v2` selectable as a baseline/opponent in self-play and in
// the arena; without it the engine throws rather than substituting.
import '@tractor/engine/rl/v2/register';

import { enumerateCandidates, type Candidate } from './candidates';
import { summarizeHand, type EpisodeSummary } from './encode';
import { buildLeaves, type SearchOptions, type SearchResult } from './search';

// A 32-bit mixer, so a deal can be addressed by coordinates rather than by
// a position in an RNG stream. Duplicate-deal evaluation needs the deal for
// (base, game, hand) to be the same cards no matter what the policy did
// beforehand, and a per-slot RNG chain cannot promise that.
function mix32(a: number, b: number, c: number): number {
  let x = (a ^ 0x9e3779b9) >>> 0;
  x = Math.imul(x ^ ((b + 0x85ebca6b) >>> 0), 0xcc9e2d51) >>> 0;
  x = (((x << 15) | (x >>> 17)) >>> 0) ^ ((c + 0xc2b2ae35) >>> 0);
  x = Math.imul(x, 0x1b873593) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x85ebca6b) >>> 0;
  x = (x ^ (x >>> 13)) >>> 0;
  x = Math.imul(x, 0xc2b2ae35) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

export type SlotOptions = {
  readonly gameId: number;
  readonly seed: number;
  // Seats whose play-phase decisions come from the learner. All four for
  // self-play; a subset when benchmarking against a built-in bot.
  readonly learnerSeats: ReadonlyArray<boolean>;
  // Bot used for non-learner seats during play.
  readonly baselineBot?: BotType;
  // Per-seat override of `baselineBot`, indexed by seat. Lets the two
  // teams be filled by *different* bots, which is what makes bot-vs-bot
  // duplicate-deal matches possible (the same instrument the agent is
  // measured with, pointed at two heuristics instead).
  readonly baselineBotBySeat?: ReadonlyArray<BotType | undefined>;
  // Bot used for declaration and kitty burial at every seat.
  readonly prePlayBot?: BotType;
  // Replace the pre-play bot with the Python generator's draw
  // (tractor/gen/episode.py): one declaration uniform over every legal
  // (seat, declaration), no overrides, and a burial uniform over the
  // landlord's cards. Evaluation uses it so neither side plays from a
  // position its own heuristic set up. In duplicate mode the draws come
  // from a stream keyed on the deal, so both halves agree.
  readonly randomPrePlay?: boolean;
  // The learner makes its own seats' pre-play decisions: the engine deals
  // card by card (``incrementalDeal``), a learner seat is asked at every
  // card it receives that it could declare with, the bots declare
  // whenever their own logic says to (as they do live), a closing round
  // follows the last card, and a learner landlord buries. Surfaced
  // through ``AdvanceResult.prePending`` and answered with
  // ``answerDeclare`` / ``answerBury``.
  readonly agentPrePlay?: boolean;
  // Probability that a hand's kitty is perturbed away from the pre-play
  // bot's choice, and the most cards that can be swapped when it is.
  readonly kittyNoise?: number;
  readonly kittyMaxSwaps?: number;
  readonly maxCandidates?: number;
  // Randomise landlord + levels per hand (on by default; off for
  // duplicate-deal evaluation, where the deal must be reproducible).
  readonly randomize?: boolean;
  // Fixed landlord/level when randomize is false, or an override of the
  // value duplicate mode would otherwise derive.
  readonly fixedLandlord?: Seat;
  readonly fixedLevel?: NaturalRank;
  // Duplicate mode. When set, hand h of this slot is dealt from
  // mix32(dealSeedBase, gameId, h) rather than from the slot's own RNG
  // chain: the shuffle seed, the landlord and the trump rank all become
  // pure functions of those three coordinates. Two slots given the same
  // base and gameId therefore play the identical sequence of deals even
  // though different seats are learner-controlled and the number of RNG
  // draws along the way differs.
  //
  // Both teams are put on the *same* level, so swapping the agent between
  // team A and team B is an exact mirror: the two halves of a pair differ
  // only in which policy sits in which seats.
  readonly dealSeedBase?: number;
  // Keep a copy of the dealt hands for the current hand (see `dealt`).
  // Off in training; used by src/dupcheck.ts to compare paired deals card
  // for card rather than trusting the digest.
  readonly captureDeal?: boolean;
};

// The deal as it stood the instant after the shuffle, before anyone
// declared or touched the kitty. Card keys, sorted, so two deals compare
// with a string equality.
export type DealtCards = {
  readonly hands: ReadonlyArray<ReadonlyArray<string>>;
  readonly kitty: ReadonlyArray<string>;
  readonly landlord: number;
  readonly trumpRank: string;
};

export type Pending = {
  readonly seat: Seat;
  readonly candidates: ReadonlyArray<Candidate>;
};

// A pre-play decision for a learner seat (``agentPrePlay`` only).
// ``mask`` covers the 11 declaration actions of py/tractor/sim/deal.py:
// pass, a single then a pair of the level rank per suit, the small- and
// big-joker pairs.
export type PrePending =
  | {
      readonly kind: 'declare';
      readonly seat: Seat;
      readonly mask: ReadonlyArray<boolean>;
      readonly closing: boolean;
    }
  | { readonly kind: 'bury'; readonly seat: Seat };

export type AdvanceResult = {
  readonly episodes: ReadonlyArray<EpisodeSummary>;
  readonly pending: Pending | null;
  readonly prePending?: PrePending | null;
};

const DECLARE_ACTIONS = 11;

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
  private dealDigestValue = 0;
  private dealtCards: DealtCards | null = null;
  private buriedKitty: Card[] = [];
  private prePlayRng: () => number = Math.random;
  private declaredThisHand = false;
  private pending: Pending | null = null;
  private prePending: PrePending | null = null;
  private dealIndex = 0;
  private closingActive = false;
  private closingCursor = 0;
  private closingPasses = 0;
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

  // Fingerprint of the current hand's deal, taken right after the shuffle.
  get dealDigest(): number {
    return this.dealDigestValue;
  }

  // The dealt cards, when `captureDeal` was requested. Null otherwise.
  get dealt(): DealtCards | null {
    return this.dealtCards;
  }

  // Runs until a learner seat must act (pending set) or the step budget
  // is exhausted. Any hands finished along the way come back as episodes.
  //
  // With no learner seats at all (bot-vs-bot duplicate matches) nothing
  // ever pends, so the slot instead yields after each completed hand --
  // otherwise it would play until the step budget and report nothing.
  advance(): AdvanceResult {
    const episodes: EpisodeSummary[] = [];
    if (this.prePending !== null) {
      return { episodes, pending: null, prePending: this.prePending };
    }
    if (this.pending !== null) return { episodes, pending: this.pending };
    const agent = this.opts.agentPrePlay === true;
    const hasLearnerSeats = this.opts.learnerSeats.some((x) => x === true);

    for (let step = 0; step < MAX_STEPS_PER_ADVANCE; step++) {
      const phase = this.state.phase;

      if (phase === 'game_over' || phase === 'between_hands') {
        this.startHand();
        continue;
      }

      if (phase === 'dealing') {
        this.runDealing();
        if (this.prePending !== null) {
          return { episodes, pending: null, prePending: this.prePending };
        }
        continue;
      }

      if (agent && (phase === 'declare_lockout' || phase === 'declare_open')) {
        this.runAgentClosing();
        if (this.prePending !== null) {
          return { episodes, pending: null, prePending: this.prePending };
        }
        continue;
      }

      if (phase === 'declare_lockout' || phase === 'declare_open') {
        this.runDeclarePhase();
        continue;
      }

      if (phase === 'kitty_replace') {
        const landlord = this.state.currentHand?.landlord ?? null;
        if (agent && landlord !== null && this.opts.learnerSeats[landlord] === true) {
          this.prePending = { kind: 'bury', seat: landlord };
          return { episodes, pending: null, prePending: this.prePending };
        }
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
              summarizeHand(
                this.state,
                finished,
                this.gameId,
                this.currentHandId,
                this.dealDigestValue,
              ),
            );
          }
          if (!hasLearnerSeats) return { episodes, pending: null };
        }
        continue;
      }

      if (phase === 'play') {
        const seat = nextToPlay(this.state);
        if (seat === null) throw new Error('play phase with no seat to act');
        if (this.opts.learnerSeats[seat] === true) {
          const candidates = enumerateCandidates(this.state, seat, {
            maxCandidates: this.opts.maxCandidates,
            myKitty: this.kittyFor(seat),
          });
          if (candidates.length > 0) {
            this.pending = { seat, candidates };
            return { episodes, pending: this.pending };
          }
        }
        this.playWithBot(seat, this.botForSeat(seat));
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

  // Positions reachable from the pending decision, for value-guided
  // search. Null when there is nothing pending or nothing to choose.
  searchLeaves(opts: SearchOptions): SearchResult | null {
    const pending = this.pending;
    const hand = this.state.currentHand;
    if (pending === null || hand === null || hand.trumpRank === null) return null;
    return buildLeaves(
      this.state,
      pending.seat,
      pending.candidates,
      this.kittyFor(pending.seat),
      { trumpRank: hand.trumpRank, trumpSuit: hand.trumpSuit },
      opts,
    );
  }

  // Forced-lead penalties seen on *bot* moves (learner moves throw).
  get forcedLeadCount(): number {
    return this.forcedLeads;
  }

  // --- internals -----------------------------------------------------------

  private nextSeed(): number {
    return Math.floor(this.rng() * 0x7fffffff);
  }

  // The buried kitty is knowledge the landlord alone holds. Candidate
  // generation subtracts `myKitty` from the unseen-card pool to sharpen
  // throw safety, so handing it to a non-landlord seat would offer that
  // seat components that read as unbeatable only because of cards it has
  // no way of knowing are out of play -- hidden information leaking into
  // the action set itself, which is what generating throws from public
  // information alone exists to prevent.
  private kittyFor(seat: Seat): ReadonlyArray<Card> {
    return seat === this.state.currentHand?.landlord ? this.buriedKitty : [];
  }

  private startHand(): void {
    const config =
      this.opts.agentPrePlay === true
        ? { ...defaultV1Config(), incrementalDeal: true }
        : defaultV1Config();
    this.dealIndex = 0;
    this.closingActive = false;
    this.prePending = null;
    let base = createGame(config);
    this.handCounter += 1;
    this.currentHandId = this.handCounter;

    let landlord: Seat;
    let levels: { A: NaturalRank; B: NaturalRank };
    let handSeed: number;

    if (this.opts.dealSeedBase !== undefined) {
      // Duplicate mode. Draw all three from a private stream keyed on
      // (base, game, hand) and only then apply any override, so the
      // shuffle seed does not shift when the landlord is pinned.
      const dealRng = mulberry32(
        mix32(this.opts.dealSeedBase, this.gameId, this.currentHandId),
      );
      const drawnLandlord = Math.floor(dealRng() * config.numPlayers);
      const drawnLevel =
        NATURAL_RANKS[Math.floor(dealRng() * NATURAL_RANKS.length)]!;
      handSeed = Math.floor(dealRng() * 0x7fffffff);
      landlord = this.opts.fixedLandlord ?? drawnLandlord;
      const level = this.opts.fixedLevel ?? drawnLevel;
      levels = { A: level, B: level };
    } else {
      const randomize = this.opts.randomize ?? true;
      landlord = randomize
        ? Math.floor(this.rng() * config.numPlayers)
        : this.opts.fixedLandlord ?? 0;
      const levelFor = (): NaturalRank => {
        if (!randomize) return this.opts.fixedLevel ?? '2';
        return NATURAL_RANKS[Math.floor(this.rng() * NATURAL_RANKS.length)]!;
      };
      levels = { A: levelFor(), B: levelFor() };
      handSeed = this.nextSeed();
    }

    base = { ...base, landlord, teamLevels: levels };
    this.state = base;
    this.buriedKitty = [];
    this.declaredThisHand = false;
    // Not the slot's own chain: that is also drawn from by the seed of
    // each half (different on purpose), so it would split the pair.
    this.prePlayRng =
      this.opts.dealSeedBase !== undefined
        ? mulberry32(
            mix32(this.opts.dealSeedBase ^ 0x5eed_b0b0, this.gameId, this.currentHandId),
          )
        : this.rng;
    this.apply({ kind: 'start_hand', seed: handSeed });
    this.dealDigestValue = this.computeDealDigest();
    this.dealtCards =
      this.opts.captureDeal === true ? this.captureDealtCards() : null;
  }

  private captureDealtCards(): DealtCards {
    const hand = this.state.currentHand;
    if (hand === null) throw new Error('captureDealtCards with no hand');
    const keys = (cards: ReadonlyArray<Card>): string[] =>
      cards.map(cardKey).sort();
    const hands: string[][] = [];
    for (let seat = 0; seat < this.state.config.numPlayers; seat++) {
      hands.push(keys(hand.hands[seat] ?? []));
    }
    return {
      hands,
      kitty: keys(hand.kitty),
      landlord: hand.landlord ?? -1,
      trumpRank: hand.trumpRank ?? 'none',
    };
  }

  // FNV-1a over the dealt hands (seat by seat), the kitty, the landlord
  // and the trump rank. Taken immediately after start_hand, before any
  // declaration or kitty pickup moves cards around, so it describes the
  // deal itself rather than what the players then did with it.
  private computeDealDigest(): number {
    const hand = this.state.currentHand;
    if (hand === null) return 0;
    let h = 0x811c9dc5;
    const push = (v: number): void => {
      h = Math.imul(h ^ (v & 0xff), 0x01000193) >>> 0;
    };
    const pushCards = (cards: ReadonlyArray<Card>): void => {
      const slots = cards.map(cardSlot).sort((a, b) => a - b);
      for (const s of slots) push(s);
    };
    for (let seat = 0; seat < this.state.config.numPlayers; seat++) {
      push(0xc0 + seat);
      pushCards(hand.hands[seat] ?? []);
    }
    push(0xfa);
    pushCards(hand.kitty);
    push(0xfb);
    push(hand.landlord ?? 255);
    push(
      hand.trumpRank === null ? 255 : NATURAL_RANKS.indexOf(hand.trumpRank),
    );
    return h >>> 0;
  }

  // --- the learner's own pre-play (agentPrePlay) --------------------------

  get dealtCount(): number {
    return this.dealIndex;
  }

  // The cards a seat can see: its revealed prefix while dealing, its
  // whole hand after.
  revealedFor(seat: Seat): ReadonlyArray<Card> {
    const hand = this.state.currentHand;
    if (hand === null) return [];
    const all = hand.hands[seat] ?? [];
    if (this.state.phase !== 'dealing') return all;
    return all.slice(0, hand.cardsDealtBySeat[seat] ?? 0);
  }

  // The cards declaration action ``a`` would show, or null if not held.
  private declareCards(seat: Seat, a: number): Card[] | null {
    const hand = this.state.currentHand;
    if (hand === null || hand.trumpRank === null) return null;
    const level = hand.trumpRank;
    const held = this.revealedFor(seat);
    const take = (pred: (c: Card) => boolean, n: number): Card[] | null => {
      const got = held.filter(pred);
      return got.length >= n ? got.slice(0, n) : null;
    };
    if (a >= 1 && a <= 4) {
      const suit = SUITS[a - 1];
      return take((c) => c.kind === 'standard' && c.suit === suit && c.rank === level, 1);
    }
    if (a >= 5 && a <= 8) {
      const suit = SUITS[a - 5];
      return take((c) => c.kind === 'standard' && c.suit === suit && c.rank === level, 2);
    }
    if (a === 9) return take((c) => c.kind === 'joker' && c.rank === 'small', 2);
    if (a === 10) return take((c) => c.kind === 'joker' && c.rank === 'big', 2);
    return null;
  }

  // Which of the 11 actions the engine would accept from ``seat`` now.
  declarationMask(seat: Seat): boolean[] {
    const mask = [true];
    for (let a = 1; a < DECLARE_ACTIONS; a++) {
      const cards = this.declareCards(seat, a);
      mask.push(
        cards !== null && applyEvent(this.state, { kind: 'declare', seat, cards }).ok,
      );
    }
    return mask;
  }

  // Tier (0 none, 1 single, 2 pair, 3/4 joker pairs), suit (0..3, 4 for
  // no-trump, -1 none) and seat (-1 none) of the standing declaration.
  standing(): { tier: number; suit: number; seat: number } {
    const d = this.state.currentHand?.standingDeclaration ?? null;
    if (d === null) return { tier: 0, suit: -1, seat: -1 };
    if (d.kind === 'joker-pair') {
      return { tier: d.jokerRank === 'big' ? 4 : 3, suit: 4, seat: d.seat };
    }
    return { tier: d.kind === 'pair' ? 2 : 1, suit: SUITS.indexOf(d.suit), seat: d.seat };
  }

  private botDeclares(seat: Seat): boolean {
    const action = pickBotAction(this.state, seat, this.prePlayBot);
    if (action.kind !== 'declare') return false;
    const event = this.toEvent(action, seat);
    if (event === null) return false;
    const result = applyEvent(this.state, event);
    if (!result.ok) return false;
    this.state = result.state;
    return true;
  }

  private runDealing(): void {
    const hand = this.state.currentHand;
    if (hand === null || hand.landlord === null) {
      throw new Error('agent pre-play needs a landlord set before the deal');
    }
    const numPlayers = this.state.config.numPlayers;
    const total =
      this.state.config.numDecks * 54 - this.state.config.kittySize;
    if (this.dealIndex >= total) {
      this.apply({ kind: 'finish_dealing' });
      return;
    }
    const seat = ((hand.landlord + this.dealIndex) % numPlayers) as Seat;
    this.apply({ kind: 'reveal_card', seat });
    this.dealIndex += 1;
    // The bots may declare at any card, as they do in a live game.
    for (let s = 0; s < numPlayers; s++) {
      if (this.opts.learnerSeats[s] === true) continue;
      this.botDeclares(s as Seat);
    }
    if (this.opts.learnerSeats[seat] === true) {
      const mask = this.declarationMask(seat);
      if (mask.slice(1).some(Boolean)) {
        this.prePending = { kind: 'declare', seat, mask, closing: false };
      }
    }
  }

  // After the last card: every seat in turn from the landlord, cycling
  // until four in a row pass, then the kitty is picked up.
  private runAgentClosing(): void {
    const hand = this.state.currentHand;
    if (hand === null || hand.landlord === null) {
      throw new Error('declaration window with no landlord');
    }
    if (this.state.phase === 'declare_open') {
      this.apply({ kind: 'pick_up_kitty', seat: hand.landlord });
      return;
    }
    const numPlayers = this.state.config.numPlayers;
    if (!this.closingActive) {
      this.closingActive = true;
      this.closingCursor = hand.landlord;
      this.closingPasses = 0;
    }
    while (this.closingPasses < numPlayers) {
      const seat = this.closingCursor as Seat;
      if (this.opts.learnerSeats[seat] === true) {
        const mask = this.declarationMask(seat);
        if (mask.slice(1).some(Boolean)) {
          this.prePending = { kind: 'declare', seat, mask, closing: true };
          return;
        }
        this.closingPasses += 1;
      } else {
        this.closingPasses = this.botDeclares(seat) ? 0 : this.closingPasses + 1;
      }
      this.closingCursor = (seat + 1) % numPlayers;
    }
    this.closingActive = false;
    this.apply({ kind: 'release_lockout', seed: this.nextSeed() });
  }

  answerDeclare(action: number): void {
    const p = this.prePending;
    if (p === null || p.kind !== 'declare') throw new Error('no declaration pending');
    if (p.mask[action] !== true) throw new Error(`action ${action} is not legal here`);
    this.prePending = null;
    if (action > 0) {
      const cards = this.declareCards(p.seat, action);
      if (cards === null) throw new Error(`seat ${p.seat} cannot show action ${action}`);
      this.apply({ kind: 'declare', seat: p.seat, cards });
    }
    if (p.closing) {
      this.closingPasses = action > 0 ? 0 : this.closingPasses + 1;
      this.closingCursor = (p.seat + 1) % this.state.config.numPlayers;
    }
  }

  // ``counts`` is the burial as copies per slot of cards.ts.
  answerBury(counts: ReadonlyArray<number>): void {
    const p = this.prePending;
    if (p === null || p.kind !== 'bury') throw new Error('no burial pending');
    const hand = this.state.currentHand;
    if (hand === null) throw new Error('burial with no hand');
    const need = [...counts];
    const cards: Card[] = [];
    for (const c of hand.hands[p.seat] ?? []) {
      const slot = cardSlot(c);
      if ((need[slot] ?? 0) > 0) {
        need[slot] = (need[slot] ?? 0) - 1;
        cards.push(c);
      }
    }
    if (cards.length !== this.state.config.kittySize || need.some((n) => n !== 0)) {
      throw new Error(`burial of ${cards.length} cards does not match the landlord's hand`);
    }
    this.prePending = null;
    this.buriedKitty = [...cards];
    this.apply({ kind: 'replace_kitty', seat: p.seat, cards });
  }

  private runDeclarePhase(): void {
    if (this.opts.randomPrePlay === true) {
      this.runRandomDeclare();
      return;
    }
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

  // The generator's declaration: every legal (seat, declaration) listed
  // in the same order as `legal_declarations` in tractor/gen/episode.py
  // -- per seat, per suit a single then a pair of the level card, then
  // the joker pairs -- one drawn uniformly, and nobody overrides it. The
  // landlord is already fixed here, so the draw sets only the trump suit.
  // With nothing to declare, the lockout release flips the kitty.
  private runRandomDeclare(): void {
    const hand = this.state.currentHand;
    if (hand === null) throw new Error('declare phase with no hand');
    if (this.state.phase === 'declare_open') {
      if (hand.landlord === null) throw new Error('declare_open with no landlord');
      this.apply({ kind: 'pick_up_kitty', seat: hand.landlord });
      return;
    }
    if (!this.declaredThisHand) {
      this.declaredThisHand = true;
      const level = hand.trumpRank;
      const options: { seat: Seat; cards: Card[] }[] = [];
      const numDecks = this.state.config.numDecks;
      for (let s = 0; s < this.state.config.numPlayers; s++) {
        const seat = s as Seat;
        const held = hand.hands[seat] ?? [];
        for (const suit of SUITS) {
          const cards = held.filter(
            (c) => c.kind === 'standard' && c.rank === level && c.suit === suit,
          );
          if (cards.length >= 1) options.push({ seat, cards: [cards[0]!] });
          if (cards.length >= numDecks) {
            options.push({ seat, cards: cards.slice(0, 2) });
          }
        }
        for (const rank of ['small', 'big'] as const) {
          const jokers = held.filter((c) => c.kind === 'joker' && c.rank === rank);
          if (jokers.length >= numDecks) {
            options.push({ seat, cards: jokers.slice(0, 2) });
          }
        }
      }
      if (options.length > 0) {
        const pick = options[Math.floor(this.prePlayRng() * options.length)]!;
        this.apply({ kind: 'declare', seat: pick.seat, cards: pick.cards });
        return;
      }
    }
    this.apply({ kind: 'release_lockout', seed: this.nextSeed() });
  }

  // Every training game so far buried its kitty with one heuristic, so the
  // agent has only ever seen the positions that heuristic produces. That is
  // a distribution the agent can quietly fit to: measured, swapping the
  // pre-play regime moves its margin by ~7 points.
  //
  // `kittyNoise` perturbs the bot's choice instead of replacing it. Burying
  // uniformly at random is the wrong treatment -- it discards jokers and
  // point cards in ways no player ever would, so the agent would spend
  // capacity on positions that never occur. Swapping k of the buried cards
  // for k random others covers the neighbourhood of sensible play.
  private runKittyReplace(): void {
    const hand = this.state.currentHand;
    if (hand === null || hand.landlord === null) {
      throw new Error('kitty_replace with no landlord');
    }
    const seat = hand.landlord;
    if (this.opts.randomPrePlay === true) {
      // A uniform eight of the landlord's cards: a partial Fisher-Yates
      // over the hand as it stands, which is the same in both halves.
      const pool = [...(hand.hands[seat] ?? [])];
      const k = this.state.config.kittySize;
      for (let i = 0; i < k; i++) {
        const j = i + Math.floor(this.prePlayRng() * (pool.length - i));
        [pool[i], pool[j]] = [pool[j]!, pool[i]!];
      }
      const cards = pool.slice(0, k);
      this.buriedKitty = [...cards];
      this.apply({ kind: 'replace_kitty', seat, cards });
      return;
    }
    const action = pickBotAction(this.state, seat, this.prePlayBot);
    if (action.kind !== 'replace_kitty') {
      throw new Error(`expected replace_kitty from bot, got ${action.kind}`);
    }
    const cards = this.perturbKitty(hand.hands[seat] ?? [], action.cards);
    this.buriedKitty = [...cards];
    this.apply({ kind: 'replace_kitty', seat, cards });
  }

  private perturbKitty(
    landlordHand: ReadonlyArray<Card>,
    chosen: ReadonlyArray<Card>,
  ): ReadonlyArray<Card> {
    const noise = this.opts.kittyNoise ?? 0;
    if (noise <= 0 || this.rng() >= noise) return chosen;

    // How many of the buried cards to re-draw. Weighted towards small
    // perturbations: a wholly random kitty is rare, a nudge is common.
    const maxSwaps = Math.min(chosen.length, this.opts.kittyMaxSwaps ?? 3);
    const swaps = 1 + Math.floor(this.rng() * maxSwaps);

    // Full identity, not `cardKey`: that ignores `deckId`, so with two
    // decks burying one copy of a card excluded the *other* copy from the
    // pool too, and a duplicate of anything buried could never be swapped
    // in -- a systematic hole in the neighbourhood this is meant to cover.
    const identity = (c: Card): string => `${cardKey(c)}#${c.deckId}`;
    const buried = new Set(chosen.map(identity));
    const pool = landlordHand.filter((c) => !buried.has(identity(c)));
    if (pool.length === 0) return chosen;

    // Distinct slots: drawing `drop` with replacement let one slot be
    // overwritten twice, so a 3-swap draw could land only 2 changes and
    // discard a card it had just swapped in.
    const slots = Array.from({ length: chosen.length }, (_, i) => i);
    const out = [...chosen];
    for (let i = 0; i < swaps && pool.length > 0 && slots.length > 0; i++) {
      const dropAt = Math.floor(this.rng() * slots.length);
      const drop = slots[dropAt]!;
      slots.splice(dropAt, 1);
      const take = Math.floor(this.rng() * pool.length);
      out[drop] = pool[take]!;
      pool.splice(take, 1);
    }
    return out;
  }

  private botForSeat(seat: Seat): BotType {
    return this.opts.baselineBotBySeat?.[seat] ?? this.baselineBot;
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
