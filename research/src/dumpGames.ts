// Dumps whole hands played through the TypeScript engine, so the Python
// simulator can be replayed against them move for move.
//
// The unit-level dumpers (dumpOrdering / dumpCombos / dumpFollow) pin the
// rules primitives. What they cannot reach is the part of the engine that
// only exists across a whole hand: who won each trick, which team the
// points went to, whether a blocked throw returned the right cards, and
// what the final score came to. That is what this emits.
//
// One JSON object per line so a large dump streams on both ends:
//
//   {
//     hand, seed, decks,
//     levelRank,            // 0..12, index into NATURAL_RANKS
//     trumpSuit,            // 0..3, or 4 for no-trump
//     landlord,             // seat 0..3
//     hands: [[slot,...] x 4],   // the four hands at the FIRST TRICK, sorted
//     kitty: [slot,...],         // the eight buried cards, sorted
//     plays: [{ seat, req, got }, ...],   // in order, for the whole hand;
//                                         // req/got keep submission order
//     tricks: [{ leader, winner, points }, ...],
//     attackerPoints,       // final, kitty bonus included
//     winner, levelDelta    // computeHandResult on that total
//   }
//
// `req` is what the policy asked to play and `got` is what the engine
// actually committed. They differ exactly when the forced-lead penalty
// fires (rules.md §7.6): an unsafe throw plays only its smallest beatable
// component and the rest goes back to the leader's hand. Recording both
// is the point -- it is the one engine behaviour that cannot be checked
// from the resulting position alone.
//
// Policy is uniform over the engine's own enumerated candidates, with a
// deliberate exception at the lead: `allLegalPlayActions` only ever
// offers *safe* throws, so a run driven by it alone would never exercise
// the penalty at all. A fraction of leads are therefore drawn as raw
// multi-component throws built straight out of the hand, safe or not.
//
//   npx tsx src/dumpGames.ts 400 12345 py/tractor/tests/fixtures/games.jsonl
//   npx tsx src/dumpGames.ts 2000 777 > games.jsonl

import { closeSync, openSync, writeSync } from 'node:fs';

import {
  NATURAL_RANKS,
  SUITS,
  allLegalPlayActions,
  applyEvent,
  cardKey,
  classifyCombo,
  computeHandResult,
  createGame,
  decomposeThrow,
  defaultV1Config,
  effectiveSuit,
  findNKinds,
  findTractors,
  mulberry32,
  nextToPlay,
  pickBotAction,
  type BotType,
  type Card,
  type EffectiveSuit,
  type GameEvent,
  type GameState,
  type NaturalRank,
  type Seat,
  type TrumpContext,
} from '@tractor/engine';

import { cardSlot } from './cards';

const DECKS = 2;
const PREPLAY: BotType = 'smart-v34';
// Share of leads drawn as a raw throw rather than from the enumerator.
// At 0.3 roughly a third of hands see the penalty fire at least once.
const THROW_PROB = 0.3;
// Wide enough that the policy sees the follow variants the enumerator
// distinguishes, small enough that enumeration is not the bottleneck.
const MAX_CANDIDATES = 512;
const MAX_STEPS_PER_HAND = 4000;

type PlayRecord = {
  readonly seat: number;
  readonly req: ReadonlyArray<number>;
  readonly got: ReadonlyArray<number>;
};

type TrickRecord = {
  readonly leader: number;
  readonly winner: number;
  readonly points: number;
};

type GameRecord = {
  readonly hand: number;
  readonly seed: number;
  readonly decks: number;
  readonly levelRank: number;
  readonly trumpSuit: number;
  readonly landlord: number;
  readonly hands: ReadonlyArray<ReadonlyArray<number>>;
  readonly kitty: ReadonlyArray<number>;
  readonly plays: ReadonlyArray<PlayRecord>;
  readonly tricks: ReadonlyArray<TrickRecord>;
  readonly attackerPoints: number;
  readonly winner: 'defenders' | 'attackers';
  readonly levelDelta: number;
};

type Stats = {
  hands: number;
  plays: number;
  tricks: number;
  forcedLeads: number;
  throwLeads: number;
  noTrump: number;
  attackerPoints: number;
};

// Slot ids for the four hands and the kitty: ascending, since a holding
// has no order.
function sortedSlots(cards: ReadonlyArray<Card>): number[] {
  return cards.map(cardSlot).sort((a, b) => a - b);
}

// Slot ids for a play, in the order the cards were submitted. NOT sorted,
// and the difference matters: `decomposeThrow` emits singleton components
// in the order the cards arrive, and `isThrowSafe` keeps the first
// beatable component of the smallest size, so which component the
// forced-lead penalty leaves on the table is a function of the order the
// leader listed their cards in. A sorted record could not reproduce it.
function playSlots(cards: ReadonlyArray<Card>): number[] {
  return cards.map(cardSlot);
}

// Full physical identity, so the two copies of a card in a 2-deck game are
// distinguishable when components are checked for disjointness.
function identityKey(c: Card): string {
  return `${cardKey(c)}#${c.deckId}`;
}

// A random multi-component same-suit lead, with no safety check. Returns
// null when the hand cannot form one in the suit drawn.
function randomThrow(
  rng: () => number,
  myCards: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): ReadonlyArray<Card> | null {
  const bySuit = new Map<EffectiveSuit, Card[]>();
  for (const c of myCards) {
    const es = effectiveSuit(c, ctx);
    const list = bySuit.get(es) ?? [];
    list.push(c);
    bySuit.set(es, list);
  }
  const suits = [...bySuit.values()].filter((cs) => cs.length >= 2);
  if (suits.length === 0) return null;
  const suitCards = suits[Math.floor(rng() * suits.length)]!;

  // Tractors first, then pairs, then singles -- the same component pool
  // the engine's own throw enumerator builds, minus the safety filter.
  const pieces: Array<ReadonlyArray<Card>> = [];
  for (const t of findTractors(suitCards, ctx, n)) pieces.push(t.cards);
  for (const p of findNKinds(suitCards, n)) pieces.push(p.cards);
  for (const c of suitCards) pieces.push([c]);
  for (let i = pieces.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [pieces[i], pieces[j]] = [pieces[j]!, pieces[i]!];
  }

  const want = 2 + Math.floor(rng() * 3);
  const used = new Set<string>();
  const chosen: Card[] = [];
  let taken = 0;
  for (const piece of pieces) {
    if (taken >= want) break;
    let disjoint = true;
    for (const c of piece) {
      if (used.has(identityKey(c))) {
        disjoint = false;
        break;
      }
    }
    if (!disjoint) continue;
    for (const c of piece) used.add(identityKey(c));
    chosen.push(...piece);
    taken += 1;
  }
  if (taken < 2) return null;
  // Components that happen to reassemble into one combo (two pairs on
  // consecutive rungs, say) are a tractor lead, not a throw; the engine
  // takes that path and the penalty never applies.
  if (classifyCombo(chosen, ctx, n).ok) return null;
  if (!decomposeThrow(chosen, ctx, n).ok) return null;
  return chosen;
}

class Dumper {
  private state: GameState;
  private readonly rng: () => number;

  constructor(seed: number) {
    this.rng = mulberry32(seed);
    this.state = createGame(defaultV1Config());
  }

  // Plays one hand end to end and returns its record. Landlord and both
  // team levels are redrawn per hand so the dump covers all 13 trump
  // ranks and all four landlord seats rather than one game's slice.
  runHand(handNo: number, stats: Stats): GameRecord {
    const config = defaultV1Config();
    const landlord = Math.floor(this.rng() * config.numPlayers) as Seat;
    const level = (): NaturalRank =>
      NATURAL_RANKS[Math.floor(this.rng() * NATURAL_RANKS.length)]!;
    this.state = {
      ...createGame(config),
      landlord,
      teamLevels: { A: level(), B: level() },
    };
    const seed = Math.floor(this.rng() * 0x7fffffff);
    this.apply({ kind: 'start_hand', seed });

    let position: {
      levelRank: number;
      trumpSuit: number;
      landlord: number;
      hands: number[][];
      kitty: number[];
    } | null = null;
    const plays: PlayRecord[] = [];
    const tricks: TrickRecord[] = [];

    for (let step = 0; step < MAX_STEPS_PER_HAND; step++) {
      const phase = this.state.phase;

      if (phase === 'declare_lockout' || phase === 'declare_open') {
        this.runDeclarePhase();
        continue;
      }
      if (phase === 'kitty_replace') {
        this.runKittyReplace();
        continue;
      }
      if (phase === 'play') {
        if (position === null) position = this.snapshot();
        plays.push(this.runPlay(stats));
        continue;
      }
      if (phase === 'trick_review') {
        this.apply({ kind: 'advance_trick' });
        const hand = this.state.currentHand;
        if (hand === null) throw new Error('advance_trick lost the hand');
        const t = hand.tricks[hand.tricks.length - 1];
        if (t === undefined) throw new Error('advance_trick recorded no trick');
        tricks.push({ leader: t.leader, winner: t.winner, points: t.points });
        continue;
      }
      if (phase === 'between_hands' || phase === 'game_over') break;
      throw new Error(`unexpected phase ${phase}`);
    }

    const hand = this.state.currentHand;
    if (hand === null) throw new Error('hand ended with no state');
    if (position === null) throw new Error('hand never reached the play phase');
    if (this.state.phase !== 'between_hands' && this.state.phase !== 'game_over') {
      throw new Error(`hand did not finish (phase ${this.state.phase})`);
    }
    const result = computeHandResult(hand.attackerPoints);

    stats.hands += 1;
    stats.plays += plays.length;
    stats.tricks += tricks.length;
    stats.attackerPoints += hand.attackerPoints;
    if (position.trumpSuit === 4) stats.noTrump += 1;

    return {
      hand: handNo,
      seed,
      decks: DECKS,
      levelRank: position.levelRank,
      trumpSuit: position.trumpSuit,
      landlord: position.landlord,
      hands: position.hands,
      kitty: position.kitty,
      plays,
      tricks,
      attackerPoints: hand.attackerPoints,
      winner: result.winner,
      levelDelta: result.levelDelta,
    };
  }

  // --- internals ---------------------------------------------------------

  // The position the Python simulator is handed: the state at the first
  // trick, after the kitty is buried and before anyone has played.
  private snapshot(): {
    levelRank: number;
    trumpSuit: number;
    landlord: number;
    hands: number[][];
    kitty: number[];
  } {
    const hand = this.state.currentHand;
    if (hand === null) throw new Error('snapshot with no hand');
    if (hand.trumpRank === null) throw new Error('snapshot with no trump rank');
    if (hand.landlord === null) throw new Error('snapshot with no landlord');
    const hands: number[][] = [];
    for (let seat = 0; seat < this.state.config.numPlayers; seat++) {
      hands.push(sortedSlots(hand.hands[seat as Seat] ?? []));
    }
    return {
      levelRank: NATURAL_RANKS.indexOf(hand.trumpRank),
      // Null means a joker declaration or a flipped joker: no-trump, which
      // the Python context stores as suit 4.
      trumpSuit: hand.trumpSuit === null ? 4 : SUITS.indexOf(hand.trumpSuit),
      landlord: hand.landlord,
      hands,
      kitty: sortedSlots(hand.kitty),
    };
  }

  private ctx(): TrumpContext {
    const hand = this.state.currentHand;
    if (hand === null || hand.trumpRank === null) {
      throw new Error('no trump context');
    }
    return { trumpRank: hand.trumpRank, trumpSuit: hand.trumpSuit };
  }

  private runPlay(stats: Stats): PlayRecord {
    const seat = nextToPlay(this.state);
    if (seat === null) throw new Error('play phase with no seat to act');
    const hand = this.state.currentHand;
    if (hand === null) throw new Error('play phase with no hand');
    const ctx = this.ctx();
    const isLead = hand.currentTrick === null;

    let requested: ReadonlyArray<Card> | null = null;
    if (isLead && this.rng() < THROW_PROB) {
      requested = randomThrow(this.rng, hand.hands[seat] ?? [], ctx, DECKS);
    }
    if (requested === null) {
      const actions = allLegalPlayActions(this.state, seat, {
        maxCandidates: MAX_CANDIDATES,
      });
      if (actions.length === 0) {
        throw new Error(`no legal action for seat ${seat}`);
      }
      requested = actions[Math.floor(this.rng() * actions.length)]!.cards;
    }
    // Counted after the choice rather than at the injection site: the
    // enumerator emits throws of its own, and its safety filter and the
    // reducer's do not agree (the enumerator lets the leader's partner
    // off the hook, `playCards` does not), so some of its "safe" throws
    // are blocked too. Both sources are throws; count both.
    if (isLead && !classifyCombo(requested, ctx, DECKS).ok) {
      stats.throwLeads += 1;
    }

    const result = applyEvent(this.state, {
      kind: 'play_cards',
      seat,
      cards: requested,
    });
    if (!result.ok) {
      throw new Error(`play_cards rejected: ${result.error}`);
    }
    const forced =
      result.notice !== undefined && result.notice.kind === 'forced_lead';
    this.state = result.state;

    const trick = this.state.currentHand?.currentTrick;
    const committed = trick?.plays[trick.plays.length - 1];
    if (committed === undefined) throw new Error('play left no trick entry');
    if (committed.seat !== seat) throw new Error('play landed on the wrong seat');

    // A shortened play and a forced-lead notice are the same event seen
    // two ways; if they ever disagree the dump is describing something
    // other than the penalty.
    if (forced !== committed.cards.length < requested.length) {
      throw new Error(
        `forced-lead notice ${forced} but played ${committed.cards.length} of ${requested.length}`,
      );
    }
    if (forced) stats.forcedLeads += 1;

    return {
      seat,
      req: playSlots(requested),
      got: playSlots(committed.cards),
    };
  }

  // Declaration and burial are delegated to a heuristic bot, exactly as
  // research/src/env.ts does: neither is part of what is being compared,
  // and a bot's choices put the play phase in positions a real game
  // reaches rather than uniformly random ones.
  private runDeclarePhase(): void {
    const numPlayers = this.state.config.numPlayers;
    for (let seat = 0; seat < numPlayers; seat++) {
      const action = pickBotAction(this.state, seat as Seat, PREPLAY);
      if (action.kind === 'wait') continue;
      const event = this.toEvent(action, seat as Seat);
      if (event === null) continue;
      const result = applyEvent(this.state, event);
      if (!result.ok) continue; // bot proposed something illegal; ignore it
      this.state = result.state;
      return;
    }
    if (this.state.phase === 'declare_lockout') {
      this.apply({
        kind: 'release_lockout',
        seed: Math.floor(this.rng() * 0x7fffffff),
      });
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
    const action = pickBotAction(this.state, seat, PREPLAY);
    if (action.kind !== 'replace_kitty') {
      throw new Error(`expected replace_kitty from bot, got ${action.kind}`);
    }
    this.apply({ kind: 'replace_kitty', seat, cards: action.cards });
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
    this.state = result.state;
  }
}

function main(): void {
  const hands = Number(process.argv[2] ?? '200');
  const seed = Number(process.argv[3] ?? '12345');
  const outPath = process.argv[4];
  if (!Number.isFinite(hands) || hands < 1) {
    throw new Error(`bad hand count ${process.argv[2]}`);
  }
  if (!Number.isFinite(seed)) throw new Error(`bad seed ${process.argv[3]}`);

  // writeSync on a file descriptor rather than process.stdout.write: the
  // latter is asynchronous when stdout is a pipe, and a dump this size
  // would otherwise buffer the whole run in memory.
  const fd = outPath === undefined ? 1 : openSync(outPath, 'w');
  const dumper = new Dumper(seed);
  const stats: Stats = {
    hands: 0,
    plays: 0,
    tricks: 0,
    forcedLeads: 0,
    throwLeads: 0,
    noTrump: 0,
    attackerPoints: 0,
  };
  const started = Date.now();
  let bytes = 0;

  try {
    // Lines are batched because one writeSync per hand is a syscall per
    // hand; at 2000 hands that is the dominant cost on Windows.
    let chunk = '';
    for (let h = 0; h < hands; h++) {
      chunk += `${JSON.stringify(dumper.runHand(h, stats))}\n`;
      if (chunk.length > 1 << 20) {
        bytes += Buffer.byteLength(chunk);
        writeSync(fd, chunk);
        chunk = '';
      }
    }
    if (chunk.length > 0) {
      bytes += Buffer.byteLength(chunk);
      writeSync(fd, chunk);
    }
  } finally {
    if (fd !== 1) closeSync(fd);
  }

  const elapsed = (Date.now() - started) / 1000;
  const line = (s: string): void => {
    process.stderr.write(`${s}\n`);
  };
  line(
    `hands            ${stats.hands} in ${elapsed.toFixed(2)}s = ${(stats.hands / elapsed).toFixed(1)} hands/s`,
  );
  line(`plays            ${stats.plays} (${(stats.plays / stats.hands).toFixed(1)}/hand)`);
  line(`tricks           ${stats.tricks} (${(stats.tricks / stats.hands).toFixed(1)}/hand)`);
  line(
    `throw leads      ${stats.throwLeads}, of which blocked ${stats.forcedLeads} ` +
      `(${(stats.forcedLeads / stats.hands).toFixed(2)}/hand)`,
  );
  line(
    `attacker points  mean ${(stats.attackerPoints / stats.hands).toFixed(1)}`,
  );
  line(`no-trump hands   ${stats.noTrump} (${((100 * stats.noTrump) / stats.hands).toFixed(1)}%)`);
  line(
    `wrote            ${(bytes / 1e6).toFixed(2)} MB to ${outPath ?? '<stdout>'}`,
  );
}

main();
