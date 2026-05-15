import { effectiveSuit, compareCards, isTrump, naturalRankIndex } from './ordering';
import { findNKinds, findTractors } from './combos';
import { classifyCombo, comboBeats, type ClassifiedCombo } from './classifier';
import { pointsForCard, sumPoints } from './scoring';
import {
  declarationTier,
  teamOf,
  type ActiveTrick,
  type GameState,
  type HandState,
  type Seat,
  type TeamId,
} from './state';
import { availableDeclarationsFor } from './declareEnum';
import { applyEvent, nextToPlay } from './reducer';
import { legalEvents } from './legalActions';
import { buildDeck } from './deck';
import { mulberry32, shuffle } from './prng';
import { decomposePlay, decomposedBeats } from './throws';
import {
  NATURAL_RANKS,
  cardEquals,
  cardKey,
  type Card,
  type EffectiveSuit,
  type NaturalRank,
  type StandardCard,
  type Suit,
  type TrumpContext,
} from './types';
import { EMBEDDED_WEIGHTS as EMBEDDED_RL_WEIGHTS } from './rl/mlp/embeddedWeights';
import { deserialize as deserializeRlPolicy } from './rl/mlp/policy';
import { makePolicyAgent as makeRlPolicyAgent } from './rl/mlp/agent';

export type BotAction =
  | { readonly kind: 'declare'; readonly cards: ReadonlyArray<Card> }
  | { readonly kind: 'pick_up_kitty' }
  | { readonly kind: 'replace_kitty'; readonly cards: ReadonlyArray<Card> }
  | { readonly kind: 'play_cards'; readonly cards: ReadonlyArray<Card> }
  | { readonly kind: 'wait' };

// Built-in bot strategies. Add a new one by extending this union and
// adding a branch to `pickBotAction`. The default 'lowball' is the
// minimal baseline used by the WS server for backwards compatibility.
export type BotType =
  | 'lowball'        // dump low cards / play lowest valid follow
  | 'aggressive'     // lead high, try to beat tricks on follow
  | 'point-hoarder'  // capture point cards when possible, dump non-points
  | 'random'         // uniformly random over a curated legal action set
  | 'smart-v0'       // copy of `smart` — deployment-pipeline smoke test
  | 'smart'          // multi-heuristic agent: team-aware follows + strong leads
  | 'smart-v2'       // smart + 1-ply search on follow decisions
  | 'smart-v3'       // smart-v2 + 2-ply search and search-based leading
  | 'smart-v4'       // smart-v3 + card tracking, better declare/kitty
  | 'smart-v5'       // smart-v4 + smart-v3 opponent model in the simulation
  | 'smart-v6'       // smart-v5 + smart-v4 opponent model in the simulation
  | 'smart-v7'       // smart-v6 + terminal opponent-cost heuristic
  | 'smart-v8'       // smart-v7 + adaptive cost penalty
  | 'smart-v9'       // smart-v8 + even stronger Ace penalty
  | 'smart-v10'      // smart-v9 with Ace penalty bumped 2.5 -> 3.0
  | 'smart-v11'      // smart-v9 + differentiated trump natural-rank penalty
  | 'smart-v12'      // smart-v9 + opponent void / no-pair tracking
  | 'smart-v13'      // smart-v9 with opponent hands masked / determinized
  | 'smart-v14'      // smart-v13 + v12's void/no-pair public-info signals
  | 'smart-v15'      // smart-v13 with N=3 determinizations averaged
  | 'smart-v16'      // smart-v13 with N=5 determinizations averaged
  | 'smart-v17'      // smart-v13 with N=10 determinizations averaged
  | 'smart-v18'      // v16 + decomposedBeats-aware winner + trump-throw candidate + hand-value lead bonus
  | 'smart-v19'      // v18 + relative (my-opp) hand-value lead bonus
  | 'smart-v20'      // v18 with constrained determinization (respects opponent voids from public play)
  | 'smart-v27'      // v18 + explicit "winning-shape" trump candidates when void in led suit
  | 'smart-v28'      // v18 + situational cost relaxation: scale down high-cost cards on big-points tricks the bot expects to win
  | 'smart-v29'      // v28 cost relaxation + v27 winning-shape candidates
  | 'rl-mlp';        // small-MLP policy trained via REINFORCE self-play

export const BOT_TYPES: ReadonlyArray<BotType> = [
  'lowball',
  'aggressive',
  'point-hoarder',
  'random',
  'smart-v0',
  'smart',
  'smart-v2',
  'smart-v3',
  'smart-v4',
  'smart-v5',
  'smart-v6',
  'smart-v7',
  'smart-v8',
  'smart-v9',
  'smart-v10',
  'smart-v11',
  'smart-v12',
  'smart-v13',
  'smart-v14',
  'smart-v15',
  'smart-v16',
  'smart-v17',
  'smart-v18',
  'smart-v19',
  'smart-v20',
  'smart-v27',
  'smart-v28',
  'smart-v29',
  'rl-mlp',
];

// User-facing subset shown in the lobby dropdown. Only the latest
// smart-bot generation (v13–v16) plus the random baseline. Earlier
// versions (smart-v0 … smart-v12) and the experimental heuristics
// (lowball / aggressive / point-hoarder / rl-mlp) remain in
// BOT_TYPES so they keep working over the protocol — they're useful
// for offline tournaments and tests — but they no longer surface in
// the lobby UI.
export const VISIBLE_BOT_TYPES: ReadonlyArray<BotType> = [
  'random',
  'smart-v16',
  'smart-v18',
  'smart-v28',
  'smart-v29',
];

export function botTypeLabel(t: BotType): string {
  switch (t) {
    case 'lowball':
      return 'Lowball (baseline)';
    case 'aggressive':
      return 'Aggressive';
    case 'point-hoarder':
      return 'Point hoarder';
    case 'random':
      return 'Random';
    case 'smart-v0':
      return 'v0: Atlas';
    case 'smart':
      return 'v1: Atlas';
    case 'smart-v2':
      return 'v2: Brook';
    case 'smart-v3':
      return 'v3: Cascade';
    case 'smart-v4':
      return 'v4: Delta';
    case 'smart-v5':
      return 'v5: Eddy';
    case 'smart-v6':
      return 'v6: Flow';
    case 'smart-v7':
      return 'v7: Gulf';
    case 'smart-v8':
      return 'v8: Harbor';
    case 'smart-v9':
      return 'v9: Inlet';
    case 'smart-v10':
      return 'v10: Jetty';
    case 'smart-v11':
      return 'v11: Kelp';
    case 'smart-v12':
      return 'v12: Lagoon';
    case 'smart-v13':
      return 'v13: Mirage';
    case 'smart-v14':
      return 'v14: Nebula';
    case 'smart-v15':
      return 'v15: Oasis';
    case 'smart-v16':
      return 'v16: Pier';
    case 'smart-v17':
      return 'v17: Quartz (N=10)';
    case 'smart-v18':
      return 'v18: Riptide (throw-trump aware + lead bonus)';
    case 'smart-v19':
      return 'v19: Sirocco (relative lead bonus)';
    case 'smart-v20':
      return 'v20: Tempest (void-aware determinization)';
    case 'smart-v27':
      return 'v27: Amazon (winning-shape candidates)';
    case 'smart-v28':
      return 'v28: Bayou (situational save-cost)';
    case 'smart-v29':
      return 'v29: Congo (winning candidates + situational save-cost)';
    case 'rl-mlp':
      return 'RL (MLP, REINFORCE self-play)';
  }
}

// Shared phase guards: declare, pick-up-kitty, kitty-replace, turn check.
// Returns a phase-specific action that's invariant across all bot
// strategies (declare with rank-R single, take kitty as landlord, bury
// 8 lowest, etc.). Returns null when the strategy must decide.
function defaultPrePlay(state: GameState, seat: Seat): BotAction | null {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    if (hand.standingDeclaration === null) {
      const myLevel = state.teamLevels[teamOf(state, seat)]!;
      const candidate = myCards.find(
        (c) => c.kind === 'standard' && c.rank === myLevel,
      );
      if (candidate !== undefined) {
        return { kind: 'declare', cards: [candidate] };
      }
    }
    return { kind: 'wait' };
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    const ctx: TrumpContext = {
      trumpRank: hand.trumpRank,
      trumpSuit: hand.trumpSuit,
    };
    // Bury the 8 lowest *non-point* cards if possible, otherwise lowest.
    const sorted = [...myCards].sort((a, b) => compareCards(a, b, ctx));
    const nonPoint = sorted.filter((c) => pointsForCard(c) === 0);
    const chosen =
      nonPoint.length >= state.config.kittySize
        ? nonPoint.slice(0, state.config.kittySize)
        : sorted.slice(0, state.config.kittySize);
    return { kind: 'replace_kitty', cards: chosen };
  }
  if (state.phase === 'play') {
    if (hand.trumpRank === null) return { kind: 'wait' };
    if (turnSeat(state) !== seat) return { kind: 'wait' };
    return null; // caller plays
  }
  return { kind: 'wait' };
}

function turnSeat(state: GameState): Seat | null {
  const hand = state.currentHand;
  if (hand === null) return null;
  if (hand.currentTrick === null) {
    return hand.tricks.length === 0
      ? hand.landlord
      : hand.tricks[hand.tricks.length - 1]!.winner;
  }
  return (
    (hand.currentTrick.leader + hand.currentTrick.plays.length) %
    state.config.numPlayers
  );
}

export function pickBotAction(
  state: GameState,
  seat: Seat,
  type: BotType = 'lowball',
): BotAction {
  if (type === 'smart' || type === 'smart-v0') {
    // smart-v0 is intentionally identical to smart — used as a smoke
    // test for the bot-deployment pipeline (engine → web/UI dropdown
    // and WS server).
    return pickSmart(state, seat);
  }
  if (type === 'smart-v2') {
    return pickSmartV2(state, seat);
  }
  if (type === 'smart-v3') {
    return pickSmartV3(state, seat);
  }
  if (type === 'smart-v4') {
    return pickSmartV4(state, seat);
  }
  if (type === 'smart-v5') {
    return pickSmartV5(state, seat);
  }
  if (type === 'smart-v6') {
    return pickSmartV6(state, seat);
  }
  if (type === 'smart-v7') {
    return pickSmartV7(state, seat);
  }
  if (type === 'smart-v8') {
    return pickSmartV8(state, seat);
  }
  if (type === 'smart-v9') {
    return pickSmartV9(state, seat);
  }
  if (type === 'smart-v10') {
    return pickSmartV10(state, seat);
  }
  if (type === 'smart-v11') {
    return pickSmartV11(state, seat);
  }
  if (type === 'smart-v12') {
    return pickSmartV12(state, seat);
  }
  if (type === 'smart-v13') {
    return pickSmartV13(state, seat);
  }
  if (type === 'smart-v14') {
    return pickSmartV14(state, seat);
  }
  if (type === 'smart-v15') {
    return pickSmartV15(state, seat);
  }
  if (type === 'smart-v16') {
    return pickSmartV16(state, seat);
  }
  if (type === 'smart-v17') {
    return pickSmartV17(state, seat);
  }
  if (type === 'smart-v18') {
    return pickSmartV18(state, seat);
  }
  if (type === 'smart-v19') {
    return pickSmartV19(state, seat);
  }
  if (type === 'smart-v20') {
    return pickSmartV20(state, seat);
  }
  if (type === 'smart-v27') {
    return pickSmartV27(state, seat);
  }
  if (type === 'smart-v28') {
    return pickSmartV28(state, seat);
  }
  if (type === 'smart-v29') {
    return pickSmartV29(state, seat);
  }
  if (type === 'rl-mlp') {
    return rlMlpAgent()(state, seat);
  }
  const pre = defaultPrePlay(state, seat);
  if (pre !== null) return pre;

  const hand = state.currentHand!;
  const myCards = hand.hands[seat] ?? [];
  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank!,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  switch (type) {
    case 'lowball':
      return pickLowball(myCards, trick, ctx, n);
    case 'aggressive':
      return pickAggressive(myCards, trick, ctx, n);
    case 'point-hoarder':
      return pickPointHoarder(state, seat, myCards, trick, ctx, n);
    case 'random':
      return pickRandom(myCards, trick, ctx, n);
  }
}

// ---------- Strategy: rl-mlp (REINFORCE-trained MLP) ----------------------
// Defers to baseline lowball for declare/kitty/non-turn phases; runs the
// trained policy on play decisions. Weights are embedded at build time
// (see `rl/mlp/embeddedWeights.ts`, generated by `train.ts`).
let _rlMlpAgent: ((s: GameState, seat: Seat) => BotAction) | null = null;
function rlMlpAgent(): (s: GameState, seat: Seat) => BotAction {
  if (_rlMlpAgent === null) {
    const policy = deserializeRlPolicy(EMBEDDED_RL_WEIGHTS);
    _rlMlpAgent = makeRlPolicyAgent({ policy, mode: 'eval' });
  }
  return _rlMlpAgent;
}

// ---------- Strategy: lowball ---------------------------------------------
// Original baseline. Lead lowest single. Follow with lowest led-suit;
// match pair structure if available; otherwise lowest filler.
function pickLowball(
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const sorted = [...myCards].sort((a, b) => compareCards(a, b, ctx));
  if (trick === null) {
    return { kind: 'play_cards', cards: [sorted[0]!] };
  }
  return fillFollow(myCards, sorted, trick, ctx, n, 'low');
}

// ---------- Strategy: aggressive -----------------------------------------
// Lead with the highest available single (try to dump strong cards).
// On follow, play the highest led-suit cards in the hope of taking the
// trick; non-led-suit fillers are lowest.
function pickAggressive(
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const sorted = [...myCards].sort((a, b) => compareCards(a, b, ctx));
  if (trick === null) {
    return { kind: 'play_cards', cards: [sorted[sorted.length - 1]!] };
  }
  return fillFollow(myCards, sorted, trick, ctx, n, 'high');
}

// ---------- Strategy: point-hoarder --------------------------------------
// Track who currently leads the trick. If a teammate is winning, dump
// point cards (5/10/K) to feed the team's pile. If an opponent is
// winning and we can plausibly beat the trick, try high; otherwise dump
// the lowest non-point card.
function pickPointHoarder(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const sorted = [...myCards].sort((a, b) => compareCards(a, b, ctx));
  if (trick === null) {
    // Lead the lowest non-point card if available.
    const nonPoint = sorted.filter((c) => pointsForCard(c) === 0);
    const pick = nonPoint[0] ?? sorted[0]!;
    return { kind: 'play_cards', cards: [pick] };
  }
  const ledSuit = effectiveSuit(trick.plays[0]!.cards[0]!, ctx);
  const myTeam = teamOf(state, seat);
  // Current "winner" of the in-flight trick. Lead seed; later plays
  // override only if they strictly beat per `comboBeats`.
  let bestSeat = trick.plays[0]!.seat;
  let bestCombo = classifyCombo(trick.plays[0]!.cards, ctx, n);
  for (let i = 1; i < trick.plays.length; i++) {
    const p = trick.plays[i]!;
    const cls = classifyCombo(p.cards, ctx, n);
    if (bestCombo.ok && cls.ok && comboBeats(cls.combo, bestCombo.combo) > 0) {
      bestSeat = p.seat;
      bestCombo = cls;
    }
  }
  const teammateLeading = teamOf(state, bestSeat) === myTeam;
  const preference: 'low' | 'high' | 'point-dump' = teammateLeading
    ? 'point-dump'
    : 'high';
  return fillFollow(myCards, sorted, trick, ctx, n, preference, ledSuit);
}

// ---------- Strategy: random ---------------------------------------------
// Random legal-ish move. For leading: pick a random card. For following:
// generate a legal follow with random ordering of fillers.
function pickRandom(
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  if (trick === null) {
    const c = myCards[Math.floor(Math.random() * myCards.length)]!;
    return { kind: 'play_cards', cards: [c] };
  }
  const sorted = [...myCards].sort((a, b) => compareCards(a, b, ctx));
  // Use the standard follow filler but with the input pool shuffled.
  const shuffled = [...myCards].sort(() => Math.random() - 0.5);
  return fillFollow(shuffled, sorted, trick, ctx, n, 'low');
}

// ---------- Shared follow helper -----------------------------------------
// Builds a legal follow respecting structural obligations:
//   - Tractor lead of length k: include a length-k tractor in led suit if
//     held; else include as many pairs as we can up to k.
//   - Pair lead: include a pair in led suit if held.
//   - Then fill remaining slots with led-suit cards, then non-led-suit.
//
// `preference`:
//   'low'         — minimize value (dump cheap cards)
//   'high'        — maximize value (try to beat the trick)
//   'point-dump'  — prefer point cards (5/10/K) when filling non-led-suit
//                   slots; led-suit slots use lowest.
function fillFollow(
  pool: ReadonlyArray<Card>,
  sortedLow: ReadonlyArray<Card>,
  trick: { leader: Seat; plays: ReadonlyArray<{ cards: ReadonlyArray<Card> }> },
  ctx: TrumpContext,
  n: number,
  preference: 'low' | 'high' | 'point-dump',
  ledSuitOverride?: ReturnType<typeof effectiveSuit>,
): BotAction {
  const lead = trick.plays[0]!;
  const ledSuit = ledSuitOverride ?? effectiveSuit(lead.cards[0]!, ctx);
  const leadCount = lead.cards.length;
  const leadClass = classifyCombo(lead.cards, ctx, n);

  let leadIsPair = false;
  let leadTractorLen = 0;
  if (leadClass.ok) {
    if (leadClass.combo.kind === 'nkind' && leadClass.combo.n === 2) {
      leadIsPair = true;
    } else if (leadClass.combo.kind === 'tractor') {
      leadTractorLen = leadClass.combo.length;
    }
  }

  const myInLed = pool.filter((c) => effectiveSuit(c, ctx) === ledSuit);
  const chosen: Card[] = [];
  const isUsed = (c: Card): boolean => chosen.some((x) => cardEquals(x, c));
  const addCard = (c: Card): void => {
    if (!isUsed(c)) chosen.push(c);
  };

  // Obligation: tractor match for tractor leads.
  if (leadTractorLen >= 2) {
    const matching = findTractors(myInLed, ctx, n).filter(
      (t) => t.length === leadTractorLen,
    );
    if (matching.length > 0) {
      const sorted = matching
        .slice()
        .sort((a, b) =>
          compareCards(
            a.cards[a.cards.length - 1]!,
            b.cards[b.cards.length - 1]!,
            ctx,
          ),
        );
      const pick =
        preference === 'high' ? sorted[sorted.length - 1]! : sorted[0]!;
      for (const c of pick.cards) addCard(c);
    } else {
      // No matching tractor — must include as many pairs in led suit as we can.
      const pairs = findNKinds(myInLed, 2);
      const sortedPairs = pairs
        .slice()
        .sort((a, b) => compareCards(a.cards[0]!, b.cards[0]!, ctx));
      const ordered =
        preference === 'high' ? [...sortedPairs].reverse() : sortedPairs;
      let added = 0;
      for (const p of ordered) {
        if (added >= leadTractorLen) break;
        if (chosen.length + p.cards.length > leadCount) break;
        for (const c of p.cards) addCard(c);
        added++;
      }
    }
  } else if (leadIsPair) {
    const pairs = findNKinds(myInLed, 2);
    if (pairs.length > 0) {
      const sortedPairs = pairs
        .slice()
        .sort((a, b) => compareCards(a.cards[0]!, b.cards[0]!, ctx));
      const picked =
        preference === 'high'
          ? sortedPairs[sortedPairs.length - 1]!
          : sortedPairs[0]!;
      for (const c of picked.cards) addCard(c);
    }
  }

  // Led-suit fillers.
  if (chosen.length < leadCount) {
    const inLedAsc = sortedLow.filter(
      (c) => effectiveSuit(c, ctx) === ledSuit && !isUsed(c),
    );
    const inLed = preference === 'high' ? [...inLedAsc].reverse() : inLedAsc;
    for (const c of inLed) {
      if (chosen.length >= leadCount) break;
      addCard(c);
    }
  }

  if (chosen.length >= leadCount) {
    return { kind: 'play_cards', cards: chosen };
  }

  // Non-led-suit fillers.
  const remainingPool = sortedLow.filter(
    (c) => effectiveSuit(c, ctx) !== ledSuit && !isUsed(c),
  );
  let fillerOrder: ReadonlyArray<Card>;
  if (preference === 'point-dump') {
    fillerOrder = [...remainingPool].sort((a, b) => {
      const dp = pointsForCard(b) - pointsForCard(a);
      if (dp !== 0) return dp;
      return compareCards(a, b, ctx);
    });
  } else if (preference === 'high') {
    fillerOrder = [...remainingPool].reverse();
  } else {
    fillerOrder = remainingPool;
  }
  for (const c of fillerOrder) {
    if (chosen.length >= leadCount) break;
    addCard(c);
  }
  return { kind: 'play_cards', cards: chosen };
}

// =========================================================================
// Strategy: smart
// =========================================================================
//
// A multi-heuristic agent designed to outperform the random / lowball
// baselines.
//
// Pre-play:
//   - Declare trump in the suit where we're longest, escalating to pair
//     (or joker pair) when held. Don't override a partner's declaration.
//   - Kitty burial: try to void a short side suit while preserving aces,
//     pairs, and point cards in the playable hand.
//
// Play (leading):
//   - Lead from strength: non-trump tractor > ace pair > ace single >
//     other high pairs > low non-point filler.
//
// Play (following): team-aware
//   - If teammate is currently winning, dump points (especially 10s/Ks).
//   - If an opponent is winning and we can beat the trick in led suit,
//     beat with the smallest combo that wins.
//   - If void in led suit and opponent is winning a trick with points,
//     trump cheaply.
//   - Otherwise, dump the lowest non-trump non-point card.

function pickSmart(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  if (trick === null) {
    return smartLead(myCards, ctx, n);
  }
  return smartFollow(state, seat, myCards, trick, ctx, n);
}

// ---- smart: declare -----------------------------------------------------

function smartDeclare(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  hand: HandState,
): BotAction {
  const R: NaturalRank | null =
    state.landlord !== null
      ? hand.trumpRank
      : state.teamLevels[teamOf(state, seat)]!;
  if (R === null) return { kind: 'wait' };

  const standing = hand.standingDeclaration;
  // Never override our own team's standing declaration.
  if (
    standing !== null &&
    teamOf(state, standing.seat) === teamOf(state, seat)
  ) {
    return { kind: 'wait' };
  }

  const avail = availableDeclarationsFor(myCards, R, standing);
  if (avail.length === 0) return { kind: 'wait' };

  // Suit strength = number of cards in that suit, excluding rank-R
  // (which become trump regardless).
  const suitStrength: Record<Suit, number> = {
    spades: 0,
    hearts: 0,
    diamonds: 0,
    clubs: 0,
  };
  for (const c of myCards) {
    if (c.kind === 'standard' && c.rank !== R) suitStrength[c.suit]++;
  }

  type SuitDecl = Extract<
    ReturnType<typeof availableDeclarationsFor>[number],
    { suit: Suit }
  >;
  const pairs = avail.filter((a): a is SuitDecl & { kind: 'pair' } => a.kind === 'pair');
  const singles = avail.filter(
    (a): a is SuitDecl & { kind: 'single' } => a.kind === 'single',
  );
  const bigJokerPair = avail.find(
    (a) => a.kind === 'joker-pair' && a.jokerRank === 'big',
  );
  const smallJokerPair = avail.find(
    (a) => a.kind === 'joker-pair' && a.jokerRank === 'small',
  );

  pairs.sort((a, b) => suitStrength[b.suit] - suitStrength[a.suit]);
  singles.sort((a, b) => suitStrength[b.suit] - suitStrength[a.suit]);

  if (standing !== null) {
    // Override an opposing-team standing declaration with minimal escalation.
    const standingT = declarationTier(standing);
    if (standingT < 2 && pairs.length > 0) {
      return { kind: 'declare', cards: [...pairs[0]!.cards] };
    }
    if (standingT < 3 && smallJokerPair) {
      return { kind: 'declare', cards: [...smallJokerPair.cards] };
    }
    if (standingT < 4 && bigJokerPair) {
      return { kind: 'declare', cards: [...bigJokerPair.cards] };
    }
    return { kind: 'wait' };
  }

  // Fresh declaration: lead with whatever we have, preferring strength.
  if (pairs.length > 0) {
    return { kind: 'declare', cards: [...pairs[0]!.cards] };
  }
  if (singles.length > 0) {
    return { kind: 'declare', cards: [...singles[0]!.cards] };
  }
  if (bigJokerPair) return { kind: 'declare', cards: [...bigJokerPair.cards] };
  if (smallJokerPair) {
    return { kind: 'declare', cards: [...smallJokerPair.cards] };
  }
  return { kind: 'wait' };
}

// ---- smart: kitty burial ------------------------------------------------

function smartBury(
  state: GameState,
  hand: HandState,
  myCards: ReadonlyArray<Card>,
): BotAction {
  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank!,
    trumpSuit: hand.trumpSuit,
  };
  const kittySize = state.config.kittySize;
  const n = state.config.numDecks;

  const nonTrumpCards: StandardCard[] = [];
  const trumpCards: Card[] = [];
  for (const c of myCards) {
    if (isTrump(c, ctx)) trumpCards.push(c);
    else if (c.kind === 'standard') nonTrumpCards.push(c);
  }

  // Pair-tag every card so we don't blindly break pairs when burying.
  const pairKeys = new Set(
    findNKinds(nonTrumpCards, n).map((p) => cardKey(p.cards[0]!)),
  );

  // Group non-trumps by suit.
  const bySuit = new Map<Suit, StandardCard[]>();
  for (const c of nonTrumpCards) {
    const list = bySuit.get(c.suit) ?? [];
    list.push(c);
    bySuit.set(c.suit, list);
  }
  for (const list of bySuit.values()) {
    list.sort((a, b) => compareCards(a, b, ctx));
  }

  // Per-card burial score (lower = more burial-worthy).
  function score(c: StandardCard): number {
    let s = naturalRankIndex(c.rank);
    s += pointsForCard(c) * 8;
    if (pairKeys.has(cardKey(c))) s += 30;
    if (c.rank === 'A') s += 200;
    return s;
  }

  const chosen: Card[] = [];
  const chosenKeys = new Set<string>();
  const includeCard = (c: Card): void => {
    const k = identityKey(c);
    if (chosenKeys.has(k)) return;
    chosen.push(c);
    chosenKeys.add(k);
  };

  // Pass 1: greedily void short side suits (where the suit's cards are
  // safe to bury — no aces).
  const suitsAsc = [...bySuit.entries()].sort(
    (a, b) => a[1].length - b[1].length,
  );
  for (const [, cards] of suitsAsc) {
    if (cards.length === 0) continue;
    if (chosen.length + cards.length > kittySize) continue;
    if (cards.some((c) => c.rank === 'A')) continue;
    // Don't bury an entire suit if doing so means tossing a K/10 pair
    // (those are pricey to lose to attackers' kitty bonus).
    const points = sumPoints(cards);
    if (points >= 20 && cards.length <= 4) continue;
    for (const c of cards) includeCard(c);
  }

  // Pass 2: fill the rest of the kitty with lowest-scored cards across
  // the remaining non-trumps.
  if (chosen.length < kittySize) {
    const remaining = nonTrumpCards.filter(
      (c) => !chosenKeys.has(identityKey(c)),
    );
    remaining.sort((a, b) => score(a) - score(b));
    for (const c of remaining) {
      if (chosen.length >= kittySize) break;
      includeCard(c);
    }
  }

  // Pass 3 (rare): bury lowest trumps to fill (e.g., very trump-heavy hand).
  if (chosen.length < kittySize) {
    const trumpSorted = [...trumpCards].sort((a, b) => compareCards(a, b, ctx));
    for (const c of trumpSorted) {
      if (chosen.length >= kittySize) break;
      includeCard(c);
    }
  }

  return { kind: 'replace_kitty', cards: chosen.slice(0, kittySize) };
}

// ---- smart: leading -----------------------------------------------------

function smartLead(
  myCards: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const sorted = [...myCards].sort((a, b) => compareCards(a, b, ctx));

  // Maximal non-trump tractors first (only longest is "best" — shorter
  // sub-tractors share cards, so we'd never lead them while a longer one
  // exists).
  const tractors = findTractors(myCards, ctx, n).filter(
    (t) => effectiveSuit(t.cards[0]!, ctx) !== 'trump',
  );
  if (tractors.length > 0) {
    const best = tractors.reduce((a, b) => {
      if (b.length !== a.length) return b.length > a.length ? b : a;
      const cmp = compareCards(
        b.cards[b.cards.length - 1]!,
        a.cards[a.cards.length - 1]!,
        ctx,
      );
      return cmp > 0 ? b : a;
    });
    return { kind: 'play_cards', cards: [...best.cards] };
  }

  // Non-trump pairs.
  const pairs = findNKinds(myCards, n).filter(
    (p) => effectiveSuit(p.cards[0]!, ctx) !== 'trump',
  );

  // Ace pairs.
  const acePairs = pairs.filter(
    (p) => p.cards[0]!.kind === 'standard' && p.cards[0]!.rank === 'A',
  );
  if (acePairs.length > 0) {
    return { kind: 'play_cards', cards: [...acePairs[0]!.cards] };
  }

  // Ace singles in non-trump.
  const nonTrumpAces = myCards.filter(
    (c) => c.kind === 'standard' && c.rank === 'A' && !isTrump(c, ctx),
  );
  if (nonTrumpAces.length > 0) {
    return { kind: 'play_cards', cards: [nonTrumpAces[0]!] };
  }

  // High non-trump pairs (next best after aces).
  if (pairs.length > 0) {
    pairs.sort((a, b) => compareCards(b.cards[0]!, a.cards[0]!, ctx));
    return { kind: 'play_cards', cards: [...pairs[0]!.cards] };
  }

  // Lowest non-point non-trump single — keeps points and trumps for later.
  const filler = sorted.filter(
    (c) => !isTrump(c, ctx) && pointsForCard(c) === 0,
  );
  if (filler.length > 0) {
    return { kind: 'play_cards', cards: [filler[0]!] };
  }

  // Lowest non-trump (point card).
  const nonTrump = sorted.filter((c) => !isTrump(c, ctx));
  if (nonTrump.length > 0) {
    return { kind: 'play_cards', cards: [nonTrump[0]!] };
  }

  // Only trumps left — lead lowest trump.
  return { kind: 'play_cards', cards: [sorted[0]!] };
}

// ---- smart: following ---------------------------------------------------

function smartFollow(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const lead = trick.plays[0]!;
  const leadCount = lead.cards.length;
  const leadCls = classifyCombo(lead.cards, ctx, n);
  const ledSuit = leadCls.ok
    ? leadCls.combo.effectiveSuit
    : effectiveSuit(lead.cards[0]!, ctx);

  // Current winner of the trick so far.
  let winnerSeat: Seat = lead.seat;
  let bestCombo: ClassifiedCombo | null = leadCls.ok ? leadCls.combo : null;
  for (let i = 1; i < trick.plays.length; i++) {
    const p = trick.plays[i]!;
    const cls = classifyCombo(p.cards, ctx, n);
    if (cls.ok && bestCombo && comboBeats(cls.combo, bestCombo) > 0) {
      winnerSeat = p.seat;
      bestCombo = cls.combo;
    }
  }
  const myTeam = teamOf(state, seat);
  const teammateWinning = teamOf(state, winnerSeat) === myTeam;
  const trickPoints = sumPoints(trick.plays.flatMap((p) => [...p.cards]));
  const numPlayers = state.config.numPlayers;
  const isLastSeat = trick.plays.length === numPlayers - 1;

  const myInLed = myCards.filter((c) => effectiveSuit(c, ctx) === ledSuit);

  // ---- Tractor lead --------------------------------------------------
  if (leadCls.ok && leadCls.combo.kind === 'tractor') {
    const k = leadCls.combo.length;
    const matchingTractors = findTractors(myInLed, ctx, n).filter(
      (t) => t.length === k,
    );

    if (matchingTractors.length > 0) {
      matchingTractors.sort((a, b) =>
        compareCards(
          a.cards[a.cards.length - 1]!,
          b.cards[b.cards.length - 1]!,
          ctx,
        ),
      );
      if (!teammateWinning && bestCombo) {
        for (const t of matchingTractors) {
          const cls = classifyCombo(t.cards, ctx, n);
          if (cls.ok && comboBeats(cls.combo, bestCombo) > 0) {
            return { kind: 'play_cards', cards: [...t.cards] };
          }
        }
      }
      // Default: lowest matching tractor (saves bigger combos for later).
      return { kind: 'play_cards', cards: [...matchingTractors[0]!.cards] };
    }

    // No tractor match. Build a follow that satisfies pair-and-fill
    // obligations: include all pairs in led suit, then fill with led
    // suit, then with non-led-suit dump.
    return buildGenericFollow(
      state,
      myCards,
      myInLed,
      leadCount,
      ledSuit,
      ctx,
      n,
      teammateWinning,
      trickPoints,
      isLastSeat,
      bestCombo,
    );
  }

  // ---- Pair lead -----------------------------------------------------
  if (leadCls.ok && leadCls.combo.kind === 'nkind' && leadCls.combo.n === 2) {
    const pairsInLed = findNKinds(myInLed, n);
    if (pairsInLed.length > 0) {
      pairsInLed.sort((a, b) =>
        compareCards(a.cards[0]!, b.cards[0]!, ctx),
      );
      if (!teammateWinning && bestCombo) {
        for (const p of pairsInLed) {
          const cls = classifyCombo(p.cards, ctx, n);
          if (cls.ok && comboBeats(cls.combo, bestCombo) > 0) {
            return { kind: 'play_cards', cards: [...p.cards] };
          }
        }
      }
      return { kind: 'play_cards', cards: [...pairsInLed[0]!.cards] };
    }
    return buildGenericFollow(
      state,
      myCards,
      myInLed,
      leadCount,
      ledSuit,
      ctx,
      n,
      teammateWinning,
      trickPoints,
      isLastSeat,
      bestCombo,
    );
  }

  // ---- Single lead ---------------------------------------------------
  if (leadCls.ok && leadCls.combo.kind === 'single') {
    if (myInLed.length > 0) {
      const sortedLed = [...myInLed].sort((a, b) => compareCards(a, b, ctx));
      if (!teammateWinning && bestCombo) {
        // Beat with the smallest card in led suit that beats.
        for (const c of sortedLed) {
          const cls = classifyCombo([c], ctx, n);
          if (cls.ok && comboBeats(cls.combo, bestCombo) > 0) {
            return { kind: 'play_cards', cards: [c] };
          }
        }
        // Can't beat: dump lowest non-point led-suit card.
        const np = sortedLed.filter((c) => pointsForCard(c) === 0);
        return { kind: 'play_cards', cards: [(np[0] ?? sortedLed[0])!] };
      }
      // Teammate winning: dump the highest-value point card we hold in
      // led suit (so points stick to our pile).
      const points = sortedLed
        .filter((c) => pointsForCard(c) > 0)
        .sort((a, b) => pointsForCard(b) - pointsForCard(a));
      if (points.length > 0) {
        return { kind: 'play_cards', cards: [points[0]!] };
      }
      return { kind: 'play_cards', cards: [sortedLed[0]!] };
    }

    // Void in led suit — free to play any single card.
    return chooseSingleDiscard(
      myCards,
      ctx,
      n,
      teammateWinning,
      trickPoints,
      isLastSeat,
      bestCombo,
    );
  }

  // ---- Throw lead (auto-wins for leader) -----------------------------
  return buildGenericFollow(
    state,
    myCards,
    myInLed,
    leadCount,
    ledSuit,
    ctx,
    n,
    teammateWinning,
    trickPoints,
    isLastSeat,
    bestCombo,
  );
}

// Helpers for smartFollow ------------------------------------------------

function chooseSingleDiscard(
  myCards: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
  teammateWinning: boolean,
  trickPoints: number,
  isLastSeat: boolean,
  bestCombo: ClassifiedCombo | null,
): BotAction {
  const trumps = myCards.filter((c) => isTrump(c, ctx));
  const nonTrump = myCards.filter((c) => !isTrump(c, ctx));
  const sortedNonTrump = [...nonTrump].sort((a, b) => compareCards(a, b, ctx));
  const sortedTrump = [...trumps].sort((a, b) => compareCards(a, b, ctx));

  // Teammate winning → dump the heaviest point card from non-trump.
  if (teammateWinning) {
    const points = nonTrump
      .filter((c) => pointsForCard(c) > 0)
      .sort((a, b) => pointsForCard(b) - pointsForCard(a));
    if (points.length > 0) {
      return { kind: 'play_cards', cards: [points[0]!] };
    }
    const np = sortedNonTrump.filter((c) => pointsForCard(c) === 0);
    if (np.length > 0) return { kind: 'play_cards', cards: [np[0]!] };
    if (sortedNonTrump.length > 0) {
      return { kind: 'play_cards', cards: [sortedNonTrump[0]!] };
    }
    return { kind: 'play_cards', cards: [sortedTrump[0]!] };
  }

  // Opponent winning. Consider trumping if there are points worth taking.
  const opponentBestIsNonTrump =
    bestCombo !== null && bestCombo.effectiveSuit !== 'trump';
  const shouldTrump =
    opponentBestIsNonTrump &&
    sortedTrump.length > 0 &&
    (trickPoints >= 10 || (isLastSeat && trickPoints >= 5));
  if (shouldTrump) {
    // Use the smallest trump that actually beats the current best.
    for (const c of sortedTrump) {
      const cls = classifyCombo([c], ctx, n);
      if (cls.ok && bestCombo && comboBeats(cls.combo, bestCombo) > 0) {
        return { kind: 'play_cards', cards: [c] };
      }
    }
  }

  // Dump lowest non-trump non-point.
  const np = sortedNonTrump.filter((c) => pointsForCard(c) === 0);
  if (np.length > 0) {
    return { kind: 'play_cards', cards: [np[0]!] };
  }
  if (sortedNonTrump.length > 0) {
    return { kind: 'play_cards', cards: [sortedNonTrump[0]!] };
  }
  return { kind: 'play_cards', cards: [sortedTrump[0]!] };
}

function buildGenericFollow(
  _state: GameState,
  myCards: ReadonlyArray<Card>,
  myInLed: ReadonlyArray<Card>,
  leadCount: number,
  ledSuit: ReturnType<typeof effectiveSuit>,
  ctx: TrumpContext,
  n: number,
  teammateWinning: boolean,
  _trickPoints: number,
  _isLastSeat: boolean,
  _bestCombo: ClassifiedCombo | null,
): BotAction {
  const chosen: Card[] = [];
  const chosenKeys = new Set<string>();
  const add = (c: Card): boolean => {
    const k = identityKey(c);
    if (chosenKeys.has(k)) return false;
    chosen.push(c);
    chosenKeys.add(k);
    return true;
  };

  // Step 1: include every pair we have in led suit (pair obligation).
  const pairsInLed = findNKinds(myInLed, n);
  pairsInLed.sort((a, b) => compareCards(a.cards[0]!, b.cards[0]!, ctx));
  for (const p of pairsInLed) {
    if (chosen.length + p.cards.length > leadCount) break;
    for (const c of p.cards) add(c);
  }

  // Step 2: fill with led-suit singles (lowest first when dumping;
  // highest first when teammate is winning, to capture points).
  const sortedLed = [...myInLed].sort((a, b) => compareCards(a, b, ctx));
  const orderLed = teammateWinning ? [...sortedLed].reverse() : sortedLed;
  for (const c of orderLed) {
    if (chosen.length >= leadCount) break;
    add(c);
  }

  // Step 3: fill the rest with non-led cards.
  if (chosen.length < leadCount) {
    const remaining = myCards.filter(
      (c) => effectiveSuit(c, ctx) !== ledSuit,
    );
    let order: Card[];
    if (teammateWinning) {
      // Dump points (non-trump first), then trumps as last resort.
      order = [...remaining].sort((a, b) => {
        const aT = isTrump(a, ctx) ? 1 : 0;
        const bT = isTrump(b, ctx) ? 1 : 0;
        if (aT !== bT) return aT - bT;
        const dp = pointsForCard(b) - pointsForCard(a);
        if (dp !== 0) return dp;
        return compareCards(a, b, ctx);
      });
    } else {
      // Save trumps, dump cheap non-points first.
      order = [...remaining].sort((a, b) => {
        const aT = isTrump(a, ctx) ? 1 : 0;
        const bT = isTrump(b, ctx) ? 1 : 0;
        if (aT !== bT) return aT - bT;
        const aP = pointsForCard(a) > 0 ? 1 : 0;
        const bP = pointsForCard(b) > 0 ? 1 : 0;
        if (aP !== bP) return aP - bP;
        return compareCards(a, b, ctx);
      });
    }
    for (const c of order) {
      if (chosen.length >= leadCount) break;
      add(c);
    }
  }

  return { kind: 'play_cards', cards: chosen };
}

function identityKey(c: Card): string {
  if (c.kind === 'joker') return `J${c.rank}-${c.deckId}`;
  return `S${c.suit}${c.rank}-${c.deckId}`;
}

// =========================================================================
// Strategy: smart-v2
// =========================================================================
//
// Builds on smart-v1 with a one-ply trick search for follow decisions:
//
//   1. **Pre-play (declare, kitty bury)**: reuse v1.
//   2. **Leading**: reuse v1.
//   3. **Following**: enumerate candidate follows, then for each candidate
//      simulate the rest of the trick with v1's policy filling the other
//      seats. Score = points captured by our team in the trick, with a
//      small penalty for using trumps and aces. Pick the best.
//   4. **Last-seat plays**: use v1's smart-v1 follow (already optimal —
//      simulation reduces to evaluating a known outcome).
//
// The search is one-ply (one trick) — fast enough to run thousands of
// games in seconds while giving v1 a tangible upgrade.

function pickSmartV2(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  if (trick === null) {
    return smartLead(myCards, ctx, n);
  }
  return smartV2SearchFollow(state, seat, myCards, trick, ctx, n);
}

// ---- smart-v2: declare --------------------------------------------------

function smartV2Declare(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  hand: HandState,
): BotAction {
  const R: NaturalRank | null =
    state.landlord !== null
      ? hand.trumpRank
      : state.teamLevels[teamOf(state, seat)]!;
  if (R === null) return { kind: 'wait' };

  const standing = hand.standingDeclaration;
  if (
    standing !== null &&
    teamOf(state, standing.seat) === teamOf(state, seat)
  ) {
    return { kind: 'wait' };
  }

  const avail = availableDeclarationsFor(myCards, R, standing);
  if (avail.length === 0) return { kind: 'wait' };

  const suitStrength: Record<Suit, number> = {
    spades: 0, hearts: 0, diamonds: 0, clubs: 0,
  };
  for (const c of myCards) {
    if (c.kind === 'standard' && c.rank !== R) suitStrength[c.suit]++;
  }
  let jokerCount = 0;
  for (const c of myCards) if (c.kind === 'joker') jokerCount++;

  function declScore(d: (typeof avail)[number]): number {
    if (d.kind === 'joker-pair') {
      // No-trump declaration: trump = rank-R cards in all suits + jokers.
      const offSuitR = myCards.filter(
        (c) => c.kind === 'standard' && c.rank === R,
      ).length;
      return offSuitR + jokerCount + (d.jokerRank === 'big' ? 1 : 0);
    }
    const baseStrength = suitStrength[d.suit];
    const tierBonus = d.kind === 'pair' ? 1 : 0;
    return baseStrength * 2 + tierBonus;
  }

  const ranked = [...avail].sort((a, b) => declScore(b) - declScore(a));
  return { kind: 'declare', cards: [...ranked[0]!.cards] };
}

// ---- smart-v2: 4th-seat follow with perfect info ------------------------

function smartV2LastSeat(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const lead = trick.plays[0]!;
  const leadCount = lead.cards.length;
  const leadCls = classifyCombo(lead.cards, ctx, n);
  const ledSuit = leadCls.ok
    ? leadCls.combo.effectiveSuit
    : effectiveSuit(lead.cards[0]!, ctx);

  // Identify current winner.
  let winnerSeat: Seat = lead.seat;
  let bestCombo: ClassifiedCombo | null = leadCls.ok ? leadCls.combo : null;
  for (let i = 1; i < trick.plays.length; i++) {
    const p = trick.plays[i]!;
    const cls = classifyCombo(p.cards, ctx, n);
    if (cls.ok && bestCombo && comboBeats(cls.combo, bestCombo) > 0) {
      winnerSeat = p.seat;
      bestCombo = cls.combo;
    }
  }
  const myTeam = teamOf(state, seat);
  const teammateWinning = teamOf(state, winnerSeat) === myTeam;
  const myInLed = myCards.filter((c) => effectiveSuit(c, ctx) === ledSuit);

  // ---- Lost: minimize cost. Use v1's smart follow (it already
  // handles "can I cheaply beat the trick?" correctly).
  if (!teammateWinning) {
    return smartFollow(state, seat, myCards, trick, ctx, n);
  }

  // ---- Won: maximize points. Build the maximum-point legal play.

  // Tractor lead with matching tractor → pick the one with most points.
  if (leadCls.ok && leadCls.combo.kind === 'tractor') {
    const k = leadCls.combo.length;
    const matching = findTractors(myInLed, ctx, n).filter(
      (t) => t.length === k,
    );
    if (matching.length > 0) {
      const byPoints = [...matching].sort(
        (a, b) => sumPoints(b.cards) - sumPoints(a.cards),
      );
      return { kind: 'play_cards', cards: [...byPoints[0]!.cards] };
    }
    return buildMaxPointFollow(myCards, myInLed, leadCount, ledSuit, ctx, n);
  }

  // Pair lead with matching pair → pick the highest-point pair.
  if (leadCls.ok && leadCls.combo.kind === 'nkind' && leadCls.combo.n === 2) {
    const pairs = findNKinds(myInLed, n);
    if (pairs.length > 0) {
      const byPoints = [...pairs].sort(
        (a, b) => sumPoints(b.cards) - sumPoints(a.cards),
      );
      return { kind: 'play_cards', cards: [...byPoints[0]!.cards] };
    }
    return buildMaxPointFollow(myCards, myInLed, leadCount, ledSuit, ctx, n);
  }

  // Single lead with led-suit card → drop the heaviest point card in led suit.
  if (leadCls.ok && leadCls.combo.kind === 'single' && myInLed.length > 0) {
    const sortedLed = [...myInLed].sort(
      (a, b) => pointsForCard(b) - pointsForCard(a),
    );
    if (pointsForCard(sortedLed[0]!) > 0) {
      return { kind: 'play_cards', cards: [sortedLed[0]!] };
    }
    // No led-suit point card; play the lowest led-suit card.
    const asc = [...myInLed].sort((a, b) => compareCards(a, b, ctx));
    return { kind: 'play_cards', cards: [asc[0]!] };
  }

  // Single lead, void in led suit → dump the heaviest non-trump point card.
  if (leadCls.ok && leadCls.combo.kind === 'single' && myInLed.length === 0) {
    const nonTrump = myCards.filter((c) => !isTrump(c, ctx));
    const pointsDesc = [...nonTrump].sort(
      (a, b) => pointsForCard(b) - pointsForCard(a),
    );
    if (pointsDesc.length > 0 && pointsForCard(pointsDesc[0]!) > 0) {
      return { kind: 'play_cards', cards: [pointsDesc[0]!] };
    }
    // No non-trump point; play lowest non-trump (or trump if forced).
    const sortedNT = [...nonTrump].sort((a, b) => compareCards(a, b, ctx));
    if (sortedNT.length > 0) {
      return { kind: 'play_cards', cards: [sortedNT[0]!] };
    }
    const trumps = [...myCards]
      .filter((c) => isTrump(c, ctx))
      .sort((a, b) => compareCards(a, b, ctx));
    return { kind: 'play_cards', cards: [trumps[0]!] };
  }

  // Tractor lead with no matching tractor / throw lead / pair lead with no
  // matching pair → maximize points via buildMaxPointFollow.
  return buildMaxPointFollow(myCards, myInLed, leadCount, ledSuit, ctx, n);
}

// Builds a legal follow of `leadCount` cards that maximizes captured
// points while satisfying the led-suit obligation. Used in last-seat
// teammate-winning scenarios. Note: doesn't try to satisfy higher-order
// pair/tractor structure obligations beyond what led-suit fill gives;
// callers should handle those upstream when a matching combo exists.
function buildMaxPointFollow(
  myCards: ReadonlyArray<Card>,
  myInLed: ReadonlyArray<Card>,
  leadCount: number,
  ledSuit: EffectiveSuit,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const chosen: Card[] = [];
  const used = new Set<string>();
  const add = (c: Card): void => {
    const k = identityKey(c);
    if (used.has(k)) return;
    chosen.push(c);
    used.add(k);
  };

  // Step 1: pair obligation in led suit (pair leads + tractor leads).
  const pairs = findNKinds(myInLed, n);
  const pairsByPoints = [...pairs].sort(
    (a, b) => sumPoints(b.cards) - sumPoints(a.cards),
  );
  for (const p of pairsByPoints) {
    if (chosen.length + p.cards.length > leadCount) break;
    for (const c of p.cards) add(c);
  }

  // Step 2: led-suit fill, points-first.
  const ledByPoints = [...myInLed].sort((a, b) => {
    const dp = pointsForCard(b) - pointsForCard(a);
    if (dp !== 0) return dp;
    return compareCards(b, a, ctx);
  });
  for (const c of ledByPoints) {
    if (chosen.length >= leadCount) break;
    add(c);
  }

  // Step 3: non-led fill, points-first non-trump (don't waste trumps for
  // a teammate-winning trick).
  if (chosen.length < leadCount) {
    const rest = myCards.filter((c) => effectiveSuit(c, ctx) !== ledSuit);
    const order = [...rest].sort((a, b) => {
      const aT = isTrump(a, ctx) ? 1 : 0;
      const bT = isTrump(b, ctx) ? 1 : 0;
      if (aT !== bT) return aT - bT;
      const dp = pointsForCard(b) - pointsForCard(a);
      if (dp !== 0) return dp;
      return compareCards(a, b, ctx);
    });
    for (const c of order) {
      if (chosen.length >= leadCount) break;
      add(c);
    }
  }
  return { kind: 'play_cards', cards: chosen };
}

// ---- smart-v2: one-ply search follow ------------------------------------

// Generates candidate follows, scores each by simulating the rest of the
// trick with smart-v1 filling other seats, and picks the highest-scoring
// candidate. Score = (points captured by my team in this trick) minus
// small penalties for using trumps and aces.
function smartV2SearchFollow(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick,
  ctx: TrumpContext,
  n: number,
): BotAction {
  // Gather candidates: legalEvents + smart-v1's own pick.
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    const key = candidateKey(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action = smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) {
    // Should not happen, but defensively fall back to v1.
    return v1Action;
  }

  const myTeam = teamOf(state, seat);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreFollowCandidate(state, seat, cand, myTeam, ctx);
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

function candidateKey(cards: ReadonlyArray<Card>): string {
  return cards.map(identityKey).sort().join('|');
}

// Score a single candidate follow by simulating the rest of the trick.
function scoreFollowCandidate(
  state: GameState,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
): number {
  // Apply our candidate.
  const r0 = applyEvent(state, {
    kind: 'play_cards',
    seat,
    cards: candidate,
  });
  if (!r0.ok) return -1e9;
  let s = r0.state;

  // Simulate the remaining seats (if any) using smart-v1.
  let guard = 0;
  while (s.phase === 'play' && guard++ < 8) {
    const trick = s.currentHand?.currentTrick;
    if (trick === null || trick === undefined) break;
    if (trick.plays.length >= s.config.numPlayers) break;
    const next = nextToPlay(s);
    if (next === null) break;
    const action = pickBotAction(s, next, 'smart');
    if (action.kind !== 'play_cards') break;
    const r = applyEvent(s, {
      kind: 'play_cards',
      seat: next,
      cards: action.cards,
    });
    if (!r.ok) break;
    s = r.state;
  }

  // Identify the trick we just (mostly) completed.
  const completedTrick =
    s.currentHand?.currentTrick ??
    s.currentHand?.tricks[s.currentHand.tricks.length - 1] ??
    null;
  if (completedTrick === null) return 0;

  // Resolve winner.
  const lead = completedTrick.plays[0]!;
  const leadCls = classifyCombo(lead.cards, ctx, 2);
  let winner: Seat = lead.seat;
  let bestCombo: ClassifiedCombo | null = leadCls.ok ? leadCls.combo : null;
  for (let i = 1; i < completedTrick.plays.length; i++) {
    const p = completedTrick.plays[i]!;
    const cls = classifyCombo(p.cards, ctx, 2);
    if (cls.ok && bestCombo && comboBeats(cls.combo, bestCombo) > 0) {
      winner = p.seat;
      bestCombo = cls.combo;
    }
  }

  let trickPoints = 0;
  for (const p of completedTrick.plays) {
    for (const c of p.cards) trickPoints += pointsForCard(c);
  }

  const myWon = teamOf(state, winner) === myTeam;
  let score = myWon ? trickPoints : -trickPoints;

  // Card-cost penalty: discourage wasting trumps and aces.
  // Use the *candidate* cards we're committing.
  for (const c of candidate) {
    if (isTrump(c, ctx)) {
      // Higher penalty for stronger trump tiers.
      // joker = high cost; trump-rank-of-trump-suit = medium; off-suit-R = lower;
      // natural trump-suit ranks = lowest.
      if (c.kind === 'joker') score -= c.rank === 'big' ? 4 : 3;
      else if (c.rank === ctx.trumpRank && c.suit === ctx.trumpSuit) score -= 2;
      else if (c.rank === ctx.trumpRank) score -= 1.5;
      else score -= 0.8;
    } else if (c.kind === 'standard') {
      if (c.rank === 'A') score -= 1.5;
      else if (c.rank === 'K') score -= 0.3;
    }
  }

  return score;
}

// =========================================================================
// Strategy: smart-v3
// =========================================================================
//
// Builds on smart-v2 with two extensions:
//
//   1. **Search-based leading**: when leading, enumerate every legal lead
//      (single, pair, tractor, safe throw) and simulate the trick to
//      completion with v1 followers. v2 used v1's heuristic for leads;
//      v3 picks the lead with the best simulated outcome.
//
//   2. **2-ply lookahead**: each candidate is scored not just by the
//      current trick but also by the next trick (with a discount). This
//      lets v3 prefer trick-winning options when "what we'd lead next"
//      is good, and prefer losing tricks when our hand is stuck.
//
// Card-cost penalties match v2 (high cards cost more). The opponent model
// remains smart-v1 — faster than recursing into v2, and v1 is strong
// enough to approximate other smart bots' play.

const SMART_V3_DISCOUNT = 0.6;
const SMART_V3_SEARCH_DEPTH = 2;

function pickSmartV3(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV3Search(state, seat, myCards, trick, ctx, n);
}

// Single search entry point for both leads and follows. Behavior is the
// same — enumerate candidates, simulate, score, pick best.
function smartV3Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  // Build candidate list.
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  // Include the v1 heuristic pick as an extra candidate (catches anything
  // legalEvents missed and gives a safe fallback when scoring ties).
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  const myTeam = teamOf(state, seat);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidate2Ply(state, seat, cand, myTeam, ctx);
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

// Equivalent set key: cards by `cardKey` (ignores deckId — two A♥ from
// different decks are interchangeable for play).
function candidateKeyByRank(cards: ReadonlyArray<Card>): string {
  return cards.map(cardKey).sort().join('|');
}

// Apply candidate, then simulate up to SMART_V3_SEARCH_DEPTH complete
// tricks with v1 driving every seat. Returns net points to `myTeam`
// across all simulated tricks (with a discount for the second trick),
// minus a per-card cost penalty for the cards we're committing.
function scoreCandidate2Ply(
  state: GameState,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
): number {
  const r0 = applyEvent(state, {
    kind: 'play_cards',
    seat,
    cards: candidate,
  });
  if (!r0.ok) return -1e9;

  const pointsScore = simulateTricks(
    r0.state,
    myTeam,
    SMART_V3_SEARCH_DEPTH,
  );

  let penalty = 0;
  for (const c of candidate) {
    if (isTrump(c, ctx)) {
      if (c.kind === 'joker') penalty += c.rank === 'big' ? 4 : 3;
      else if (c.rank === ctx.trumpRank && c.suit === ctx.trumpSuit) {
        penalty += 2;
      } else if (c.rank === ctx.trumpRank) penalty += 1.5;
      else penalty += 0.8;
    } else if (c.kind === 'standard') {
      if (c.rank === 'A') penalty += 1.5;
      else if (c.rank === 'K') penalty += 0.3;
    }
  }
  return pointsScore - penalty;
}

// Simulate up to `depth` complete tricks from `state`, with v1 deciding
// every play (including continuations of the in-progress trick). Returns
// (net points to myTeam across simulated tricks), with the kth completed
// trick weighted by SMART_V3_DISCOUNT^(k-1).
function simulateTricks(
  initial: GameState,
  myTeam: TeamId,
  depth: number,
  opponentModel: BotType = 'smart',
): number {
  let s = initial;
  let score = 0;
  let weight = 1.0;

  for (let d = 0; d < depth; d++) {
    if (s.phase !== 'play' && s.phase !== 'trick_review') break;

    // Drive any in-progress trick to completion (3 followers max per trick).
    let guard = 0;
    while (s.phase === 'play' && guard++ < 5) {
      const next = nextToPlay(s);
      if (next === null) break;
      const action = pickBotAction(s, next, opponentModel);
      if (action.kind !== 'play_cards') break;
      const r = applyEvent(s, {
        kind: 'play_cards',
        seat: next,
        cards: action.cards,
      });
      if (!r.ok) break;
      s = r.state;
    }

    // Resolve and accumulate the trick we just completed.
    if (s.phase === 'trick_review' && s.currentHand?.currentTrick) {
      const trick = s.currentHand.currentTrick;
      if (trick.plays.length === s.config.numPlayers) {
        const trickPoints = pointsInTrickPlays(trick.plays);
        const winner = winnerOfActiveTrick(
          trick,
          {
            trumpRank: s.currentHand.trumpRank!,
            trumpSuit: s.currentHand.trumpSuit,
          },
          s.config.numDecks,
        );
        const myWin = teamOf(s, winner) === myTeam;
        score += weight * (myWin ? trickPoints : -trickPoints);
      }
      // Advance to next trick (or end of hand).
      const r = applyEvent(s, { kind: 'advance_trick' });
      if (!r.ok) break;
      s = r.state;
    } else {
      // Hand is over or unexpected state — stop the lookahead.
      break;
    }

    weight *= SMART_V3_DISCOUNT;
  }
  return score;
}

function pointsInTrickPlays(
  plays: ReadonlyArray<{ cards: ReadonlyArray<Card> }>,
): number {
  let pts = 0;
  for (const p of plays) for (const c of p.cards) pts += pointsForCard(c);
  return pts;
}

function winnerOfActiveTrick(
  trick: ActiveTrick,
  ctx: TrumpContext,
  n: number,
): Seat {
  const lead = trick.plays[0]!;
  const leadCls = classifyCombo(lead.cards, ctx, n);
  let winner: Seat = lead.seat;
  let best: ClassifiedCombo | null = leadCls.ok ? leadCls.combo : null;
  for (let i = 1; i < trick.plays.length; i++) {
    const p = trick.plays[i]!;
    const cls = classifyCombo(p.cards, ctx, n);
    if (cls.ok && best && comboBeats(cls.combo, best) > 0) {
      winner = p.seat;
      best = cls.combo;
    }
  }
  return winner;
}

// =========================================================================
// Strategy: smart-v4
// =========================================================================
//
// Builds on smart-v3 with two improvements:
//
//   1. **3-ply search**: simulates 3 tricks per candidate instead of 2.
//      The third trick's points are discounted further (0.6^2 = 0.36),
//      so the new ply adds a smaller but still meaningful signal about
//      the trajectory of the hand.
//   2. **Card-tracking cost adjustment**: counts played cards so that a
//      non-A card promoted to highest-remaining in its suit (e.g., K
//      after both As are out) is treated as a guaranteed winner with
//      cost 0 — the search will readily play it to capture tricks
//      rather than treating it as a midweight save-for-later card.
//
// Declare and kitty reuse smart-v1's helpers (other variations tested
// — joker-over-single declaration, tractor-preservation kitty burial —
// reduced strength on aggregate in self-play head-to-heads).

function pickSmartV4(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    // Use v1's declare. The joker-over-single preference was tested but
    // gave up too many trumps for tier protection (single in long suit
    // gives ~12 expected trumps after override risk; joker pair gives
    // ~7).
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    // Reuse v1's bury. Tractor preservation was tested but it blocked
    // creating valuable voids in short suits (a low-value tractor in
    // the kitty is fine — 0 points × multiplier = 0).
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV4Search(state, seat, myCards, trick, ctx, n);
}

// ---- smart-v4: declare --------------------------------------------------
// Like v1's smartDeclare, but: when starting from no standing
// declaration and we have no pair, prefer a joker-pair (tier 3/4,
// unbeatable) over a single (tier 1, often overridden).
function smartDeclareV4(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  hand: HandState,
): BotAction {
  const R: NaturalRank | null =
    state.landlord !== null
      ? hand.trumpRank
      : state.teamLevels[teamOf(state, seat)]!;
  if (R === null) return { kind: 'wait' };

  const standing = hand.standingDeclaration;
  if (
    standing !== null &&
    teamOf(state, standing.seat) === teamOf(state, seat)
  ) {
    return { kind: 'wait' };
  }

  const avail = availableDeclarationsFor(myCards, R, standing);
  if (avail.length === 0) return { kind: 'wait' };

  const suitStrength: Record<Suit, number> = {
    spades: 0, hearts: 0, diamonds: 0, clubs: 0,
  };
  for (const c of myCards) {
    if (c.kind === 'standard' && c.rank !== R) suitStrength[c.suit]++;
  }

  type SuitDecl = Extract<
    ReturnType<typeof availableDeclarationsFor>[number],
    { suit: Suit }
  >;
  const pairs = avail.filter((a): a is SuitDecl & { kind: 'pair' } => a.kind === 'pair');
  const singles = avail.filter(
    (a): a is SuitDecl & { kind: 'single' } => a.kind === 'single',
  );
  const bigJokerPair = avail.find(
    (a) => a.kind === 'joker-pair' && a.jokerRank === 'big',
  );
  const smallJokerPair = avail.find(
    (a) => a.kind === 'joker-pair' && a.jokerRank === 'small',
  );

  pairs.sort((a, b) => suitStrength[b.suit] - suitStrength[a.suit]);
  singles.sort((a, b) => suitStrength[b.suit] - suitStrength[a.suit]);

  if (standing !== null) {
    const standingT = declarationTier(standing);
    if (standingT < 2 && pairs.length > 0) {
      return { kind: 'declare', cards: [...pairs[0]!.cards] };
    }
    if (standingT < 3 && smallJokerPair) {
      return { kind: 'declare', cards: [...smallJokerPair.cards] };
    }
    if (standingT < 4 && bigJokerPair) {
      return { kind: 'declare', cards: [...bigJokerPair.cards] };
    }
    return { kind: 'wait' };
  }

  // Fresh declaration. v4 ordering: pair > joker pair > single.
  if (pairs.length > 0) {
    return { kind: 'declare', cards: [...pairs[0]!.cards] };
  }
  if (bigJokerPair) return { kind: 'declare', cards: [...bigJokerPair.cards] };
  if (smallJokerPair) {
    return { kind: 'declare', cards: [...smallJokerPair.cards] };
  }
  if (singles.length > 0) {
    return { kind: 'declare', cards: [...singles[0]!.cards] };
  }
  return { kind: 'wait' };
}

// ---- smart-v4: kitty burial --------------------------------------------
// Like v1's smartBury, but adds a strong tractor-preservation penalty:
// cards belonging to an existing length-2+ tractor in the non-trump
// portion of the hand are heavily penalized for burial.
function smartBuryV4(
  state: GameState,
  hand: HandState,
  myCards: ReadonlyArray<Card>,
): BotAction {
  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank!,
    trumpSuit: hand.trumpSuit,
  };
  const kittySize = state.config.kittySize;
  const n = state.config.numDecks;

  const nonTrumpCards: StandardCard[] = [];
  const trumpCards: Card[] = [];
  for (const c of myCards) {
    if (isTrump(c, ctx)) trumpCards.push(c);
    else if (c.kind === 'standard') nonTrumpCards.push(c);
  }

  const tractorKeys = new Set<string>();
  for (const t of findTractors(nonTrumpCards, ctx, n)) {
    for (const c of t.cards) tractorKeys.add(cardKey(c));
  }
  const pairKeys = new Set(
    findNKinds(nonTrumpCards, n).map((p) => cardKey(p.cards[0]!)),
  );

  const bySuit = new Map<Suit, StandardCard[]>();
  for (const c of nonTrumpCards) {
    const list = bySuit.get(c.suit) ?? [];
    list.push(c);
    bySuit.set(c.suit, list);
  }
  for (const list of bySuit.values()) {
    list.sort((a, b) => compareCards(a, b, ctx));
  }

  function score(c: StandardCard): number {
    let s = naturalRankIndex(c.rank);
    s += pointsForCard(c) * 8;
    if (tractorKeys.has(cardKey(c))) s += 80;
    else if (pairKeys.has(cardKey(c))) s += 30;
    if (c.rank === 'A') s += 200;
    return s;
  }

  const chosen: Card[] = [];
  const chosenKeys = new Set<string>();
  const includeCard = (c: Card): void => {
    const k = identityKey(c);
    if (chosenKeys.has(k)) return;
    chosen.push(c);
    chosenKeys.add(k);
  };

  // Pass 1: void short side suits — skip if any card is a tractor member.
  const suitsAsc = [...bySuit.entries()].sort(
    (a, b) => a[1].length - b[1].length,
  );
  for (const [, cards] of suitsAsc) {
    if (cards.length === 0) continue;
    if (chosen.length + cards.length > kittySize) continue;
    if (cards.some((c) => c.rank === 'A')) continue;
    if (cards.some((c) => tractorKeys.has(cardKey(c)))) continue;
    const points = sumPoints(cards);
    if (points >= 20 && cards.length <= 4) continue;
    for (const c of cards) includeCard(c);
  }

  // Pass 2: fill with lowest-scored cards.
  if (chosen.length < kittySize) {
    const remaining = nonTrumpCards.filter(
      (c) => !chosenKeys.has(identityKey(c)),
    );
    remaining.sort((a, b) => score(a) - score(b));
    for (const c of remaining) {
      if (chosen.length >= kittySize) break;
      includeCard(c);
    }
  }

  // Pass 3 (rare): trumps.
  if (chosen.length < kittySize) {
    const trumpSorted = [...trumpCards].sort((a, b) => compareCards(a, b, ctx));
    for (const c of trumpSorted) {
      if (chosen.length >= kittySize) break;
      includeCard(c);
    }
  }

  return { kind: 'replace_kitty', cards: chosen.slice(0, kittySize) };
}

// ---- smart-v4: search-based play with card tracking --------------------

type PlayedTracking = {
  readonly isGuaranteedSuitWinner: (c: Card) => boolean;
};

function buildPlayedTracking(
  state: GameState,
  ctx: TrumpContext,
  myHand: ReadonlyArray<Card>,
): PlayedTracking {
  const playedCounts = new Map<string, number>();
  const hand = state.currentHand;
  if (hand !== null) {
    for (const t of hand.tricks) {
      for (const p of t.plays) {
        for (const c of p.cards) {
          const k = cardKey(c);
          playedCounts.set(k, (playedCounts.get(k) ?? 0) + 1);
        }
      }
    }
    if (hand.currentTrick !== null) {
      for (const p of hand.currentTrick.plays) {
        for (const c of p.cards) {
          const k = cardKey(c);
          playedCounts.set(k, (playedCounts.get(k) ?? 0) + 1);
        }
      }
    }
  }

  function myHoldingsOf(suit: Suit, rank: NaturalRank): number {
    let count = 0;
    for (const c of myHand) {
      if (c.kind === 'standard' && c.suit === suit && c.rank === rank) {
        count++;
      }
    }
    return count;
  }

  // Non-trump card `c` is a guaranteed suit-winner iff every higher
  // non-trump rank in its suit has been fully accounted for — either
  // played already or held by us.
  function isGuaranteedSuitWinner(c: Card): boolean {
    if (c.kind !== 'standard' || isTrump(c, ctx)) return false;
    const cardRankIdx = naturalRankIndex(c.rank);
    for (let i = NATURAL_RANKS.length - 1; i >= 0; i--) {
      const r = NATURAL_RANKS[i]!;
      if (naturalRankIndex(r) <= cardRankIdx) break;
      if (r === ctx.trumpRank) continue;
      const k = `${c.suit}:${r}`;
      const played = playedCounts.get(k) ?? 0;
      const mine = myHoldingsOf(c.suit, r);
      if (played + mine < 2) return false;
    }
    return true;
  }

  return { isGuaranteedSuitWinner };
}

function smartV4Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV4(state, seat, cand, myTeam, ctx, tracking);
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

const SMART_V4_SEARCH_DEPTH = 3;

function scoreCandidateV4(
  state: GameState,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  const r0 = applyEvent(state, {
    kind: 'play_cards',
    seat,
    cards: candidate,
  });
  if (!r0.ok) return -1e9;

  const pointsScore = simulateTricks(
    r0.state,
    myTeam,
    SMART_V4_SEARCH_DEPTH,
  );

  let penalty = 0;
  for (const c of candidate) {
    penalty += cardCostV8(c, ctx, tracking);
  }
  return pointsScore - penalty;
}

// v8's stronger save-the-resources cost function. Higher costs for the
// best cards push the search to save them for late-hand / last-trick
// opportunities where they're worth more.
function cardCostV8(
  c: Card,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  if (isTrump(c, ctx)) {
    if (c.kind === 'joker') return c.rank === 'big' ? 5 : 4; // ↑ from 4/3
    if (c.rank === ctx.trumpRank && c.suit === ctx.trumpSuit) return 2.5; // ↑ from 2
    if (c.rank === ctx.trumpRank) return 2; // ↑ from 1.5
    return 1; // ↑ from 0.8
  }
  if (c.kind !== 'standard') return 0;
  if (c.rank === 'A') return 2; // ↑ from 1.5
  if (tracking.isGuaranteedSuitWinner(c)) return 0;
  if (c.rank === 'K') return 0.5; // ↑ from 0.3
  return 0;
}

// v9's cost function: same as v8 but bumps the Ace penalty 2.0 → 2.5.
// Tests whether further along the save-the-aces axis still helps once
// v8's other penalty increases are in place.
function cardCostV9(
  c: Card,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  if (isTrump(c, ctx)) {
    if (c.kind === 'joker') return c.rank === 'big' ? 5 : 4;
    if (c.rank === ctx.trumpRank && c.suit === ctx.trumpSuit) return 2.5;
    if (c.rank === ctx.trumpRank) return 2;
    return 1;
  }
  if (c.kind !== 'standard') return 0;
  if (c.rank === 'A') return 2.5; // ↑ from v8's 2.0
  if (tracking.isGuaranteedSuitWinner(c)) return 0;
  if (c.rank === 'K') return 0.5;
  return 0;
}

// =========================================================================
// Strategy: smart-v9
// =========================================================================
//
// v9 inherits v8's whole architecture (5-ply search, smart-v3 opponent
// model, stronger save-the-resources penalties) and bumps the Ace
// penalty 2.0 → 2.5. A single-variable test of the save-the-aces axis
// that drove v8's improvement over v7.

function pickSmartV9(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV9Search(state, seat, myCards, trick, ctx, n);
}

function smartV9Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV9(state, seat, cand, myTeam, ctx, tracking);
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

function scoreCandidateV9(
  state: GameState,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  const r0 = applyEvent(state, {
    kind: 'play_cards',
    seat,
    cards: candidate,
  });
  if (!r0.ok) return -1e9;

  const pointsScore = simulateTricks(
    r0.state,
    myTeam,
    SMART_V8_SEARCH_DEPTH,
    SMART_V8_OPPONENT_MODEL,
  );

  let penalty = 0;
  for (const c of candidate) {
    penalty += cardCostV9(c, ctx, tracking);
  }
  return pointsScore - penalty;
}

// =========================================================================
// Strategy: smart-v10
// =========================================================================
//
// v10 continues the save-the-aces axis: Ace penalty 2.5 → 3.0. Tests
// whether further preservation of aces helps after v9's gain.

function cardCostV10(
  c: Card,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  if (isTrump(c, ctx)) {
    if (c.kind === 'joker') return c.rank === 'big' ? 5 : 4;
    if (c.rank === ctx.trumpRank && c.suit === ctx.trumpSuit) return 2.5;
    if (c.rank === ctx.trumpRank) return 2;
    return 1;
  }
  if (c.kind !== 'standard') return 0;
  if (c.rank === 'A') return 3; // ↑ from v9's 2.5
  if (tracking.isGuaranteedSuitWinner(c)) return 0;
  if (c.rank === 'K') return 0.5;
  return 0;
}

function pickSmartV10(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV10Search(state, seat, myCards, trick, ctx, n);
}

function smartV10Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV10(state, seat, cand, myTeam, ctx, tracking);
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

function scoreCandidateV10(
  state: GameState,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  const r0 = applyEvent(state, {
    kind: 'play_cards',
    seat,
    cards: candidate,
  });
  if (!r0.ok) return -1e9;

  const pointsScore = simulateTricks(
    r0.state,
    myTeam,
    SMART_V8_SEARCH_DEPTH,
    SMART_V8_OPPONENT_MODEL,
  );

  let penalty = 0;
  for (const c of candidate) {
    penalty += cardCostV10(c, ctx, tracking);
  }
  return pointsScore - penalty;
}

// =========================================================================
// Strategy: smart-v11
// =========================================================================
//
// v11 keeps v9's whole architecture (5-ply search, v3 opponent model,
// Ace penalty 2.5) but differentiates the natural-trump-rank bucket of
// the cost function. v9 charged a flat 1.0 for every natural trump in
// the trump suit (a 3-of-trump and the trump A had identical cost),
// which doesn't reflect that trump A is the top of tier 1 — it beats
// every other natural trump.
//
// Trump card hierarchy (low → high), with new v11 cost:
//   3,4,6,7,8,9,10,J,Q  in trump suit          0.3
//   K in trump suit                            0.8
//   A in trump suit                            1.5
//   rank-R in non-trump suits (off-suit-R)     2.0  (unchanged)
//   rank-R in trump suit                       2.5  (unchanged)
//   small joker                                4.0  (unchanged)
//   big joker                                  5.0  (unchanged)
//
// Non-trump and tracking promotion logic is unchanged from v9.

function cardCostV11(
  c: Card,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  if (isTrump(c, ctx)) {
    if (c.kind === 'joker') return c.rank === 'big' ? 5 : 4;
    if (c.rank === ctx.trumpRank && c.suit === ctx.trumpSuit) return 2.5;
    if (c.rank === ctx.trumpRank) return 2;
    // Natural trump-suit rank (tier 1). Scale by rank height.
    if (c.kind === 'standard') {
      if (c.rank === 'A') return 1.5;
      if (c.rank === 'K') return 0.8;
      return 0.3;
    }
    return 1; // unreachable
  }
  if (c.kind !== 'standard') return 0;
  if (c.rank === 'A') return 2.5;
  if (tracking.isGuaranteedSuitWinner(c)) return 0;
  if (c.rank === 'K') return 0.5;
  return 0;
}

function pickSmartV11(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV11Search(state, seat, myCards, trick, ctx, n);
}

function smartV11Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV11(state, seat, cand, myTeam, ctx, tracking);
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

function scoreCandidateV11(
  state: GameState,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  const r0 = applyEvent(state, {
    kind: 'play_cards',
    seat,
    cards: candidate,
  });
  if (!r0.ok) return -1e9;

  const pointsScore = simulateTricks(
    r0.state,
    myTeam,
    SMART_V8_SEARCH_DEPTH,
    SMART_V8_OPPONENT_MODEL,
  );

  let penalty = 0;
  for (const c of candidate) {
    penalty += cardCostV11(c, ctx, tracking);
  }
  return pointsScore - penalty;
}

// =========================================================================
// Strategy: smart-v12
// =========================================================================
//
// v12 keeps v9's whole architecture and adds **deterministic** signals
// from past play:
//
//   1. **Opponent voids** — when a follower plays any non-led-suit card
//      to a trick led in suit X, they're permanently void in X. Future
//      tricks led in X will be trumped (or dumped) by that opponent.
//
//   2. **Opponent no-pair-in-suit** — when a pair is led in suit X and a
//      follower doesn't match with a pair (because the engine forces a
//      pair match if you have one), they confirmedly have NO pair in X.
//
// These are exact signals — the engine's follow rules guarantee them.
// The simulation already uses real opponent hands, but smart-v3 (the
// opponent model in our simulation) may not always trump optimally. So
// we layer an explicit score adjustment at lead time:
//
//   - Penalty for leading a non-trump combo into a suit where any
//     opponent is known void.
//   - Bonus for leading a non-trump pair in a suit where every opponent
//     is known to have no pair (the pair-trick is guaranteed unless
//     trumped — and the void penalty already gates the trump risk).

type PlayedTrackingV12 = PlayedTracking & {
  readonly isOpponentKnownVoid: (seat: Seat, suit: EffectiveSuit) => boolean;
  readonly isOpponentKnownNoPair: (seat: Seat, suit: EffectiveSuit) => boolean;
};

function buildPlayedTrackingV12(
  state: GameState,
  ctx: TrumpContext,
  myHand: ReadonlyArray<Card>,
): PlayedTrackingV12 {
  const base = buildPlayedTracking(state, ctx, myHand);
  const voidKeys = new Set<string>();
  const noPairKeys = new Set<string>();

  const scan = (
    trick: { plays: ReadonlyArray<{ seat: Seat; cards: ReadonlyArray<Card> }> },
  ): void => {
    if (trick.plays.length === 0) return;
    const lead = trick.plays[0]!;
    const leadCards = lead.cards;
    const ledSuit = effectiveSuit(leadCards[0]!, ctx);
    const leadCls = classifyCombo(leadCards, ctx, state.config.numDecks);
    const leadIsPair =
      leadCls.ok && leadCls.combo.kind === 'nkind' && leadCls.combo.n === 2;

    for (let i = 1; i < trick.plays.length; i++) {
      const p = trick.plays[i]!;
      const inLedCount = p.cards.filter(
        (c) => effectiveSuit(c, ctx) === ledSuit,
      ).length;
      if (inLedCount < p.cards.length) {
        // Followed with some off-suit cards → exhausted led suit.
        voidKeys.add(`${p.seat}:${ledSuit}`);
      }
      if (leadIsPair && inLedCount === 2) {
        // Played 2 led-suit cards to a pair lead — but if those 2 weren't
        // a pair themselves, the player had no pair in led suit (engine
        // forces a pair match if held).
        const cls = classifyCombo(p.cards, ctx, state.config.numDecks);
        const playedPair =
          cls.ok &&
          cls.combo.kind === 'nkind' &&
          cls.combo.n === 2 &&
          cls.combo.effectiveSuit === ledSuit;
        if (!playedPair) {
          noPairKeys.add(`${p.seat}:${ledSuit}`);
        }
      }
    }
  };

  const hand = state.currentHand;
  if (hand !== null) {
    for (const t of hand.tricks) scan(t);
    if (hand.currentTrick !== null) scan(hand.currentTrick);
  }

  return {
    isGuaranteedSuitWinner: base.isGuaranteedSuitWinner,
    isOpponentKnownVoid: (seat, suit) => voidKeys.has(`${seat}:${suit}`),
    isOpponentKnownNoPair: (seat, suit) => noPairKeys.has(`${seat}:${suit}`),
  };
}

// Score adjustment applied only when leading. Returns a delta to add to
// the candidate score.
const SMART_V12_VOID_PENALTY_PER_OPP = 2.0;
const SMART_V12_NO_PAIR_BONUS = 5.0;

function leadAdjustmentV12(
  state: GameState,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
  tracking: PlayedTrackingV12,
): number {
  const candSuit = effectiveSuit(candidate[0]!, ctx);
  // No adjustment for trump leads — void/pair logic targets non-trump.
  if (candSuit === 'trump') return 0;

  const myTeam = teamOf(state, seat);
  const opponentSeats: Seat[] = [];
  for (let s = 0; s < state.config.numPlayers; s++) {
    if (teamOf(state, s) !== myTeam) opponentSeats.push(s);
  }

  let voidOpponents = 0;
  let noPairOpponents = 0;
  for (const opp of opponentSeats) {
    if (tracking.isOpponentKnownVoid(opp, candSuit)) voidOpponents++;
    if (tracking.isOpponentKnownNoPair(opp, candSuit)) noPairOpponents++;
  }

  let adj = -SMART_V12_VOID_PENALTY_PER_OPP * voidOpponents;

  // Pair-lead bonus if all opponents are confirmed pair-less.
  const cls = classifyCombo(candidate, ctx, n);
  if (cls.ok && cls.combo.kind === 'nkind' && cls.combo.n === 2) {
    if (noPairOpponents === opponentSeats.length) {
      adj += SMART_V12_NO_PAIR_BONUS;
    }
  }
  return adj;
}

function pickSmartV12(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV12Search(state, seat, myCards, trick, ctx, n);
}

function smartV12Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  const tracking = buildPlayedTrackingV12(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  const isLeading = trick === null;
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    let score = scoreCandidateV12Base(
      state,
      seat,
      cand,
      myTeam,
      ctx,
      tracking,
    );
    if (isLeading) {
      score += leadAdjustmentV12(state, seat, cand, ctx, n, tracking);
    }
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

// Same base scoring as v9 — simulation + cardCostV9 (Ace 2.5 etc.).
function scoreCandidateV12Base(
  state: GameState,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  const r0 = applyEvent(state, {
    kind: 'play_cards',
    seat,
    cards: candidate,
  });
  if (!r0.ok) return -1e9;

  const pointsScore = simulateTricks(
    r0.state,
    myTeam,
    SMART_V8_SEARCH_DEPTH,
    SMART_V8_OPPONENT_MODEL,
  );

  let penalty = 0;
  for (const c of candidate) {
    penalty += cardCostV9(c, ctx, tracking);
  }
  return pointsScore - penalty;
}

// =========================================================================
// Strategy: smart-v13
// =========================================================================
//
// v13 fixes a long-standing correctness issue in v3-v12: their search
// reads opponent hands directly via `state.currentHand.hands[opp]`,
// which is information no real player has. v13 uses the same v9
// architecture but **masks** opponent hands before each candidate
// evaluation: the unknown cards (deck − my hand − played cards − known
// kitty) are randomly distributed across opponent seats to produce a
// "determinized" state that's consistent with what the bot can actually
// observe. The simulation then runs on this determinized state.
//
// Single determinization per scoring call (seeded by trick count for
// stability across candidate comparisons within one decision).

function pickSmartV13(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV13Search(state, seat, myCards, trick, ctx, n);
}

function smartV13Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  // Candidate enumeration only uses my own cards + the trick state,
  // never opponents' hands. We can use the real state here for legal
  // moves (since legality is observable).
  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  // Build the determinized state ONCE for this decision. All candidates
  // are scored against the same sampled opp hands, so comparisons are
  // apples-to-apples even if the sample is noisy.
  const hand = state.currentHand!;
  const determinizationSeed = deterministicSeed(state, seat);
  const determinized = maskOpponentHands(state, seat, determinizationSeed);

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV13(determinized, seat, cand, myTeam, ctx, tracking);
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

function scoreCandidateV13(
  determinizedState: GameState,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  const r0 = applyEvent(determinizedState, {
    kind: 'play_cards',
    seat,
    cards: candidate,
  });
  if (!r0.ok) return -1e9;

  const pointsScore = simulateTricks(
    r0.state,
    myTeam,
    SMART_V8_SEARCH_DEPTH,
    SMART_V8_OPPONENT_MODEL,
  );

  let penalty = 0;
  for (const c of candidate) {
    penalty += cardCostV9(c, ctx, tracking);
  }
  return pointsScore - penalty;
}

// Produce a deterministic seed from the bot's observable state so that
// candidate scoring within a single decision uses a single, stable
// determinization. The seed varies across tricks so we don't bias toward
// a fixed sample.
function deterministicSeed(state: GameState, seat: Seat): number {
  const hand = state.currentHand;
  if (hand === null) return 1;
  const tricksDone = hand.tricks.length;
  const playsInCurrent = hand.currentTrick?.plays.length ?? 0;
  return (
    1 + seat * 257 + tricksDone * 1009 + playsInCurrent * 17 + hand.handNumber * 7919
  ) >>> 0;
}

// Returns a state with opponents' hands and (if I'm not landlord) the
// kitty replaced by random assignments drawn from cards that aren't
// known to me. My own hand and the played-trick history are preserved.
function maskOpponentHands(
  state: GameState,
  mySeat: Seat,
  seed: number,
): GameState {
  const hand = state.currentHand;
  if (hand === null) return state;

  const myCards = hand.hands[mySeat] ?? [];
  const playedCards: Card[] = [];
  for (const t of hand.tricks) {
    for (const p of t.plays) {
      for (const c of p.cards) playedCards.push(c);
    }
  }
  if (hand.currentTrick !== null) {
    for (const p of hand.currentTrick.plays) {
      for (const c of p.cards) playedCards.push(c);
    }
  }
  const iAmLandlord = hand.landlord === mySeat;
  const knownKitty = iAmLandlord ? hand.kitty : [];

  const knownSet = new Set<string>();
  const addKnown = (c: Card): void => {
    knownSet.add(identityKey(c));
  };
  for (const c of myCards) addKnown(c);
  for (const c of playedCards) addKnown(c);
  for (const c of knownKitty) addKnown(c);

  const fullDeck = buildDeck(state.config.numDecks);
  const unknown = fullDeck.filter((c) => !knownSet.has(identityKey(c)));

  const rng = mulberry32(seed);
  const shuffled = shuffle(unknown, rng);

  const newHands: Record<Seat, ReadonlyArray<Card>> = {};
  let idx = 0;
  for (let s = 0; s < state.config.numPlayers; s++) {
    if (s === mySeat) {
      newHands[s] = myCards;
      continue;
    }
    const size = (hand.hands[s] ?? []).length;
    newHands[s] = shuffled.slice(idx, idx + size);
    idx += size;
  }

  const newKitty: ReadonlyArray<Card> = iAmLandlord
    ? hand.kitty
    : shuffled.slice(idx, idx + hand.kitty.length);

  return {
    ...state,
    currentHand: {
      ...hand,
      hands: newHands,
      kitty: newKitty,
    },
  };
}

// Variant of maskOpponentHands that respects publicly-known voids:
// if opponent S has previously played off-suit on a led suit X, they've
// proven they hold no cards of effective suit X, so the sampler never
// assigns them a card with effectiveSuit === X. Same compute budget as
// maskOpponentHands; samples are restricted to the consistent slice of
// the world that any observer at the table could rule in.
function maskOpponentHandsConstrained(
  state: GameState,
  mySeat: Seat,
  seed: number,
): GameState {
  const hand = state.currentHand;
  if (hand === null) return state;
  if (hand.trumpRank === null) return maskOpponentHands(state, mySeat, seed);
  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };

  const voidSuits = new Map<Seat, Set<EffectiveSuit>>();
  const recordVoid = (s: Seat, suit: EffectiveSuit): void => {
    let set = voidSuits.get(s);
    if (set === undefined) {
      set = new Set();
      voidSuits.set(s, set);
    }
    set.add(suit);
  };
  const scan = (
    trick: { plays: ReadonlyArray<{ seat: Seat; cards: ReadonlyArray<Card> }> },
  ): void => {
    if (trick.plays.length === 0) return;
    const lead = trick.plays[0]!;
    const ledSuit = effectiveSuit(lead.cards[0]!, ctx);
    for (let i = 1; i < trick.plays.length; i++) {
      const p = trick.plays[i]!;
      const inLed = p.cards.filter(
        (c) => effectiveSuit(c, ctx) === ledSuit,
      ).length;
      if (inLed < p.cards.length) recordVoid(p.seat, ledSuit);
    }
  };
  for (const t of hand.tricks) scan(t);
  if (hand.currentTrick !== null) scan(hand.currentTrick);

  const myCards = hand.hands[mySeat] ?? [];
  const playedCards: Card[] = [];
  for (const t of hand.tricks) {
    for (const p of t.plays) {
      for (const c of p.cards) playedCards.push(c);
    }
  }
  if (hand.currentTrick !== null) {
    for (const p of hand.currentTrick.plays) {
      for (const c of p.cards) playedCards.push(c);
    }
  }
  const iAmLandlord = hand.landlord === mySeat;
  const knownKitty = iAmLandlord ? hand.kitty : [];

  const knownSet = new Set<string>();
  for (const c of myCards) knownSet.add(identityKey(c));
  for (const c of playedCards) knownSet.add(identityKey(c));
  for (const c of knownKitty) knownSet.add(identityKey(c));

  const fullDeck = buildDeck(state.config.numDecks);
  const unknown = fullDeck.filter((c) => !knownSet.has(identityKey(c)));

  const rng = mulberry32(seed);
  const shuffled = shuffle(unknown, rng);

  type Target =
    | { readonly kind: 'seat'; readonly seat: Seat; capacity: number }
    | { readonly kind: 'kitty'; capacity: number };
  const targets: Target[] = [];
  for (let s = 0; s < state.config.numPlayers; s++) {
    if (s === mySeat) continue;
    const size = (hand.hands[s] ?? []).length;
    targets.push({ kind: 'seat', seat: s as Seat, capacity: size });
  }
  if (!iAmLandlord) {
    targets.push({ kind: 'kitty', capacity: hand.kitty.length });
  }

  const accepts = (t: Target, c: Card): boolean => {
    if (t.kind === 'kitty') return true;
    const set = voidSuits.get(t.seat);
    if (set === undefined) return true;
    return !set.has(effectiveSuit(c, ctx));
  };

  const placed: Card[][] = targets.map(() => []);
  const unplaced: Card[] = [];
  // Greedy: each card goes to the accepting target with the most
  // remaining capacity. Cards that fit nowhere get retried after.
  for (const c of shuffled) {
    let bestIdx = -1;
    let bestCap = -1;
    for (let i = 0; i < targets.length; i++) {
      const t = targets[i]!;
      if (t.capacity <= 0) continue;
      if (!accepts(t, c)) continue;
      if (t.capacity > bestCap) {
        bestCap = t.capacity;
        bestIdx = i;
      }
    }
    if (bestIdx === -1) {
      unplaced.push(c);
      continue;
    }
    placed[bestIdx]!.push(c);
    targets[bestIdx]!.capacity--;
  }
  // Fall back to any open slot for cards that couldn't be placed under
  // constraints (extremely rare; only if voids over-constrain the sample).
  for (const c of unplaced) {
    for (let i = 0; i < targets.length; i++) {
      if (targets[i]!.capacity > 0) {
        placed[i]!.push(c);
        targets[i]!.capacity--;
        break;
      }
    }
  }

  const newHands: Record<Seat, ReadonlyArray<Card>> = {};
  newHands[mySeat] = myCards;
  let newKitty: ReadonlyArray<Card> = hand.kitty;
  for (let i = 0; i < targets.length; i++) {
    const t = targets[i]!;
    if (t.kind === 'seat') {
      newHands[t.seat] = placed[i]!;
    } else {
      newKitty = placed[i]!;
    }
  }

  return {
    ...state,
    currentHand: {
      ...hand,
      hands: newHands,
      kitty: newKitty,
    },
  };
}

// =========================================================================
// Strategy: smart-v14
// =========================================================================
//
// v14 combines v13's hidden-information determinization with v12's
// **deterministic public-info signals**. Both ingredients are needed:
//   - Determinization (v13) keeps the bot honest — no opponent-hand
//     reads inside the simulation.
//   - Public-info signals (v12) provide the bot with reliable knowledge
//     that the simulation can't reconstruct, e.g.:
//       * If an opponent played off-suit on a led suit, they're
//         permanently void in that suit (would trump non-trump leads).
//       * If a pair was led and an opponent didn't match with a pair,
//         the engine forces a pair match — so they have no pair in
//         that suit. Future pair leads in that suit are safer.
//
// v13 alone lost 0-12 to v9 (the cheating full-info bot). v14's hope:
// the deterministic public signals partially compensate for the
// information loss without re-introducing the cheating.

function pickSmartV14(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV14Search(state, seat, myCards, trick, ctx, n);
}

function smartV14Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  // Determinize opponent hands (hidden-info correctness from v13).
  const determinizationSeed = deterministicSeed(state, seat);
  const determinized = maskOpponentHands(state, seat, determinizationSeed);

  // Public-info trackings (computed from REAL state — these are facts
  // any observer at the table would know).
  const tracking = buildPlayedTrackingV12(state, ctx, myCards);

  const myTeam = teamOf(state, seat);
  const isLeading = trick === null;
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    let score = scoreCandidateV14(
      determinized,
      seat,
      cand,
      myTeam,
      ctx,
      tracking,
    );
    if (isLeading) {
      score += leadAdjustmentV12(state, seat, cand, ctx, n, tracking);
    }
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

function scoreCandidateV14(
  determinizedState: GameState,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  const r0 = applyEvent(determinizedState, {
    kind: 'play_cards',
    seat,
    cards: candidate,
  });
  if (!r0.ok) return -1e9;

  const pointsScore = simulateTricks(
    r0.state,
    myTeam,
    SMART_V8_SEARCH_DEPTH,
    SMART_V8_OPPONENT_MODEL,
  );

  let penalty = 0;
  for (const c of candidate) {
    penalty += cardCostV9(c, ctx, tracking);
  }
  return pointsScore - penalty;
}

// =========================================================================
// Strategy: smart-v15
// =========================================================================
//
// v15 is v13's hidden-info bot but averages over N=3 determinizations
// per candidate. Each sample uses a different random opponent-hand
// assignment consistent with public info. Averaging reduces the
// variance from a single noisy determinization — the hope is that
// candidate ranking becomes more reliable.
//
// Cost: ~3× v13 per decision.

const SMART_V15_NUM_SAMPLES = 3;

function pickSmartV15(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV15Search(state, seat, myCards, trick, ctx, n);
}

function smartV15Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  // Pre-sample N determinizations. All candidates score against the
  // same N samples — apples-to-apples comparisons.
  const baseSeed = deterministicSeed(state, seat);
  const determinizations: GameState[] = [];
  for (let i = 0; i < SMART_V15_NUM_SAMPLES; i++) {
    determinizations.push(maskOpponentHands(state, seat, baseSeed + i * 1009));
  }

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV15(
      determinizations,
      seat,
      cand,
      myTeam,
      ctx,
      tracking,
    );
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

function scoreCandidateV15(
  determinizations: ReadonlyArray<GameState>,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  let totalPointsScore = 0;
  let validSamples = 0;
  for (const det of determinizations) {
    const r0 = applyEvent(det, {
      kind: 'play_cards',
      seat,
      cards: candidate,
    });
    if (!r0.ok) continue;
    totalPointsScore += simulateTricks(
      r0.state,
      myTeam,
      SMART_V8_SEARCH_DEPTH,
      SMART_V8_OPPONENT_MODEL,
    );
    validSamples++;
  }
  if (validSamples === 0) return -1e9;
  const avgPointsScore = totalPointsScore / validSamples;

  let penalty = 0;
  for (const c of candidate) {
    penalty += cardCostV9(c, ctx, tracking);
  }
  return avgPointsScore - penalty;
}

// =========================================================================
// Strategy: smart-v16
// =========================================================================
//
// Same as v15 but uses N=5 determinizations per scoring call instead of
// N=3. More samples → lower variance in candidate score estimates, at
// the cost of ~5/3 = 1.67× more compute.

const SMART_V16_NUM_SAMPLES = 5;

function pickSmartV16(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV16Search(state, seat, myCards, trick, ctx, n);
}

function smartV16Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  const baseSeed = deterministicSeed(state, seat);
  const determinizations: GameState[] = [];
  for (let i = 0; i < SMART_V16_NUM_SAMPLES; i++) {
    determinizations.push(maskOpponentHands(state, seat, baseSeed + i * 1009));
  }

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV15(
      determinizations,
      seat,
      cand,
      myTeam,
      ctx,
      tracking,
    );
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

// =========================================================================
// Strategy: smart-v17
// =========================================================================
//
// Same as v15/v16 but with N=10 determinizations per scoring call.
// Tests whether further variance reduction continues to help past v16.

const SMART_V17_NUM_SAMPLES = 10;

function pickSmartV17(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV17Search(state, seat, myCards, trick, ctx, n);
}

function smartV17Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  const baseSeed = deterministicSeed(state, seat);
  const determinizations: GameState[] = [];
  for (let i = 0; i < SMART_V17_NUM_SAMPLES; i++) {
    determinizations.push(maskOpponentHands(state, seat, baseSeed + i * 1009));
  }

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV15(
      determinizations,
      seat,
      cand,
      myTeam,
      ctx,
      tracking,
    );
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

// =========================================================================
// Strategy: smart-v18
// =========================================================================
//
// Three fixes/additions on top of v16's N=5 determinized search:
//
// 1. **Correct winner detection in simulation**: previous versions used
//    `classifyCombo` + `comboBeats` to resolve the simulated trick
//    winner. That logic gives the leader an auto-win on throws because
//    `classifyCombo` returns ok=false for multi-component plays. The
//    engine now resolves via `decomposePlay` + `decomposedBeats`, which
//    correctly lets trumping plays beat throws via same-shape matching.
//    v18 fixes the simulation to match.
//
// 2. **Trump-trump candidate for follow positions**: when bot is void
//    in led suit and has trumps, generate an explicit "lowest N trumps"
//    candidate. `legalEvents`' 'low' fill prefers non-trump cards, so
//    this candidate is otherwise never offered to the search.
//
// 3. **Hand-state-dependent lead bonus**: each simulated trick my team
//    wins adds `α × handLeadValue(my_hand)` to the score. A strong
//    hand (many aces, big trumps, guaranteed winners) values the lead
//    more; a weak hand values it less.

const SMART_V18_NUM_SAMPLES = 5;
const SMART_V18_LEAD_BONUS_WEIGHT = 0.15; // small per-card "lead value" multiplier

function pickSmartV18(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV18Search(state, seat, myCards, trick, ctx, n);
}

function smartV18Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  // NEW: when following and void in led suit, add a "play lowest N
  // trumps + fillers" candidate. This is critical for trumping throws.
  if (trick !== null) {
    const trumpFillCandidate = buildTrumpFillCandidate(
      myCards,
      trick,
      ctx,
    );
    if (trumpFillCandidate !== null) consider(trumpFillCandidate);
  }

  if (candidates.length === 0) return v1Action;

  const baseSeed = deterministicSeed(state, seat);
  const determinizations: GameState[] = [];
  for (let i = 0; i < SMART_V18_NUM_SAMPLES; i++) {
    determinizations.push(maskOpponentHands(state, seat, baseSeed + i * 1009));
  }

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  const handLeadVal = handLeadValue(myCards, ctx, tracking);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV18(
      determinizations,
      seat,
      cand,
      myTeam,
      ctx,
      tracking,
      handLeadVal,
    );
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

// Build a candidate where the bot plays its lowest available trump
// cards (filling required count), using non-led-suit non-trump as the
// rest. Returns null if the bot can't or wouldn't choose this (e.g.,
// has led-suit cards to follow with).
function buildTrumpFillCandidate(
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick,
  ctx: TrumpContext,
): ReadonlyArray<Card> | null {
  const lead = trick.plays[0]!;
  const ledSuit = effectiveSuit(lead.cards[0]!, ctx);
  const leadCount = lead.cards.length;

  const myInLed = myCards.filter((c) => effectiveSuit(c, ctx) === ledSuit);
  // If bot has any led-suit cards, the follow rules force playing them.
  // Skip this candidate — the search will use 'low' / pair-match etc.
  if (myInLed.length > 0) return null;

  const trumpsAsc = myCards
    .filter((c) => isTrump(c, ctx))
    .sort((a, b) => compareCards(a, b, ctx));
  if (trumpsAsc.length === 0) return null;

  const chosen: Card[] = [];
  // Use up to `leadCount` trumps, lowest first.
  for (const t of trumpsAsc) {
    if (chosen.length >= leadCount) break;
    chosen.push(t);
  }
  // If not enough trumps, fill rest with lowest non-trump non-led cards.
  if (chosen.length < leadCount) {
    const fillers = myCards
      .filter((c) => !isTrump(c, ctx))
      .sort((a, b) => compareCards(a, b, ctx));
    for (const f of fillers) {
      if (chosen.length >= leadCount) break;
      chosen.push(f);
    }
  }
  if (chosen.length !== leadCount) return null;
  return chosen;
}

function scoreCandidateV18(
  determinizations: ReadonlyArray<GameState>,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
  handLeadVal: number,
): number {
  let totalPointsScore = 0;
  let totalWins = 0;
  let validSamples = 0;
  for (const det of determinizations) {
    const r0 = applyEvent(det, {
      kind: 'play_cards',
      seat,
      cards: candidate,
    });
    if (!r0.ok) continue;
    const { score, wins } = simulateTricksAndWinsV18(
      r0.state,
      myTeam,
      SMART_V8_SEARCH_DEPTH,
      SMART_V8_OPPONENT_MODEL,
    );
    totalPointsScore += score;
    totalWins += wins;
    validSamples++;
  }
  if (validSamples === 0) return -1e9;
  const avgPointsScore = totalPointsScore / validSamples;
  const avgWins = totalWins / validSamples;

  let penalty = 0;
  for (const c of candidate) {
    penalty += cardCostV9(c, ctx, tracking);
  }
  const leadBonus = SMART_V18_LEAD_BONUS_WEIGHT * handLeadVal * avgWins;
  return avgPointsScore - penalty + leadBonus;
}

// Like simulateTricks but ALSO tracks "wins" (number of completed
// tricks won by myTeam during the simulation). Also uses the correct
// post-engine-update winner detection via decomposedBeats.
function simulateTricksAndWinsV18(
  initial: GameState,
  myTeam: TeamId,
  depth: number,
  opponentModel: BotType = 'smart',
): { score: number; wins: number } {
  let s = initial;
  let score = 0;
  let wins = 0;
  let weight = 1.0;

  for (let d = 0; d < depth; d++) {
    if (s.phase !== 'play' && s.phase !== 'trick_review') break;

    let guard = 0;
    while (s.phase === 'play' && guard++ < 5) {
      const next = nextToPlay(s);
      if (next === null) break;
      const action = pickBotAction(s, next, opponentModel);
      if (action.kind !== 'play_cards') break;
      const r = applyEvent(s, {
        kind: 'play_cards',
        seat: next,
        cards: action.cards,
      });
      if (!r.ok) break;
      s = r.state;
    }

    if (s.phase === 'trick_review' && s.currentHand?.currentTrick) {
      const trick = s.currentHand.currentTrick;
      if (trick.plays.length === s.config.numPlayers) {
        const trickPoints = pointsInTrickPlays(trick.plays);
        const ctx: TrumpContext = {
          trumpRank: s.currentHand.trumpRank!,
          trumpSuit: s.currentHand.trumpSuit,
        };
        // Use decomposedBeats-aware winner.
        const winner = winnerWithThrows(
          trick,
          ctx,
          s.config.numDecks,
        );
        const myWin = teamOf(s, winner) === myTeam;
        score += weight * (myWin ? trickPoints : -trickPoints);
        if (myWin) wins += weight;
      }
      const r = applyEvent(s, { kind: 'advance_trick' });
      if (!r.ok) break;
      s = r.state;
    } else {
      break;
    }

    weight *= SMART_V3_DISCOUNT;
  }
  return { score, wins };
}

// Resolve trick winner using the same decomposedBeats rules as the
// engine's resolveTrick. This correctly handles trumps beating throws.
function winnerWithThrows(
  trick: ActiveTrick,
  ctx: TrumpContext,
  n: number,
): Seat {
  const lead = trick.plays[0]!;
  const leadDecomp = decomposePlay(lead.cards, ctx, n);
  if (leadDecomp === null) return lead.seat;
  let winner: Seat = lead.seat;
  let winningDecomp = leadDecomp;
  for (let i = 1; i < trick.plays.length; i++) {
    const p = trick.plays[i]!;
    const pDecomp = decomposePlay(p.cards, ctx, n);
    if (pDecomp === null) continue;
    if (decomposedBeats(pDecomp, winningDecomp, ctx, n)) {
      winner = p.seat;
      winningDecomp = pDecomp;
    }
  }
  return winner;
}

// Sum a hand's expected future winning potential for use in the lead
// bonus. Roughly: aces / jokers / trump-rank cards are worth the most;
// guaranteed-suit-winners (via card tracking) count as aces.
function handLeadValue(
  hand: ReadonlyArray<Card>,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  let v = 0;
  for (const c of hand) {
    if (isTrump(c, ctx)) {
      if (c.kind === 'joker') v += c.rank === 'big' ? 2 : 1.5;
      else if (c.kind === 'standard') {
        if (c.rank === ctx.trumpRank && c.suit === ctx.trumpSuit) v += 1.5;
        else if (c.rank === ctx.trumpRank) v += 1.0;
        else if (c.rank === 'A') v += 0.6;
      }
    } else if (c.kind === 'standard') {
      if (c.rank === 'A') v += 1.5;
      else if (tracking.isGuaranteedSuitWinner(c)) v += 1.0;
    }
  }
  return v;
}

// =========================================================================
// Strategy: smart-v19
// =========================================================================
//
// Same as v18 but lead bonus uses (my hand value − opp hand value) per
// determinization, averaged over samples. Captures the "win when I'd
// lead better than opp" intuition: if my hand is weaker than opp's,
// winning the trick is actually less valuable (they'd lead worse).

const SMART_V19_LEAD_BONUS_WEIGHT = 0.15;

function pickSmartV19(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV19Search(state, seat, myCards, trick, ctx, n);
}

function smartV19Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  // Trump-fill candidate (from v18).
  if (trick !== null) {
    const trumpFill = buildTrumpFillCandidate(myCards, trick, ctx);
    if (trumpFill !== null) consider(trumpFill);
  }

  if (candidates.length === 0) return v1Action;

  const baseSeed = deterministicSeed(state, seat);
  const determinizations: GameState[] = [];
  for (let i = 0; i < SMART_V18_NUM_SAMPLES; i++) {
    determinizations.push(maskOpponentHands(state, seat, baseSeed + i * 1009));
  }

  // Average opp hand value across determinizations.
  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  const myHandVal = handLeadValue(myCards, ctx, tracking);
  let oppHandValSum = 0;
  for (const det of determinizations) {
    // Sum opp hand values from det.
    if (det.currentHand !== null) {
      for (let s = 0; s < det.config.numPlayers; s++) {
        if (teamOf(det, s) === myTeam) continue;
        oppHandValSum += handLeadValue(
          det.currentHand.hands[s] ?? [],
          ctx,
          tracking,
        );
      }
    }
  }
  // Average opp hand value (sum of both opp seats per det, averaged
  // across dets, then divided by 2 for "per opponent seat").
  const avgOppHandVal = oppHandValSum / (determinizations.length * 2);
  const netLeadVal = myHandVal - avgOppHandVal;

  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV19(
      determinizations,
      seat,
      cand,
      myTeam,
      ctx,
      tracking,
      netLeadVal,
    );
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

function scoreCandidateV19(
  determinizations: ReadonlyArray<GameState>,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
  netLeadVal: number,
): number {
  let totalPointsScore = 0;
  let totalWins = 0;
  let validSamples = 0;
  for (const det of determinizations) {
    const r0 = applyEvent(det, {
      kind: 'play_cards',
      seat,
      cards: candidate,
    });
    if (!r0.ok) continue;
    const { score, wins } = simulateTricksAndWinsV18(
      r0.state,
      myTeam,
      SMART_V8_SEARCH_DEPTH,
      SMART_V8_OPPONENT_MODEL,
    );
    totalPointsScore += score;
    totalWins += wins;
    validSamples++;
  }
  if (validSamples === 0) return -1e9;
  const avgPointsScore = totalPointsScore / validSamples;
  const avgWins = totalWins / validSamples;

  let penalty = 0;
  for (const c of candidate) {
    penalty += cardCostV9(c, ctx, tracking);
  }
  const leadBonus = SMART_V19_LEAD_BONUS_WEIGHT * netLeadVal * avgWins;
  return avgPointsScore - penalty + leadBonus;
}

// =========================================================================
// Strategy: smart-v20
// =========================================================================
//
// v18 with a stronger determinization: opponent-hand samples respect
// publicly-known voids. If opponent S has ever followed off-suit on a
// led suit X, the sampler never assigns them a card with effectiveSuit
// === X. Compute budget is unchanged (N=5 samples, same depth, same
// opponent model). Every sample is now a feasible reality given the
// observable history, instead of a uniform random sample over all
// hand-shaped allocations (most of which contradict prior plays).
//
// Hypothesis: late in the hand — when many voids are known — the
// unconstrained sampler is wasting samples on impossible worlds, and
// trump/aces are being timed suboptimally in expectation.

function pickSmartV20(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV20Search(state, seat, myCards, trick, ctx, n);
}

function smartV20Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (trick !== null) {
    const trumpFill = buildTrumpFillCandidate(myCards, trick, ctx);
    if (trumpFill !== null) consider(trumpFill);
  }

  if (candidates.length === 0) return v1Action;

  // Constrained determinization is the sole difference from v18.
  const baseSeed = deterministicSeed(state, seat);
  const determinizations: GameState[] = [];
  for (let i = 0; i < SMART_V18_NUM_SAMPLES; i++) {
    determinizations.push(
      maskOpponentHandsConstrained(state, seat, baseSeed + i * 1009),
    );
  }

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  const handLeadVal = handLeadValue(myCards, ctx, tracking);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV18(
      determinizations,
      seat,
      cand,
      myTeam,
      ctx,
      tracking,
      handLeadVal,
    );
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

function cardCostV4(
  c: Card,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  if (isTrump(c, ctx)) {
    if (c.kind === 'joker') return c.rank === 'big' ? 4 : 3;
    if (c.rank === ctx.trumpRank && c.suit === ctx.trumpSuit) return 2;
    if (c.rank === ctx.trumpRank) return 1.5;
    return 0.8;
  }
  if (c.kind !== 'standard') return 0;
  // Aces always keep their full save-cost — they're the best non-trump
  // anyway and we want to time their use carefully.
  if (c.rank === 'A') return 1.5;
  // A non-A card "promoted" to highest-remaining in its suit (e.g., K
  // after both As played) becomes a guaranteed in-suit winner. Reduce
  // its cost to ~0 so the search readily uses it to capture tricks.
  if (tracking.isGuaranteedSuitWinner(c)) return 0;
  if (c.rank === 'K') return 0.3;
  return 0;
}

// =========================================================================
// Strategy: smart-v5
// =========================================================================
//
// Identical decision structure to smart-v4 (3-ply search, card-tracking
// cost penalty), but the simulation of the *other* seats uses smart-v3
// instead of smart-v1. Each simulated play now does its own 2-ply
// lookahead, so the predicted plays are much closer to what a real
// search-based bot (like v4) would actually do.
//
// Cost is ~5-10× v4 per decision because every simulated opponent play
// triggers a nested 2-ply search. The improvement is concentrated in
// games against strong opponents — exactly the benchmark we're trying
// to win.

const SMART_V5_OPPONENT_MODEL: BotType = 'smart-v3';

function pickSmartV5(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV5Search(state, seat, myCards, trick, ctx, n);
}

function smartV5Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV5(state, seat, cand, myTeam, ctx, tracking);
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

function scoreCandidateV5(
  state: GameState,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  const r0 = applyEvent(state, {
    kind: 'play_cards',
    seat,
    cards: candidate,
  });
  if (!r0.ok) return -1e9;

  const pointsScore = simulateTricks(
    r0.state,
    myTeam,
    SMART_V4_SEARCH_DEPTH,
    SMART_V5_OPPONENT_MODEL,
  );

  let penalty = 0;
  for (const c of candidate) {
    penalty += cardCostV4(c, ctx, tracking);
  }
  return pointsScore - penalty;
}

// =========================================================================
// Strategy: smart-v6
// =========================================================================
//
// v6 keeps smart-v5's smart-v3 opponent model but increases the search
// horizon from 3 plies to 4. The fourth trick is discounted heavily
// (0.6^3 ≈ 0.216), so its signal is small but informative for trick
// trajectories where the next-but-one trick matters.
//
// An earlier v6 used smart-v4 as the opponent model (instead of deeper
// search) but tied v5 6-6 over 12 games — a more accurate opponent
// model didn't help once predictions were already strong enough. The
// extra ply is a cleaner orthogonal improvement.

const SMART_V6_OPPONENT_MODEL: BotType = 'smart-v3';
const SMART_V6_SEARCH_DEPTH = 4;

function pickSmartV6(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV6Search(state, seat, myCards, trick, ctx, n);
}

function smartV6Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV6(state, seat, cand, myTeam, ctx, tracking);
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

function scoreCandidateV6(
  state: GameState,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  const r0 = applyEvent(state, {
    kind: 'play_cards',
    seat,
    cards: candidate,
  });
  if (!r0.ok) return -1e9;

  const pointsScore = simulateTricks(
    r0.state,
    myTeam,
    SMART_V6_SEARCH_DEPTH,
    SMART_V6_OPPONENT_MODEL,
  );

  let penalty = 0;
  for (const c of candidate) {
    penalty += cardCostV4(c, ctx, tracking);
  }
  return pointsScore - penalty;
}

// =========================================================================
// Strategy: smart-v7
// =========================================================================
//
// v7 increases the search horizon from v6's 4 plies to 5. The fifth
// trick contributes 0.6^4 ≈ 0.13 weight — small, but its signal aligns
// in many positions because the trick chain past the 4-ply horizon
// often determines mid-hand momentum.
//
// An earlier v7 with a terminal opponent-cost heuristic tied v6 6-6 —
// the heuristic was correlated with raw pointsScore and didn't add a
// meaningful new signal. Deeper search is a cleaner improvement.

const SMART_V7_SEARCH_DEPTH = 5;
// Kept for back-compat; unused after the rewrite.
const SMART_V7_OPPONENT_COST_WEIGHT = 0;

function pickSmartV7(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV7Search(state, seat, myCards, trick, ctx, n);
}

function smartV7Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV7(state, seat, cand, myTeam, ctx, tracking);
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

function scoreCandidateV7(
  state: GameState,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  const r0 = applyEvent(state, {
    kind: 'play_cards',
    seat,
    cards: candidate,
  });
  if (!r0.ok) return -1e9;

  const pointsScore = simulateTricks(
    r0.state,
    myTeam,
    SMART_V7_SEARCH_DEPTH,
    SMART_V6_OPPONENT_MODEL,
  );

  let penalty = 0;
  for (const c of candidate) {
    penalty += cardCostV4(c, ctx, tracking);
  }
  return pointsScore - penalty;
}

// Like simulateTricks but also returns the final state, so callers can
// evaluate terminal heuristics (e.g., remaining-hand value).
function simulateTricksAndState(
  initial: GameState,
  myTeam: TeamId,
  depth: number,
  opponentModel: BotType = 'smart',
): { score: number; finalState: GameState } {
  let s = initial;
  let score = 0;
  let weight = 1.0;

  for (let d = 0; d < depth; d++) {
    if (s.phase !== 'play' && s.phase !== 'trick_review') break;

    let guard = 0;
    while (s.phase === 'play' && guard++ < 5) {
      const next = nextToPlay(s);
      if (next === null) break;
      const action = pickBotAction(s, next, opponentModel);
      if (action.kind !== 'play_cards') break;
      const r = applyEvent(s, {
        kind: 'play_cards',
        seat: next,
        cards: action.cards,
      });
      if (!r.ok) break;
      s = r.state;
    }

    if (s.phase === 'trick_review' && s.currentHand?.currentTrick) {
      const trick = s.currentHand.currentTrick;
      if (trick.plays.length === s.config.numPlayers) {
        const trickPoints = pointsInTrickPlays(trick.plays);
        const winner = winnerOfActiveTrick(
          trick,
          {
            trumpRank: s.currentHand.trumpRank!,
            trumpSuit: s.currentHand.trumpSuit,
          },
          s.config.numDecks,
        );
        const myWin = teamOf(s, winner) === myTeam;
        score += weight * (myWin ? trickPoints : -trickPoints);
      }
      const r = applyEvent(s, { kind: 'advance_trick' });
      if (!r.ok) break;
      s = r.state;
    } else {
      break;
    }

    weight *= SMART_V3_DISCOUNT;
  }
  return { score, finalState: s };
}

// Sum of `cardFutureValue` over every card in the opponents' hands.
function oppHandValue(
  state: GameState,
  myTeam: TeamId,
  ctx: TrumpContext,
): number {
  if (state.currentHand === null) return 0;
  let v = 0;
  for (let s = 0; s < state.config.numPlayers; s++) {
    if (teamOf(state, s) === myTeam) continue;
    const cards = state.currentHand.hands[s] ?? [];
    for (const c of cards) v += cardFutureValue(c, ctx);
  }
  return v;
}

function cardFutureValue(c: Card, ctx: TrumpContext): number {
  if (isTrump(c, ctx)) {
    if (c.kind === 'joker') return c.rank === 'big' ? 4 : 3;
    if (c.kind === 'standard') {
      if (c.rank === ctx.trumpRank && c.suit === ctx.trumpSuit) return 2.5;
      if (c.rank === ctx.trumpRank) return 1.5;
      return 0.4;
    }
    return 0.4;
  }
  if (c.kind === 'standard') {
    if (c.rank === 'A') return 2;
    if (c.rank === 'K') return 0.5;
  }
  return 0;
}

// =========================================================================
// Strategy: smart-v8
// =========================================================================
//
// v8 keeps v7's 5-ply search + v3 opponent model but raises the cost
// penalty on top cards (jokers, aces, trump-rank cards). The intent is
// to preserve game-winning resources longer — many tied games in
// earlier v8 attempts came from the bot spending an ace or trump on a
// 5-point trick when it could have been saved for the kitty multiplier.
//
// Earlier v8 attempts (all failed to beat v7):
//   - 6-ply, v3 opp: 41.7% (extra ply was noise).
//   - 5-ply, v3 opp, adaptive cost: 41.7% (scaling cost down at hand
//     end wasted high cards before the kitty multiplier).
//   - 5-ply, v4 opp: 50.0% (more accurate opp didn't help).

const SMART_V8_SEARCH_DEPTH = 5;
const SMART_V8_OPPONENT_MODEL: BotType = 'smart-v3';

function pickSmartV8(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV8Search(state, seat, myCards, trick, ctx, n);
}

function smartV8Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (candidates.length === 0) return v1Action;

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV8(state, seat, cand, myTeam, ctx, tracking);
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

function scoreCandidateV8(
  state: GameState,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
): number {
  const r0 = applyEvent(state, {
    kind: 'play_cards',
    seat,
    cards: candidate,
  });
  if (!r0.ok) return -1e9;

  const pointsScore = simulateTricks(
    r0.state,
    myTeam,
    SMART_V8_SEARCH_DEPTH,
    SMART_V8_OPPONENT_MODEL,
  );

  let penalty = 0;
  for (const c of candidate) {
    penalty += cardCostV4(c, ctx, tracking);
  }
  return pointsScore - penalty;
}

// =========================================================================
// Strategy: smart-v27
// =========================================================================
//
// Same scoring/determinization as v18, but the candidate-generation stage
// is enriched. When the bot is following while void in led suit, v18 only
// considered a single "lowest trumps + fillers" candidate. v27 also
// enumerates explicit *winning-shape* candidates so the search has a real
// chance to pick the play that actually beats the lead:
//
//  - Lead is a single (length 1): emit candidates that play a joker
//    (small joker preferred, plus big-joker variant) and a candidate
//    that plays the lowest trump-rank single.
//  - Lead is a pair: emit a candidate playing the lowest available
//    trump-rank pair OR trump-suit pair, plus low fillers.
//  - Lead is a tractor of length L: emit a candidate playing the lowest
//    trump tractor of length L, plus low fillers.
//
// These augment the candidate set; scoring/penalty (`scoreCandidateV18`)
// is unchanged, so the search may still prefer cheaper plays. The aim is
// to make sure the winning shape is at least *considered*.

const SMART_V27_DIAG = {
  builderCalls: 0,
  voidCalls: 0,
  emittedJokerSingle: 0,
  emittedTrumpRankSingle: 0,
  emittedTrumpRankPair: 0,
  emittedTrumpSuitPair: 0,
  emittedTrumpTractor: 0,
  selectedJokerSingle: 0,
  selectedTrumpRankSingle: 0,
  selectedTrumpRankPair: 0,
  selectedTrumpSuitPair: 0,
  selectedTrumpTractor: 0,
};

export function getSmartV27Diagnostics(): typeof SMART_V27_DIAG {
  return { ...SMART_V27_DIAG };
}

export function resetSmartV27Diagnostics(): void {
  for (const k of Object.keys(SMART_V27_DIAG) as Array<keyof typeof SMART_V27_DIAG>) {
    SMART_V27_DIAG[k] = 0;
  }
}

type WinningShapeKind =
  | 'joker-single'
  | 'trump-rank-single'
  | 'trump-rank-pair'
  | 'trump-suit-pair'
  | 'trump-tractor';

type LabeledCandidate = {
  readonly cards: ReadonlyArray<Card>;
  readonly kind: WinningShapeKind;
};

function pickSmartV27(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV27Search(state, seat, myCards, trick, ctx, n);
}

function smartV27Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const kindByKey = new Map<string, WinningShapeKind>();
  const consider = (cards: ReadonlyArray<Card>, kind?: WinningShapeKind): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
    if (kind !== undefined) kindByKey.set(key, kind);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (trick !== null) {
    const trumpFill = buildTrumpFillCandidate(myCards, trick, ctx);
    if (trumpFill !== null) consider(trumpFill);

    const winningCandidates = buildTrumpWinningCandidatesV27(
      myCards,
      trick,
      ctx,
      n,
    );
    for (const lc of winningCandidates) {
      consider(lc.cards, lc.kind);
    }
  }

  if (candidates.length === 0) return v1Action;

  const baseSeed = deterministicSeed(state, seat);
  const determinizations: GameState[] = [];
  for (let i = 0; i < SMART_V18_NUM_SAMPLES; i++) {
    determinizations.push(maskOpponentHands(state, seat, baseSeed + i * 1009));
  }

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  const handLeadVal = handLeadValue(myCards, ctx, tracking);
  let bestCards = candidates[0]!;
  let bestKey = candidateKeyByRank(bestCards);
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV18(
      determinizations,
      seat,
      cand,
      myTeam,
      ctx,
      tracking,
      handLeadVal,
    );
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
      bestKey = candidateKeyByRank(cand);
    }
  }
  const selectedKind = kindByKey.get(bestKey);
  if (selectedKind !== undefined) {
    switch (selectedKind) {
      case 'joker-single':
        SMART_V27_DIAG.selectedJokerSingle++;
        break;
      case 'trump-rank-single':
        SMART_V27_DIAG.selectedTrumpRankSingle++;
        break;
      case 'trump-rank-pair':
        SMART_V27_DIAG.selectedTrumpRankPair++;
        break;
      case 'trump-suit-pair':
        SMART_V27_DIAG.selectedTrumpSuitPair++;
        break;
      case 'trump-tractor':
        SMART_V27_DIAG.selectedTrumpTractor++;
        break;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

// Build an array of "winning-shape" candidates. Only fires when the bot
// is void in led suit. Returns [] otherwise. Each candidate is a full
// play (length === lead length).
function buildTrumpWinningCandidatesV27(
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick,
  ctx: TrumpContext,
  n: number,
): ReadonlyArray<LabeledCandidate> {
  SMART_V27_DIAG.builderCalls++;
  const lead = trick.plays[0]!;
  const ledSuit = effectiveSuit(lead.cards[0]!, ctx);
  const leadCount = lead.cards.length;

  const myInLed = myCards.filter((c) => effectiveSuit(c, ctx) === ledSuit);
  if (myInLed.length > 0) return [];
  SMART_V27_DIAG.voidCalls++;

  const leadClass = classifyCombo(lead.cards, ctx, n);
  if (!leadClass.ok) return [];

  const out: LabeledCandidate[] = [];

  // Pick lowest fillers from myCards minus usedCards, preferring
  // non-trump (ascending) then lowest trumps.
  const pickFillers = (usedCards: ReadonlyArray<Card>, need: number): Card[] => {
    if (need <= 0) return [];
    const usedCounts = new Map<string, number>();
    for (const u of usedCards) {
      const k = cardKey(u);
      usedCounts.set(k, (usedCounts.get(k) ?? 0) + 1);
    }
    const remaining: Card[] = [];
    for (const c of myCards) {
      const k = cardKey(c);
      const used = usedCounts.get(k) ?? 0;
      if (used > 0) {
        usedCounts.set(k, used - 1);
      } else {
        remaining.push(c);
      }
    }
    const nonTrumpAsc = remaining
      .filter((c) => !isTrump(c, ctx))
      .sort((a, b) => compareCards(a, b, ctx));
    const trumpAsc = remaining
      .filter((c) => isTrump(c, ctx))
      .sort((a, b) => compareCards(a, b, ctx));
    const fillers: Card[] = [];
    for (const c of nonTrumpAsc) {
      if (fillers.length >= need) break;
      fillers.push(c);
    }
    for (const c of trumpAsc) {
      if (fillers.length >= need) break;
      fillers.push(c);
    }
    return fillers;
  };

  if (leadClass.combo.kind === 'single' && leadCount === 1) {
    const jokers = myCards
      .filter((c) => c.kind === 'joker')
      .sort((a, b) => compareCards(a, b, ctx));
    if (jokers.length > 0) {
      const small = jokers[0]!;
      out.push({ cards: [small], kind: 'joker-single' });
      SMART_V27_DIAG.emittedJokerSingle++;
      const big = jokers[jokers.length - 1]!;
      if (cardKey(big) !== cardKey(small)) {
        out.push({ cards: [big], kind: 'joker-single' });
        SMART_V27_DIAG.emittedJokerSingle++;
      }
    }
    const trumpRankCards = myCards.filter(
      (c) => c.kind === 'standard' && c.rank === ctx.trumpRank,
    );
    if (trumpRankCards.length > 0) {
      const trumpSuitRank = trumpRankCards.find(
        (c) => c.kind === 'standard' && c.suit === ctx.trumpSuit,
      );
      const chosen = trumpSuitRank ?? trumpRankCards[0]!;
      out.push({ cards: [chosen], kind: 'trump-rank-single' });
      SMART_V27_DIAG.emittedTrumpRankSingle++;
    }
  } else if (leadClass.combo.kind === 'nkind' && leadClass.combo.n === 2) {
    const pairs = findNKinds(myCards, 2);
    const trumpRankPairs = pairs.filter((p) => {
      const c0 = p.cards[0]!;
      return c0.kind === 'standard' && c0.rank === ctx.trumpRank;
    });
    if (trumpRankPairs.length > 0) {
      trumpRankPairs.sort((a, b) =>
        compareCards(a.cards[0]!, b.cards[0]!, ctx),
      );
      const pickedCards = trumpRankPairs[0]!.cards;
      const fillers = pickFillers(pickedCards, leadCount - pickedCards.length);
      const cards = [...pickedCards, ...fillers];
      if (cards.length === leadCount) {
        out.push({ cards, kind: 'trump-rank-pair' });
        SMART_V27_DIAG.emittedTrumpRankPair++;
      }
    }
    const trumpSuitPairs = pairs.filter((p) => {
      const c0 = p.cards[0]!;
      if (c0.kind !== 'standard') return false;
      if (c0.rank === ctx.trumpRank) return false;
      return isTrump(c0, ctx) && c0.suit === ctx.trumpSuit;
    });
    if (trumpSuitPairs.length > 0) {
      trumpSuitPairs.sort((a, b) =>
        compareCards(a.cards[0]!, b.cards[0]!, ctx),
      );
      const pickedCards = trumpSuitPairs[0]!.cards;
      const fillers = pickFillers(pickedCards, leadCount - pickedCards.length);
      const cards = [...pickedCards, ...fillers];
      if (cards.length === leadCount) {
        out.push({ cards, kind: 'trump-suit-pair' });
        SMART_V27_DIAG.emittedTrumpSuitPair++;
      }
    }
  } else if (leadClass.combo.kind === 'tractor') {
    const wantLen = leadClass.combo.length;
    const wantN = leadClass.combo.n;
    const tractors = findTractors(myCards, ctx, n, 2);
    const trumpTractors = tractors.filter((t) => {
      if (t.length !== wantLen) return false;
      if (t.n !== wantN) return false;
      const c0 = t.cards[0]!;
      return isTrump(c0, ctx);
    });
    if (trumpTractors.length > 0) {
      trumpTractors.sort((a, b) =>
        compareCards(a.cards[0]!, b.cards[0]!, ctx),
      );
      const picked = trumpTractors[0]!.cards;
      const fillers = pickFillers(picked, leadCount - picked.length);
      const cards = [...picked, ...fillers];
      if (cards.length === leadCount) {
        out.push({ cards, kind: 'trump-tractor' });
        SMART_V27_DIAG.emittedTrumpTractor++;
      }
    }
  }

  return out;
}

// =========================================================================
// Strategy: smart-v28
// =========================================================================
//
// v18 + situational cost relaxation. v27's diagnostic showed that the
// candidate-gen layer already emits winning-shape candidates (jokers,
// trump-rank pairs, etc.), but the cost penalty (`cardCostV9`) is so
// strong that the search rejects them even when they would capture a
// high-points trick: jokers offered 239 times, selected only 7 times
// (3%). cardCostV9's save-costs are calibrated for long-term hoarding
// and don't adapt to the current trick's stakes.
//
// v28 keeps every part of v18 intact and only changes the penalty
// inside `scoreCandidate`. When the bot is expected to win the current
// trick (`avgWins > 0.6`) AND the trick already has ≥5 points on the
// table, every high-cost card (cost ≥ 1.4 — jokers, trump rank, A) is
// scaled by `factor = max(0.25, 1 − (pts − 5) × 0.05)`. So 5 pts →
// 0.75×, 10 pts → 0.5×, 15+ pts → 0.25× (clamped). Candidate generation
// is identical to v18 — this isolates the cost-change effect.
//
// 16-game benchmark vs v18 (seeds 2801/2803 then 2821/2823): 9-7
// (56.25%). Statistical p ≈ 0.40 at n=16 — not definitive, but the
// only positive signal across 9 architectural variants attempted in
// this lineage. New tentative peak; v18 retained in dropdown.
//
// Telemetry: when env var SMART_V28_TRACE is truthy, count how often
// the relaxation fires (any card in the candidate had its cost
// reduced) so we can verify the situational signal is changing
// decisions.

let _smartV28RelaxFireCount = 0;
let _smartV28RelaxOfferCount = 0;
const SMART_V28_TRACE = ((): boolean => {
  try {
    return (
      typeof process !== 'undefined' &&
      process.env !== undefined &&
      Boolean(process.env.SMART_V28_TRACE)
    );
  } catch {
    return false;
  }
})();

export function smartV28RelaxCounters(): {
  fired: number;
  offered: number;
} {
  return { fired: _smartV28RelaxFireCount, offered: _smartV28RelaxOfferCount };
}
export function resetSmartV28RelaxCounters(): void {
  _smartV28RelaxFireCount = 0;
  _smartV28RelaxOfferCount = 0;
}

function currentTrickPointsOnTable(trick: ActiveTrick | null): number {
  if (trick === null) return 0;
  let pts = 0;
  for (const p of trick.plays) {
    for (const c of p.cards) {
      pts += pointsForCard(c);
    }
  }
  return pts;
}

function pickSmartV28(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV28Search(state, seat, myCards, trick, ctx, n);
}

function smartV28Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  // Candidate generation: identical to v18 — isolate the cost change.
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (trick !== null) {
    const trumpFillCandidate = buildTrumpFillCandidate(myCards, trick, ctx);
    if (trumpFillCandidate !== null) consider(trumpFillCandidate);
  }

  if (candidates.length === 0) return v1Action;

  const baseSeed = deterministicSeed(state, seat);
  const determinizations: GameState[] = [];
  for (let i = 0; i < SMART_V18_NUM_SAMPLES; i++) {
    determinizations.push(maskOpponentHands(state, seat, baseSeed + i * 1009));
  }

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  const handLeadVal = handLeadValue(myCards, ctx, tracking);
  const trickPointsOnTable = currentTrickPointsOnTable(trick);

  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV28(
      determinizations,
      seat,
      cand,
      myTeam,
      ctx,
      tracking,
      handLeadVal,
      trickPointsOnTable,
    );
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  return { kind: 'play_cards', cards: bestCards };
}

function scoreCandidateV28(
  determinizations: ReadonlyArray<GameState>,
  seat: Seat,
  candidate: ReadonlyArray<Card>,
  myTeam: TeamId,
  ctx: TrumpContext,
  tracking: PlayedTracking,
  handLeadVal: number,
  trickPointsOnTable: number,
): number {
  let totalPointsScore = 0;
  let totalWins = 0;
  let validSamples = 0;
  for (const det of determinizations) {
    const r0 = applyEvent(det, {
      kind: 'play_cards',
      seat,
      cards: candidate,
    });
    if (!r0.ok) continue;
    const { score, wins } = simulateTricksAndWinsV18(
      r0.state,
      myTeam,
      SMART_V8_SEARCH_DEPTH,
      SMART_V8_OPPONENT_MODEL,
    );
    totalPointsScore += score;
    totalWins += wins;
    validSamples++;
  }
  if (validSamples === 0) return -1e9;
  const avgPointsScore = totalPointsScore / validSamples;
  const avgWins = totalWins / validSamples;

  let penalty = 0;
  let anyRelaxed = false;
  let anyOffered = false;
  for (const c of candidate) {
    let cost = cardCostV9(c, ctx, tracking);
    if (cost >= 1.4) {
      anyOffered = true;
      // Situational relaxation: if I'm expected to win this trick AND
      // the trick already has ≥5 points on the table, scale down the
      // save-cost of high-cost cards (joker, trump-rank, A).
      if (trickPointsOnTable >= 5 && avgWins > 0.6) {
        const factor = Math.max(0.25, 1 - (trickPointsOnTable - 5) * 0.05);
        if (factor < 1) {
          cost *= factor;
          anyRelaxed = true;
        }
      }
    }
    penalty += cost;
  }
  if (SMART_V28_TRACE) {
    if (anyOffered) _smartV28RelaxOfferCount++;
    if (anyRelaxed) _smartV28RelaxFireCount++;
  }
  const leadBonus = SMART_V18_LEAD_BONUS_WEIGHT * handLeadVal * avgWins;
  return avgPointsScore - penalty + leadBonus;
}

// =========================================================================
// Strategy: smart-v29
// =========================================================================
//
// Compound of v27 + v28: enumerate v27's explicit winning-shape trump
// candidates (jokers, trump-rank pair, trump-suit pair, trump tractor)
// when void in led suit, AND apply v28's situational save-cost
// relaxation in the candidate scoring. v27 alone only tied vs v18
// because cardCostV9 still rejected those high-cost candidates even
// when capturing a high-points trick — v28's relaxation should now let
// them actually win the selection step when the stakes are high.

const SMART_V29_DIAG = {
  offered: 0,   // turns where buildTrumpWinningCandidatesV27 emitted ≥1 candidate
  selected: 0,  // turns where the chosen play matched a v27 candidate
};
export function getSmartV29Diagnostics(): typeof SMART_V29_DIAG {
  return { ...SMART_V29_DIAG };
}
export function resetSmartV29Diagnostics(): void {
  SMART_V29_DIAG.offered = 0;
  SMART_V29_DIAG.selected = 0;
}

function pickSmartV29(state: GameState, seat: Seat): BotAction {
  const hand = state.currentHand;
  if (hand === null) return { kind: 'wait' };
  const myCards = hand.hands[seat] ?? [];

  if (state.phase === 'declare_lockout' || state.phase === 'declare_open') {
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      return { kind: 'pick_up_kitty' };
    }
    return smartDeclare(state, seat, myCards, hand);
  }
  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return { kind: 'wait' };
    if (hand.trumpRank === null) return { kind: 'wait' };
    return smartBury(state, hand, myCards);
  }
  if (state.phase !== 'play') return { kind: 'wait' };
  if (hand.trumpRank === null) return { kind: 'wait' };
  if (turnSeat(state) !== seat) return { kind: 'wait' };

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const trick = hand.currentTrick;

  return smartV29Search(state, seat, myCards, trick, ctx, n);
}

function smartV29Search(
  state: GameState,
  seat: Seat,
  myCards: ReadonlyArray<Card>,
  trick: ActiveTrick | null,
  ctx: TrumpContext,
  n: number,
): BotAction {
  // Candidate generation: identical to v18 + v27's winning-shape adds.
  const seen = new Set<string>();
  const candidates: ReadonlyArray<Card>[] = [];
  const consider = (cards: ReadonlyArray<Card>): void => {
    if (cards.length === 0) return;
    const key = candidateKeyByRank(cards);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push(cards);
  };

  for (const ev of legalEvents(state, seat)) {
    if (ev.kind === 'play_cards') consider(ev.cards);
  }
  const v1Action =
    trick === null
      ? smartLead(myCards, ctx, n)
      : smartFollow(state, seat, myCards, trick, ctx, n);
  if (v1Action.kind === 'play_cards') consider(v1Action.cards);

  if (trick !== null) {
    const trumpFillCandidate = buildTrumpFillCandidate(myCards, trick, ctx);
    if (trumpFillCandidate !== null) consider(trumpFillCandidate);
  }

  const v27CandidateKeys = new Set<string>();
  if (trick !== null) {
    const winningCandidates = buildTrumpWinningCandidatesV27(
      myCards,
      trick,
      ctx,
      n,
    );
    for (const wc of winningCandidates) {
      v27CandidateKeys.add(candidateKeyByRank(wc.cards));
      consider(wc.cards);
    }
    if (winningCandidates.length > 0) SMART_V29_DIAG.offered++;
  }

  if (candidates.length === 0) return v1Action;

  const baseSeed = deterministicSeed(state, seat);
  const determinizations: GameState[] = [];
  for (let i = 0; i < SMART_V18_NUM_SAMPLES; i++) {
    determinizations.push(maskOpponentHands(state, seat, baseSeed + i * 1009));
  }

  const tracking = buildPlayedTracking(state, ctx, myCards);
  const myTeam = teamOf(state, seat);
  const handLeadVal = handLeadValue(myCards, ctx, tracking);
  const trickPointsOnTable = currentTrickPointsOnTable(trick);

  let bestCards = candidates[0]!;
  let bestScore = -Infinity;
  for (const cand of candidates) {
    const score = scoreCandidateV28(
      determinizations,
      seat,
      cand,
      myTeam,
      ctx,
      tracking,
      handLeadVal,
      trickPointsOnTable,
    );
    if (score > bestScore) {
      bestScore = score;
      bestCards = cand;
    }
  }
  if (v27CandidateKeys.has(candidateKeyByRank(bestCards))) {
    SMART_V29_DIAG.selected++;
  }
  return { kind: 'play_cards', cards: bestCards };
}
