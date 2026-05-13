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
  'rl-mlp',
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
      return 'Smart v0 (== Smart)';
    case 'smart':
      return 'Smart';
    case 'smart-v2':
      return 'Smart v2';
    case 'smart-v3':
      return 'Smart v3 (2-ply search)';
    case 'smart-v4':
      return 'Smart v4 (tracking + better decl/kitty)';
    case 'smart-v5':
      return 'Smart v5 (v3 opponent model)';
    case 'smart-v6':
      return 'Smart v6 (4-ply search, v3 opponent)';
    case 'smart-v7':
      return 'Smart v7 (5-ply search)';
    case 'smart-v8':
      return 'Smart v8 (5-ply, stronger save-the-aces penalty)';
    case 'smart-v9':
      return 'Smart v9 (+ even stronger Ace penalty)';
    case 'smart-v10':
      return 'Smart v10 (Ace penalty 3.0)';
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
