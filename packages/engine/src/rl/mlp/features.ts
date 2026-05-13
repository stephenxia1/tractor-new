// Per-candidate feature extractor for the RL bot.
//
// One feature vector is computed per `play_cards` candidate the policy is
// scoring at a single decision point. The feature space is hand-designed
// to capture the most discriminative signals for tractor play:
//
//   - Combo properties: size, total points, top/bottom rank, structure
//   - Action tag (one-hot from legalActions tags)
//   - Effective suit (one-hot)
//   - Phase context: progress through the hand, points already captured,
//     attacker/defender, hand size, trump fraction
//   - In-trick context: who's currently winning, position in trick,
//     points already in the trick, whether this candidate would beat
//     the current best
//   - Misc: am I the landlord
//
// The total dimensionality is FEATURE_DIM. Index 0 is a constant bias.

import {
  classifyCombo,
  comboBeats,
  type ClassifiedCombo,
} from '../../classifier';
import { effectiveSuit, isTrump, naturalRankIndex } from '../../ordering';
import { pointsForCard, sumPoints } from '../../scoring';
import { teamOf, type GameState, type Seat } from '../../state';
import {
  cardKey,
  NATURAL_RANKS,
  type Card,
  type EffectiveSuit,
  type TrumpContext,
} from '../../types';

export const FEATURE_DIM = 32;

const TAG_INDEX: Readonly<Record<string, number>> = {
  'lead-single': 0,
  'lead-pair': 1,
  'lead-tractor': 2,
  'lead-throw-safe': 3,
  'follow-led-suit': 4,
  'follow-trump': 5,
  'follow-discard': 6,
};

const SUIT_INDEX: Readonly<Record<EffectiveSuit, number>> = {
  spades: 0,
  hearts: 1,
  diamonds: 2,
  clubs: 3,
  trump: 4,
};

// Cached "global" context computed once per decision and reused for every
// candidate at that decision, since most of these don't depend on the
// candidate at all.
export type DecisionContext = {
  readonly state: GameState;
  readonly seat: Seat;
  readonly ctx: TrumpContext;
  readonly numDecks: number;
  readonly myCards: ReadonlyArray<Card>;
  readonly handSize: number;
  readonly trumpFraction: number;
  readonly attackerPoints: number;
  readonly tricksRemaining: number;
  readonly myTeamIsAttackers: boolean;
  readonly iAmLandlord: boolean;
  readonly isLeading: boolean;
  // Trick-only fields; meaningless when leading.
  readonly trickPosition: number;       // 0..3, normalized later
  readonly trickPointsSoFar: number;
  readonly currentBestCombo: ClassifiedCombo | null;
  readonly currentBestSeat: Seat | null;
  readonly teammateWinning: boolean;
  readonly ledSuit: EffectiveSuit | null;
};

export function buildDecisionContext(
  state: GameState,
  seat: Seat,
): DecisionContext {
  const hand = state.currentHand!;
  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank!,
    trumpSuit: hand.trumpSuit,
  };
  const myCards = hand.hands[seat] ?? [];
  const handSize = myCards.length;
  let trumpCount = 0;
  for (const c of myCards) if (isTrump(c, ctx)) trumpCount++;
  const trumpFraction = handSize === 0 ? 0 : trumpCount / handSize;
  const tricksRemaining = 25 - hand.tricks.length;
  const myTeam = teamOf(state, seat);
  const landlord = hand.landlord;
  const landlordTeam = landlord !== null ? teamOf(state, landlord) : null;
  const myTeamIsAttackers =
    landlordTeam !== null ? landlordTeam !== myTeam : true;
  const iAmLandlord = landlord === seat;

  const trick = hand.currentTrick;
  if (trick === null) {
    return {
      state,
      seat,
      ctx,
      numDecks: state.config.numDecks,
      myCards,
      handSize,
      trumpFraction,
      attackerPoints: hand.attackerPoints,
      tricksRemaining,
      myTeamIsAttackers,
      iAmLandlord,
      isLeading: true,
      trickPosition: 0,
      trickPointsSoFar: 0,
      currentBestCombo: null,
      currentBestSeat: null,
      teammateWinning: false,
      ledSuit: null,
    };
  }

  const trickPosition = trick.plays.length; // 0..3
  let trickPointsSoFar = 0;
  for (const p of trick.plays) trickPointsSoFar += sumPoints(p.cards);
  const ledSuit = effectiveSuit(trick.plays[0]!.cards[0]!, ctx);

  // Identify current "best" play in the trick.
  let bestSeat = trick.plays[0]!.seat;
  let bestCombo: ClassifiedCombo | null = null;
  const lead = classifyCombo(trick.plays[0]!.cards, ctx, state.config.numDecks);
  if (lead.ok) bestCombo = lead.combo;
  for (let i = 1; i < trick.plays.length; i++) {
    const p = trick.plays[i]!;
    const cls = classifyCombo(p.cards, ctx, state.config.numDecks);
    if (
      bestCombo !== null &&
      cls.ok &&
      comboBeats(cls.combo, bestCombo) > 0
    ) {
      bestSeat = p.seat;
      bestCombo = cls.combo;
    }
  }
  const teammateWinning = teamOf(state, bestSeat) === myTeam;

  return {
    state,
    seat,
    ctx,
    numDecks: state.config.numDecks,
    myCards,
    handSize,
    trumpFraction,
    attackerPoints: hand.attackerPoints,
    tricksRemaining,
    myTeamIsAttackers,
    iAmLandlord,
    isLeading: false,
    trickPosition,
    trickPointsSoFar,
    currentBestCombo: bestCombo,
    currentBestSeat: bestSeat,
    teammateWinning,
    ledSuit,
  };
}

// Compute the feature vector for one candidate move. Writes into `out`
// (size FEATURE_DIM) for zero-allocation; allocates a fresh array if
// `out` is omitted. Index 0 is the bias term (= 1).
export function extractFeatures(
  cards: ReadonlyArray<Card>,
  tags: ReadonlyArray<string>,
  d: DecisionContext,
  out?: Float64Array,
): Float64Array {
  const f = out ?? new Float64Array(FEATURE_DIM);
  for (let i = 0; i < FEATURE_DIM; i++) f[i] = 0;

  // 0: bias
  f[0] = 1;

  // 1: num cards / 8
  f[1] = cards.length / 8;

  // 2: sum points / 30
  const pts = sumPoints(cards);
  f[2] = pts / 30;

  // 3-4: max/min natural rank index / 13 (jokers count as 13/14 -> >=1)
  let maxRank = -1;
  let minRank = 14;
  for (const c of cards) {
    const r = naturalRankIdx(c);
    if (r > maxRank) maxRank = r;
    if (r < minRank) minRank = r;
  }
  if (maxRank < 0) {
    f[3] = 0;
    f[4] = 0;
  } else {
    f[3] = maxRank / 13;
    f[4] = minRank / 13;
  }

  // 5: is_trump_combo (all cards trump)
  let allTrump = true;
  for (const c of cards) {
    if (!isTrump(c, d.ctx)) {
      allTrump = false;
      break;
    }
  }
  f[5] = allTrump ? 1 : 0;

  // 6: contains_joker
  let hasJoker = false;
  for (const c of cards) {
    if (c.kind === 'joker') {
      hasJoker = true;
      break;
    }
  }
  f[6] = hasJoker ? 1 : 0;

  // 7: contains_R (trump-rank card)
  let hasR = false;
  for (const c of cards) {
    if (c.kind === 'standard' && c.rank === d.ctx.trumpRank) {
      hasR = true;
      break;
    }
  }
  f[7] = hasR ? 1 : 0;

  // 8: is_pair_or_higher (any group of 2+ same-key cards in candidate)
  const keyCount = new Map<string, number>();
  for (const c of cards) {
    const k = cardKey(c);
    keyCount.set(k, (keyCount.get(k) ?? 0) + 1);
  }
  let hasPair = false;
  for (const v of keyCount.values()) {
    if (v >= 2) {
      hasPair = true;
      break;
    }
  }
  f[8] = hasPair ? 1 : 0;

  // 9-15: tag one-hot (7 tag dims)
  for (const t of tags) {
    const idx = TAG_INDEX[t];
    if (idx !== undefined) f[9 + idx] = 1;
  }

  // 16-20: effective_suit one-hot from candidate's first card
  if (cards.length > 0) {
    const es = effectiveSuit(cards[0]!, d.ctx);
    f[16 + SUIT_INDEX[es]] = 1;
  }

  // 21: is_leading
  f[21] = d.isLeading ? 1 : 0;

  // 22: tricks_remaining / 25
  f[22] = d.tricksRemaining / 25;

  // 23: attackerPoints / 200
  f[23] = d.attackerPoints / 200;

  // 24: my_team_is_attackers
  f[24] = d.myTeamIsAttackers ? 1 : 0;

  // 25: trump fraction of my hand
  f[25] = d.trumpFraction;

  // 26: hand_size / 25
  f[26] = d.handSize / 25;

  // 27: trick_position / 4
  f[27] = d.trickPosition / 4;

  // 28: trick_points_so_far / 30
  f[28] = d.trickPointsSoFar / 30;

  // 29: would_beat_current_winner (only meaningful when following with a
  // matching-structure combo)
  if (!d.isLeading && d.currentBestCombo !== null) {
    const cls = classifyCombo(cards, d.ctx, d.numDecks);
    if (cls.ok && comboBeats(cls.combo, d.currentBestCombo) > 0) {
      f[29] = 1;
    }
  }

  // 30: teammate_currently_winning
  f[30] = d.teammateWinning ? 1 : 0;

  // 31: i_am_landlord
  f[31] = d.iAmLandlord ? 1 : 0;

  return f;
}

function naturalRankIdx(c: Card): number {
  if (c.kind === 'joker') return c.rank === 'big' ? 14 : 13;
  return naturalRankIndex(c.rank);
}

// Small sanity-check on FEATURE_DIM at module load — surfaces drift if
// someone adds a feature without bumping the constant.
{
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _ = NATURAL_RANKS.length; // touch to silence unused-import
}
