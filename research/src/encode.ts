// Serialises one decision (and one finished hand) into the wire format
// defined in record.ts. Everything written here is information the acting
// seat legitimately has: its own hand, the public play history, and its
// own buried kitty. Nothing reads another seat's cards.

import {
  NATURAL_RANKS,
  SUITS,
  computeHandResult,
  effectiveSuit,
  sumPoints,
  teamOf,
  type Card,
  type GameState,
  type HandState,
  type Seat,
  type TrumpContext,
} from '@tractor/engine';

import { addToPlane, cardSlot, NUM_SLOTS } from './cards';
import { winningSeat, type Candidate } from './candidates';
import {
  ByteWriter,
  CANDIDATE_BYTES,
  CANDIDATE_HEADER_BYTES,
  MAX_CANDIDATE_CARDS,
  NO_CARD,
  E_ATTACKER_POINTS,
  E_DEAL_DIGEST,
  E_DEFENDERS_WON,
  E_HAND_ID,
  E_KITTY_POINTS,
  E_GAME_ID,
  E_LANDLORD_SEAT,
  E_LEVEL_DELTA,
  E_NUM_TRICKS,
  E_TRICKS_BASE,
  E_TRUMP_RANK,
  E_TRUMP_SUIT,
  EPISODE_BYTES,
  H_ATTACKER_POINTS,
  H_CARDS_SEEN,
  H_GAME_ID,
  H_HAND_ID,
  H_HAND_SIZE,
  H_I_AM_ATTACKER,
  H_IS_LEAD,
  H_LANDLORD_REL,
  H_NUM_CANDIDATES,
  H_ROLE,
  H_SEAT,
  H_TABLE_POINTS,
  H_TRICK_INDEX,
  H_TRICK_POS,
  H_TRUMP_RANK,
  H_TRUMP_SUIT,
  H_VERSION,
  H_VOID_MASK,
  H_WINNER_REL,
  HEADER_BYTES,
  HISTORY_BYTES,
  HISTORY_MAX_CARDS,
  HISTORY_STEP_BYTES,
  HISTORY_STEPS,
  H_HAS_PRIVILEGED,
  H_HISTORY_LEN,
  H_LEAF_OF,
  H_LEAF_POINTS,
  NOT_A_LEAF,
  MAX_TRICKS,
  PRIVILEGED_BYTES,
  PLANE_MY_HAND,
  PLANE_MY_KITTY,
  PLANE_PLAYED_BASE,
  PLANE_TRICK_BASE,
  PLANES_BYTES,
  RECORD_VERSION,
} from './record';

export const ROLE_LANDLORD = 0;
export const ROLE_DEFENDER = 1;
export const ROLE_ATTACKER = 2;

export const NO_TRUMP_SUIT_INDEX = 4;

export function roleOf(state: GameState, seat: Seat, landlord: Seat): number {
  if (seat === landlord) return ROLE_LANDLORD;
  return teamOf(state, seat) === teamOf(state, landlord)
    ? ROLE_DEFENDER
    : ROLE_ATTACKER;
}

export function trumpSuitIndex(hand: HandState): number {
  if (hand.trumpSuit === null) return NO_TRUMP_SUIT_INDEX;
  return SUITS.indexOf(hand.trumpSuit);
}

// Suits each seat is publicly known to be out of: they followed a trick
// led in suit X with something other than X, which the follow rules only
// permit once their X cards are exhausted. Bit 0..3 are SUITS, bit 4 trump.
export function voidMasks(
  state: GameState,
  ctx: TrumpContext,
): Uint8Array {
  const masks = new Uint8Array(state.config.numPlayers);
  const hand = state.currentHand;
  if (hand === null) return masks;
  const suitBit = (es: string): number => {
    if (es === 'trump') return 1 << 4;
    const idx = SUITS.indexOf(es as (typeof SUITS)[number]);
    return idx < 0 ? 0 : 1 << idx;
  };
  const scan = (plays: ReadonlyArray<{ seat: Seat; cards: ReadonlyArray<Card> }>): void => {
    const lead = plays[0];
    if (lead === undefined) return;
    const led = effectiveSuit(lead.cards[0]!, ctx);
    const bit = suitBit(led);
    for (let i = 1; i < plays.length; i++) {
      const play = plays[i]!;
      const offSuit = play.cards.some((c) => effectiveSuit(c, ctx) !== led);
      if (offSuit) masks[play.seat] = masks[play.seat]! | bit;
    }
  };
  for (const t of hand.tricks) scan(t.plays);
  if (hand.currentTrick !== null) scan(hand.currentTrick.plays);
  return masks;
}

export type DecisionMeta = {
  readonly gameId: number;
  readonly handId: number;
  readonly seat: Seat;
  readonly myKitty: ReadonlyArray<Card>;
  // Training only. When true the record carries the other seats' hands and
  // the kitty for a centralised critic. Never set this for a record whose
  // decision will be made by the policy in a real game.
  readonly privileged?: boolean;
  // Search leaf metadata: which candidate this position is evidence for,
  // and the points the simulated trick settled (signed for this seat).
  readonly leafOf?: number;
  readonly leafPoints?: number;
};

// Writes: u16 recordBytes | header | planes | candidates.
export function writeDecision(
  w: ByteWriter,
  state: GameState,
  meta: DecisionMeta,
  candidates: ReadonlyArray<Candidate>,
): void {
  const hand = state.currentHand;
  if (hand === null || hand.trumpRank === null || hand.landlord === null) {
    throw new Error('writeDecision requires a live hand with trump set');
  }
  const seat = meta.seat;
  const numPlayers = state.config.numPlayers;
  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const trick = hand.currentTrick;
  const isLead = trick === null || trick.plays.length === 0;

  const lenAt = w.reserve(2);
  const start = w.length;

  // ---- header ----
  const header = w.claim(HEADER_BYTES);
  header[H_VERSION] = RECORD_VERSION;
  header[H_SEAT] = seat;
  header[H_ROLE] = roleOf(state, seat, hand.landlord);
  header[H_LANDLORD_REL] = (hand.landlord - seat + numPlayers) % numPlayers;
  header[H_TRUMP_RANK] = NATURAL_RANKS.indexOf(hand.trumpRank);
  header[H_TRUMP_SUIT] = trumpSuitIndex(hand);
  header[H_TRICK_INDEX] = hand.tricks.length;
  header[H_TRICK_POS] = trick === null ? 0 : trick.plays.length;
  header[H_IS_LEAD] = isLead ? 1 : 0;
  header[H_HAND_SIZE] = (hand.hands[seat] ?? []).length;

  let tablePoints = 0;
  let winnerRel = 255;
  if (trick !== null && trick.plays.length > 0) {
    for (const p of trick.plays) tablePoints += sumPoints(p.cards);
    const ws = winningSeat(trick, ctx, state.config.numDecks);
    if (ws !== null) winnerRel = (ws - seat + numPlayers) % numPlayers;
  }
  header[H_WINNER_REL] = winnerRel;
  header[H_TABLE_POINTS] = Math.min(tablePoints, 255);

  let cardsSeen = 0;
  for (const t of hand.tricks) for (const p of t.plays) cardsSeen += p.cards.length;
  if (trick !== null) for (const p of trick.plays) cardsSeen += p.cards.length;

  const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
  view.setUint16(H_ATTACKER_POINTS, Math.min(hand.attackerPoints, 65535), true);
  view.setUint16(H_CARDS_SEEN, cardsSeen, true);
  header[H_NUM_CANDIDATES] = Math.min(candidates.length, 255);
  header[H_I_AM_ATTACKER] =
    teamOf(state, seat) === teamOf(state, hand.landlord) ? 0 : 1;

  header[H_LEAF_OF] = meta.leafOf === undefined ? NOT_A_LEAF : meta.leafOf;
  view.setInt16(H_LEAF_POINTS, Math.max(-30000, Math.min(30000, meta.leafPoints ?? 0)), true);

  const masks = voidMasks(state, ctx);
  for (let rel = 0; rel < 4; rel++) {
    const other = (seat + rel) % numPlayers;
    header[H_VOID_MASK + rel] = masks[other] ?? 0;
  }
  view.setUint16(H_GAME_ID, meta.gameId, true);
  view.setUint32(H_HAND_ID, meta.handId, true);

  // ---- planes ----
  const planes = w.claim(PLANES_BYTES);
  addToPlane(hand.hands[seat] ?? [], planes, PLANE_MY_HAND * NUM_SLOTS);
  for (const t of hand.tricks) {
    for (const p of t.plays) {
      const rel = (p.seat - seat + numPlayers) % numPlayers;
      addToPlane(p.cards, planes, (PLANE_PLAYED_BASE + rel) * NUM_SLOTS);
    }
  }
  if (trick !== null) {
    for (const p of trick.plays) {
      const rel = (p.seat - seat + numPlayers) % numPlayers;
      addToPlane(p.cards, planes, (PLANE_PLAYED_BASE + rel) * NUM_SLOTS);
      addToPlane(p.cards, planes, (PLANE_TRICK_BASE + rel) * NUM_SLOTS);
    }
  }
  if (seat === hand.landlord && meta.myKitty.length > 0) {
    addToPlane(meta.myKitty, planes, PLANE_MY_KITTY * NUM_SLOTS);
  }

  // ---- move history (most recent last) ----
  // Ordering, timing and authorship, which the aggregate planes drop.
  type Step = { seat: Seat; cards: ReadonlyArray<Card>; trickIndex: number; pos: number };
  const steps: Step[] = [];
  hand.tricks.forEach((t, ti) => {
    t.plays.forEach((p, pi) => {
      steps.push({ seat: p.seat, cards: p.cards, trickIndex: ti, pos: pi });
    });
  });
  if (trick !== null) {
    trick.plays.forEach((p, pi) => {
      steps.push({ seat: p.seat, cards: p.cards, trickIndex: hand.tricks.length, pos: pi });
    });
  }
  const recent = steps.slice(-HISTORY_STEPS);
  const historyBuf = w.claim(HISTORY_BYTES);
  recent.forEach((step, i) => {
    const off = i * HISTORY_STEP_BYTES;
    const n = Math.min(step.cards.length, HISTORY_MAX_CARDS);
    historyBuf[off] = n;
    historyBuf[off + 1] = (step.seat - seat + numPlayers) % numPlayers;
    historyBuf[off + 2] = Math.min(step.trickIndex, 255);
    historyBuf[off + 3] = step.pos;
    for (let c = 0; c < n; c++) {
      historyBuf[off + 4 + c] = cardSlot(step.cards[c]!);
    }
    for (let c = n; c < HISTORY_MAX_CARDS; c++) {
      historyBuf[off + 4 + c] = NO_CARD;
    }
  });
  for (let i = recent.length; i < HISTORY_STEPS; i++) {
    const off = i * HISTORY_STEP_BYTES;
    for (let c = 0; c < HISTORY_MAX_CARDS; c++) historyBuf[off + 4 + c] = NO_CARD;
  }
  header[H_HISTORY_LEN] = recent.length;

  // ---- privileged block (training only) ----
  header[H_HAS_PRIVILEGED] = meta.privileged === true ? 1 : 0;
  if (meta.privileged === true) {
    const priv = w.claim(PRIVILEGED_BYTES);
    for (let rel = 1; rel < numPlayers; rel++) {
      const other = (seat + rel) % numPlayers;
      addToPlane(hand.hands[other] ?? [], priv, (rel - 1) * NUM_SLOTS);
    }
    addToPlane(hand.kitty, priv, 3 * NUM_SLOTS);
  }

  // ---- candidates (fixed width, padded) ----
  for (const c of candidates) {
    const slots = w.claim(CANDIDATE_BYTES);
    slots[0] = c.cards.length;
    slots[1] = c.shape;
    slots[2] = c.flags;
    slots[3] = c.points;
    slots.fill(NO_CARD, CANDIDATE_HEADER_BYTES, CANDIDATE_BYTES);
    for (let i = 0; i < c.cards.length && i < MAX_CANDIDATE_CARDS; i++) {
      slots[CANDIDATE_HEADER_BYTES + i] = cardSlot(c.cards[i]!);
    }
  }

  w.patchU16(lenAt, w.length - start);
}

export type EpisodeSummary = {
  readonly gameId: number;
  readonly handId: number;
  readonly landlordSeat: Seat;
  readonly numTricks: number;
  readonly attackerPoints: number;
  readonly levelDelta: number;
  readonly defendersWon: boolean;
  readonly trumpRankIdx: number;
  readonly trumpSuitIdx: number;
  // Per trick, in play order: points captured and whether the attackers
  // took them. The kitty bonus is folded into `attackerPoints` only.
  readonly trickPoints: ReadonlyArray<number>;
  readonly trickAttackerWon: ReadonlyArray<boolean>;
  // Points sitting in the buried kitty. Trick points sum to 200 minus
  // this; if the attackers take the last trick these are multiplied in.
  readonly kittyPoints: number;
  // Fingerprint of the deal (GameSlot.dealDigest), so duplicate-deal
  // evaluation can verify that the two halves of a pair really were the
  // same cards instead of assuming it.
  readonly dealDigest: number;
};

export function summarizeHand(
  state: GameState,
  hand: HandState,
  gameId: number,
  handId: number,
  dealDigest = 0,
): EpisodeSummary {
  const landlord = hand.landlord!;
  const landlordTeam = teamOf(state, landlord);
  const trickPoints: number[] = [];
  const trickAttackerWon: boolean[] = [];
  for (const t of hand.tricks) {
    trickPoints.push(t.points);
    trickAttackerWon.push(teamOf(state, t.winner) !== landlordTeam);
  }
  const result = computeHandResult(hand.attackerPoints);
  return {
    gameId,
    handId,
    landlordSeat: landlord,
    numTricks: hand.tricks.length,
    attackerPoints: hand.attackerPoints,
    levelDelta: result.levelDelta,
    defendersWon: result.winner === 'defenders',
    trumpRankIdx: hand.trumpRank === null ? 0 : NATURAL_RANKS.indexOf(hand.trumpRank),
    trumpSuitIdx: trumpSuitIndex(hand),
    trickPoints,
    trickAttackerWon,
    kittyPoints: sumPoints(hand.kitty),
    dealDigest,
  };
}

export function writeEpisode(w: ByteWriter, e: EpisodeSummary): void {
  const buf = w.claim(EPISODE_BYTES);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  view.setUint32(E_HAND_ID, e.handId, true);
  view.setUint16(E_GAME_ID, e.gameId, true);
  view.setUint32(E_DEAL_DIGEST, e.dealDigest >>> 0, true);
  buf[E_LANDLORD_SEAT] = e.landlordSeat;
  buf[E_NUM_TRICKS] = Math.min(e.numTricks, MAX_TRICKS);
  view.setUint16(E_ATTACKER_POINTS, Math.min(e.attackerPoints, 65535), true);
  view.setInt8(E_LEVEL_DELTA, e.levelDelta);
  buf[E_DEFENDERS_WON] = e.defendersWon ? 1 : 0;
  buf[E_TRUMP_RANK] = e.trumpRankIdx;
  buf[E_TRUMP_SUIT] = e.trumpSuitIdx;
  buf[E_KITTY_POINTS] = Math.min(e.kittyPoints, 255);
  for (let i = 0; i < Math.min(e.numTricks, MAX_TRICKS); i++) {
    buf[E_TRICKS_BASE + i * 2] = Math.min(e.trickPoints[i] ?? 0, 255);
    buf[E_TRICKS_BASE + i * 2 + 1] = e.trickAttackerWon[i] === true ? 1 : 0;
  }
}
