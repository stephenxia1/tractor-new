// Wire records -> network inputs. The TypeScript twin of
// `research/py/tractor_rl/features2.py`.
//
// This must agree with the Python featuriser *exactly*, because the
// weights were fitted to what that produced. A one-slot disagreement is
// not a crash, it is a bot that quietly plays worse — so
// `research/py/test_ts_parity.py` diffs the two on real records rather
// than trusting the reading of it.
//
// Only the policy and history views are here. The privileged view exists
// for the centralised critic and is training-only; inference never has
// those bytes and must never need them.

import {
  CANDIDATE_HEADER_BYTES,
  FLAG_ALL_LED_SUIT,
  FLAG_ALL_TRUMP,
  FLAG_BEATS_WINNER,
  FLAG_HAS_POINTS,
  HEADER_BYTES,
  HISTORY_BYTES,
  HISTORY_MAX_CARDS,
  HISTORY_STEP_BYTES,
  HISTORY_STEPS,
  H_ATTACKER_POINTS,
  H_CARDS_SEEN,
  H_HAND_SIZE,
  H_HISTORY_LEN,
  H_IS_LEAD,
  H_I_AM_ATTACKER,
  H_LANDLORD_REL,
  H_ROLE,
  H_TABLE_POINTS,
  H_TRICK_INDEX,
  H_TRICK_POS,
  H_TRUMP_RANK,
  H_TRUMP_SUIT,
  H_VOID_MASK,
  H_WINNER_REL,
  NO_CARD,
  NUM_PLANES,
  NUM_SHAPES,
  PLANES_BYTES,
  PLANE_MY_HAND,
  PLANE_MY_KITTY,
  PLANE_PLAYED_BASE,
  PLANE_TRICK_BASE,
} from './record';
import { NUM_SLOTS } from './cards';
import { NUM_ENC_SLOTS, capacityFor, encMap, encodePlaneInto } from './encoding';

export const NUM_ROLES = 3;
const NUM_RANKS = 13;

const POLICY_PLANES =
  2 + // my hand: has-one, has-pair
  2 + // unseen pool: has-one, has-pair
  4 + // cards played this hand, per relative seat
  4 + // cards in the current trick, per relative seat
  1; // my buried kitty (landlord only)
export const PLANE_DIM = POLICY_PLANES * NUM_ENC_SLOTS;

export const SCALAR_DIM =
  NUM_ROLES + // role one-hot
  4 + // landlord relative seat
  4 + // position in trick
  1 + // is-lead
  5 + // relative seat currently winning (4 = nobody yet)
  1 + // am I an attacker
  5 + // hand size, trick index, banked points, table points, cards seen
  4 * 5 + // void mask: 4 relative seats x (4 suits + trump)
  NUM_RANKS + // trump rank one-hot
  5 + // trump suit one-hot (4 = no-trump)
  1; // bias

export const STATE_DIM = PLANE_DIM + SCALAR_DIM; // 920
export const ACTION_DIM = 2 * NUM_ENC_SLOTS + NUM_SHAPES + 4 + 2; // 144
export const HISTORY_TOKEN_DIM = NUM_ENC_SLOTS + 4 + 4 + 2; // 76

function u16(rec: Uint8Array, at: number): number {
  return rec[at]! + rec[at + 1]! * 256;
}

// The ten policy planes, chain-encoded: [10, 66] flattened.
function policyPlanes(rec: Uint8Array, map: Int32Array): Float32Array {
  const out = new Float32Array(NUM_PLANES * NUM_ENC_SLOTS);
  for (let p = 0; p < NUM_PLANES; p++) {
    encodePlaneInto(rec, HEADER_BYTES + p * NUM_SLOTS, map, out, p * NUM_ENC_SLOTS);
  }
  return out;
}

export function stateFeatures(rec: Uint8Array): Float32Array {
  const rank = rec[H_TRUMP_RANK]!;
  const suit = rec[H_TRUMP_SUIT]!;
  const map = encMap(rank, suit);
  const planes = policyPlanes(rec, map);
  const capacity = capacityFor(rank, suit);

  const out = new Float32Array(STATE_DIM);
  const myHandAt = PLANE_MY_HAND * NUM_ENC_SLOTS;
  const kittyAt = PLANE_MY_KITTY * NUM_ENC_SLOTS;

  let at = 0;
  // my hand: has-one, has-pair
  for (let i = 0; i < NUM_ENC_SLOTS; i++) out[at + i] = planes[myHandAt + i]! >= 1 ? 1 : 0;
  at += NUM_ENC_SLOTS;
  for (let i = 0; i < NUM_ENC_SLOTS; i++) out[at + i] = planes[myHandAt + i]! >= 2 ? 1 : 0;
  at += NUM_ENC_SLOTS;

  // The unseen pool is the most informative derived quantity. After chain
  // encoding a position can hold up to 4 cards (the off-suit trump-rank
  // tier), so the ceiling is per-position rather than a flat 2.
  const unseen = new Float32Array(NUM_ENC_SLOTS);
  for (let i = 0; i < NUM_ENC_SLOTS; i++) {
    let playedSum = 0;
    for (let s = 0; s < 4; s++) playedSum += planes[(PLANE_PLAYED_BASE + s) * NUM_ENC_SLOTS + i]!;
    const v = capacity[i]! - planes[myHandAt + i]! - playedSum - planes[kittyAt + i]!;
    unseen[i] = v > 0 ? v : 0;
  }
  for (let i = 0; i < NUM_ENC_SLOTS; i++) out[at + i] = unseen[i]! >= 1 ? 1 : 0;
  at += NUM_ENC_SLOTS;
  for (let i = 0; i < NUM_ENC_SLOTS; i++) out[at + i] = unseen[i]! >= 2 ? 1 : 0;
  at += NUM_ENC_SLOTS;

  for (let s = 0; s < 4; s++) {
    const base = (PLANE_PLAYED_BASE + s) * NUM_ENC_SLOTS;
    for (let i = 0; i < NUM_ENC_SLOTS; i++) out[at + i] = planes[base + i]! * 0.5;
    at += NUM_ENC_SLOTS;
  }
  for (let s = 0; s < 4; s++) {
    const base = (PLANE_TRICK_BASE + s) * NUM_ENC_SLOTS;
    for (let i = 0; i < NUM_ENC_SLOTS; i++) out[at + i] = planes[base + i]! * 0.5;
    at += NUM_ENC_SLOTS;
  }
  for (let i = 0; i < NUM_ENC_SLOTS; i++) out[at + i] = planes[kittyAt + i]! * 0.5;
  at += NUM_ENC_SLOTS;

  // --- scalars ---
  const oneHot = (value: number, width: number): void => {
    const clamped = value < 0 ? 0 : value > width - 1 ? width - 1 : value;
    out[at + clamped] = 1;
    at += width;
  };

  oneHot(rec[H_ROLE]!, NUM_ROLES);
  oneHot(rec[H_LANDLORD_REL]!, 4);
  oneHot(rec[H_TRICK_POS]!, 4);
  out[at++] = rec[H_IS_LEAD]!;
  const winner = rec[H_WINNER_REL]!;
  oneHot(winner > 3 ? 4 : winner, 5);
  out[at++] = rec[H_I_AM_ATTACKER]!;

  out[at++] = rec[H_HAND_SIZE]! / 25.0;
  out[at++] = rec[H_TRICK_INDEX]! / 25.0;
  out[at++] = u16(rec, H_ATTACKER_POINTS) / 200.0;
  out[at++] = rec[H_TABLE_POINTS]! / 40.0;
  out[at++] = u16(rec, H_CARDS_SEEN) / 108.0;

  // Void mask, rotated so suit index 0 is always the trump suit. Under
  // no-trump there is nothing to rotate about, so suits stay absolute.
  for (let seat = 0; seat < 4; seat++) {
    const bits = rec[H_VOID_MASK + seat]!;
    for (let s = 0; s < 4; s++) {
      const rotated = suit < 4 ? (suit + s) % 4 : s;
      out[at + seat * 5 + s] = (bits >> rotated) & 1;
    }
    out[at + seat * 5 + 4] = (bits >> 4) & 1;
  }
  at += 20;

  oneHot(rank, NUM_RANKS);
  oneHot(suit, 5);
  out[at++] = 1.0; // bias

  if (at !== STATE_DIM) throw new Error(`state feature width ${at} != ${STATE_DIM}`);
  return out;
}

export type HistoryFeatures = {
  readonly tokens: Float32Array; // [HISTORY_STEPS, HISTORY_TOKEN_DIM]
  readonly mask: Float32Array; // [HISTORY_STEPS]
};

export function historyFeatures(rec: Uint8Array): HistoryFeatures {
  const map = encMap(rec[H_TRUMP_RANK]!, rec[H_TRUMP_SUIT]!);
  const tokens = new Float32Array(HISTORY_STEPS * HISTORY_TOKEN_DIM);
  const mask = new Float32Array(HISTORY_STEPS);
  const base = HEADER_BYTES + PLANES_BYTES;
  const length = rec[H_HISTORY_LEN]!;

  const rawCounts = new Uint8Array(NUM_SLOTS);
  for (let w = 0; w < HISTORY_STEPS; w++) {
    if (w >= length) break; // padded steps stay zero, as does their mask
    mask[w] = 1;
    const step = base + w * HISTORY_STEP_BYTES;
    const nCards = rec[step]!;
    const seatRel = rec[step + 1]!;
    const trickIdx = rec[step + 2]!;
    const pos = rec[step + 3]!;

    rawCounts.fill(0);
    for (let c = 0; c < HISTORY_MAX_CARDS; c++) {
      const slot = rec[step + 4 + c]!;
      if (slot !== NO_CARD) rawCounts[slot]! += 1;
    }
    const tokenAt = w * HISTORY_TOKEN_DIM;
    const enc = new Float32Array(NUM_ENC_SLOTS);
    encodePlaneInto(rawCounts, 0, map, enc, 0);
    for (let i = 0; i < NUM_ENC_SLOTS; i++) tokens[tokenAt + i] = enc[i]! * 0.5;
    tokens[tokenAt + NUM_ENC_SLOTS + Math.min(Math.max(seatRel, 0), 3)] = 1;
    tokens[tokenAt + NUM_ENC_SLOTS + 4 + Math.min(Math.max(pos, 0), 3)] = 1;
    tokens[tokenAt + NUM_ENC_SLOTS + 8] = trickIdx / 25.0;
    tokens[tokenAt + NUM_ENC_SLOTS + 9] = nCards / 8.0;
  }
  return { tokens, mask };
}

// One candidate's action features. `cand` is a CANDIDATE_BYTES slice.
export function actionFeatures(cand: Uint8Array, trumpRank: number, trumpSuit: number): Float32Array {
  const map = encMap(trumpRank, trumpSuit);
  const nCards = cand[0]!;
  const shape = cand[1]!;
  const flags = cand[2]!;
  const points = cand[3]!;

  const rawCounts = new Uint8Array(NUM_SLOTS);
  for (let i = CANDIDATE_HEADER_BYTES; i < cand.length; i++) {
    const slot = cand[i]!;
    if (slot !== NO_CARD) rawCounts[slot]! += 1;
  }
  const counts = new Float32Array(NUM_ENC_SLOTS);
  encodePlaneInto(rawCounts, 0, map, counts, 0);

  const out = new Float32Array(ACTION_DIM);
  for (let i = 0; i < NUM_ENC_SLOTS; i++) out[i] = counts[i]! >= 1 ? 1 : 0;
  for (let i = 0; i < NUM_ENC_SLOTS; i++) out[NUM_ENC_SLOTS + i] = counts[i]! >= 2 ? 1 : 0;
  let base = 2 * NUM_ENC_SLOTS;
  out[base + Math.min(Math.max(shape, 0), NUM_SHAPES - 1)] = 1;
  base += NUM_SHAPES;
  out[base + 0] = (flags & FLAG_ALL_TRUMP) !== 0 ? 1 : 0;
  out[base + 1] = (flags & FLAG_ALL_LED_SUIT) !== 0 ? 1 : 0;
  out[base + 2] = (flags & FLAG_HAS_POINTS) !== 0 ? 1 : 0;
  out[base + 3] = (flags & FLAG_BEATS_WINNER) !== 0 ? 1 : 0;
  base += 4;
  out[base] = nCards / 8.0;
  out[base + 1] = points / 30.0;
  return out;
}

export { HISTORY_STEPS, HISTORY_BYTES };
