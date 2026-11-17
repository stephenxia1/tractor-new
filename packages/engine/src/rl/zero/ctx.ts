// The trump context the PPO networks read, ported from
// research/py/tractor/rules/cards.py (`Ctx`: effective suit and strength of
// each of the 54 card types) and nn/features.py (`static_card_features`).
//
// Card types are the 54-slot index of `rl/v2/cards.ts`: 13 * suit + rank
// for SUITS x NATURAL_RANKS (2..A), 52 the small joker, 53 the big joker.
// A trump context is (level rank 0..12, trump suit 0..3 or 4 = no-trump).

export const NUM_TYPES = 54;
export const NUM_RANKS = 13;
export const SMALL_JOKER = 52;
export const BIG_JOKER = 53;
export const NO_TRUMP = 4;
/** Effective suit 4 is trump; 0..3 are the natural suits. */
export const TRUMP = 4;

export const NUM_EFF_SUITS = 5;
export const NUM_STRENGTH = 16; // MAX_STRENGTH (15) + 1
export const NUM_POINT_VALUES = 3;
export const NUM_CARD_FLAGS = 4;
export const STATIC_SUIT = 0;
export const STATIC_STRENGTH = STATIC_SUIT + NUM_EFF_SUITS;
export const STATIC_POINTS = STATIC_STRENGTH + NUM_STRENGTH;
export const STATIC_FLAGS = STATIC_POINTS + NUM_POINT_VALUES;
export const D_STATIC = STATIC_FLAGS + NUM_CARD_FLAGS; // 28

const RANK_5 = 3;
const RANK_10 = 8;
const RANK_K = 11;

/** Point value of each card type. */
export const POINTS: ReadonlyArray<number> = Array.from({ length: NUM_TYPES }, (_, t) => {
  if (t >= SMALL_JOKER) return 0;
  const r = t % NUM_RANKS;
  return r === RANK_5 ? 5 : r === RANK_10 || r === RANK_K ? 10 : 0;
});

export type TrumpCtx = {
  readonly levelRank: number;
  readonly trumpSuit: number;
  /** Effective suit per card type, 4 = trump. */
  readonly eff: Int8Array;
  /** Strength rung per card type within its effective suit, 0..15. */
  readonly strength: Int8Array;
};

const CTX_CACHE = new Map<number, TrumpCtx>();

/** `rules/cards.py::Ctx(level_rank, trump_suit)`. */
export function trumpCtx(levelRank: number, trumpSuit: number): TrumpCtx {
  if (!(levelRank >= 0 && levelRank < NUM_RANKS)) throw new Error(`level rank ${levelRank}`);
  if (!(trumpSuit >= 0 && trumpSuit <= NO_TRUMP)) throw new Error(`trump suit ${trumpSuit}`);
  const key = levelRank * 5 + trumpSuit;
  const hit = CTX_CACHE.get(key);
  if (hit !== undefined) return hit;
  const eff = new Int8Array(NUM_TYPES);
  const strength = new Int8Array(NUM_TYPES);
  const lr = levelRank;
  const ts = trumpSuit;
  for (let t = 0; t < NUM_TYPES; t++) {
    const joker = t >= SMALL_JOKER;
    const rank = t % NUM_RANKS;
    const suit = Math.floor(t / NUM_RANKS);
    const trump = joker || rank === lr || (ts !== NO_TRUMP && suit === ts);
    eff[t] = trump ? TRUMP : suit;
    if (!trump) {
      strength[t] = rank > lr ? rank - 1 : rank;
    } else if (ts === NO_TRUMP) {
      strength[t] = t === BIG_JOKER ? 2 : t === SMALL_JOKER ? 1 : 0;
    } else if (t === BIG_JOKER) {
      strength[t] = 15;
    } else if (t === SMALL_JOKER) {
      strength[t] = 14;
    } else if (rank === lr && suit === ts) {
      strength[t] = 13;
    } else if (rank === lr) {
      strength[t] = 12;
    } else {
      strength[t] = rank > lr ? rank - 1 : rank;
    }
  }
  const ctx: TrumpCtx = { levelRank, trumpSuit, eff, strength };
  CTX_CACHE.set(key, ctx);
  return ctx;
}

const STATIC_CACHE = new Map<number, Float32Array>();

/** float32[54 * D_STATIC], row-major: `features.static_card_features`.
 *  One-hot effective suit, one-hot strength, one-hot point value over
 *  {0, 5, 10}, then is-level-card, is-trump-suit-level, small joker, big
 *  joker. Shared; do not write to it. */
export function staticCardFeatures(ctx: TrumpCtx): Float32Array {
  const key = ctx.levelRank * 5 + ctx.trumpSuit;
  const hit = STATIC_CACHE.get(key);
  if (hit !== undefined) return hit;
  const out = new Float32Array(NUM_TYPES * D_STATIC);
  for (let t = 0; t < NUM_TYPES; t++) {
    const row = t * D_STATIC;
    out[row + STATIC_SUIT + ctx.eff[t]!] = 1;
    out[row + STATIC_STRENGTH + ctx.strength[t]!] = 1;
    const p = POINTS[t]!;
    out[row + STATIC_POINTS + (p === 0 ? 0 : p === 5 ? 1 : 2)] = 1;
    const isLevel = t < SMALL_JOKER && t % NUM_RANKS === ctx.levelRank;
    if (isLevel) out[row + STATIC_FLAGS + 0] = 1;
    if (isLevel && Math.floor(t / NUM_RANKS) === ctx.trumpSuit) out[row + STATIC_FLAGS + 1] = 1;
    if (t === SMALL_JOKER) out[row + STATIC_FLAGS + 2] = 1;
    if (t === BIG_JOKER) out[row + STATIC_FLAGS + 3] = 1;
  }
  STATIC_CACHE.set(key, out);
  return out;
}
