// Dynamic 54×2 card encoding for the DMC agent.
//
// Layout (108 binary values, row-major, two rows of 54):
//
//   slot 0..11   = non-trump suit A (alpha order, trump excluded), ranks 2..A
//                  with rank-R removed (12 entries)
//   slot 12..23  = non-trump suit B, ranks 2..A skip R
//   slot 24..35  = non-trump suit C, ranks 2..A skip R
//   slot 36..47  = trump suit, ranks 2..A skip R
//   slot 48..50  = off-suit rank-R, one per non-trump suit (alpha order)
//   slot 51      = trump-suit rank-R
//   slot 52      = small joker
//   slot 53      = big joker
//
// Row 0 = "count ≥ 1", row 1 = "count ≥ 2". A set with one copy fills
// row 0 only; a pair fills both rows.
//
// "Dynamic" = the trump suit always lives in the same slot range
// regardless of the actual suit. The network learns one representation
// for "the trump suit" and reuses it across all 13 possible trump ranks.
//
// No-trump (joker trump) mode is not yet supported by this encoding;
// callers should ensure trumpSuit !== null. The DMC training wrapper
// excludes joker-trump declarations.

import { SUITS, type Card, type NaturalRank, type Suit } from '../../types';

export const ENCODING_LEN = 108; // 54 slots × 2 rows

// Slot constants — kept in one place so the network input layout is
// trivial to inspect.
export const SLOTS_PER_NON_TRUMP_SUIT = 12;
export const TRUMP_SUIT_OFFSET = 36;
export const OFFSUIT_R_OFFSET = 48; // 3 slots
export const TRUMP_R_SLOT = 51;
export const SMALL_JOKER_SLOT = 52;
export const BIG_JOKER_SLOT = 53;
export const NUM_SLOTS = 54;

// Order non-trump suits alphabetically (so the encoding is deterministic).
const SUIT_ORDER: ReadonlyArray<Suit> = ['clubs', 'diamonds', 'hearts', 'spades'];

// Ranks 2..A in order; we'll skip the trump rank when building per-suit
// rank lists.
const ALL_RANKS: ReadonlyArray<NaturalRank> = [
  '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A',
];

export type EncodingContext = {
  readonly trumpRank: NaturalRank;
  readonly trumpSuit: Suit; // must be non-null; no-trump unsupported here
};

// Map (suit, rank) → which suit "slot" in the encoding (0..2 = non-trump,
// 3 = trump-suit slot range).
function suitSlot(suit: Suit, ctx: EncodingContext): number {
  if (suit === ctx.trumpSuit) return 3;
  // Non-trump suits in alpha order, with trump removed.
  const nonTrump = SUIT_ORDER.filter((s) => s !== ctx.trumpSuit);
  return nonTrump.indexOf(suit);
}

// Map (suit, rank) → encoding slot index in [0, 53].
// rank-R cards route to the off-suit-R / trump-R slots, not the regular
// per-suit ranks.
export function slotForCard(card: Card, ctx: EncodingContext): number {
  if (card.kind === 'joker') {
    return card.rank === 'small' ? SMALL_JOKER_SLOT : BIG_JOKER_SLOT;
  }
  if (card.rank === ctx.trumpRank) {
    if (card.suit === ctx.trumpSuit) return TRUMP_R_SLOT;
    // Off-suit-R: slot among 48..50, indexed by non-trump-suit order.
    const slot = suitSlot(card.suit, ctx);
    return OFFSUIT_R_OFFSET + slot;
  }
  // Regular per-suit rank. Skip rank R when indexing within the suit.
  const ranks = ALL_RANKS.filter((r) => r !== ctx.trumpRank);
  const rankIdx = ranks.indexOf(card.rank);
  if (rankIdx === -1) throw new Error(`rank not indexed: ${card.rank}`);
  const sSlot = suitSlot(card.suit, ctx);
  // Trump suit goes at TRUMP_SUIT_OFFSET; non-trump suits at 0/12/24.
  if (sSlot === 3) return TRUMP_SUIT_OFFSET + rankIdx;
  return sSlot * SLOTS_PER_NON_TRUMP_SUIT + rankIdx;
}

// Encode a set of cards into a length-108 Float32Array (interpretable as
// 54×2 one-hot of {count ≥ 1, count ≥ 2}).
//
// `out` is optional zero-allocation buffer; callers running in hot loops
// pre-allocate and reuse.
export function encodeCardSet(
  cards: ReadonlyArray<Card>,
  ctx: EncodingContext,
  out?: Float32Array,
): Float32Array {
  const buf = out ?? new Float32Array(ENCODING_LEN);
  buf.fill(0);
  // counts[slot] = number of cards in that slot (0, 1, or 2).
  const counts = new Uint8Array(NUM_SLOTS);
  for (const c of cards) {
    const slot = slotForCard(c, ctx);
    counts[slot] = Math.min(2, (counts[slot] ?? 0) + 1);
  }
  for (let s = 0; s < NUM_SLOTS; s++) {
    if (counts[s]! >= 1) buf[s] = 1;
    if (counts[s]! >= 2) buf[NUM_SLOTS + s] = 1;
  }
  return buf;
}
