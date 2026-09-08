// Chain-position card encoding — the TypeScript twin of
// `research/py/tractor_rl/encoding.py`.
//
// The raw wire slot is (suit, rank), which is the wrong basis for
// learning: with trump rank 5 the 6 sits next to the 4, and each of the
// 13 possible trump ranks rearranges what "one higher" means. Cards are
// therefore placed at their *chain position*, so a position means a
// strength in every trump context.
//
// The Python side reads this table from `chains.json`, which is dumped
// out of this engine. Here it is derived from `chainPosition` directly at
// module load: 13 x 5 x 54 lookups, once, and no data file that can fall
// out of step with the rules it came from.
//
// Layout, 66 positions:
//   0..17   trump ladder by chain position
//   18..65  four 12-slot suit blocks, indexed by offset from the trump
//           suit (absolute suit under no-trump) with the trump rank removed

import { chainPosition } from '../../combos';
import { isTrump } from '../../ordering';
import { NATURAL_RANKS, SUITS, type NaturalRank, type Suit, type TrumpContext } from '../../types';

import { cardForSlot, NUM_RANKS, NUM_SLOTS } from './cards';

export const TRUMP_BLOCK = 18;
export const SUIT_BLOCK = 12;
export const NUM_ENC_SLOTS = TRUMP_BLOCK + 4 * SUIT_BLOCK; // 66
export const NO_TRUMP = 4;
const NUM_TRUMP_CONTEXTS = 5; // four suits + no-trump
const NUM_DECKS = 2;

function contextFor(rankIndex: number, suitIndex: number): TrumpContext {
  return {
    trumpRank: NATURAL_RANKS[rankIndex] as NaturalRank,
    trumpSuit: suitIndex < SUITS.length ? (SUITS[suitIndex] as Suit) : null,
  };
}

// encOfRaw[rank][suit] is a 54-entry map from raw slot to encoded position.
// capacity[rank][suit] is how many physical cards can occupy each encoded
// position — usually 2, but 4 at the shared off-suit trump-rank tier.
function buildTables(): {
  encOfRaw: ReadonlyArray<ReadonlyArray<Int32Array>>;
  capacity: ReadonlyArray<ReadonlyArray<Float32Array>>;
} {
  const encOfRaw: Int32Array[][] = [];
  const capacity: Float32Array[][] = [];
  for (let r = 0; r < NUM_RANKS; r++) {
    const bySuitEnc: Int32Array[] = [];
    const bySuitCap: Float32Array[] = [];
    for (let t = 0; t < NUM_TRUMP_CONTEXTS; t++) {
      const ctx = contextFor(r, t);
      const enc = new Int32Array(NUM_SLOTS).fill(-1);
      const cap = new Float32Array(NUM_ENC_SLOTS);
      for (let slot = 0; slot < NUM_SLOTS; slot++) {
        const card = cardForSlot(slot);
        const pos = chainPosition(card, ctx);
        let target: number;
        if (isTrump(card, ctx)) {
          if (pos < 0 || pos >= TRUMP_BLOCK) {
            throw new Error(`trump chain position ${pos} outside the block`);
          }
          target = pos;
        } else {
          if (pos < 0 || pos >= SUIT_BLOCK) {
            throw new Error(`suit chain position ${pos} outside the block`);
          }
          const suitIdx = Math.floor(slot / NUM_RANKS);
          const block = t === NO_TRUMP ? suitIdx : (suitIdx - t + 4) % 4;
          target = TRUMP_BLOCK + block * SUIT_BLOCK + pos;
        }
        enc[slot] = target;
        cap[target]! += NUM_DECKS;
      }
      bySuitEnc.push(enc);
      bySuitCap.push(cap);
    }
    encOfRaw.push(bySuitEnc);
    capacity.push(bySuitCap);
  }
  return { encOfRaw, capacity };
}

const TABLES = buildTables();
export const ENC_OF_RAW = TABLES.encOfRaw;
export const CAPACITY = TABLES.capacity;

export function encMap(trumpRank: number, trumpSuit: number): Int32Array {
  const row = ENC_OF_RAW[trumpRank];
  if (row === undefined) throw new Error(`trump rank ${trumpRank} out of range`);
  const map = row[trumpSuit];
  if (map === undefined) throw new Error(`trump suit ${trumpSuit} out of range`);
  return map;
}

export function capacityFor(trumpRank: number, trumpSuit: number): Float32Array {
  const row = CAPACITY[trumpRank];
  if (row === undefined) throw new Error(`trump rank ${trumpRank} out of range`);
  const cap = row[trumpSuit];
  if (cap === undefined) throw new Error(`trump suit ${trumpSuit} out of range`);
  return cap;
}

// Accumulates one 54-slot raw count plane into 66 encoded positions.
// Several raw slots share a position (the three off-suit trump ranks are
// one interchangeable tier), so counts add rather than overwrite.
export function encodePlaneInto(
  raw: Uint8Array,
  rawOffset: number,
  map: Int32Array,
  out: Float32Array,
  outOffset: number,
): void {
  for (let slot = 0; slot < NUM_SLOTS; slot++) {
    const count = raw[rawOffset + slot]!;
    if (count !== 0) out[outOffset + map[slot]!]! += count;
  }
}
