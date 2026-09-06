"""Chain-position card encoding, built from the engine's own ordering.

The raw wire slot is (suit, rank), which is the wrong basis for learning:
with trump rank 5, the 6 sits next to the 4, and every one of the 13
possible trump ranks rearranges what "one higher" means. A network given
raw slots has to learn the ordering separately in each of 65 trump
contexts.

Here each card is placed at its *chain position* instead -- the index the
engine uses to decide adjacency and strength (`chainPosition` in
combos.ts) -- so position means strength, consistently, in every context.

The table is derived from `chains.json`, which `src/dumpChains.ts` dumps
straight out of the engine. Nothing here re-derives the rules, so the
encoding cannot drift from them; regenerate that file if the ordering ever
changes.

Layout, 66 positions:

    0..17    trump ladder, by chain position
             (normal trump uses 0..15: suit ranks, off-suit-R, trump-R,
              small joker, big joker; no-trump uses 0..2)
    18..65   four 12-slot suit blocks, by rank position with the trump
             rank removed. Under a normal trump the blocks are indexed by
             offset from the trump suit, so block 0 is always empty and
             "block 1" always means the same thing relative to trump; in
             no-trump they are indexed by absolute suit.

Two useful consequences. The three off-suit trump-rank cards share one
position, which is correct -- they are one tier, mutually interchangeable
for following and for beating. And a pair is always two counts in a single
position, so "pair" is a magnitude rather than a pattern to recognise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESEARCH_DIR = Path(__file__).resolve().parents[2]
CHAINS_PATH = RESEARCH_DIR / "chains.json"

NUM_RAW_SLOTS = 54
TRUMP_BLOCK = 18
SUIT_BLOCK = 12
NUM_ENC_SLOTS = TRUMP_BLOCK + 4 * SUIT_BLOCK  # 66
NUM_RANKS = 13
NUM_SUITS = 4
NO_TRUMP = 4

# Index used for "nothing here", so a gather can express an empty slot.
PAD_SLOT = NUM_RAW_SLOTS


def _load_chains() -> dict:
    if not CHAINS_PATH.exists():
        raise RuntimeError(
            f"{CHAINS_PATH} is missing. Run `npx tsx src/dumpChains.ts > chains.json` "
            f"in {RESEARCH_DIR}."
        )
    return json.loads(CHAINS_PATH.read_text(encoding="utf-8"))


def build_tables() -> tuple[np.ndarray, np.ndarray]:
    """Returns (enc_of_raw, src_of_enc).

    enc_of_raw[R, T, raw]  -> encoded position for that card
    src_of_enc[R, T, enc]  -> a raw slot feeding that position, or PAD_SLOT

    Several raw slots can share an encoded position (the off-suit trump
    ranks), so the gather table keeps only one representative and the
    scatter table is the authoritative one; counts are accumulated with
    the scatter form.
    """
    chains = _load_chains()
    enc_of_raw = np.full((NUM_RANKS, 5, NUM_RAW_SLOTS), -1, dtype=np.int64)
    for r in range(NUM_RANKS):
        for t in range(5):
            rows = chains[f"{r}:{t}"]
            for raw, (eff_suit, is_trump, chain_pos) in enumerate(rows):
                if is_trump:
                    enc = chain_pos
                    if not 0 <= enc < TRUMP_BLOCK:
                        raise ValueError(f"trump chain position {enc} outside the block")
                else:
                    suit_idx = raw // NUM_RANKS
                    block = suit_idx if t == NO_TRUMP else (suit_idx - t) % NUM_SUITS
                    if not 0 <= chain_pos < SUIT_BLOCK:
                        raise ValueError(f"suit chain position {chain_pos} outside the block")
                    enc = TRUMP_BLOCK + block * SUIT_BLOCK + chain_pos
                enc_of_raw[r, t, raw] = enc
    if (enc_of_raw < 0).any():
        raise ValueError("some card was never assigned an encoded position")

    src_of_enc = np.full((NUM_RANKS, 5, NUM_ENC_SLOTS), PAD_SLOT, dtype=np.int64)
    for r in range(NUM_RANKS):
        for t in range(5):
            for raw in range(NUM_RAW_SLOTS):
                src_of_enc[r, t, enc_of_raw[r, t, raw]] = raw
    return enc_of_raw, src_of_enc


ENC_OF_RAW, SRC_OF_ENC = build_tables()

# One-hot scatter matrices, [13, 5, 54, 66]: encoded = raw @ M. Doing the
# permutation as a matmul keeps the whole batch in one vectorised op and
# handles the many-to-one positions (off-suit trump ranks) by summing,
# which is exactly the count semantics we want.
_SCATTER = np.zeros((NUM_RANKS, 5, NUM_RAW_SLOTS, NUM_ENC_SLOTS), dtype=np.float32)
for _r in range(NUM_RANKS):
    for _t in range(5):
        _SCATTER[_r, _t, np.arange(NUM_RAW_SLOTS), ENC_OF_RAW[_r, _t]] = 1.0
SCATTER = _SCATTER


def encode_planes(planes: np.ndarray, trump_rank: np.ndarray, trump_suit: np.ndarray) -> np.ndarray:
    """[B, P, 54] raw counts -> [B, P, 66] chain-position counts."""
    mats = SCATTER[trump_rank, trump_suit]  # [B, 54, 66]
    return np.einsum("bps,bse->bpe", planes, mats, optimize=True)


def describe(enc_index: int, trump_rank: int, trump_suit: int) -> str:
    """Human-readable name for an encoded position, for debugging."""
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    suits = ["spades", "hearts", "diamonds", "clubs"]
    raws = [
        raw for raw in range(NUM_RAW_SLOTS)
        if ENC_OF_RAW[trump_rank, trump_suit, raw] == enc_index
    ]
    if not raws:
        return f"enc{enc_index}: (empty in this context)"
    names = []
    for raw in raws:
        if raw == 52:
            names.append("small joker")
        elif raw == 53:
            names.append("big joker")
        else:
            names.append(f"{ranks[raw % NUM_RANKS]}{suits[raw // NUM_RANKS][0]}")
    return f"enc{enc_index}: " + ", ".join(names)
