"""Wire records -> network inputs, v2.

Three views come out of the same record, and keeping them apart is the
point of this module:

* **policy view** -- what the agent may see: its own hand, public play,
  its own buried kitty, derived voids and the unseen pool.
* **history view** -- the recent move sequence as tokens, for the causal
  attention encoder. Also policy-legal; it is public information.
* **privileged view** -- the other seats' hands and the kitty. TRAINING
  ONLY, for the centralised critic. `state_features` and
  `history_features` never touch those bytes, and `test_no_leak.py`
  checks that by zeroing the block and asserting the policy view is
  bit-identical.

Cards are placed by chain position (see encoding.py), so a position means
a strength rather than a (suit, rank) pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import layout as L
from .encoding import CAPACITY, ENC_OF_RAW, NUM_ENC_SLOTS, encode_planes

# Frame merging, worker splitting and episode parsing do not depend on the
# card representation, so they are shared with the v1 featuriser rather
# than duplicated.
from .features import (  # noqa: F401
    Episode,
    Merged,
    merge_batches,
    parse_episodes,
    split_per_worker,
)

NUM_DECKS = 2

# --- dimensions -------------------------------------------------------------

_POLICY_PLANES = (
    2  # my hand: has-one, has-pair
    + 2  # unseen pool: has-one, has-pair
    + 4  # cards played this hand, per relative seat
    + 4  # cards in the current trick, per relative seat
    + 1  # my buried kitty (landlord only)
)
PLANE_DIM = _POLICY_PLANES * NUM_ENC_SLOTS

SCALAR_DIM = (
    L.NUM_ROLES  # role one-hot
    + 4  # landlord relative seat
    + 4  # position in trick
    + 1  # is-lead
    + 5  # relative seat currently winning (4 = nobody yet)
    + 1  # am I an attacker
    + 5  # hand size, trick index, banked points, table points, cards seen
    + 4 * 5  # void mask: 4 relative seats x (4 suits + trump)
    + L.NUM_RANKS  # trump rank one-hot
    + 5  # trump suit one-hot (4 = no-trump)
    + 1  # bias
)
STATE_DIM = PLANE_DIM + SCALAR_DIM

ACTION_DIM = 2 * NUM_ENC_SLOTS + L.NUM_SHAPES + 4 + 2

# History token: the cards played, who played them, where in the trick, when.
HISTORY_TOKEN_DIM = NUM_ENC_SLOTS + 4 + 4 + 2
HISTORY_STEPS = L.HISTORY_STEPS

PRIVILEGED_DIM = L.PRIVILEGED_PLANES * NUM_ENC_SLOTS


@dataclass(slots=True)
class Batch:
    states: np.ndarray  # [B, state_bytes] uint8
    cands: np.ndarray  # [N, CANDIDATE_BYTES] uint8
    owner: np.ndarray  # [N] int64
    cand_counts: np.ndarray  # [B]
    episodes: np.ndarray  # [E, EPISODE_BYTES]
    has_privileged: bool = False

    @property
    def num_decisions(self) -> int:
        return int(self.states.shape[0])


def _check_wire_layout(payload: bytes, first_rec: int, rec_bytes: int, state_bytes: int) -> None:
    """Fail loudly when the actor wrote a layout other than the one we parse.

    layout.py's import-time check compares spec.json against LAYOUT_VERSION, so
    it can only catch "spec regenerated, layout.py not updated": a record.ts
    edit that skipped `npm run spec` leaves both artifacts stale and agreeing
    with each other, and a field inserted past the header then shifts every
    plane, history step and candidate by a few bytes with nothing raising. The
    version byte the actor stamps into each record is the only witness of the
    layout a frame was actually written with, and the record size is the only
    witness of a size drift that the floor division in parse_frame would
    otherwise absorb. Checked once per frame, on the first record, because a
    frame comes from one worker running one build; raised rather than asserted
    because `python -O` strips asserts and this must never be optional.
    """
    version = payload[first_rec + L.H_VERSION]
    if version != L.LAYOUT_VERSION:
        raise ValueError(
            f"records were written with wire layout version {version} but this "
            f"build parses version {L.LAYOUT_VERSION}; regenerate spec.json with "
            f"`npm run spec` and update LAYOUT_VERSION in layout.py."
        )
    if rec_bytes < state_bytes or (rec_bytes - state_bytes) % L.CANDIDATE_BYTES:
        raise ValueError(
            f"record is {rec_bytes} bytes, which is not {state_bytes} state bytes "
            f"plus a whole number of {L.CANDIDATE_BYTES}-byte candidates; "
            f"regenerate spec.json with `npm run spec` and update layout.py."
        )


def parse_frame(payload: bytes) -> Batch:
    """Split one actor frame into decisions + episodes.

    Record size depends on whether the privileged block is present. One
    worker runs one setting, so the layout is resolved once from the first
    record and the rest is vectorised -- but nothing on the wire forced
    that uniformity, and search leaves once hardcoded the flag to false
    while the ordinary records beside them followed ``--privileged``. The
    loop below therefore checks every record's flag against the resolved
    one instead of taking the uniformity on trust.
    """
    buf = np.frombuffer(payload, dtype=np.uint8)
    n_dec = int.from_bytes(payload[0:2], "little")
    n_ep = int.from_bytes(payload[2:4], "little")

    off = L.BATCH_HEADER_BYTES
    if n_dec == 0:
        state_bytes = L.STATE_BYTES
        starts = np.zeros(0, dtype=np.int64)
        counts = np.zeros(0, dtype=np.int64)
        has_priv = False
    else:
        first_rec = off + 2
        has_priv = bool(payload[first_rec + L.H_HAS_PRIVILEGED])
        state_bytes = L.STATE_BYTES_PRIVILEGED if has_priv else L.STATE_BYTES
        _check_wire_layout(
            payload, first_rec, int.from_bytes(payload[off : off + 2], "little"), state_bytes
        )
        starts = np.empty(n_dec, dtype=np.int64)
        counts = np.empty(n_dec, dtype=np.int64)
        for i in range(n_dec):
            rec_bytes = int.from_bytes(payload[off : off + 2], "little")
            starts[i] = off + 2
            rec_priv = bool(payload[off + 2 + L.H_HAS_PRIVILEGED])
            if rec_priv != has_priv:
                raise ValueError(
                    f"record {i} of {n_dec} says has_privileged={rec_priv} but the frame "
                    f"was resolved as has_privileged={has_priv} from record 0; every "
                    f"record in a frame must agree, because the {state_bytes}-byte state "
                    f"block is subtracted from all of them to count candidates and a "
                    f"narrower record makes that count negative. Check that the actor "
                    f"writes the same privileged flag for search leaves as for ordinary "
                    f"decisions."
                )
            counts[i] = (rec_bytes - state_bytes) // L.CANDIDATE_BYTES
            off += 2 + rec_bytes

    if n_dec:
        # A slice-and-copy loop, not the obvious `buf[starts[:, None] +
        # arange(state_bytes)]`. That form materialises an
        # [n_dec, state_bytes] int64 index -- 8 bytes of index per byte
        # fetched, so 0.5 MB of scratch to move 63 KB at a typical 64
        # decisions a worker -- and then takes the slow fancy-indexing
        # path over it. Each record's bytes are contiguous, so a plain
        # slice assignment per row is a memcpy, and the loop overhead is
        # amortised over 984 bytes. Measured 8.5x faster at 64 records
        # and 4.7x at 1024; the crossover where fancy indexing would win
        # is far above any frame this protocol produces.
        states = np.empty((n_dec, state_bytes), dtype=np.uint8)
        for i in range(n_dec):
            begin = int(starts[i])
            states[i] = buf[begin:begin + state_bytes]
        total = int(counts.sum())
        owner = np.repeat(np.arange(n_dec, dtype=np.int64), counts)
        within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        cand_start = np.repeat(starts + state_bytes, counts) + within * L.CANDIDATE_BYTES
        cands = buf[cand_start[:, None] + np.arange(L.CANDIDATE_BYTES)]
    else:
        states = np.zeros((0, state_bytes), dtype=np.uint8)
        cands = np.zeros((0, L.CANDIDATE_BYTES), dtype=np.uint8)
        owner = np.zeros(0, dtype=np.int64)

    episodes = (
        buf[off : off + n_ep * L.EPISODE_BYTES].reshape(n_ep, L.EPISODE_BYTES)
        if n_ep
        else np.zeros((0, L.EPISODE_BYTES), dtype=np.uint8)
    )
    return Batch(states, cands, owner, counts, episodes, has_priv)


# --- shared helpers ---------------------------------------------------------


def field_u8(states: np.ndarray, offset: int) -> np.ndarray:
    return states[:, offset].astype(np.int64)


def field_u16(states: np.ndarray, offset: int) -> np.ndarray:
    return states[:, offset].astype(np.int64) + states[:, offset + 1].astype(np.int64) * 256


def field_u32(states: np.ndarray, offset: int) -> np.ndarray:
    b = states[:, offset : offset + 4].astype(np.int64)
    return b[:, 0] + (b[:, 1] << 8) + (b[:, 2] << 16) + (b[:, 3] << 24)


def _trump(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return field_u8(states, L.H_TRUMP_RANK), field_u8(states, L.H_TRUMP_SUIT)


def _policy_planes(states: np.ndarray) -> np.ndarray:
    """[B, 10, 66] chain-encoded counts. Reads only the policy plane block."""
    b = states.shape[0]
    raw = states[:, L.HEADER_BYTES : L.HEADER_BYTES + L.PLANES_BYTES]
    raw = raw.reshape(b, L.NUM_PLANES, L.NUM_SLOTS).astype(np.float32)
    rank, suit = _trump(states)
    return encode_planes(raw, rank, suit)


# --- policy view ------------------------------------------------------------


def state_features(states: np.ndarray) -> np.ndarray:
    """[B, state_bytes] -> [B, STATE_DIM] float32. Policy-legal only."""
    b = states.shape[0]
    if b == 0:
        return np.zeros((0, STATE_DIM), dtype=np.float32)

    planes = _policy_planes(states)
    my_hand = planes[:, L.PLANE_MY_HAND]
    played = planes[:, L.PLANE_PLAYED_BASE : L.PLANE_PLAYED_BASE + 4]
    trick = planes[:, L.PLANE_TRICK_BASE : L.PLANE_TRICK_BASE + 4]
    kitty = planes[:, L.PLANE_MY_KITTY]

    # The unseen pool is the single most informative derived quantity, and
    # after chain encoding a position can hold up to 4 cards (the off-suit
    # trump-rank tier), so the ceiling is per-position rather than 2.
    rank, suit = _trump(states)
    capacity = CAPACITY[rank, suit]
    unseen = np.clip(capacity - my_hand - played.sum(axis=1) - kitty, 0.0, None)

    plane_feats = np.concatenate(
        [
            (my_hand >= 1).astype(np.float32),
            (my_hand >= 2).astype(np.float32),
            (unseen >= 1).astype(np.float32),
            (unseen >= 2).astype(np.float32),
            played.reshape(b, -1) * 0.5,
            trick.reshape(b, -1) * 0.5,
            kitty * 0.5,
        ],
        axis=1,
    )

    scalars = np.zeros((b, SCALAR_DIM), dtype=np.float32)
    rows = np.arange(b)
    at = 0

    def one_hot(values: np.ndarray, width: int) -> None:
        nonlocal at
        scalars[rows, at + np.clip(values.astype(np.int64), 0, width - 1)] = 1.0
        at += width

    one_hot(field_u8(states, L.H_ROLE), L.NUM_ROLES)
    one_hot(field_u8(states, L.H_LANDLORD_REL), 4)
    one_hot(field_u8(states, L.H_TRICK_POS), 4)
    scalars[:, at] = field_u8(states, L.H_IS_LEAD)
    at += 1
    winner = field_u8(states, L.H_WINNER_REL)
    one_hot(np.where(winner > 3, 4, winner), 5)
    scalars[:, at] = field_u8(states, L.H_I_AM_ATTACKER)
    at += 1

    scalars[:, at] = field_u8(states, L.H_HAND_SIZE) / 25.0
    scalars[:, at + 1] = field_u8(states, L.H_TRICK_INDEX) / 25.0
    scalars[:, at + 2] = field_u16(states, L.H_ATTACKER_POINTS) / 200.0
    scalars[:, at + 3] = field_u8(states, L.H_TABLE_POINTS) / 40.0
    scalars[:, at + 4] = field_u16(states, L.H_CARDS_SEEN) / 108.0
    at += 5

    void = states[:, L.H_VOID_MASK : L.H_VOID_MASK + 4].astype(np.int64)
    suit_rot = np.where(
        suit[:, None] < 4,
        (suit[:, None] + np.arange(4)[None, :]) % 4,
        np.arange(4)[None, :],
    )
    suit_bits = (void[:, :, None] >> suit_rot[:, None, :]) & 1
    trump_bits = (void >> 4) & 1
    scalars[:, at : at + 20] = np.concatenate(
        [suit_bits, trump_bits[:, :, None]], axis=2
    ).reshape(b, 20).astype(np.float32)
    at += 20

    one_hot(rank, L.NUM_RANKS)
    one_hot(suit, 5)
    scalars[:, at] = 1.0
    at += 1
    assert at == SCALAR_DIM, (at, SCALAR_DIM)

    return np.concatenate([plane_feats, scalars], axis=1)


def history_features(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[B, W, HISTORY_TOKEN_DIM] tokens and a [B, W] validity mask.

    Public information: what was played, by whom, in which trick and at
    which position. The planes lose all of that ordering.
    """
    b = states.shape[0]
    if b == 0:
        return (
            np.zeros((0, HISTORY_STEPS, HISTORY_TOKEN_DIM), dtype=np.float32),
            np.zeros((0, HISTORY_STEPS), dtype=np.float32),
        )
    base = L.HEADER_BYTES + L.PLANES_BYTES
    block = states[:, base : base + L.HISTORY_BYTES].reshape(b, HISTORY_STEPS, L.HISTORY_STEP_BYTES)
    n_cards = block[:, :, 0].astype(np.float32)
    seat_rel = block[:, :, 1].astype(np.int64)
    trick_idx = block[:, :, 2].astype(np.float32)
    pos = block[:, :, 3].astype(np.int64)
    slots = block[:, :, 4 : 4 + L.HISTORY_MAX_CARDS].astype(np.int64)

    # Card counts per step, in chain positions. Counting into the 54 raw
    # slots and permuting afterwards does the same work twice: each card is
    # sent to its encoded position first, so the bincount lands directly in
    # the 66 output bins and the permutation disappears.
    valid_card = slots != L.NO_CARD
    flat_rows = np.repeat(np.arange(b * HISTORY_STEPS), L.HISTORY_MAX_CARDS).reshape(
        b, HISTORY_STEPS, L.HISTORY_MAX_CARDS
    )
    rank, suit = _trump(states)
    enc_map = ENC_OF_RAW[rank, suit]  # [B, 54]
    # NO_CARD would index out of range; those entries are masked out below.
    safe_slots = np.where(valid_card, slots, 0)
    enc_slots = np.take_along_axis(enc_map[:, None, :], safe_slots, axis=2)
    idx = flat_rows[valid_card] * NUM_ENC_SLOTS + enc_slots[valid_card]
    enc = np.bincount(idx, minlength=b * HISTORY_STEPS * NUM_ENC_SLOTS).astype(np.float32)
    enc = enc.reshape(b, HISTORY_STEPS, NUM_ENC_SLOTS)  # [B, W, 66]

    tokens = np.zeros((b, HISTORY_STEPS, HISTORY_TOKEN_DIM), dtype=np.float32)
    tokens[:, :, :NUM_ENC_SLOTS] = enc * 0.5
    bb, ww = np.meshgrid(np.arange(b), np.arange(HISTORY_STEPS), indexing="ij")
    tokens[bb, ww, NUM_ENC_SLOTS + np.clip(seat_rel, 0, 3)] = 1.0
    tokens[bb, ww, NUM_ENC_SLOTS + 4 + np.clip(pos, 0, 3)] = 1.0
    tokens[:, :, NUM_ENC_SLOTS + 8] = trick_idx / 25.0
    tokens[:, :, NUM_ENC_SLOTS + 9] = n_cards / 8.0

    lengths = field_u8(states, L.H_HISTORY_LEN)
    mask = (np.arange(HISTORY_STEPS)[None, :] < lengths[:, None]).astype(np.float32)
    tokens *= mask[:, :, None]
    return tokens, mask


def action_features(cands: np.ndarray, states: np.ndarray, owner: np.ndarray) -> np.ndarray:
    n = cands.shape[0]
    if n == 0:
        return np.zeros((0, ACTION_DIM), dtype=np.float32)
    n_cards = cands[:, 0].astype(np.float32)
    shape = cands[:, 1].astype(np.int64)
    flags = cands[:, 2].astype(np.int64)
    points = cands[:, 3].astype(np.float32)

    # Same shortcut as the history tokens: send each card straight to its
    # chain position so the bincount lands in the 66 output bins, instead of
    # counting into 54 raw slots and permuting afterwards. This is the
    # hottest featuriser -- a decision carries ~26 candidates -- so the
    # intermediate it removes was the largest one in the pipeline.
    slots = cands[:, L.CANDIDATE_HEADER_BYTES :].astype(np.int64)
    valid = slots != L.NO_CARD
    rows, _ = np.nonzero(valid)
    rank, suit = _trump(states)
    enc_map = ENC_OF_RAW[rank[owner], suit[owner]]  # [n, 54]
    safe_slots = np.where(valid, slots, 0)
    enc_slots = np.take_along_axis(enc_map, safe_slots, axis=1)
    flat = rows * NUM_ENC_SLOTS + enc_slots[valid]
    counts = np.bincount(flat, minlength=n * NUM_ENC_SLOTS).astype(np.float32)
    counts = counts.reshape(n, NUM_ENC_SLOTS)

    out = np.zeros((n, ACTION_DIM), dtype=np.float32)
    out[:, :NUM_ENC_SLOTS] = counts >= 1
    out[:, NUM_ENC_SLOTS : 2 * NUM_ENC_SLOTS] = counts >= 2
    base = 2 * NUM_ENC_SLOTS
    out[np.arange(n), base + np.clip(shape, 0, L.NUM_SHAPES - 1)] = 1.0
    base += L.NUM_SHAPES
    out[:, base + 0] = (flags & L.FLAG_ALL_TRUMP) != 0
    out[:, base + 1] = (flags & L.FLAG_ALL_LED_SUIT) != 0
    out[:, base + 2] = (flags & L.FLAG_HAS_POINTS) != 0
    out[:, base + 3] = (flags & L.FLAG_BEATS_WINNER) != 0
    base += 4
    out[:, base] = n_cards / 8.0
    out[:, base + 1] = points / 30.0
    return out


# --- privileged view (TRAINING ONLY) ----------------------------------------


def privileged_features(states: np.ndarray) -> np.ndarray:
    """[B, PRIVILEGED_DIM]: the other seats' hands and the kitty.

    Never call this on the policy path. It exists so a centralised critic
    can be trained on full information and then distilled into the
    imperfect-information network.
    """
    b = states.shape[0]
    if b == 0:
        return np.zeros((0, PRIVILEGED_DIM), dtype=np.float32)
    if states.shape[1] < L.STATE_BYTES_PRIVILEGED:
        raise ValueError(
            "records do not carry the privileged block; start the worker with --privileged 1"
        )
    base = L.HEADER_BYTES + L.PLANES_BYTES + L.HISTORY_BYTES
    raw = states[:, base : base + L.PRIVILEGED_BYTES]
    raw = raw.reshape(b, L.PRIVILEGED_PLANES, L.NUM_SLOTS).astype(np.float32)
    rank, suit = _trump(states)
    enc = encode_planes(raw, rank, suit)
    return (enc * 0.5).reshape(b, -1)


def belief_targets(states: np.ndarray, priv: np.ndarray | None = None) -> np.ndarray:
    """[B, 3 * 66] in {0,1}: does relative seat r hold a card at position p.

    The belief head's supervision, and training-only for the same reason.

    `priv` lets a caller that already built the privileged view hand it back
    instead of paying for it twice; the learner does exactly that on every
    step, which was a full redundant featurisation per gradient step.
    """
    if priv is None:
        priv = privileged_features(states)
    priv = priv.reshape(states.shape[0], L.PRIVILEGED_PLANES, NUM_ENC_SLOTS)
    return (priv[:, :3] > 0).astype(np.float32).reshape(states.shape[0], -1)
