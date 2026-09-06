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
from .encoding import NUM_ENC_SLOTS, encode_planes

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


def parse_frame(payload: bytes) -> Batch:
    """Split one actor frame into decisions + episodes.

    Record size depends on whether the privileged block is present, which
    is uniform within a frame (one worker, one setting), so the layout is
    resolved once and the rest is vectorised.
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
        starts = np.empty(n_dec, dtype=np.int64)
        counts = np.empty(n_dec, dtype=np.int64)
        for i in range(n_dec):
            rec_bytes = int.from_bytes(payload[off : off + 2], "little")
            starts[i] = off + 2
            counts[i] = (rec_bytes - state_bytes) // L.CANDIDATE_BYTES
            off += 2 + rec_bytes

    if n_dec:
        states = buf[starts[:, None] + np.arange(state_bytes)]
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
    capacity = np.zeros_like(my_hand)
    capacity[:] = 0
    ones = np.ones((b, 1, L.NUM_SLOTS), dtype=np.float32) * NUM_DECKS
    rank, suit = _trump(states)
    capacity = encode_planes(ones, rank, suit)[:, 0]
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

    # Card counts per step, then the same chain-position encoding as the
    # planes so a card means the same thing everywhere.
    valid_card = slots != L.NO_CARD
    flat_rows = np.repeat(np.arange(b * HISTORY_STEPS), L.HISTORY_MAX_CARDS).reshape(
        b, HISTORY_STEPS, L.HISTORY_MAX_CARDS
    )
    idx = flat_rows[valid_card] * L.NUM_SLOTS + slots[valid_card]
    raw_counts = np.bincount(idx, minlength=b * HISTORY_STEPS * L.NUM_SLOTS).astype(np.float32)
    raw_counts = raw_counts.reshape(b, HISTORY_STEPS, L.NUM_SLOTS)
    rank, suit = _trump(states)
    enc = encode_planes(raw_counts, rank, suit)  # [B, W, 66]

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

    slots = cands[:, L.CANDIDATE_HEADER_BYTES :].astype(np.int64)
    valid = slots != L.NO_CARD
    rows, _ = np.nonzero(valid)
    flat = rows * L.NUM_SLOTS + slots[valid]
    raw = np.bincount(flat, minlength=n * L.NUM_SLOTS).astype(np.float32)
    raw = raw.reshape(n, 1, L.NUM_SLOTS)
    rank, suit = _trump(states)
    counts = encode_planes(raw, rank[owner], suit[owner])[:, 0]

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


def belief_targets(states: np.ndarray) -> np.ndarray:
    """[B, 3 * 66] in {0,1}: does relative seat r hold a card at position p.

    The belief head's supervision, and training-only for the same reason.
    """
    priv = privileged_features(states).reshape(states.shape[0], L.PRIVILEGED_PLANES, NUM_ENC_SLOTS)
    return (priv[:, :3] > 0).astype(np.float32).reshape(states.shape[0], -1)
