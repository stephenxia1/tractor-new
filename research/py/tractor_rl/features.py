"""Wire records -> network inputs.

Every modelling decision about *how the board is presented to the network*
lives in this file: which planes exist, how trump is anchored, what the
scalars are. The TypeScript side never sees any of it, so changing the
representation is a Python-only edit.

Two ideas are doing the real work here:

* **Trump anchoring.** Slots are permuted so the trump suit always occupies
  the first 13 positions and the other suits follow in a fixed rotation.
  One "trump" representation is then shared across all four suits instead
  of the network having to learn each separately. No-trump hands (joker
  declarations, ~20% of hands under smart declaration) use the identity
  permutation and are flagged in the scalars.

* **Derived unseen pool.** `unseen = 2 - mine - played - my_kitty` is the
  single most informative feature for card counting and costs one
  subtraction here rather than another 54 bytes on the wire.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import layout as L

# --- slot permutations ------------------------------------------------------


def _build_suit_perms() -> np.ndarray:
    """[5, 54] permutation: index 0..3 anchors that trump suit first,
    index 4 (no-trump) is the identity."""
    perms = np.zeros((5, L.NUM_SLOTS), dtype=np.int64)
    for trump in range(5):
        suit_order = (
            [(trump + k) % L.NUM_SUITS for k in range(L.NUM_SUITS)]
            if trump < L.NUM_SUITS
            else list(range(L.NUM_SUITS))
        )
        order: list[int] = []
        for suit in suit_order:
            order.extend(range(suit * L.NUM_RANKS, suit * L.NUM_RANKS + L.NUM_RANKS))
        order.extend([52, 53])
        perms[trump] = np.asarray(order, dtype=np.int64)
    return perms


SUIT_PERM = _build_suit_perms()

# --- dimensions -------------------------------------------------------------

_PLANE_FEATURES = (
    2 * L.NUM_SLOTS  # my hand: has-one, has-pair
    + 2 * L.NUM_SLOTS  # unseen pool: has-one, has-pair
    + 4 * L.NUM_SLOTS  # cards played this hand, per relative seat
    + 4 * L.NUM_SLOTS  # cards in the current trick, per relative seat
    + L.NUM_SLOTS  # my buried kitty (landlord only)
)
SCALAR_DIM = (
    L.NUM_ROLES  # role one-hot
    + 4  # landlord relative seat
    + 4  # position in trick
    + 1  # is-lead
    + 5  # relative seat currently winning (index 4 = nobody yet)
    + 1  # am I an attacker
    + 5  # hand size, trick index, banked points, table points, cards seen
    + 4 * 5  # void mask: 4 relative seats x (4 suits + trump)
    + L.NUM_RANKS  # trump rank one-hot
    + 5  # trump suit one-hot (index 4 = no-trump)
    + 1  # bias
)
STATE_DIM = _PLANE_FEATURES + SCALAR_DIM
ACTION_DIM = 2 * L.NUM_SLOTS + L.NUM_SHAPES + 4 + 2

NUM_DECKS = 2


@dataclass(slots=True)
class Batch:
    """One frame's worth of pending decisions, still in wire form."""

    states: np.ndarray  # [B, STATE_BYTES] uint8
    cands: np.ndarray  # [N, CANDIDATE_BYTES] uint8
    owner: np.ndarray  # [N] int64, index into states
    cand_counts: np.ndarray  # [B] int64
    episodes: np.ndarray  # [E, EPISODE_BYTES] uint8

    @property
    def num_decisions(self) -> int:
        return int(self.states.shape[0])


def parse_frame(payload: bytes) -> Batch:
    """Split one actor frame into decisions + episodes.

    The only Python-level loop is over decisions (a few hundred per frame)
    to recover record offsets; everything after that is vectorised.
    """
    buf = np.frombuffer(payload, dtype=np.uint8)
    n_dec = int.from_bytes(payload[0:2], "little")
    n_ep = int.from_bytes(payload[2:4], "little")

    starts = np.empty(n_dec, dtype=np.int64)
    counts = np.empty(n_dec, dtype=np.int64)
    off = L.BATCH_HEADER_BYTES
    for i in range(n_dec):
        rec_bytes = int.from_bytes(payload[off : off + 2], "little")
        starts[i] = off + 2
        counts[i] = (rec_bytes - L.STATE_BYTES) // L.CANDIDATE_BYTES
        off += 2 + rec_bytes

    if n_dec:
        states = buf[starts[:, None] + np.arange(L.STATE_BYTES)]
        total = int(counts.sum())
        owner = np.repeat(np.arange(n_dec, dtype=np.int64), counts)
        within = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        cand_start = np.repeat(starts + L.STATE_BYTES, counts) + within * L.CANDIDATE_BYTES
        cands = buf[cand_start[:, None] + np.arange(L.CANDIDATE_BYTES)]
    else:
        states = np.zeros((0, L.STATE_BYTES), dtype=np.uint8)
        cands = np.zeros((0, L.CANDIDATE_BYTES), dtype=np.uint8)
        owner = np.zeros(0, dtype=np.int64)

    episodes = (
        buf[off : off + n_ep * L.EPISODE_BYTES].reshape(n_ep, L.EPISODE_BYTES)
        if n_ep
        else np.zeros((0, L.EPISODE_BYTES), dtype=np.uint8)
    )
    return Batch(states=states, cands=cands, owner=owner, cand_counts=counts, episodes=episodes)


# --- state features ---------------------------------------------------------


def field_u8(states: np.ndarray, offset: int) -> np.ndarray:
    """One uint8 header field for a batch of records."""
    return states[:, offset].astype(np.int64)


def field_u16(states: np.ndarray, offset: int) -> np.ndarray:
    return states[:, offset].astype(np.int64) + states[:, offset + 1].astype(np.int64) * 256


def field_u32(states: np.ndarray, offset: int) -> np.ndarray:
    b = states[:, offset : offset + 4].astype(np.int64)
    return b[:, 0] + (b[:, 1] << 8) + (b[:, 2] << 16) + (b[:, 3] << 24)


def _u16(states: np.ndarray, offset: int) -> np.ndarray:
    return (
        states[:, offset].astype(np.int32)
        + states[:, offset + 1].astype(np.int32) * 256
    )


def state_features(states: np.ndarray) -> np.ndarray:
    """[B, STATE_BYTES] uint8 -> [B, STATE_DIM] float32."""
    b = states.shape[0]
    if b == 0:
        return np.zeros((0, STATE_DIM), dtype=np.float32)

    header = states[:, : L.HEADER_BYTES]
    planes = states[:, L.HEADER_BYTES :].reshape(b, L.NUM_PLANES, L.NUM_SLOTS)

    trump_suit = header[:, L.H_TRUMP_SUIT].astype(np.int64)
    perm = SUIT_PERM[trump_suit]  # [B, 54]
    planes = np.take_along_axis(planes, perm[:, None, :], axis=2).astype(np.float32)

    my_hand = planes[:, L.PLANE_MY_HAND]
    played = planes[:, L.PLANE_PLAYED_BASE : L.PLANE_PLAYED_BASE + 4]
    trick = planes[:, L.PLANE_TRICK_BASE : L.PLANE_TRICK_BASE + 4]
    kitty = planes[:, L.PLANE_MY_KITTY]

    unseen = np.clip(NUM_DECKS - my_hand - played.sum(axis=1) - kitty, 0.0, NUM_DECKS)

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
        idx = np.clip(values.astype(np.int64), 0, width - 1)
        scalars[rows, at + idx] = 1.0
        at += width

    one_hot(header[:, L.H_ROLE], L.NUM_ROLES)
    one_hot(header[:, L.H_LANDLORD_REL], 4)
    one_hot(header[:, L.H_TRICK_POS], 4)
    scalars[:, at] = header[:, L.H_IS_LEAD]
    at += 1
    winner = header[:, L.H_WINNER_REL].astype(np.int64)
    one_hot(np.where(winner > 3, 4, winner), 5)
    scalars[:, at] = header[:, L.H_I_AM_ATTACKER]
    at += 1

    scalars[:, at] = header[:, L.H_HAND_SIZE] / 25.0
    scalars[:, at + 1] = header[:, L.H_TRICK_INDEX] / 25.0
    scalars[:, at + 2] = _u16(states, L.H_ATTACKER_POINTS) / 200.0
    scalars[:, at + 3] = header[:, L.H_TABLE_POINTS] / 40.0
    scalars[:, at + 4] = _u16(states, L.H_CARDS_SEEN) / 108.0
    at += 5

    # Void bits, with the suit axis rotated the same way as the planes so
    # "suit 0" means trump-suit in both places.
    void = header[:, L.H_VOID_MASK : L.H_VOID_MASK + 4].astype(np.int64)  # [B, 4]
    suit_rot = np.where(
        trump_suit[:, None] < L.NUM_SUITS,
        (trump_suit[:, None] + np.arange(L.NUM_SUITS)[None, :]) % L.NUM_SUITS,
        np.arange(L.NUM_SUITS)[None, :],
    )  # [B, 4] source suit for each output position
    suit_bits = (void[:, :, None] >> suit_rot[:, None, :]) & 1  # [B, rel, suit]
    trump_bits = (void >> 4) & 1  # [B, rel]
    void_feats = np.concatenate([suit_bits, trump_bits[:, :, None]], axis=2)
    scalars[:, at : at + 20] = void_feats.reshape(b, 20).astype(np.float32)
    at += 20

    one_hot(header[:, L.H_TRUMP_RANK], L.NUM_RANKS)
    one_hot(trump_suit, 5)
    scalars[:, at] = 1.0
    at += 1
    assert at == SCALAR_DIM, (at, SCALAR_DIM)

    return np.concatenate([plane_feats, scalars], axis=1)


# --- action features --------------------------------------------------------


def action_features(
    cands: np.ndarray, states: np.ndarray, owner: np.ndarray
) -> np.ndarray:
    """[N, CANDIDATE_BYTES] uint8 -> [N, ACTION_DIM] float32.

    Candidate card lists are fixed-width and padded, so the whole batch
    converts to count planes with one bincount.
    """
    n = cands.shape[0]
    if n == 0:
        return np.zeros((0, ACTION_DIM), dtype=np.float32)

    n_cards = cands[:, 0].astype(np.float32)
    shape = cands[:, 1].astype(np.int64)
    flags = cands[:, 2].astype(np.int64)
    points = cands[:, 3].astype(np.float32)

    slots = cands[:, L.CANDIDATE_HEADER_BYTES :].astype(np.int64)
    valid = slots != L.NO_CARD
    rowsr, colsr = np.nonzero(valid)
    flat = rowsr * L.NUM_SLOTS + slots[rowsr, colsr]
    counts = np.bincount(flat, minlength=n * L.NUM_SLOTS).reshape(n, L.NUM_SLOTS)

    # Anchor trump the same way the state planes are anchored.
    trump_suit = states[owner, L.H_TRUMP_SUIT].astype(np.int64)
    counts = np.take_along_axis(counts, SUIT_PERM[trump_suit], axis=1)

    out = np.zeros((n, ACTION_DIM), dtype=np.float32)
    out[:, : L.NUM_SLOTS] = counts >= 1
    out[:, L.NUM_SLOTS : 2 * L.NUM_SLOTS] = counts >= 2
    base = 2 * L.NUM_SLOTS
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


# --- episodes ---------------------------------------------------------------


@dataclass(slots=True)
class Episode:
    game_id: int
    hand_id: int
    landlord_seat: int
    num_tricks: int
    attacker_points: int
    level_delta: int
    defenders_won: bool
    kitty_points: int
    trick_points: np.ndarray  # [num_tricks] int
    trick_attacker_won: np.ndarray  # [num_tricks] bool
    # Fingerprint of the deal this hand came from (env.ts computeDealDigest).
    # Duplicate-deal evaluation joins pairs on (game, hand) and checks this.
    deal_digest: int = 0
    trump_rank: int = 0
    trump_suit: int = 0


def parse_episodes(episodes: np.ndarray) -> list[Episode]:
    out: list[Episode] = []
    for row in episodes:
        raw = row.tobytes()
        num_tricks = int(row[L.E_NUM_TRICKS])
        trick_bytes = row[L.E_TRICKS_BASE : L.E_TRICKS_BASE + 2 * L.MAX_TRICKS]
        pts = trick_bytes[0::2][:num_tricks].astype(np.int32)
        won = trick_bytes[1::2][:num_tricks].astype(bool)
        out.append(
            Episode(
                game_id=int.from_bytes(raw[L.E_GAME_ID : L.E_GAME_ID + 2], "little"),
                hand_id=int.from_bytes(raw[L.E_HAND_ID : L.E_HAND_ID + 4], "little"),
                landlord_seat=int(row[L.E_LANDLORD_SEAT]),
                num_tricks=num_tricks,
                attacker_points=int.from_bytes(
                    raw[L.E_ATTACKER_POINTS : L.E_ATTACKER_POINTS + 2], "little"
                ),
                level_delta=int(np.int8(row[L.E_LEVEL_DELTA])),
                defenders_won=bool(row[L.E_DEFENDERS_WON]),
                kitty_points=int(row[L.E_KITTY_POINTS]),
                trick_points=pts,
                trick_attacker_won=won,
                deal_digest=int.from_bytes(
                    raw[L.E_DEAL_DIGEST : L.E_DEAL_DIGEST + 4], "little"
                ),
                trump_rank=int(row[L.E_TRUMP_RANK]),
                trump_suit=int(row[L.E_TRUMP_SUIT]),
            )
        )
    return out


# --- merging worker frames -------------------------------------------------


@dataclass(slots=True)
class Merged:
    """Every worker's pending decisions concatenated, so one featurise call
    and one forward pass cover the whole round."""

    states: np.ndarray  # [B, STATE_BYTES]
    cands: np.ndarray  # [N, CANDIDATE_BYTES]
    owner: np.ndarray  # [N] decision index
    cand_counts: np.ndarray  # [B]
    cand_offsets: np.ndarray  # [B] first candidate row per decision
    bounds: np.ndarray  # [W+1] decision-index slice per worker

    @property
    def num_decisions(self) -> int:
        return int(self.states.shape[0])


def merge_batches(batches: list[Batch]) -> Merged:
    sizes = [b.num_decisions for b in batches]
    bounds = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    total = int(bounds[-1])
    if total == 0:
        return Merged(
            states=np.zeros((0, L.STATE_BYTES), dtype=np.uint8),
            cands=np.zeros((0, L.CANDIDATE_BYTES), dtype=np.uint8),
            owner=np.zeros(0, dtype=np.int64),
            cand_counts=np.zeros(0, dtype=np.int64),
            cand_offsets=np.zeros(0, dtype=np.int64),
            bounds=bounds,
        )
    states = np.concatenate([b.states for b in batches if b.num_decisions])
    cands = np.concatenate([b.cands for b in batches if b.num_decisions])
    counts = np.concatenate([b.cand_counts for b in batches if b.num_decisions])
    owner = np.repeat(np.arange(total, dtype=np.int64), counts)
    offsets = np.cumsum(counts) - counts
    return Merged(
        states=states,
        cands=cands,
        owner=owner,
        cand_counts=counts,
        cand_offsets=offsets,
        bounds=bounds,
    )


def split_per_worker(values: np.ndarray, bounds: np.ndarray) -> list[np.ndarray]:
    return [values[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
