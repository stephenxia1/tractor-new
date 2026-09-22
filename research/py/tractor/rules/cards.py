"""Card identity, trump context, and the strength ladder.

A card is a *type* in ``0..53``: ``13 * suit + rank`` for the standard
cards, 52 for the small joker, 53 for the big joker. Suits are ordered
spades, hearts, diamonds, clubs and ranks 2..A, matching the slot
numbering the TypeScript engine emits, so the two representations index
each other directly.

Multisets of cards -- hands, plays, the kitty -- are ``int8[54]`` count
vectors with entries in ``0..num_decks``. Nothing carries a per-copy
identity. Two copies of a card are indistinguishable by construction,
which removes a class of bug the identity-carrying representation is
prone to: combo detection comparing by rank-and-suit while legality
comparison compares by copy, so one side sees a pair where the other
sees two unrelated cards.

Everything that depends on the trump context is precomputed. There are
only 65 contexts (13 level ranks by 4 trump suits plus no-trump), so the
effective suit, the strength and the trump flag of all 54 types in every
one of them are built once into lookup tables at import.
"""

from __future__ import annotations

import numpy as np

NUM_SUITS = 4
NUM_RANKS = 13
NUM_TYPES = 54

SMALL_JOKER = 52
BIG_JOKER = 53

SUIT_NAMES = ("spades", "hearts", "diamonds", "clubs")
RANK_NAMES = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")

# Effective suit 4 means trump; 0..3 are the natural suits.
TRUMP = 4

# Rank indices of the point cards.
RANK_5 = RANK_NAMES.index("5")
RANK_10 = RANK_NAMES.index("10")
RANK_K = RANK_NAMES.index("K")

# No-trump is stored as trump suit 4 so that a context is a pair of small
# integers rather than an optional.
NO_TRUMP = 4
NUM_TRUMP_CHOICES = 5
NUM_CONTEXTS = NUM_RANKS * NUM_TRUMP_CHOICES


def suit_of(t: int) -> int:
    """Natural suit of a standard card. Undefined for jokers."""
    return t // NUM_RANKS


def rank_of(t: int) -> int:
    """Rank index of a standard card. Undefined for jokers."""
    return t % NUM_RANKS


def card_id(suit: int, rank: int) -> int:
    return suit * NUM_RANKS + rank


def context_id(level_rank: int, trump_suit: int) -> int:
    """Index of ``(level_rank, trump_suit)`` into the lookup tables."""
    return level_rank * NUM_TRUMP_CHOICES + trump_suit


def card_name(t: int) -> str:
    if t == SMALL_JOKER:
        return "SJ"
    if t == BIG_JOKER:
        return "BJ"
    return f"{RANK_NAMES[rank_of(t)]}{SUIT_NAMES[suit_of(t)][0].upper()}"


# --- point values ---------------------------------------------------------

POINTS = np.zeros(NUM_TYPES, dtype=np.int16)
for _t in range(NUM_SUITS * NUM_RANKS):
    _r = rank_of(_t)
    if _r == RANK_5:
        POINTS[_t] = 5
    elif _r in (RANK_10, RANK_K):
        POINTS[_t] = 10


# --- per-context tables ---------------------------------------------------


def _build_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eff = np.zeros((NUM_CONTEXTS, NUM_TYPES), dtype=np.int8)
    strength = np.zeros((NUM_CONTEXTS, NUM_TYPES), dtype=np.int8)
    is_trump = np.zeros((NUM_CONTEXTS, NUM_TYPES), dtype=bool)

    for lr in range(NUM_RANKS):
        for ts in range(NUM_TRUMP_CHOICES):
            cid = context_id(lr, ts)
            for t in range(NUM_TYPES):
                joker = t >= SMALL_JOKER
                if joker:
                    trump = True
                else:
                    trump = rank_of(t) == lr or (ts != NO_TRUMP and suit_of(t) == ts)
                is_trump[cid, t] = trump
                eff[cid, t] = TRUMP if trump else suit_of(t)

                if not trump:
                    # Ranks ascending with the level rank removed.
                    r = rank_of(t)
                    strength[cid, t] = r - 1 if r > lr else r
                elif ts == NO_TRUMP:
                    # Only the level cards and the jokers are trump, and
                    # the four level cards are equal in strength.
                    if t == BIG_JOKER:
                        strength[cid, t] = 2
                    elif t == SMALL_JOKER:
                        strength[cid, t] = 1
                    else:
                        strength[cid, t] = 0
                else:
                    if t == BIG_JOKER:
                        strength[cid, t] = 15
                    elif t == SMALL_JOKER:
                        strength[cid, t] = 14
                    elif rank_of(t) == lr and suit_of(t) == ts:
                        strength[cid, t] = 13
                    elif rank_of(t) == lr:
                        # The three off-suit level cards share one rung,
                        # so two of them never chain into a tractor.
                        strength[cid, t] = 12
                    else:
                        r = rank_of(t)
                        strength[cid, t] = r - 1 if r > lr else r
    return eff, strength, is_trump


EFF_SUIT, STRENGTH, IS_TRUMP = _build_tables()

# Highest strength any card can reach, per effective suit, for bounds.
MAX_STRENGTH = int(STRENGTH.max())


class Ctx:
    """A trump context: the level rank and the trump suit.

    ``trump_suit`` is ``NO_TRUMP`` for a joker declaration. The instance
    caches the row of each lookup table so callers index a 54-vector
    rather than a 2-D array.
    """

    __slots__ = (
        "level_rank",
        "trump_suit",
        "cid",
        "eff",
        "strength",
        "is_trump",
        "suit_mask",
    )

    def __init__(self, level_rank: int, trump_suit: int) -> None:
        if not 0 <= level_rank < NUM_RANKS:
            raise ValueError(f"level_rank {level_rank} out of range")
        if not 0 <= trump_suit < NUM_TRUMP_CHOICES:
            raise ValueError(f"trump_suit {trump_suit} out of range")
        self.level_rank = level_rank
        self.trump_suit = trump_suit
        self.cid = context_id(level_rank, trump_suit)
        self.eff: np.ndarray = EFF_SUIT[self.cid]
        self.strength: np.ndarray = STRENGTH[self.cid]
        self.is_trump: np.ndarray = IS_TRUMP[self.cid]
        # One int8 row per effective suit, holding 1 where a card type
        # belongs to it. Restricting a holding to a suit is then a
        # multiply: the comparison it replaces is constant for the
        # context, and the enumerator asks for it several times per
        # decision.
        self.suit_mask: np.ndarray = np.stack(
            [(self.eff == suit).astype(np.int8) for suit in range(NUM_SUITS + 1)]
        )

    def __repr__(self) -> str:
        ts = "NT" if self.trump_suit == NO_TRUMP else SUIT_NAMES[self.trump_suit]
        return f"Ctx(level={RANK_NAMES[self.level_rank]}, trump={ts})"


def full_deck(num_decks: int = 2) -> np.ndarray:
    """Count vector for a complete deck."""
    return np.full(NUM_TYPES, num_decks, dtype=np.int8)


def counts_from_ids(ids: "list[int] | np.ndarray") -> np.ndarray:
    """Count vector for a list of card types."""
    out = np.zeros(NUM_TYPES, dtype=np.int8)
    for t in ids:
        out[t] += 1
    return out


def ids_from_counts(counts: np.ndarray) -> list[int]:
    """Card types in a count vector, ascending, repeated by count."""
    out: list[int] = []
    for t in range(NUM_TYPES):
        out.extend([t] * int(counts[t]))
    return out


def total_points(counts: np.ndarray) -> int:
    return int(POINTS @ counts.astype(np.int16))
