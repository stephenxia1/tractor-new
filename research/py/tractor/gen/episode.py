"""Random generation of play-phase starting positions.

The trump suit, the level rank, the landlord and the kitty are inputs to
the play phase rather than outcomes of it, so they are drawn here. No
scripted policy is consulted at any point.

The declaration is drawn uniformly over the *legal* declarations rather
than uniformly over the four suits: a trump suit nobody could have
declared is a position the rules cannot produce. Within that constraint
the draw carries no judgement about which declaration is good.

The burial is a uniform eight-card subset of the landlord's thirty-three.
This is also what makes the uniform belief sampler correct: it models the
kitty as an ordinary hidden location, which is true of a uniform burial
and false of a chosen one.
"""

from __future__ import annotations

import numpy as np

from ..rules.cards import (
    BIG_JOKER,
    NO_TRUMP,
    NUM_TYPES,
    SMALL_JOKER,
    Ctx,
    card_id,
    suit_of,
)
from ..sim.state import PlayState

KITTY_SIZE = 8
NUM_SEATS = 4


def deal_sizes(num_decks: int) -> tuple[int, int]:
    """Cards per hand and kitty size for a deck count.

    The kitty is fixed at eight and the rest is split four ways, so a
    deck count that does not divide evenly has no deal and is rejected
    rather than dealt into hands of unequal size.
    """
    total = NUM_TYPES * num_decks - KITTY_SIZE
    if num_decks < 1 or total <= 0 or total % NUM_SEATS:
        raise ValueError(
            f"{num_decks} decks leaves {total} cards after the kitty, "
            f"which does not divide among {NUM_SEATS} seats"
        )
    return total // NUM_SEATS, KITTY_SIZE


HAND_SIZE = deal_sizes(2)[0]


def _uniform_subset(
    rng: np.random.Generator, pool: np.ndarray, k: int
) -> np.ndarray:
    """A uniform k-card subset of a count vector."""
    cards = np.repeat(np.arange(NUM_TYPES), pool.astype(np.int64))
    picked = rng.choice(cards, size=k, replace=False)
    out = np.zeros(NUM_TYPES, dtype=np.int8)
    for t in picked:
        out[t] += 1
    return out


def legal_declarations(
    hands: np.ndarray, level_rank: int, num_decks: int
) -> list[tuple[int, int, int]]:
    """Every declaration available, as ``(seat, trump_suit, tier)``.

    Tier 1 is a single level card, 2 a pair of them, 3 a small-joker pair
    and 4 a big-joker pair. Tiers 3 and 4 declare no-trump.
    """
    out: list[tuple[int, int, int]] = []
    for seat in range(NUM_SEATS):
        for suit in range(4):
            held = int(hands[seat, card_id(suit, level_rank)])
            if held >= 1:
                out.append((seat, suit, 1))
            if held >= num_decks:
                out.append((seat, suit, 2))
        if int(hands[seat, SMALL_JOKER]) >= num_decks:
            out.append((seat, NO_TRUMP, 3))
        if int(hands[seat, BIG_JOKER]) >= num_decks:
            out.append((seat, NO_TRUMP, 4))
    return out


def _flip_trump(rng: np.random.Generator, kitty: np.ndarray) -> int:
    """rules.md §5.4: with no declaration, the kitty's top card sets the
    suit. Jokers carry no suit, so the first non-joker is taken."""
    cards = np.repeat(np.arange(NUM_TYPES), kitty.astype(np.int64))
    rng.shuffle(cards)
    for t in cards:
        if t < SMALL_JOKER:
            return suit_of(int(t))
    return NO_TRUMP


def generate(
    rng: np.random.Generator, num_decks: int = 2
) -> tuple[PlayState, dict]:
    """One starting position, plus a record of how it was drawn."""
    hand_size, kitty_size = deal_sizes(num_decks)
    deck = np.repeat(np.arange(NUM_TYPES), num_decks)
    rng.shuffle(deck)

    hands = np.zeros((NUM_SEATS, NUM_TYPES), dtype=np.int8)
    for seat in range(NUM_SEATS):
        for t in deck[seat * hand_size : (seat + 1) * hand_size]:
            hands[seat, t] += 1
    kitty = np.zeros(NUM_TYPES, dtype=np.int8)
    for t in deck[NUM_SEATS * hand_size :]:
        kitty[t] += 1

    landlord = int(rng.integers(NUM_SEATS))
    level_rank = int(rng.integers(13))

    options = legal_declarations(hands, level_rank, num_decks)
    if options:
        seat, trump_suit, tier = options[int(rng.integers(len(options)))]
        declarer = seat
    else:
        trump_suit = _flip_trump(rng, kitty)
        declarer, tier = -1, 0

    # The landlord takes the kitty and buries eight at random.
    pool = (hands[landlord] + kitty).astype(np.int8)
    buried = _uniform_subset(rng, pool, kitty_size)
    hands[landlord] = (pool - buried).astype(np.int8)

    ctx = Ctx(level_rank, trump_suit)
    state = PlayState.new(ctx, hands, buried, landlord, num_decks)
    info = {
        "level_rank": level_rank,
        "trump_suit": trump_suit,
        "landlord": landlord,
        "declarer": declarer,
        "tier": tier,
        "declaration_options": len(options),
    }
    return state, info
