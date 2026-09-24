"""The deal and the trump declaration, as a sequence of decisions.

Landlord-set hands only (rules.md §4.1): the landlord is fixed before
the deal and receives the kitty whoever declares, so a declaration
chooses the trump suit and nothing else. The trump rank is the
landlord's team's level.

The deal is 100 cards, one at a time and clockwise from the landlord
(§4.2); the last 8 of the stack are the kitty. **Every card dealt is a
decision** for the seat that received it: pass, or make one of the
declarations its revealed cards allow. A seat with nothing legal passes
without being asked. After the last card there is a closing round --
each seat in turn, cycling until four in a row pass -- because
declarations stay open until the landlord picks up the kitty (§5.3),
and without it the last seat dealt a card would be the only one that
ever saw a full hand before deciding.

Actions are a fixed set of 11:

    0       pass
    1..4    single of the level rank in suit 0..3            (tier 1)
    5..8    pair of the level rank in suit 0..3              (tier 2)
    9       small-joker pair, no-trump                       (tier 3)
    10      big-joker pair, no-trump                         (tier 4)

A declaration is legal when the seat holds the cards among those
revealed to it and its tier is strictly above the standing one (§5.1,
§5.2; the engine's ``declare`` refuses ``newTier <= currentTier``). With
no declaration at all, the top card of the kitty sets the suit, a
joker meaning no-trump (§5.4).

``tests/test_deal.py`` checks the legal set against the TypeScript
engine on scenarios dumped by ``src/dumpDeclare.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..rules.cards import BIG_JOKER, NO_TRUMP, NUM_TYPES, SMALL_JOKER, Ctx, card_id, suit_of
from .state import NUM_SEATS, team_of

__all__ = [
    "NUM_ACTIONS",
    "PASS",
    "Declaration",
    "DealEnv",
    "legal_mask",
    "action_tier",
    "action_suit",
]

NUM_ACTIONS = 11
PASS = 0
KITTY_SIZE = 8

_TIER = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4], dtype=np.int64)
_SUIT = np.array([-1, 0, 1, 2, 3, 0, 1, 2, 3, NO_TRUMP, NO_TRUMP], dtype=np.int64)


def action_tier(action: int) -> int:
    return int(_TIER[action])


def action_suit(action: int) -> int:
    return int(_SUIT[action])


@dataclass(frozen=True)
class Declaration:
    seat: int
    action: int

    @property
    def tier(self) -> int:
        return action_tier(self.action)

    @property
    def suit(self) -> int:
        return action_suit(self.action)


def legal_mask(
    revealed: np.ndarray, level_rank: int, standing_tier: int, num_decks: int = 2
) -> np.ndarray:
    """bool[11]: what a seat holding ``revealed`` may do now.

    Pass is always legal. Everything else needs the cards and a tier
    strictly above ``standing_tier`` (0 when nothing stands).
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[PASS] = True
    for s in range(4):
        held = int(revealed[card_id(s, level_rank)])
        if held >= 1 and 1 > standing_tier:
            mask[1 + s] = True
        if held >= num_decks and 2 > standing_tier:
            mask[5 + s] = True
    if int(revealed[SMALL_JOKER]) >= num_decks and 3 > standing_tier:
        mask[9] = True
    if int(revealed[BIG_JOKER]) >= num_decks and 4 > standing_tier:
        mask[10] = True
    return mask


class DealEnv:
    """One hand from the shuffle to the kitty pickup.

    Drive it with ``next_decision`` (the seat whose choice is pending, or
    ``None`` once declarations have closed) and ``act``. ``pool`` and
    ``trump_suit`` are then the landlord's 33 cards and the trump suit.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        landlord: int,
        level_rank: int,
        num_decks: int = 2,
    ) -> None:
        self.landlord = int(landlord)
        self.level_rank = int(level_rank)
        self.num_decks = int(num_decks)
        deck = np.repeat(np.arange(NUM_TYPES), num_decks)
        rng.shuffle(deck)
        self.stack = deck
        self.n_dealt_total = len(deck) - KITTY_SIZE
        self.final = np.zeros((NUM_SEATS, NUM_TYPES), dtype=np.int8)
        for i in range(self.n_dealt_total):
            self.final[(self.landlord + i) % NUM_SEATS, deck[i]] += 1
        self.kitty = np.bincount(
            deck[self.n_dealt_total :], minlength=NUM_TYPES
        ).astype(np.int8)
        self.revealed = np.zeros((NUM_SEATS, NUM_TYPES), dtype=np.int8)
        self.dealt = 0
        self.standing: Declaration | None = None
        self.log: list[Declaration] = []
        self.closing = False
        self.cursor = 0
        self.passes = 0
        self.pending: int | None = None
        self.done = False

    # -- queries -----------------------------------------------------------

    @property
    def standing_tier(self) -> int:
        return 0 if self.standing is None else self.standing.tier

    def mask(self, seat: int) -> np.ndarray:
        return legal_mask(
            self.revealed[seat], self.level_rank, self.standing_tier, self.num_decks
        )

    def _can_declare(self, seat: int) -> bool:
        return bool(self.mask(seat)[1:].any())

    @property
    def trump_suit(self) -> int:
        """The trump suit once declarations close (a kitty flip if none)."""
        if self.standing is not None:
            return self.standing.suit
        top = int(self.stack[self.n_dealt_total])
        return NO_TRUMP if top >= SMALL_JOKER else suit_of(top)

    @property
    def ctx(self) -> Ctx:
        return Ctx(self.level_rank, self.trump_suit)

    @property
    def pool(self) -> np.ndarray:
        """int8[54]: the landlord's hand with the kitty picked up."""
        return (self.final[self.landlord] + self.kitty).astype(np.int8)

    # -- driving -----------------------------------------------------------

    def next_decision(self) -> int | None:
        """Deal on until a seat has a real choice; ``None`` when closed."""
        if self.pending is not None:
            return self.pending
        while not self.done:
            if not self.closing:
                if self.dealt == self.n_dealt_total:
                    self.closing = True
                    self.cursor = (self.landlord + self.dealt) % NUM_SEATS
                    self.passes = 0
                    continue
                seat = (self.landlord + self.dealt) % NUM_SEATS
                self.revealed[seat, self.stack[self.dealt]] += 1
                self.dealt += 1
                if self._can_declare(seat):
                    self.pending = seat
                    return seat
                continue
            if self.passes >= NUM_SEATS:
                self.done = True
                break
            seat = self.cursor
            if self._can_declare(seat):
                self.pending = seat
                return seat
            self.passes += 1
            self.cursor = (seat + 1) % NUM_SEATS
        return None

    def act(self, seat: int, action: int) -> None:
        if self.pending != seat:
            raise RuntimeError(f"seat {seat} acted, {self.pending} was asked")
        if not self.mask(seat)[action]:
            raise ValueError(f"seat {seat} may not take action {action}")
        self.pending = None
        if action != PASS:
            self.standing = Declaration(seat, int(action))
            self.log.append(self.standing)
        if self.closing:
            self.passes = 0 if action != PASS else self.passes + 1
            self.cursor = (seat + 1) % NUM_SEATS


# --- the declaring seat's view --------------------------------------------

# Features: revealed counts (54), per-suit summaries (4 x 5), jokers (2),
# progress (2), closing flag (1), level rank (13), landlord's relative
# seat (4), standing tier (5), standing suit (5), declarer's relative
# seat (4), legal mask (11).
DECLARE_OBS_DIM = 54 + 20 + 2 + 2 + 1 + 13 + 4 + 5 + 5 + 4 + NUM_ACTIONS
# The baseline may read everything: the four final hands in relative
# order and the kitty.
DECLARE_PRIV_DIM = 5 * NUM_TYPES


def declare_obs(env: DealEnv, seat: int) -> np.ndarray:
    """float32[DECLARE_OBS_DIM]: what ``seat`` knows when asked."""
    st = env.standing
    return declare_features(
        revealed=env.revealed[seat],
        num_decks=env.num_decks,
        level_rank=env.level_rank,
        landlord=env.landlord,
        seat=seat,
        dealt=env.dealt,
        total=env.n_dealt_total,
        closing=env.closing,
        standing_tier=env.standing_tier,
        standing_suit=-1 if st is None else st.suit,
        standing_seat=-1 if st is None else st.seat,
        mask=env.mask(seat),
    )


def declare_features(
    *,
    revealed: np.ndarray,
    num_decks: int,
    level_rank: int,
    landlord: int,
    seat: int,
    dealt: int,
    total: int,
    closing: bool,
    standing_tier: int,
    standing_suit: int,
    standing_seat: int,
    mask: np.ndarray,
) -> np.ndarray:
    """The same features from their parts, for a caller without a
    ``DealEnv`` -- the evaluation host describes a live engine hand."""
    hand = np.asarray(revealed).astype(np.float32)
    n = num_decks
    out = np.zeros(DECLARE_OBS_DIM, dtype=np.float32)
    at = 0
    out[at : at + NUM_TYPES] = hand / n
    at += NUM_TYPES
    for s in range(4):
        cards = hand[s * 13 : (s + 1) * 13]
        out[at + 0] = cards.sum() / 13.0
        out[at + 1] = float(cards[level_rank]) / n
        out[at + 2] = float((cards >= n).sum()) / 6.0
        out[at + 3] = float(cards[8:].sum()) / 10.0  # 10 and above
        out[at + 4] = float(cards[3] * 5 + cards[8] * 10 + cards[11] * 10) / 50.0
        at += 5
    out[at] = hand[SMALL_JOKER] / n
    out[at + 1] = hand[BIG_JOKER] / n
    at += 2
    out[at] = dealt / total
    out[at + 1] = hand.sum() / (total / NUM_SEATS)
    at += 2
    out[at] = float(closing)
    at += 1
    out[at + level_rank] = 1.0
    at += 13
    out[at + (landlord - seat) % NUM_SEATS] = 1.0
    at += 4
    out[at + standing_tier] = 1.0
    at += 5
    if standing_tier > 0:
        out[at + standing_suit] = 1.0
    at += 5
    if standing_tier > 0:
        out[at + (standing_seat - seat) % NUM_SEATS] = 1.0
    at += 4
    out[at : at + NUM_ACTIONS] = np.asarray(mask, dtype=np.float32)
    at += NUM_ACTIONS
    assert at == DECLARE_OBS_DIM
    return out


def declare_privileged(env: DealEnv, seat: int) -> np.ndarray:
    """float32[DECLARE_PRIV_DIM]: the whole deal, from ``seat``'s chair."""
    out = np.zeros((5, NUM_TYPES), dtype=np.float32)
    for rel in range(NUM_SEATS):
        out[rel] = env.final[(seat + rel) % NUM_SEATS]
    out[4] = env.kitty
    return out.reshape(-1)


def same_team(a: int, b: int) -> bool:
    return team_of(a) == team_of(b)
