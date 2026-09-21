"""Observation encoding: one seat's legal view of a play-phase position.

Everything here is built from what a single seat may know -- its own
hand, its own burial when it is the landlord, and every card that has
hit the table. Nothing reads another seat's hand, and nothing reads the
kitty for a seat that did not bury it. ``tests/test_features.py``
enforces that by re-dealing the hidden cards among the hidden locations
and demanding a bit-identical observation.

Seats are relative: ``rel = (other - me) % 4``, so 0 is me, 1 the seat
that acts after me, 2 my partner and 3 the seat before me. A network fed
relative seats sees one position the same way from all four chairs.

The per-card features are split in two because the larger half never
moves. Effective suit, strength and point value are fixed for a whole
hand once the trump context is known, so ``Cache`` builds that block
once per context and every later decision in the hand reuses it; only
the two-column dynamic block is rebuilt per decision. The history tokens
summarise past plays with the *static* block alone -- a token must mean
the same thing tomorrow as it did when it was written, and the dynamic
columns are relative to a moving unseen set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..rules.cards import (
    BIG_JOKER,
    MAX_STRENGTH,
    NUM_RANKS,
    NUM_TRUMP_CHOICES,
    NUM_TYPES,
    POINTS,
    SMALL_JOKER,
    TRUMP,
    Ctx,
)
from ..rules.combos import covers, decompose, play_suit
from ..sim.state import NUM_SEATS, PlayState, team_of

# --- plane layout ---------------------------------------------------------

NUM_PLANES = 10
PLANE_HAND = 0  # my hand
PLANE_PLAYED = 1  # + relative seat: completed tricks
PLANE_TRICK = 5  # + relative seat: the trick in progress
PLANE_KITTY = 9  # my burial, zero unless I am the landlord

# --- card feature layout --------------------------------------------------

NUM_EFF_SUITS = 5  # the four natural suits plus trump
NUM_STRENGTH = MAX_STRENGTH + 1  # strength rungs, 0..15
NUM_POINT_VALUES = 3  # {0, 5, 10}
NUM_CARD_FLAGS = 4

STATIC_SUIT = 0
STATIC_STRENGTH = STATIC_SUIT + NUM_EFF_SUITS
STATIC_POINTS = STATIC_STRENGTH + NUM_STRENGTH
STATIC_FLAGS = STATIC_POINTS + NUM_POINT_VALUES
FLAG_LEVEL = 0  # a card of the level rank, in any suit
FLAG_TRUMP_LEVEL = 1  # the level card of the trump suit
FLAG_SMALL_JOKER = 2
FLAG_BIG_JOKER = 3
D_STATIC = STATIC_FLAGS + NUM_CARD_FLAGS  # 28

DYN_COUNT = 0  # copies held, over num_decks
DYN_TOP_UNSEEN = 1  # nothing unseen in this effective suit outranks it
D_DYNAMIC = 2

D_CARD = D_STATIC + D_DYNAMIC  # 30

# --- context layout -------------------------------------------------------

CTX_ROLE = 0  # 4: landlord, landlord's partner, the two attacker seats
CTX_ATTACKER = CTX_ROLE + NUM_SEATS  # my team is attacking
CTX_LEVEL = CTX_ATTACKER + 1  # 13
CTX_TRUMP = CTX_LEVEL + NUM_RANKS  # 5, index 4 = no-trump
CTX_ATTACKER_POINTS = CTX_TRUMP + NUM_TRUMP_CHOICES
CTX_UNPLAYED_POINTS = CTX_ATTACKER_POINTS + 1
CTX_TRICKS_PLAYED = CTX_UNPLAYED_POINTS + 1
CTX_CARDS_LEFT = CTX_TRICKS_PLAYED + 1
CTX_LEAD_SIZE = CTX_CARDS_LEFT + 1  # the follow-only fields start here
CTX_LED_SUIT = CTX_LEAD_SIZE + 1  # 5
CTX_WINNING_SEAT = CTX_LED_SUIT + NUM_EFF_SUITS  # 4, relative
CONTEXT_DIM = CTX_WINNING_SEAT + NUM_SEATS  # 37
FOLLOW_FIELDS = slice(CTX_LEAD_SIZE, CONTEXT_DIM)

# --- history layout -------------------------------------------------------

# Tokens kept. A two-deck hand runs to a hundred plays, so the window
# only ever slides for longer variants; the oldest fall off the front.
MAX_HISTORY = 128

HIST_SUMMARY = 0  # D_STATIC: sum of the static rows of the cards played
HIST_SEAT = D_STATIC  # 4, relative
HIST_TRICK = D_STATIC + 4
HIST_POSITION = D_STATIC + 5
HIST_IS_LEAD = D_STATIC + 6
HIST_WON = D_STATIC + 7
HIST_POINTS = D_STATIC + 8
HISTORY_DIM = D_STATIC + 9  # 37

# --- normalisers ----------------------------------------------------------

MAX_TRICKS = 25.0
MAX_HAND = 25.0
MAX_LEAD_SIZE = 8.0
POINT_NORM = 200.0  # the two-deck point total
PLAY_POINT_NORM = 10.0
POINTS_PER_DECK = 100

# Relative distance from the landlord to the role slot: the landlord,
# then their partner, then the attacker who plays after the landlord and
# the one who plays before.
_ROLE_OF_OFFSET = np.array([0, 2, 1, 3], dtype=np.int64)

_POINT_SLOT = np.where(POINTS == 0, 0, np.where(POINTS == 5, 1, 2)).astype(np.int64)
_POINTS_F = POINTS.astype(np.float32)
_TYPES = np.arange(NUM_TYPES)

# Per-context [54, 54] matrix: entry (t, u) is 1 when u shares t's
# effective suit and outranks it. One matrix-vector product against the
# unseen counts then answers "is t the top of its suit" for all 54 types
# at once. Only 65 contexts exist, so the table is bounded.
_BEATERS: dict[int, np.ndarray] = {}


def beater_matrix(ctx: Ctx) -> np.ndarray:
    """Which card types outrank which, within an effective suit."""
    got = _BEATERS.get(ctx.cid)
    if got is None:
        same_suit = ctx.eff[:, None] == ctx.eff[None, :]
        higher = ctx.strength[None, :] > ctx.strength[:, None]
        got = (same_suit & higher).astype(np.float32)
        got.flags.writeable = False
        _BEATERS[ctx.cid] = got
    return got


# --- card features --------------------------------------------------------


def static_card_features(ctx: Ctx, num_decks: int = 2) -> np.ndarray:
    """float32[54, D_STATIC]: what a card is, in this trump context.

    One-hot effective suit, one-hot strength, one-hot point value over
    {0, 5, 10}, then the flags is-level-card, is-trump-suit-level,
    is-small-joker, is-big-joker.

    ``num_decks`` takes part in no feature today; it is in the signature
    to match the dynamic half and to key the cache, so that a future
    deck-count-dependent feature does not silently reuse a stale block.
    """
    eff = ctx.eff.astype(np.int64)
    strength = ctx.strength.astype(np.int64)
    out = np.zeros((NUM_TYPES, D_STATIC), dtype=np.float32)
    out[_TYPES, STATIC_SUIT + eff] = 1.0
    out[_TYPES, STATIC_STRENGTH + strength] = 1.0
    out[_TYPES, STATIC_POINTS + _POINT_SLOT] = 1.0
    is_level = (_TYPES < SMALL_JOKER) & (_TYPES % NUM_RANKS == ctx.level_rank)
    # A no-trump context has trump_suit 4, which no natural suit equals,
    # so the trump-suit-level flag is empty there without a special case.
    out[:, STATIC_FLAGS + FLAG_LEVEL] = is_level
    out[:, STATIC_FLAGS + FLAG_TRUMP_LEVEL] = is_level & (
        _TYPES // NUM_RANKS == ctx.trump_suit
    )
    out[:, STATIC_FLAGS + FLAG_SMALL_JOKER] = _TYPES == SMALL_JOKER
    out[:, STATIC_FLAGS + FLAG_BIG_JOKER] = _TYPES == BIG_JOKER
    return out


def dynamic_card_features(
    counts_in_hand: np.ndarray,
    unseen: np.ndarray,
    ctx: Ctx,
    num_decks: int = 2,
) -> np.ndarray:
    """float32[54, 2]: what a card is *to me, now*.

    Column 0 is how many copies I hold over ``num_decks`` -- 0, 0.5 or 1
    with two decks. Column 1 is 1.0 when no unseen card of the same
    effective suit has strictly greater strength, so the card would win
    its suit against every card still out there. A second copy of the
    card itself does not disqualify it: the comparison is strict.
    """
    out = np.empty((NUM_TYPES, D_DYNAMIC), dtype=np.float32)
    out[:, DYN_COUNT] = counts_in_hand.astype(np.float32) / float(num_decks)
    beaters = beater_matrix(ctx) @ unseen.astype(np.float32)
    out[:, DYN_TOP_UNSEEN] = beaters == 0.0
    return out


def card_features(
    counts_in_hand: np.ndarray,
    unseen: np.ndarray,
    ctx: Ctx,
    num_decks: int = 2,
    static: np.ndarray | None = None,
) -> np.ndarray:
    """float32[54, D_CARD]: the static block followed by the dynamic one.

    Pass ``static`` (from a :class:`Cache`) to skip rebuilding the half
    that is constant for the hand.
    """
    if static is None:
        static = static_card_features(ctx, num_decks)
    dynamic = dynamic_card_features(counts_in_hand, unseen, ctx, num_decks)
    return np.concatenate((static, dynamic), axis=1, dtype=np.float32)


class Cache:
    """Per-hand store for the static card features.

    Keyed by the trump context, so one cache can serve a whole run of
    hands and can never hand back a block built for another context.
    The arrays it returns are read-only; callers that want to write copy
    first.
    """

    __slots__ = ("_static",)

    def __init__(self) -> None:
        self._static: dict[tuple[int, int], np.ndarray] = {}

    def static_card_features(self, ctx: Ctx, num_decks: int = 2) -> np.ndarray:
        key = (ctx.cid, num_decks)
        got = self._static.get(key)
        if got is None:
            got = static_card_features(ctx, num_decks)
            got.flags.writeable = False
            self._static[key] = got
        return got

    def clear(self) -> None:
        self._static.clear()

    def __len__(self) -> int:
        return len(self._static)


_DEFAULT_CACHE = Cache()


# --- position readers -----------------------------------------------------

# Columns of the per-play meta table built by `plays_so_far`.
META_SEAT = 0  # relative seat that made the play
META_TRICK = 1  # index of the trick within the hand
META_POSITION = 2  # position within that trick, 0 for the lead
META_WON = 3  # the play took the trick; 0 while the trick is open
META_COLS = 4


def plays_so_far(state: PlayState, seat: int) -> tuple[np.ndarray, np.ndarray]:
    """Every play of the hand, oldest first, seen from `seat`.

    Returns the cards as float32[n, 54] and the meta table as
    int64[n, META_COLS]. One pass builds what the planes, the point
    count and the history tokens all need; the alternative is three
    walks over the same hundred plays.

    Plays in the trick still in progress carry the index of the trick
    that has not closed and a zero won-flag, because the trick is
    undecided. A token is therefore final from the moment its trick
    closes, which is what lets a caller keep a prefix and append.
    """
    plays: list[np.ndarray] = []
    meta: list[tuple[int, int, int, int]] = []

    for index, trick in enumerate(state.tricks):
        leader = trick.leader
        winner = trick.winner
        for i, play in enumerate(trick.plays):
            player = (leader + i) % NUM_SEATS
            plays.append(play)
            meta.append(
                ((player - seat) % NUM_SEATS, index, i, int(player == winner))
            )
    index = len(state.tricks)
    for i, play in enumerate(state.trick_plays):
        player = (state.trick_leader + i) % NUM_SEATS
        plays.append(play)
        meta.append(((player - seat) % NUM_SEATS, index, i, 0))

    if not plays:
        return (
            np.zeros((0, NUM_TYPES), dtype=np.float32),
            np.zeros((0, META_COLS), dtype=np.int64),
        )
    return np.array(plays, dtype=np.float32), np.array(meta, dtype=np.int64)


def _points_played(counts: np.ndarray) -> float:
    """Points on the table, from the stacked plays."""
    if counts.shape[0] == 0:
        return 0.0
    return float(counts.sum(axis=0) @ _POINTS_F)


def _planes_from_plays(
    state: PlayState, seat: int, counts: np.ndarray, meta: np.ndarray
) -> np.ndarray:
    planes = np.zeros((NUM_PLANES, NUM_TYPES), dtype=np.int8)
    planes[PLANE_HAND] = state.hands[seat]
    n = counts.shape[0]
    if n:
        # Scatter by matrix product: an [8, n] selector drops each play
        # into its plane in one pass. Counts never exceed num_decks, so
        # the float32 arithmetic is exact.
        open_trick = meta[:, META_TRICK] >= len(state.tricks)
        slot = meta[:, META_SEAT] + np.where(
            open_trick, PLANE_TRICK - PLANE_PLAYED, 0
        )
        select = np.zeros((PLANE_KITTY - PLANE_PLAYED, n), dtype=np.float32)
        select[slot, np.arange(n)] = 1.0
        planes[PLANE_PLAYED:PLANE_KITTY] = (select @ counts).astype(np.int8)
    if seat == state.landlord:
        planes[PLANE_KITTY] = state.kitty
    return planes


def card_planes(state: PlayState, seat: int) -> np.ndarray:
    """int8[10, 54]: the cards `seat` can see, by where they are.

    Plane 0 is my hand, planes 1..4 the completed-trick plays by
    relative seat, planes 5..8 the plays in the trick in progress, and
    plane 9 my burial -- all zeros unless I am the landlord.
    """
    counts, meta = plays_so_far(state, seat)
    return _planes_from_plays(state, seat, counts, meta)


def _suit_and_rank(
    counts: np.ndarray, ctx: Ctx
) -> tuple[int, tuple[int, int]] | None:
    """A play's effective suit and its ordering key, or None when the
    play spans more than one suit.

    The key is what ``play_rank`` returns, reached without decomposing:
    a component's ``top`` is the strength of its highest rung, every
    rung holds a card, so the maximum over the components is the
    maximum strength over the cards played.
    """
    present = np.nonzero(counts)[0]
    if present.size == 0:
        return None
    suits = ctx.eff[present]
    suit = int(suits[0])
    if np.any(suits != suit):
        return None
    return suit, (1 if suit == TRUMP else 0, int(ctx.strength[present].max()))


def current_winner(state: PlayState) -> int | None:
    """The seat winning the trick in progress, or None between tricks.

    Every response is measured against the lead rather than against the
    running winner, matching ``PlayState._close_trick`` so that the seat
    named here is the one who takes the trick if it ends now.
    """
    if not state.trick_plays:
        return None
    ctx = state.ctx
    lead = state.trick_plays[0]
    got = _suit_and_rank(lead, ctx)
    if got is None:
        return state.trick_leader
    suit, best = got

    winner = state.trick_leader
    lead_parts = None  # decomposed only once a response can outrank it
    for i in range(1, len(state.trick_plays)):
        play = state.trick_plays[i]
        got = _suit_and_rank(play, ctx)
        if got is None:
            continue
        _cand_suit, rank = got
        # A play that cannot outrank the incumbent cannot take the
        # trick, and that test is far cheaper than asking whether the
        # play covers the lead at all.
        if rank <= best:
            continue
        if lead_parts is None:
            lead_parts = decompose(lead, ctx, state.num_decks)
            if lead_parts is None:
                return state.trick_leader
        if not covers(play, lead_parts, suit, ctx, state.num_decks):
            continue
        best = rank
        winner = (state.trick_leader + i) % NUM_SEATS
    return winner


def _context_vector(
    state: PlayState, seat: int, points_played: float
) -> np.ndarray:
    ctx = state.ctx
    out = np.zeros(CONTEXT_DIM, dtype=np.float32)

    offset = (seat - state.landlord) % NUM_SEATS
    out[CTX_ROLE + int(_ROLE_OF_OFFSET[offset])] = 1.0
    out[CTX_ATTACKER] = float(team_of(seat) != state.defenders)
    out[CTX_LEVEL + ctx.level_rank] = 1.0
    out[CTX_TRUMP + ctx.trump_suit] = 1.0

    # Recomputed from the trick records rather than read off the state:
    # once the hand ends ``state.attacker_points`` has absorbed the
    # kitty bonus, which a seat that did not bury it may not see.
    attacker_points = sum(
        t.points for t in state.tricks if team_of(t.winner) != state.defenders
    )
    deck_points = POINTS_PER_DECK * state.num_decks
    out[CTX_ATTACKER_POINTS] = attacker_points / POINT_NORM
    out[CTX_UNPLAYED_POINTS] = (deck_points - points_played) / POINT_NORM
    out[CTX_TRICKS_PLAYED] = len(state.tricks) / MAX_TRICKS
    out[CTX_CARDS_LEFT] = int(state.hands[seat].sum()) / MAX_HAND

    if state.trick_plays:
        lead = state.trick_plays[0]
        out[CTX_LEAD_SIZE] = int(lead.sum()) / MAX_LEAD_SIZE
        suit = play_suit(lead, ctx)
        if suit is not None:
            out[CTX_LED_SUIT + suit] = 1.0
        winner = current_winner(state)
        if winner is not None:
            out[CTX_WINNING_SEAT + (winner - seat) % NUM_SEATS] = 1.0
    return out


def context_vector(state: PlayState, seat: int) -> np.ndarray:
    """float32[CONTEXT_DIM]: the scalar view of the position.

    The follow-only tail -- lead size, led suit, who is winning -- is
    zero on a lead.
    """
    counts, _meta = plays_so_far(state, seat)
    return _context_vector(state, seat, _points_played(counts))


def _history_tokens(
    counts: np.ndarray, meta: np.ndarray, static: np.ndarray
) -> tuple[np.ndarray, int]:
    """float32[MAX_HISTORY, HISTORY_DIM] and the number of rows filled.

    The cards of a play are summarised by the *static* feature rows,
    weighted by count. The dynamic half is deliberately left out: it is
    measured against an unseen set that moves every trick, so a token
    carrying it would mean something different each time it was read.
    """
    tokens = np.zeros((MAX_HISTORY, HISTORY_DIM), dtype=np.float32)
    kept = min(counts.shape[0], MAX_HISTORY)
    if kept == 0:
        return tokens, 0

    recent = counts[-kept:]
    info = meta[-kept:]
    rows = np.arange(kept)

    tokens[:kept, HIST_SUMMARY : HIST_SUMMARY + D_STATIC] = recent @ static
    tokens[rows, HIST_SEAT + info[:, META_SEAT]] = 1.0
    tokens[:kept, HIST_TRICK] = info[:, META_TRICK] / MAX_TRICKS
    tokens[:kept, HIST_POSITION] = info[:, META_POSITION] / NUM_SEATS
    tokens[:kept, HIST_IS_LEAD] = info[:, META_POSITION] == 0
    tokens[:kept, HIST_WON] = info[:, META_WON]
    tokens[:kept, HIST_POINTS] = (recent @ _POINTS_F) / PLAY_POINT_NORM
    return tokens, kept


# --- the observation ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Observation:
    """One seat's view of a position. Every array is freshly allocated.

    ``planes``  int8[NUM_PLANES, 54] -- see :func:`card_planes`.
    ``unseen``  int8[54] -- copies of each type this seat has not seen.
    ``cards``   float32[54, D_CARD] -- static features then dynamic.
    ``context`` float32[CONTEXT_DIM] -- see :func:`context_vector`.
    ``history`` float32[MAX_HISTORY, HISTORY_DIM] -- one token per play,
                oldest first, zero-padded past ``history_len``.
    """

    seat: int
    planes: np.ndarray
    unseen: np.ndarray
    cards: np.ndarray
    context: np.ndarray
    history: np.ndarray
    history_len: np.int32

    def arrays(self) -> dict[str, np.ndarray]:
        """The array fields by name, for comparison and batching."""
        return {
            "planes": self.planes,
            "unseen": self.unseen,
            "cards": self.cards,
            "context": self.context,
            "history": self.history,
        }


def encode(state: PlayState, seat: int, cache: Cache | None = None) -> Observation:
    """Everything `seat` may legally know about the position.

    ``cache`` carries the static card features across the decisions of
    one hand. A module-level cache is used when none is given, which is
    always safe: the cache is keyed by the trump context.
    """
    if cache is None:
        cache = _DEFAULT_CACHE
    static = cache.static_card_features(state.ctx, state.num_decks)

    counts, meta = plays_so_far(state, seat)
    planes = _planes_from_plays(state, seat, counts, meta)
    unseen = state.unseen_from(seat)
    cards = card_features(
        planes[PLANE_HAND], unseen, state.ctx, state.num_decks, static
    )
    context = _context_vector(state, seat, _points_played(counts))
    history, length = _history_tokens(counts, meta, static)

    return Observation(
        seat=seat,
        planes=planes,
        unseen=unseen,
        cards=cards,
        context=context,
        history=history,
        history_len=np.int32(length),
    )
