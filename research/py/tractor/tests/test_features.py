"""Tests for the observation encoder.

The one that matters is ``test_no_hidden_information``. At every
decision of every hand the cards the acting seat cannot see are dealt
out again among the places they could be -- the other three hands keep
their sizes, the kitty keeps its own -- and the observation is
re-encoded. Every array must come back bit-identical. Anything the
encoder reads that it has no right to read moves under that re-deal.
"""

from __future__ import annotations

import numpy as np

from tractor.gen.episode import generate
from tractor.nn import features as F
from tractor.rules.cards import (
    BIG_JOKER,
    NUM_TYPES,
    RANK_NAMES,
    SMALL_JOKER,
    Ctx,
    card_id,
    counts_from_ids,
    total_points,
)
from tractor.sim.state import NUM_SEATS, TrickRecord

DECKS = 2
LEAK_DECISIONS = 2000


def _rollout(rng: np.random.Generator, state, cap: int = 32) -> np.ndarray:
    """Play one uniformly random legal move. Returns what was played."""
    moves = state.legal_moves(cap=cap)
    return state.step(moves.moves[int(rng.integers(len(moves.moves)))])


def _redeal_hidden(rng: np.random.Generator, state, seat: int):
    """A clone with everything `seat` cannot see dealt out afresh.

    The other three hands keep their sizes and the kitty keeps its own,
    but which card sits where is redrawn. The acting seat's own hand is
    untouched, and so is the kitty when the acting seat is the landlord
    who buried it.
    """
    shuffled = state.clone()
    others = [s for s in range(NUM_SEATS) if s != seat]
    sizes = [int(shuffled.hands[s].sum()) for s in others]
    pools = [
        np.repeat(np.arange(NUM_TYPES), shuffled.hands[s].astype(np.int64))
        for s in others
    ]
    hide_kitty = seat != shuffled.landlord
    if hide_kitty:
        pools.append(
            np.repeat(np.arange(NUM_TYPES), shuffled.kitty.astype(np.int64))
        )

    cards = np.concatenate(pools)
    rng.shuffle(cards)
    at = 0
    for other, size in zip(others, sizes):
        shuffled.hands[other] = counts_from_ids(cards[at : at + size])
        at += size
    if hide_kitty:
        shuffled.kitty = counts_from_ids(cards[at:])
    return shuffled


def _check_observation(obs, state, seat: int) -> None:
    """Shapes, dtypes, and the invariants that tie the blocks together."""
    assert obs.seat == seat
    assert obs.planes.shape == (F.NUM_PLANES, NUM_TYPES)
    assert obs.planes.dtype == np.int8
    assert obs.unseen.shape == (NUM_TYPES,)
    assert obs.unseen.dtype == np.int8
    assert obs.cards.shape == (NUM_TYPES, F.D_CARD)
    assert obs.cards.dtype == np.float32
    assert obs.context.shape == (F.CONTEXT_DIM,)
    assert obs.context.dtype == np.float32
    assert obs.history.shape == (F.MAX_HISTORY, F.HISTORY_DIM)
    assert obs.history.dtype == np.float32
    assert obs.history_len.dtype == np.int32

    np.testing.assert_array_equal(obs.planes[F.PLANE_HAND], state.hands[seat])
    if seat != state.landlord:
        assert not obs.planes[F.PLANE_KITTY].any(), "the kitty plane leaked"
    else:
        np.testing.assert_array_equal(obs.planes[F.PLANE_KITTY], state.kitty)

    # The unseen vector is the deck less everything on the planes.
    seen = obs.planes.sum(axis=0, dtype=np.int16)
    expect = np.maximum(state.num_decks - seen, 0).astype(np.int8)
    np.testing.assert_array_equal(obs.unseen, expect)

    # One-hots are one-hot.
    assert obs.context[F.CTX_ROLE : F.CTX_ROLE + NUM_SEATS].sum() == 1.0
    assert obs.context[F.CTX_LEVEL : F.CTX_LEVEL + 13].sum() == 1.0
    assert obs.context[F.CTX_TRUMP : F.CTX_TRUMP + 5].sum() == 1.0
    suits = obs.cards[:, F.STATIC_SUIT : F.STATIC_SUIT + F.NUM_EFF_SUITS]
    assert suits.sum() == NUM_TYPES
    strengths = obs.cards[:, F.STATIC_STRENGTH : F.STATIC_STRENGTH + F.NUM_STRENGTH]
    assert strengths.sum() == NUM_TYPES


def test_layout_matches_the_documented_widths() -> None:
    assert F.D_STATIC == 5 + 16 + 3 + 4 == 28
    assert F.D_DYNAMIC == 2
    assert F.D_CARD == 30
    assert F.NUM_PLANES == 10
    assert F.CONTEXT_DIM == 4 + 1 + 13 + 5 + 4 + 1 + 5 + 4 == 37
    assert F.HISTORY_DIM == F.D_STATIC + 4 + 1 + 1 + 2 + 1
    assert F.MAX_HISTORY == 128


def test_no_hidden_information() -> None:
    rng = np.random.default_rng(20260920)
    cache = F.Cache()
    decisions = 0
    hands = 0
    moved = 0

    while decisions < LEAK_DECISIONS:
        state, _info = generate(rng, DECKS)
        hands += 1
        while not state.done:
            seat = state.to_act
            obs = F.encode(state, seat, cache)
            shuffled = _redeal_hidden(rng, state, seat)
            other = F.encode(shuffled, seat, cache)

            np.testing.assert_array_equal(shuffled.hands[seat], state.hands[seat])
            if seat == state.landlord:
                np.testing.assert_array_equal(shuffled.kitty, state.kitty)
            if not np.array_equal(shuffled.hands, state.hands) or not np.array_equal(
                shuffled.kitty, state.kitty
            ):
                moved += 1

            theirs = other.arrays()
            for name, mine in obs.arrays().items():
                assert mine.dtype == theirs[name].dtype, f"{name} changed dtype"
                if not np.array_equal(mine, theirs[name]):
                    where = np.transpose(np.nonzero(mine != theirs[name]))
                    raise AssertionError(
                        f"{name} depends on hidden cards at "
                        f"{[tuple(int(i) for i in w) for w in where[:4]]} "
                        f"(hand {hands}, decision {decisions}, seat {seat})"
                    )
            assert obs.history_len == other.history_len

            decisions += 1
            _rollout(rng, state)

    assert moved > decisions // 2, "the re-deal barely moved cards; the test is vacuous"
    print(
        f"    {decisions} decisions over {hands} hands, "
        f"{moved} re-deals moved cards, every array identical"
    )


def test_shapes_and_dtypes_for_leads_and_follows() -> None:
    rng = np.random.default_rng(101)
    leads = 0
    follows = 0

    for _ in range(4):
        state, _info = generate(rng, DECKS)
        while not state.done:
            seat = state.to_act
            obs = F.encode(state, seat)
            _check_observation(obs, state, seat)

            if state.is_lead:
                leads += 1
                assert np.all(
                    obs.context[F.FOLLOW_FIELDS] == 0.0
                ), "follow-only fields are set on a lead"
            else:
                follows += 1
                lead_size = obs.context[F.CTX_LEAD_SIZE] * F.MAX_LEAD_SIZE
                assert lead_size == int(state.trick_plays[0].sum())
                led = obs.context[F.CTX_LED_SUIT : F.CTX_LED_SUIT + F.NUM_EFF_SUITS]
                assert led.sum() == 1.0
                winning = obs.context[
                    F.CTX_WINNING_SEAT : F.CTX_WINNING_SEAT + NUM_SEATS
                ]
                assert winning.sum() == 1.0
            _rollout(rng, state)

    assert leads > 0 and follows > 0
    print(f"    {leads} leads and {follows} follows checked")


def test_dynamic_top_of_suit_flag() -> None:
    level = RANK_NAMES.index("2")
    ctx = Ctx(level, 0)  # spades are trump, the twos are the level rank
    hearts = 1
    ace = card_id(hearts, RANK_NAMES.index("A"))
    king = card_id(hearts, RANK_NAMES.index("K"))
    queen = card_id(hearts, RANK_NAMES.index("Q"))

    hand = np.zeros(NUM_TYPES, dtype=np.int8)
    hand[ace] = 2
    unseen = np.zeros(NUM_TYPES, dtype=np.int8)
    unseen[king] = 2
    unseen[queen] = 1

    dyn = F.dynamic_card_features(hand, unseen, ctx, DECKS)
    top = dyn[:, F.DYN_TOP_UNSEEN]
    assert top[ace] == 1.0
    # The kings are top of hearts only because both aces sit in my hand.
    assert top[king] == 1.0
    assert top[queen] == 0.0

    # Counts normalise to {0, 0.5, 1}.
    assert dyn[ace, F.DYN_COUNT] == 1.0
    assert dyn[king, F.DYN_COUNT] == 0.0
    hand[queen] = 1
    assert F.dynamic_card_features(hand, unseen, ctx, DECKS)[queen, F.DYN_COUNT] == 0.5

    # The comparison is strict: another copy of the card itself out
    # there does not stop it being the top of its suit.
    same = np.zeros(NUM_TYPES, dtype=np.int8)
    same[ace] = 1
    assert F.dynamic_card_features(hand, same, ctx, DECKS)[ace, F.DYN_TOP_UNSEEN] == 1.0

    # Trump is its own effective suit, so an unseen joker says nothing
    # about hearts -- but it does outrank every other trump.
    with_joker = unseen.copy()
    with_joker[BIG_JOKER] = 1
    dyn = F.dynamic_card_features(hand, with_joker, ctx, DECKS)
    top = dyn[:, F.DYN_TOP_UNSEEN]
    assert top[king] == 1.0
    assert top[BIG_JOKER] == 1.0
    assert top[SMALL_JOKER] == 0.0
    assert top[card_id(0, level)] == 0.0  # the trump-suit two, under the joker

    # The three off-suit level cards share one strength rung, so neither
    # outranks the other and both stay top with nothing higher unseen.
    level_only = np.zeros(NUM_TYPES, dtype=np.int8)
    level_only[card_id(1, level)] = 2  # the heart twos, trump in this context
    top = F.dynamic_card_features(hand, level_only, ctx, DECKS)[:, F.DYN_TOP_UNSEEN]
    assert top[card_id(2, level)] == 1.0  # diamond two, the same rung
    assert top[card_id(0, level)] == 1.0  # spade two, the rung above
    assert top[card_id(0, RANK_NAMES.index("A"))] == 0.0  # trump ace, below both


def test_static_features_are_constant_within_a_hand() -> None:
    rng = np.random.default_rng(77)
    state, _info = generate(rng, DECKS)
    cache = F.Cache()
    seat = state.to_act

    first = F.encode(state, seat, cache)
    for _ in range(6):
        _rollout(rng, state)
    later = F.encode(state, seat, cache)

    np.testing.assert_array_equal(
        first.cards[:, : F.D_STATIC], later.cards[:, : F.D_STATIC]
    )
    # The same block, not merely an equal one, and not writable.
    block = cache.static_card_features(state.ctx, DECKS)
    assert block is cache.static_card_features(state.ctx, DECKS)
    assert not block.flags.writeable
    assert first.cards.flags.writeable, "the observation must own its arrays"
    assert len(cache) == 1

    # The dynamic half does move: this seat has played and has seen more.
    assert not np.array_equal(
        first.cards[:, F.D_STATIC :], later.cards[:, F.D_STATIC :]
    )

    # A different trump is a different hand and a different block.
    other = Ctx(state.ctx.level_rank, (state.ctx.trump_suit + 1) % 5)
    assert not np.array_equal(block, cache.static_card_features(other, DECKS))
    assert len(cache) == 2
    np.testing.assert_array_equal(
        cache.static_card_features(other, DECKS),
        F.static_card_features(other, DECKS),
    )


def test_history_length_tracks_plays() -> None:
    rng = np.random.default_rng(31)
    state, _info = generate(rng, DECKS)
    static = F.static_card_features(state.ctx, DECKS)
    plays = 0
    last = None

    while not state.done:
        seat = state.to_act
        obs = F.encode(state, seat)
        length = int(obs.history_len)
        assert length == min(plays, F.MAX_HISTORY), f"{length} tokens for {plays} plays"
        assert np.all(obs.history[length:] == 0.0), "the padding is not zero"

        if last is not None:
            player, cards, trick_index, position = last
            token = obs.history[length - 1]
            assert np.any(token != 0.0), "the newest token is empty"
            np.testing.assert_array_equal(
                token[: F.D_STATIC], cards.astype(np.float32) @ static
            )
            rel = (player - seat) % NUM_SEATS
            assert token[F.HIST_SEAT + rel] == 1.0
            assert token[F.HIST_SEAT : F.HIST_SEAT + NUM_SEATS].sum() == 1.0
            assert token[F.HIST_TRICK] == np.float32(trick_index / F.MAX_TRICKS)
            assert token[F.HIST_POSITION] == np.float32(position / NUM_SEATS)
            assert token[F.HIST_IS_LEAD] == float(position == 0)
            assert token[F.HIST_POINTS] == np.float32(
                total_points(cards) / F.PLAY_POINT_NORM
            )

        trick_index = len(state.tricks)
        position = len(state.trick_plays)
        played = _rollout(rng, state)
        last = (seat, played, trick_index, position)
        plays += 1

    print(f"    {plays} plays, the token count tracked every step")


def test_history_is_capped_at_the_token_budget() -> None:
    rng = np.random.default_rng(5)
    state, _info = generate(rng, DECKS)

    # A synthetic history: more plays than the budget, so the window has
    # to slide. A real two-deck hand stops at a hundred plays, well
    # inside the cap, so nothing but a fabricated state exercises this.
    tricks = F.MAX_HISTORY // NUM_SEATS + 5
    state.tricks = [
        TrickRecord(
            leader=index % NUM_SEATS,
            plays=[counts_from_ids([index % NUM_TYPES]) for _ in range(NUM_SEATS)],
            winner=(index + 1) % NUM_SEATS,
            points=0,
        )
        for index in range(tricks)
    ]
    total = tricks * NUM_SEATS
    assert total > F.MAX_HISTORY

    obs = F.encode(state, 0)
    assert obs.history.shape == (F.MAX_HISTORY, F.HISTORY_DIM)
    assert int(obs.history_len) == F.MAX_HISTORY

    # The window keeps the most recent plays, oldest of them first.
    dropped = total - F.MAX_HISTORY
    assert obs.history[0, F.HIST_TRICK] == np.float32(
        (dropped // NUM_SEATS) / F.MAX_TRICKS
    )
    assert obs.history[0, F.HIST_POSITION] == np.float32(
        (dropped % NUM_SEATS) / NUM_SEATS
    )
    assert obs.history[-1, F.HIST_TRICK] == np.float32((tricks - 1) / F.MAX_TRICKS)
    assert obs.history[-1, F.HIST_POSITION] == np.float32(
        (NUM_SEATS - 1) / NUM_SEATS
    )
    print(f"    {total} plays kept {int(obs.history_len)} tokens")


def test_current_winner_matches_the_engine() -> None:
    rng = np.random.default_rng(404)
    checked = 0

    for _ in range(20):
        state, _info = generate(rng, DECKS)
        while not state.done:
            if len(state.trick_plays) == 1:
                assert F.current_winner(state) == state.trick_leader
            if state.is_lead:
                assert F.current_winner(state) is None

            closing = len(state.trick_plays) == NUM_SEATS - 1
            probe = state.clone() if closing else None
            played = _rollout(rng, state)
            if closing:
                # The engine empties the trick as it closes it, so the
                # complete trick only exists on the clone.
                probe.trick_plays.append(played)
                assert F.current_winner(probe) == state.tricks[-1].winner
                checked += 1

    print(f"    {checked} closed tricks agreed with the engine")


def main() -> int:
    import time

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        started = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - reporting harness
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}"[:1500])
        else:
            print(f"ok    {fn.__name__}  ({time.time() - started:.1f}s)")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
