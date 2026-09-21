"""Hand-built positions with outcomes worked out on paper.

These are deliberately short rather than full twenty-five-trick hands.
A transcribed full hand is mostly filler between the two or three
moments that pin a rule, and the filler is what the conservation tests
in test_selfplay.py already cover. Each position below isolates one
mechanism and states its expected outcome exactly.

The positions are not complete deals -- seats hold a few cards each and
the kitty is stocked to a chosen point total -- so deck conservation
does not apply here.
"""

from __future__ import annotations

import numpy as np

from tractor.rules.cards import (
    BIG_JOKER,
    NUM_TYPES,
    RANK_NAMES,
    SMALL_JOKER,
    SUIT_NAMES,
    Ctx,
    card_id,
    total_points,
)
from tractor.rules.combos import TRACTOR, classify
from tractor.sim.state import PlayState, hand_result

DECKS = 2
SUIT_LETTERS = {name[0].upper(): i for i, name in enumerate(SUIT_NAMES)}


def card(text: str) -> int:
    """Parse 'AS', '10H', 'SJ', 'BJ'."""
    if text == "SJ":
        return SMALL_JOKER
    if text == "BJ":
        return BIG_JOKER
    rank, suit = text[:-1], text[-1]
    return card_id(SUIT_LETTERS[suit], RANK_NAMES.index(rank))


def counts(*names: str) -> np.ndarray:
    out = np.zeros(NUM_TYPES, dtype=np.int8)
    for n in names:
        out[card(n)] += 1
    return out


def build(
    level: str,
    trump: str,
    landlord: int,
    hands: list[list[str]],
    kitty: list[str],
) -> PlayState:
    ctx = Ctx(RANK_NAMES.index(level), SUIT_LETTERS[trump] if trump != "NT" else 4)
    arr = np.zeros((4, NUM_TYPES), dtype=np.int8)
    for seat, cards in enumerate(hands):
        arr[seat] = counts(*cards)
    return PlayState.new(ctx, arr, counts(*kitty), landlord, DECKS)


# Eight cards with no point value, to stock a kitty to a chosen total.
BLANKS = ["3C", "4C", "6C", "7C", "8C", "JC", "QC", "AC"]


def _play(state: PlayState, seat_cards: list[tuple[int, list[str]]]) -> None:
    for seat, names in seat_cards:
        assert state.to_act == seat, f"expected seat {state.to_act}, got {seat}"
        state.step(counts(*names))


def test_kitty_multiplier_single() -> None:
    """Attackers take the last trick with one card: the kitty doubles.

    Level 9, hearts trump, landlord seat 0, so seats 0 and 2 defend and
    seats 1 and 3 attack. Seat 1 is void in spades and ruffs. The trick
    carries the five of spades, worth five. The kitty holds one king,
    worth ten, and the winning play is a single, so the bonus is
    10 x 2 x 1 = 20.
    """
    state = build(
        "9", "H", 0,
        [["3S"], ["3H"], ["4S"], ["5S"]],
        ["KS"] + BLANKS[:7],
    )
    assert total_points(state.kitty) == 10
    _play(state, [(0, ["3S"]), (1, ["3H"]), (2, ["4S"]), (3, ["5S"])])
    assert state.done
    assert state.tricks[0].winner == 1
    assert state.tricks[0].points == 5
    assert state.attacker_points == 25
    assert hand_result(25) == (2, 0)


def test_kitty_multiplier_pair() -> None:
    """The same, won with a pair: the multiplier is four, not two."""
    state = build(
        "9", "H", 0,
        [["3S", "3S"], ["4H", "4H"], ["5S", "6S"], ["7S", "8S"]],
        ["KS"] + BLANKS[:7],
    )
    _play(
        state,
        [
            (0, ["3S", "3S"]),
            (1, ["4H", "4H"]),
            (2, ["5S", "6S"]),
            (3, ["7S", "8S"]),
        ],
    )
    assert state.done
    assert state.tricks[0].winner == 1
    assert state.tricks[0].points == 5
    # 5 from the trick, plus 10 x 2 x 2 from the kitty.
    assert state.attacker_points == 45
    assert hand_result(45) == (1, 0)


def test_defenders_take_the_last_trick() -> None:
    """No bonus at all when the defending team wins the last trick."""
    state = build(
        "9", "H", 0,
        [["3S"], ["4S"], ["KS"], ["5S"]],
        ["KS"] + BLANKS[:7],
    )
    _play(state, [(0, ["3S"]), (1, ["4S"]), (2, ["KS"]), (3, ["5S"])])
    assert state.done
    assert state.tricks[0].winner == 2
    assert state.attacker_points == 0
    assert hand_result(0) == (3, 0)


def test_trump_pair_beats_a_higher_plain_pair() -> None:
    state = build(
        "9", "H", 0,
        [["AS", "AS"], ["3H", "3H"], ["4S", "5S"], ["6S", "7S"]],
        BLANKS,
    )
    _play(
        state,
        [
            (0, ["AS", "AS"]),
            (1, ["3H", "3H"]),
            (2, ["4S", "5S"]),
            (3, ["6S", "7S"]),
        ],
    )
    assert state.tricks[0].winner == 1


def test_two_trump_singles_do_not_beat_a_pair() -> None:
    """The incumbent's structure is fixed; singles never merge."""
    state = build(
        "9", "H", 0,
        [["AS", "AS"], ["3H", "4H"], ["4S", "5S"], ["6S", "7S"]],
        BLANKS,
    )
    _play(
        state,
        [
            (0, ["AS", "AS"]),
            (1, ["3H", "4H"]),
            (2, ["4S", "5S"]),
            (3, ["6S", "7S"]),
        ],
    )
    assert state.tricks[0].winner == 0


def test_blocked_throw_forces_the_smallest_component() -> None:
    """Seat 0 throws the ace and king of spades. The second ace is in
    another hand, so the king is beatable and the throw is rejected: the
    king is led alone and the ace goes back."""
    state = build(
        "9", "H", 0,
        [["AS", "KS"], ["AS", "3H"], ["4S", "5S"], ["6S", "7S"]],
        BLANKS,
    )
    played = state.step(counts("AS", "KS"))
    np.testing.assert_array_equal(played, counts("KS"))
    assert state.last_forced_lead is not None
    np.testing.assert_array_equal(state.last_forced_lead, counts("AS"))
    # The ace stays in hand and only one card left it.
    assert state.hands[0][card("AS")] == 1
    assert state.hands[0][card("KS")] == 0
    assert int(state.trick_plays[0].sum()) == 1


def test_safe_throw_stands() -> None:
    """Both aces of spades in one hand: nothing can beat either
    component, so the throw is led whole."""
    state = build(
        "9", "H", 0,
        [["AS", "AS", "KS", "KS"], ["3H", "4H", "5H", "6H"],
         ["4S", "5S", "6S", "7S"], ["8S", "9S", "JS", "QS"]],
        BLANKS,
    )
    played = state.step(counts("AS", "AS", "KS", "KS"))
    np.testing.assert_array_equal(played, counts("AS", "AS", "KS", "KS"))
    assert state.last_forced_lead is None


def test_tier_bridging_tractor() -> None:
    """rules.md 7.2: a run may cross the trump tiers. With hearts trump
    and level five, the king and ace of hearts sit at rungs 10 and 11
    and an off-suit five at rung 12, so three pairs chain."""
    ctx = Ctx(RANK_NAMES.index("5"), SUIT_LETTERS["H"])
    play = counts("KH", "KH", "AH", "AH", "5C", "5C")
    got = classify(play, ctx, DECKS)
    assert got is not None
    assert got.kind == TRACTOR
    assert got.length == 3
    assert got.top == 12


def test_score_brackets() -> None:
    """rules.md 8.3, at every boundary."""
    expected = {
        0: (3, 0),
        5: (2, 0),
        35: (2, 0),
        39: (2, 0),
        40: (1, 0),
        75: (1, 0),
        79: (1, 0),
        80: (0, 0),
        119: (0, 0),
        120: (0, 1),
        159: (0, 1),
        160: (0, 2),
        199: (0, 2),
        200: (0, 3),
        # The table caps: a very large kitty bonus is still three levels.
        640: (0, 3),
    }
    for points, want in expected.items():
        assert hand_result(points) == want, f"P={points}"


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
