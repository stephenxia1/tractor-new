"""End-to-end play with a uniform random policy.

Checks the invariants that hold at every step of every hand regardless
of how anyone plays: cards are conserved, points are conserved, a legal
move always exists, and the hand terminates.
"""

from __future__ import annotations

import collections

import numpy as np

from tractor.gen.episode import generate
from tractor.rules.cards import NUM_TYPES, full_deck, total_points
from tractor.sim.state import hand_result, team_of

DECKS = 2
TOTAL_POINTS = 200


def _account(state) -> np.ndarray:
    """Every card in the game: hands, kitty, and everything played."""
    seen = state.hands.sum(axis=0).astype(np.int16) + state.kitty.astype(np.int16)
    for trick in state.tricks:
        for play in trick.plays:
            seen += play
    for play in state.trick_plays:
        seen += play
    return seen


def play_hands(count: int, seed: int, cap: int = 64) -> dict:
    rng = np.random.default_rng(seed)
    deck = full_deck(DECKS).astype(np.int16)
    stats: dict = {
        "hands": 0,
        "decisions": 0,
        "leads": 0,
        "lead_moves": 0,
        "follows": 0,
        "follow_moves": 0,
        "tricks": 0,
        "truncated": 0,
        "unenumerated": 0,
        "forced_leads": 0,
        "attacker_points": [],
        "kitty_points": [],
        "level_deltas": collections.Counter(),
        "trump_suits": collections.Counter(),
        "landlords": collections.Counter(),
        "no_declaration": 0,
    }

    for _ in range(count):
        state, info = generate(rng, DECKS)
        stats["trump_suits"][info["trump_suit"]] += 1
        stats["landlords"][info["landlord"]] += 1
        if info["declaration_options"] == 0:
            stats["no_declaration"] += 1
        stats["kitty_points"].append(total_points(state.kitty))

        guard = 0
        while not state.done:
            guard += 1
            assert guard < 400, "hand did not terminate"
            np.testing.assert_array_equal(
                _account(state), deck, "cards were not conserved"
            )

            moves = state.legal_moves(cap=cap)
            assert len(moves) > 0, "no legal move offered"
            stats["truncated"] += moves.truncated
            stats["unenumerated"] += int(moves.unenumerated)
            if state.is_lead:
                stats["leads"] += 1
                stats["lead_moves"] += len(moves)
            else:
                stats["follows"] += 1
                stats["follow_moves"] += len(moves)
            stats["decisions"] += 1

            choice = moves.moves[int(rng.integers(len(moves.moves)))]
            before = state.hands[state.to_act].copy()
            played = state.step(choice)
            assert not np.any(played > before)
            if state.last_forced_lead is not None:
                stats["forced_leads"] += 1

        np.testing.assert_array_equal(
            _account(state), deck, "cards were not conserved at the end"
        )
        assert int(state.hands.sum()) == 0

        trick_points = sum(t.points for t in state.tricks)
        assert trick_points + total_points(state.kitty) == TOTAL_POINTS, (
            f"points not conserved: tricks {trick_points} "
            f"+ kitty {total_points(state.kitty)}"
        )

        stats["hands"] += 1
        stats["tricks"] += len(state.tricks)
        stats["attacker_points"].append(state.attacker_points)
        stats["level_deltas"][hand_result(state.attacker_points)] += 1

    return stats


def test_random_selfplay_conserves_everything() -> None:
    stats = play_hands(300, seed=20260920)
    hands = stats["hands"]
    assert hands == 300
    ap = np.array(stats["attacker_points"])
    print(
        f"    hands {hands}  decisions {stats['decisions']} "
        f"({stats['decisions'] / hands:.1f}/hand)"
    )
    print(
        f"    tricks/hand {stats['tricks'] / hands:.1f}  "
        f"lead moves {stats['lead_moves'] / max(stats['leads'], 1):.1f}  "
        f"follow moves {stats['follow_moves'] / max(stats['follows'], 1):.1f}"
    )
    print(
        f"    attacker points mean {ap.mean():.1f}  "
        f"kitty points mean {np.mean(stats['kitty_points']):.1f}  "
        f"forced leads {stats['forced_leads']}"
    )
    print(
        f"    truncated {stats['truncated']}  unenumerated {stats['unenumerated']}  "
        f"no-declaration hands {stats['no_declaration']}"
    )
    print(f"    trump suits {dict(sorted(stats['trump_suits'].items()))}")


def test_defenders_and_attackers_are_opposite_teams() -> None:
    rng = np.random.default_rng(5)
    for _ in range(50):
        state, _info = generate(rng, DECKS)
        assert team_of(state.landlord) == state.defenders
        assert team_of((state.landlord + 1) % 4) != state.defenders


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
