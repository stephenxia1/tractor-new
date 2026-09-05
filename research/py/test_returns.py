"""Checks on the Monte-Carlo target, which is the thing DMC actually
optimises. A sign error here trains a confidently wrong agent, and the
symptom (mediocre play) looks the same as "needs more data".

    PYTHONPATH=py python py/test_returns.py
"""

from __future__ import annotations

import numpy as np

from tractor_rl.dmc import DMCConfig, hand_returns
from tractor_rl.features import Episode


def make_episode(
    trick_points: list[int],
    attacker_won: list[bool],
    attacker_points: int,
    level_delta: int,
    defenders_won: bool,
    kitty_points: int = 0,
) -> Episode:
    return Episode(
        game_id=0,
        hand_id=1,
        landlord_seat=0,
        num_tricks=len(trick_points),
        attacker_points=attacker_points,
        level_delta=level_delta,
        defenders_won=defenders_won,
        kitty_points=kitty_points,
        trick_points=np.asarray(trick_points, dtype=np.int32),
        trick_attacker_won=np.asarray(attacker_won, dtype=bool),
    )


def check(name: str, got: float, want: float, tol: float = 1e-4) -> None:
    ok = abs(got - want) <= tol
    print(f"{'ok ' if ok else 'FAIL'} {name}: got {got:+.4f} want {want:+.4f}")
    if not ok:
        raise AssertionError(name)


def main() -> None:
    cfg = DMCConfig()
    g = cfg.gamma

    # Defenders hold the attackers to 20 points: attackers took only trick 1.
    # Level bracket 1-39 is defenders +2, so the attacker-POV terminal is -2.
    ep = make_episode([0, 20, 0], [False, True, False], 20, 2, True)
    r = hand_returns(ep, cfg)
    want2 = g * -2.0
    want1 = 0.2 + g * want2
    want0 = g * want1
    check("defended hand, last trick", float(r[2]), want2)
    check("defended hand, middle trick", float(r[1]), want1)
    check("defended hand, first trick", float(r[0]), want0)
    assert (r < 0).all(), "a hand the defenders won must be negative attacker-POV"

    # Attackers take everything: 200 points, +3 levels.
    ep = make_episode([100, 100], [True, True], 200, 3, False)
    r = hand_returns(ep, cfg)
    want1 = 1.0 + g * 3.0
    want0 = 1.0 + g * want1
    check("swept hand, last trick", float(r[1]), want1)
    check("swept hand, first trick", float(r[0]), want0)

    # The kitty multiplier lands on the last trick: tricks award 60, the
    # final total is 100, so 40 points of kitty bonus belong at the end.
    ep = make_episode([30, 30], [True, True], 100, 0, False, kitty_points=10)
    r = hand_returns(ep, cfg)
    want1 = (0.3 + 0.4) + g * 0.0  # 40 bonus / 100 scale, terminal delta 0
    check("kitty bonus on last trick", float(r[1]), want1)

    # Sign convention: the same hand is worth the opposite to each side.
    ep = make_episode([0, 25], [False, True], 25, 2, True)
    r = hand_returns(ep, cfg)
    attacker_target = float(r[0])
    defender_target = -attacker_target
    check("attacker target is negative here", np.sign(attacker_target), -1.0)
    check("defender target is positive here", np.sign(defender_target), 1.0)

    # Capturing points the defenders take is worth nothing to the attackers
    # and nothing to the defenders either -- only denial shows up.
    won_by_defenders = make_episode([50, 0], [False, False], 0, 3, True)
    r_def = hand_returns(won_by_defenders, cfg)
    check("defender captures do not reward attackers", float(r_def[0]), g * g * -3.0)

    # A hand with no tricks yields no targets rather than an exception.
    empty = make_episode([], [], 0, 3, True)
    assert hand_returns(empty, cfg).shape == (0,)
    print("ok  empty hand handled")

    print("\nall return checks passed")


if __name__ == "__main__":
    main()
