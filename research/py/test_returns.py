"""Checks on the Monte-Carlo target, which is the thing DMC actually
optimises. A sign error here trains a confidently wrong agent, and the
symptom (mediocre play) looks the same as "needs more data".

    PYTHONPATH=py python py/test_returns.py
"""

from __future__ import annotations

import numpy as np

from tractor_rl.dmc import DMCConfig, hand_returns
from tractor_rl.dmc2 import (
    DMC2Config, decision_signs, hand_returns as hand_returns_v2)
from tractor_rl import layout as L
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


def main_v2() -> None:
    """The same ground, on dmc2 -- which is what the live trainers call.

    `dmc.hand_returns` above is the legacy v1 path, reached only by
    train_dmc.py. train2 and train_pop call `dmc2.hand_returns`, which
    hands back the point return and the level outcome separately, both
    attacker-POV. Everything below checks that pair and stops there.

    The per-decision sign that turns those attacker-POV scalars into a
    defending seat's target is checked in `main_signs` below. Both trainers
    used to inline it as `1.0 if attacker else -1.0` in their collection
    loops, where nothing could reach it; it is now `dmc2.decision_signs`,
    called by both, so flipping it to a bare 1.0 -- which puts half the
    seats on an exactly inverted objective while the loss curve goes on
    looking normal -- fails here instead of shipping.
    """
    cfg = DMC2Config()
    g = cfg.gamma
    scale = cfg.point_scale

    # Attackers take 20 points on the middle trick of three and nothing
    # else, so the defenders hold them under 40 and take the bracket: +2
    # levels to them is -2 attacker-POV.
    ep = make_episode([0, 20, 0], [False, True, False], 20, 2, True)
    r, level = hand_returns_v2(ep, cfg)
    capture = 20.0 / scale
    check("v2 capture credits the trick it happened on", float(r[1]), capture)
    check("v2 capture discounts back one trick", float(r[0]), g * capture)
    check("v2 nothing is credited after the capture", float(r[2]), 0.0)
    check("v2 level is attacker-POV", level, -2.0)

    # The kitty multiplier is whatever the attackers finished with beyond
    # the tricks, and it belongs on the final trick alone: 60 taken in
    # tricks against 100 in total leaves 40 to land at the end. Stripping
    # the discounted last trick off the first leaves the first trick's own
    # reward, which must carry none of that bonus.
    ep = make_episode([30, 30], [True, True], 100, 0, False, kitty_points=10)
    r, level = hand_returns_v2(ep, cfg)
    check("v2 kitty bonus lands on the last trick", float(r[1]), (30.0 + 40.0) / scale)
    check("v2 kitty bonus lands there only once",
          float(r[0]) - g * float(r[1]), 30.0 / scale)
    check("v2 a hand that moved no levels reports none", level, 0.0)

    # v2 keeps the level out of the point return entirely -- it is a second
    # scalar feeding a second head, which is the whole difference from the
    # v1 target checked above. So a hand with no points at all must produce
    # returns that are identically zero however far the levels moved, and
    # the outcome must survive exactly once, in that scalar.
    ep = make_episode([0, 0, 0], [False, False, False], 0, 3, True)
    r, level = hand_returns_v2(ep, cfg)
    assert float(np.abs(r).max()) == 0.0, f"the level term leaked into the point return: {r}"
    check("v2 level appears once, as its own scalar", level, -3.0)

    # A hand with no tricks still has an outcome; the trainer skips it on
    # the empty return rather than on an exception.
    empty = make_episode([], [], 0, 3, True)
    r, level = hand_returns_v2(empty, cfg)
    assert r.shape == (0,), r.shape
    check("v2 empty hand still reports its level", level, -3.0)

    print("\nall dmc2 return checks passed")


def main_signs() -> None:
    """The attacker-POV -> seat-POV flip, on states built here by hand.

    The expected values are written out directly rather than derived from
    anything in dmc2, so this fails if the convention is inverted, dropped,
    or read off the wrong header byte.
    """
    states = np.zeros((4, L.STATE_BYTES), dtype=np.uint8)
    states[0, L.H_I_AM_ATTACKER] = 1
    states[1, L.H_I_AM_ATTACKER] = 0
    states[2, L.H_I_AM_ATTACKER] = 1
    states[3, L.H_I_AM_ATTACKER] = 0

    sg = decision_signs(states)
    assert sg.dtype == np.float32, sg.dtype
    assert sg.shape == (4,), sg.shape
    check("attacking seat keeps the attacker-POV sign", float(sg[0]), 1.0)
    check("defending seat inverts it", float(sg[1]), -1.0)
    check("and again, so the sign is not a row index", float(sg[2]), 1.0)
    check("and again", float(sg[3]), -1.0)

    # The flip has to reach both targets: a defender's reward for the
    # attackers taking 20 points is the negation of the attackers' own.
    ep = make_episode([0, 20, 0], [False, True, False], 20, 2, True)
    r, level = hand_returns_v2(ep, DMC2Config())
    check("attackers are credited for their own capture",
          float(r[1]) * float(sg[0]), 0.2)
    check("defenders are debited for that same capture",
          float(r[1]) * float(sg[1]), -0.2)
    check("the level outcome flips with it", float(level) * float(sg[1]), 2.0)

    # Only the byte the convention is defined on may decide the sign; a
    # seat's role and its position in the trick must not move it.
    other = np.zeros((1, L.STATE_BYTES), dtype=np.uint8)
    other[0, L.H_ROLE] = 2
    other[0, L.H_TRICK_POS] = 3
    check("no attacker byte means defending", float(decision_signs(other)[0]), -1.0)

    print("\nall sign-convention checks passed")


if __name__ == "__main__":
    main()
    print()
    main_v2()
    print()
    main_signs()
