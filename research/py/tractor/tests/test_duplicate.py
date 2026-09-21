"""The duplicate-deal harness, checked against itself.

Four of these five tests are about the harness rather than about any
policy. A comparison harness that is wrong does not announce itself: it
returns a plausible number with a plausible error bar, and the number is
wrong by a factor or by a sign. So the properties that must hold by
construction are asserted as identities -- a self-comparison centred on
zero, an exact antisymmetry, bit-identical reruns -- and only the last
test asks the harness to measure something.

    python -m tractor.tests.test_duplicate
"""

from __future__ import annotations

import numpy as np

from tractor.eval.duplicate import (
    compare,
    lowest_index_policy,
    salted,
    self_check,
    uniform_random_policy,
)

# 400 deals is 800 hands, about a minute. The seed is the date the file
# was written, fixed so the suite is deterministic.
SEED = 20260921
DEALS = 400

# Two independently seeded copies of uniform random: equal in strength by
# construction, independent in their choices.
RANDOM_A = salted(uniform_random_policy, 0xA5A5)
RANDOM_B = salted(uniform_random_policy, 0x5A5A)

_cache: dict[str, object] = {}


def random_vs_weak():
    """Uniform random against the lowest-index policy, computed once.

    Two tests need this run and it is the expensive one, so it is shared
    rather than paid for twice.
    """
    if "rand_low" not in _cache:
        _cache["rand_low"] = compare(
            uniform_random_policy, lowest_index_policy, DEALS, SEED
        )
    return _cache["rand_low"]


def test_self_comparison_is_zero() -> None:
    """A policy against itself must score zero, and be seen to.

    This is the harness's own validity check. The true difference between
    a policy and an independently seeded copy of itself is exactly zero,
    so if the measured mean misses zero by more than its own error bar
    the fault is in the scoring, the seating or the clustering -- not in
    the policy. The test is a two-sigma check on a fixed seed, which
    under the null would trip for about one seed in twenty; that is the
    price of the check being a real one.
    """
    r = self_check(uniform_random_policy, DEALS, SEED)
    z = abs(r.mean) / r.stderr
    print(
        f"    levels {r.mean:+.4f}  se {r.stderr:.4f}  "
        f"95% [{r.ci_low:+.4f}, {r.ci_high:+.4f}]  ({z:.2f} se from zero)"
    )
    print(
        f"    points {r.point_mean:+.2f}  se {r.point_stderr:.2f}  "
        f"95% [{r.point_ci_low:+.2f}, {r.point_ci_high:+.2f}]  "
        f"n={r.num_deals} deals, {2 * r.num_deals} hands"
    )
    assert r.num_deals == DEALS
    assert abs(r.mean) <= 2.0 * r.stderr, f"{r.mean:+.4f} is {z:.2f} se from zero"
    assert abs(r.point_mean) <= 2.0 * r.point_stderr, (
        f"points {r.point_mean:+.2f} is "
        f"{abs(r.point_mean) / r.point_stderr:.2f} se from zero"
    )
    assert r.ci_low < 0.0 < r.ci_high

    # The degenerate form of the same statement, for completeness. Handing
    # `compare` one factory twice makes the two replays of each deal
    # byte-identical, so every deal scores exactly zero. That is a real
    # property of the seeding scheme and worth pinning down, but it is
    # zero by construction and so cannot stand in for the check above.
    same = compare(uniform_random_policy, uniform_random_policy, 20, SEED)
    assert np.all(same.scores == 0.0)
    assert np.all(same.point_scores == 0.0)
    print(f"    one factory twice: all {same.num_deals} deals exactly 0.0")


def test_antisymmetry_is_exact() -> None:
    """Swapping the arguments must negate the answer exactly.

    Not approximately, and not on average. Because each seat pair draws
    from a stream fixed by the deal and the pair, `compare(y, x)` plays
    the same four hands as `compare(x, y)` with the two seatings in the
    other order, so every per-deal score flips sign bit for bit. An
    equality that held only in distribution would leave room for a
    seat-to-team mix-up that cancels in the mean.
    """
    for name, px, py in (
        ("random vs weak", uniform_random_policy, lowest_index_policy),
        ("random vs random", RANDOM_A, RANDOM_B),
    ):
        fwd = compare(px, py, 60, SEED)
        rev = compare(py, px, 60, SEED)

        assert fwd.mean == -rev.mean, f"{name}: {fwd.mean} vs {-rev.mean}"
        assert fwd.point_mean == -rev.point_mean
        assert fwd.stderr == rev.stderr
        assert fwd.ci_low == -rev.ci_high and fwd.ci_high == -rev.ci_low
        np.testing.assert_array_equal(fwd.scores, -rev.scores)
        np.testing.assert_array_equal(fwd.point_scores, -rev.point_scores)
        # The same deals, and the same two hands with the seatings
        # exchanged: column 0 of one run is column 1 of the other.
        np.testing.assert_array_equal(fwd.digests, rev.digests)
        np.testing.assert_array_equal(fwd.hand_levels, -rev.hand_levels[:, ::-1])
        np.testing.assert_array_equal(
            fwd.attacker_points, rev.attacker_points[:, ::-1]
        )
        print(f"    {name}: {fwd.mean:+.4f} = -({rev.mean:+.4f})  exact over 60 deals")


def test_determinism() -> None:
    """The same seed gives the same numbers, down to the last hand."""
    a = compare(RANDOM_A, RANDOM_B, 40, SEED)
    b = compare(RANDOM_A, RANDOM_B, 40, SEED)
    assert a.mean == b.mean and a.stderr == b.stderr
    np.testing.assert_array_equal(a.scores, b.scores)
    np.testing.assert_array_equal(a.point_scores, b.point_scores)
    np.testing.assert_array_equal(a.hand_levels, b.hand_levels)
    np.testing.assert_array_equal(a.attacker_points, b.attacker_points)
    np.testing.assert_array_equal(a.digests, b.digests)
    assert a.decisions == b.decisions

    # A different seed must not merely reshuffle the same deals.
    other = compare(RANDOM_A, RANDOM_B, 40, SEED + 1)
    shared = set(a.digests.tolist()) & set(other.digests.tolist())
    assert not shared, f"{len(shared)} deals repeated across seeds"
    assert len(set(a.digests.tolist())) == a.num_deals, "deals repeated within a run"
    print(
        f"    seed {SEED} twice: identical over {a.num_deals} deals, "
        f"{a.decisions} decisions; {len(set(a.digests.tolist()))} distinct deals, "
        f"0 shared with seed {SEED + 1}"
    )


def test_pairing_cancels_deal_luck() -> None:
    """The point of the design: the paired score is the quieter statistic.

    The raw per-hand signed result carries the whole spread of the deal.
    Averaging the two replays of one deal cancels most of that, and by
    much more than averaging two unrelated hands would: two independent
    hands would divide the standard deviation by root two, and this does
    better because the two halves of a pair are negatively correlated
    through the cards they share.
    """
    r = random_vs_weak()
    hand_sd = r.hand_stdev
    deal_sd = r.deal_stdev
    independent = hand_sd / np.sqrt(2.0)
    # Hands needed for a given precision, paired against unpaired.
    saved = hand_sd**2 / (2.0 * deal_sd**2)
    pairs = np.corrcoef(r.hand_levels[:, 0], r.hand_levels[:, 1])[0, 1]

    print(
        f"    per-hand signed result  sd {hand_sd:.3f}  "
        f"(n={r.hand_levels.size} hands)"
    )
    print(
        f"    per-deal paired score   sd {deal_sd:.3f}  "
        f"({deal_sd / hand_sd:.3f}x)  n={r.num_deals} deals"
    )
    print(
        f"    averaging two unrelated hands would give {independent:.3f}; "
        f"correlation within a pair {pairs:+.3f}"
    )
    print(
        f"    equal precision costs {saved:.2f}x fewer hands paired "
        f"than unpaired"
    )
    assert deal_sd < 0.7 * hand_sd, "pairing did not materially reduce the spread"
    assert deal_sd < 0.9 * independent, "pairing beat nothing but plain averaging"
    assert pairs < 0.0, "the two halves of a pair should be negatively correlated"


def test_weak_policy_is_resolved() -> None:
    """A real difference must come back with an interval that excludes zero.

    The lowest-index policy leads its lowest card and never spends a high
    one, so uniform random should beat it and beat it clearly. A harness
    that cannot separate these two has nothing to say about two policies
    that differ by less.
    """
    r = random_vs_weak()
    z = r.mean / r.stderr
    print(
        f"    random - lowest-index   {r.mean:+.4f} levels/deal  "
        f"se {r.stderr:.4f}  95% [{r.ci_low:+.4f}, {r.ci_high:+.4f}]  ({z:.1f} se)"
    )
    print(
        f"                            {r.point_mean:+.2f} points/deal  "
        f"se {r.point_stderr:.2f}  "
        f"95% [{r.point_ci_low:+.2f}, {r.point_ci_high:+.2f}]"
    )
    print(
        f"    attacker points: random {r.attacker_points.mean():.1f} mean, "
        f"{r.num_deals} deals / {2 * r.num_deals} hands / {r.decisions} decisions"
    )
    assert r.significant
    assert r.ci_low > 0.0, "uniform random failed to beat the weakened policy"
    assert r.point_ci_low > 0.0, "the point metric disagrees with the level metric"

    # And the reverse comparison must lose by the same margin.
    rev = compare(lowest_index_policy, uniform_random_policy, 60, SEED)
    assert rev.mean < 0.0


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
