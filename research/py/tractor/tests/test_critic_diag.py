"""The critic's holdout measurement, and the two ways it could lie.

The measurement exists because the learner's own ``explained_variance``
is in-sample, so these tests are about the parts that make the holdout
number mean what it says: the split must not leak a hand across both
sides, and the interval must be clustered by hand. Both failures are
silent -- they produce a plausible number, not an error -- so each one
is pinned against a case where the right answer is known in advance.
"""

from __future__ import annotations

import numpy as np

from tractor.train.critic_diag import (
    _clustered_ev_interval,
    _slope,
    split_hands,
)
from tractor.train.ppo import explained_variance


def _episodes(n_hands: int, rows_each: int) -> np.ndarray:
    """``episode`` as a rollout writes it: runs of one hand's rows."""
    return np.repeat(np.arange(n_hands, dtype=np.int64), rows_each)


# --- 1. the split ---------------------------------------------------------


def test_split_hands_never_puts_one_hand_on_both_sides() -> None:
    """The failure that would turn a holdout score into an in-sample one.

    The critic reads a privileged view of all four hands, so a hand with
    rows in the fit is fully known to it; scoring that hand's other rows
    as "held out" measures memorisation and calls it generalisation.
    """
    rng = np.random.default_rng(0)
    for trial in range(25):
        n_hands = int(rng.integers(2, 40))
        # Ragged on purpose: real hands differ in length, and a split
        # that happens to work on equal-sized groups is not tested.
        lengths = rng.integers(1, 30, size=n_hands)
        episode = np.repeat(np.arange(n_hands, dtype=np.int64), lengths)
        holdout = float(rng.uniform(0.05, 0.95))

        fit, held = split_hands(episode, holdout, np.random.default_rng(trial))
        shared = np.intersect1d(episode[fit], episode[held])
        assert shared.size == 0, f"hands {shared} appear on both sides"


def test_split_hands_covers_every_row_exactly_once() -> None:
    """No row is dropped and none is counted twice."""
    episode = _episodes(12, 7)
    fit, held = split_hands(episode, 0.25, np.random.default_rng(1))
    both = np.sort(np.concatenate([fit, held]))
    assert np.array_equal(both, np.arange(episode.size))


def test_split_hands_keeps_both_sides_populated() -> None:
    """A rounding that empties a side is no measurement, not a small one."""
    episode = _episodes(3, 4)
    for holdout in (0.01, 0.05, 0.5, 0.95, 0.99):
        fit, held = split_hands(episode, holdout, np.random.default_rng(2))
        assert fit.size > 0 and held.size > 0, holdout
        assert np.unique(episode[held]).size >= 1
        assert np.unique(episode[fit]).size >= 1


def test_split_hands_refuses_what_it_cannot_split() -> None:
    for bad_episode in (np.zeros(10, dtype=np.int64),):
        try:
            split_hands(bad_episode, 0.25, np.random.default_rng(3))
        except ValueError as exc:
            assert "2 hands" in str(exc), exc
        else:
            raise AssertionError("a single-hand buffer was split")

    for bad_holdout in (0.0, 1.0, -0.1, 1.5):
        try:
            split_hands(_episodes(4, 3), bad_holdout, np.random.default_rng(4))
        except ValueError as exc:
            assert "holdout" in str(exc), exc
        else:
            raise AssertionError(f"holdout {bad_holdout} was accepted")


# --- 2. the interval ------------------------------------------------------


def _row_interval(
    predicted: np.ndarray, actual: np.ndarray, rng: np.random.Generator,
    draws: int = 2000,
) -> tuple[float, float]:
    """The naive bootstrap the clustered one exists to replace."""
    stats = np.empty(draws)
    for d in range(draws):
        idx = rng.integers(0, predicted.size, size=predicted.size)
        stats[d] = explained_variance(predicted[idx], actual[idx])
    lo, hi = np.percentile(stats[np.isfinite(stats)], [2.5, 97.5])
    return float(lo), float(hi)


def test_clustered_interval_is_much_wider_than_resampling_rows() -> None:
    """Rows inside a hand are dependent, so row resampling lies.

    Built at the limit the real thing approaches: with ``gamma = 1`` and
    an end-loaded reward, every row of a hand carries the same return,
    so 40 hands of 20 rows is 40 independent observations and not 800.
    A bootstrap over rows believes it has 800 and reports an interval
    roughly ``sqrt(20)`` times too narrow. This is the error that once
    made a 24-hand measurement read ``+0.42`` when 96 hands said
    ``-0.013``.
    """
    rng = np.random.default_rng(7)
    n_hands, rows_each = 40, 20
    episode = _episodes(n_hands, rows_each)
    truth = rng.standard_normal(n_hands)
    actual = np.repeat(truth, rows_each)
    # A noisy per-hand prediction: the noise is per hand too, because a
    # critic reading the same hand twice gives nearly the same answer.
    predicted = np.repeat(truth + 0.8 * rng.standard_normal(n_hands), rows_each)

    c_lo, c_hi = _clustered_ev_interval(
        predicted, actual, episode, np.random.default_rng(8)
    )
    r_lo, r_hi = _row_interval(predicted, actual, np.random.default_rng(9))

    clustered_width = c_hi - c_lo
    row_width = r_hi - r_lo
    assert clustered_width > 2.0 * row_width, (
        f"clustered width {clustered_width:.4f} should dwarf the row "
        f"bootstrap's {row_width:.4f}; the clustering is not working"
    )
    point = explained_variance(predicted, actual)
    assert c_lo <= point <= c_hi


def test_clustered_interval_brackets_a_known_answer() -> None:
    """A critic that is exactly right sits at 1.0, tightly."""
    rng = np.random.default_rng(11)
    episode = _episodes(30, 10)
    actual = np.repeat(rng.standard_normal(30), 10)
    lo, hi = _clustered_ev_interval(
        actual.copy(), actual, episode, np.random.default_rng(12)
    )
    assert 0.99 <= lo <= 1.0 and 0.99 <= hi <= 1.0, (lo, hi)


def test_clustered_interval_declines_to_guess_from_one_hand() -> None:
    episode = np.zeros(20, dtype=np.int64)
    values = np.arange(20, dtype=np.float64)
    lo, hi = _clustered_ev_interval(
        values, values, episode, np.random.default_rng(13)
    )
    assert np.isnan(lo) and np.isnan(hi)


# --- 3. calibration -------------------------------------------------------


def test_slope_reports_over_confidence() -> None:
    """Predictions twice as spread as the truth give a slope of one half.

    Which is the reading the real critic produced (0.49), and the reason
    it is reported alongside the explained variance: a critic can be
    over-confident and still explain some variance, and the two numbers
    fail in different directions.
    """
    rng = np.random.default_rng(15)
    truth = rng.standard_normal(5000)
    assert abs(_slope(2.0 * truth, truth) - 0.5) < 0.02
    assert abs(_slope(truth, truth) - 1.0) < 0.02
    # Under-confident: predictions squashed towards the mean.
    assert abs(_slope(0.5 * truth, truth) - 2.0) < 0.05


def test_slope_is_undefined_for_a_constant_prediction() -> None:
    truth = np.arange(50, dtype=np.float64)
    assert np.isnan(_slope(np.zeros(50), truth))
