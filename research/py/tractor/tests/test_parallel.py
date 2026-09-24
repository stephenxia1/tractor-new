"""The parallel rollout, against the invariants the serial one guarantees.

Moving the simulator into worker processes cannot change what a row
means. The failures it *can* introduce are all silent: the parent
writing an action into the wrong row, two workers sharing a row, a hand
split across two ranges, a reward credited in a worker's window but
read back at the wrong offset. Every one of those produces a buffer
that loads, trains, and is wrong.

So these tests check the things a corrupted row range would break --
that every stored action indexes a legal slot, that a hand's rows stay
together and in order, that the credit rule still sums to what the
rollout's docstring says it sums to -- and compare the aggregate
statistics against the serial rollout, which is the reference
implementation.
"""

from __future__ import annotations

import numpy as np
import torch

from tractor.nn.model import ActorCritic, ModelConfig
from tractor.sim.state import NUM_SEATS, team_of
from tractor.train.buffer import RolloutBuffer
from tractor.train.parallel import collect_parallel
from tractor.train.rollout import collect

# Real ``max_history`` (the buffer accepts no other) with everything
# else cut down, so a test plays real hands through a small network.
SMALL = ModelConfig(
    d_model=32,
    n_layers_history=1,
    n_layers_query=1,
    n_heads=2,
    d_ff=64,
    critic_d=16,
    critic_layers=1,
    critic_heads=2,
    critic_d_ff=32,
    belief_d=16,
    scalar_d=8,
    head_d=16,
    max_candidates=16,
)
DEVICE = torch.device("cpu")


def _net(seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(SMALL)


def _parallel(hands: int, workers: int, seed: int = 1, level_weight: float = 1.0):
    return collect_parallel(
        _net(),
        np.random.default_rng(seed),
        None,
        n_hands=hands,
        device=DEVICE,
        cfg=SMALL,
        level_weight=level_weight,
        pool=12,
        workers=workers,
    )


def _serial(hands: int, seed: int = 1, level_weight: float = 1.0):
    buf = RolloutBuffer(hands * 100, SMALL)
    stats = collect(
        _net(),
        np.random.default_rng(seed),
        buf,
        n_hands=hands,
        device=DEVICE,
        cfg=SMALL,
        level_weight=level_weight,
        pool=12,
    )
    return buf, stats


# --- 1. the write-back ----------------------------------------------------


def test_every_stored_action_indexes_a_legal_slot() -> None:
    """The sharpest test of the parent writing into the right row.

    A worker writes the observation and the parent writes the action
    into the row it was told. Off-by-one anywhere in that path lands an
    action from one decision beside another decision's candidate list,
    and the odds it still points at a filled slot fall with every row.
    """
    buf, stats = _parallel(hands=8, workers=3)
    n = len(buf)
    assert n == int(stats["rows"]) > 0
    action = np.asarray(buf.action)[:n]
    mask = np.asarray(buf.candidate_mask)[:n]
    legal = mask[np.arange(n), action]
    assert legal.all(), f"{int((~legal).sum())} of {n} rows index a padded slot"


def test_the_parent_writes_a_log_probability_into_every_row() -> None:
    """Rows are created with placeholder scalars; none may survive.

    ``logp = 0`` is probability 1, which is an importance ratio of
    exactly one forever -- so a row the parent never reached would
    train as a permanent no-op rather than raise. It is also the
    *correct* value for a forced decision, where the seat has one legal
    move and no choice was made, so the two cases are separated by the
    candidate mask rather than lumped together. Over 704 rows this is
    not a technicality: 14.6% of decisions are forced.
    """
    buf, stats = _parallel(hands=8, workers=3)
    n = len(buf)
    logp = np.asarray(buf.logp)[:n]
    legal_count = np.asarray(buf.candidate_mask)[:n].sum(axis=1)

    forced = legal_count == 1
    assert forced.any(), "no forced decisions; this test proves nothing here"
    assert (logp[forced] == 0.0).all(), "a forced decision is not certain"
    assert (logp[~forced] < 0.0).all(), (
        f"{int((logp[~forced] >= 0).sum())} rows with a real choice kept the "
        "placeholder log-probability"
    )
    # And the mask agrees with what the workers counted independently.
    assert abs(forced.mean() - stats["forced_fraction"]) < 1e-9


# --- 2. hands survive the partition ---------------------------------------


def test_a_hand_stays_whole_and_in_order() -> None:
    """GAE orders a seat's trajectory by row index, so order is meaning.

    A hand lives entirely inside one worker's range and compaction moves
    whole ranges, so this must hold after compaction as well as before.
    """
    for workers in (1, 2, 5):
        buf, _stats = _parallel(hands=10, workers=workers)
        n = len(buf)
        episode = np.asarray(buf.episode)[:n]
        seat = np.asarray(buf.seat)[:n]
        done = np.asarray(buf.done)[:n]

        for hand in np.unique(episode):
            rows = np.nonzero(episode == hand)[0]
            assert np.array_equal(rows, np.sort(rows)), (workers, hand)
            assert np.unique(seat[rows]).size == NUM_SEATS, (workers, hand)
            # Exactly one terminal row per seat, and it is that seat's last.
            for s in range(NUM_SEATS):
                mine = rows[seat[rows] == s]
                assert mine.size > 0
                assert done[mine].sum() == 1, (workers, hand, s)
                assert done[mine[-1]], (workers, hand, s)


def test_every_hand_gets_its_own_episode_id() -> None:
    """Workers number their hands independently; the bases must not clash.

    Two hands sharing an id become one trajectory in ``compute_gae``,
    and the join is invisible -- the earlier hand's last row simply
    bootstraps off the later hand's first value.
    """
    for workers in (2, 3, 7):
        buf, stats = _parallel(hands=14, workers=workers)
        n = len(buf)
        episode = np.asarray(buf.episode)[:n]
        assert np.unique(episode).size == int(stats["hands"]) == 14, workers


# --- 3. the reward rule is unchanged --------------------------------------


def test_trick_credit_still_sums_to_the_points_a_team_took() -> None:
    """``rollout``'s documented invariant, checked through the new path.

    A team's trick credits over a hand sum to exactly the fraction of
    the deck's points that team captured, so the two teams' sums are
    equal and opposite and each lies in [-1, 1].
    """
    buf, _stats = _parallel(hands=10, workers=3)
    n = len(buf)
    episode = np.asarray(buf.episode)[:n]
    seat = np.asarray(buf.seat)[:n]
    points = np.asarray(buf.reward_points)[:n]

    for hand in np.unique(episode):
        rows = np.nonzero(episode == hand)[0]
        teams = np.array([team_of(int(s)) for s in seat[rows]])
        first = float(points[rows[teams == 0]].sum())
        second = float(points[rows[teams == 1]].sum())
        assert abs(first + second) < 1e-4, (hand, first, second)
        assert -1.0001 <= first <= 1.0001, (hand, first)


def test_reward_is_its_two_terms_combined() -> None:
    """``reward == reward_points + level_weight * reward_level`` exactly."""
    for level_weight in (0.0, 0.5, 1.0):
        buf, _stats = _parallel(hands=6, workers=3, level_weight=level_weight)
        n = len(buf)
        want = (
            np.asarray(buf.reward_points)[:n]
            + np.float32(level_weight) * np.asarray(buf.reward_level)[:n]
        )
        assert np.allclose(np.asarray(buf.reward)[:n], want, atol=1e-6), (
            level_weight
        )


def test_the_terminal_reward_lands_only_on_terminal_rows() -> None:
    buf, _stats = _parallel(hands=8, workers=3)
    n = len(buf)
    done = np.asarray(buf.done)[:n]
    level = np.asarray(buf.reward_level)[:n]
    assert not level[~done].any(), "a non-terminal row carries a level reward"
    assert (np.abs(level[done]) <= 1.0 + 1e-6).all()


# --- 4. against the serial rollout ----------------------------------------


def test_parallel_and_serial_agree_on_what_the_game_is() -> None:
    """Statistics that describe the game, not the parallelism, must match.

    The two draw different deals -- separate generators, hands split
    across workers -- so rows cannot be compared one to one. What can be
    compared is anything that is a property of Tractor rather than of
    the scheduler: how often a seat has exactly one legal move, and how
    long a hand is.
    """
    hands = 24
    par_buf, par = _parallel(hands=hands, workers=4)
    ser_buf, ser = _serial(hands=hands)

    assert par["hands"] == ser["hands"] == hands
    # ~88 decisions a hand, and the two should not differ by much more
    # than the deal-to-deal spread.
    par_len = par["decisions"] / hands
    ser_len = ser["decisions"] / hands
    assert abs(par_len - ser_len) < 6.0, (par_len, ser_len)
    assert abs(par["forced_fraction"] - ser["forced_fraction"]) < 0.06, (
        par["forced_fraction"], ser["forced_fraction"]
    )
    assert len(par_buf) == par["decisions"]
    assert len(ser_buf) == ser["decisions"]


def test_one_worker_is_still_the_whole_round() -> None:
    """The degenerate split must not lose or duplicate a hand."""
    buf, stats = _parallel(hands=5, workers=1)
    assert stats["hands"] == 5
    assert stats["workers"] == 1.0
    assert np.unique(np.asarray(buf.episode)[: len(buf)]).size == 5


def test_more_workers_than_hands_is_clamped() -> None:
    """Asking for eight workers to play three hands must not hang."""
    buf, stats = _parallel(hands=3, workers=8)
    assert stats["hands"] == 3
    assert stats["workers"] <= 3.0
    assert np.unique(np.asarray(buf.episode)[: len(buf)]).size == 3


def test_it_fills_a_buffer_the_caller_brought() -> None:
    """A caller holding its own buffer should not have to change."""
    mine = RolloutBuffer(4000, SMALL)
    out, stats = collect_parallel(
        _net(), np.random.default_rng(3), mine, n_hands=6, device=DEVICE,
        cfg=SMALL, pool=8, workers=2,
    )
    assert out is mine
    assert len(mine) == stats["decisions"] > 0


def test_workers_play_the_starts_they_are_given() -> None:
    """Hand ``i`` of a round begins from ``starts[i]``, across workers."""
    from tractor.train.parallel import RolloutPool
    from tractor.train.starts import make_starts

    starts, _rows, _stats = make_starts(np.random.default_rng(9), 4, None, None)
    pool = RolloutPool(cfg=SMALL, workers=2, hands_per_round=4)
    try:
        buf, stats = pool.collect(
            _net(), np.random.default_rng(1), None, n_hands=4, device=DEVICE,
            pool=4, starts=starts,
        )
    finally:
        pool.close()
    assert stats["hands"] == 4
    n = len(buf)
    for i, st in enumerate(starts):
        rows = np.nonzero(buf.episode[:n] == i)[0]
        first = int(rows[0])
        assert int(buf.seat[first]) == st.landlord  # the landlord leads
        view = buf.privileged[first].astype(np.int8)
        for rel in range(NUM_SEATS):
            assert np.array_equal(view[rel], st.hands[(st.landlord + rel) % NUM_SEATS])
        assert np.array_equal(view[NUM_SEATS], st.kitty)
