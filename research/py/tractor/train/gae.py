"""Credit assignment over per-seat trajectories, not over buffer rows.

The thing this file exists to get right is grouping. A rollout buffer
fills in the order decisions happen at the table, so four consecutive
rows are four *different* seats, and with several hands in flight they
are several different hands as well. A seat's own trajectory -- the
sequence of positions it acted in and the returns it earned -- is a
subsequence of the buffer, roughly every fourth row within one hand.
Running the GAE recursion down the buffer in row order therefore mixes
four players and N hands into one imaginary episode, and the resulting
advantages are not wrong by a constant: they are a seat's own TD error
plus three strangers'. The bug is invisible in every shape check and in
every finite-metrics test, which is why the recursion here is written
against an explicit ``(episode, seat)`` grouping and why there is a slow
reference implementation to check the fast one against.

Ordering within a group is by buffer row index. A rollout writes a
hand's decisions in the order they are made, so for a fixed
``(episode, seat)`` the row index is the seat's clock. Nothing else in
the contract records a within-hand position, and inventing one would
only be a second, less trustworthy copy of that fact.

``gamma`` defaults to 1.0. A hand is a finite episode of at most a few
dozen decisions and the reward that matters arrives at the end of it;
discounting would tell the network that the last trick is worth less
than the first, which is the opposite of true.

Two boundary rules, both load-bearing.

*A ``done`` terminates.* ``done=True`` marks a seat's last decision in a
hand, so at that row the bootstrap value is zero and the accumulated
advantage resets. Carrying either across the boundary would credit a
hand with the next hand's outcome. The reset is applied at *every*
``done``, not only at the end of a group, which is what makes a reused
episode id survivable: two hands sharing an id become one group, and
the join is clean *provided the earlier hand's last row for that seat
carries ``done``*. That is the only thing standing between them. Where
it does not hold -- a hand truncated by a full buffer, a caller that
restarts its episode counter mid-round -- the two hands really are one
trajectory here, and the earlier one's last step bootstraps off the
later one's first value. Unique ids are the invariant; the ``done``
reset is a second line, not a substitute.

*A group that does not end in ``done`` is truncated, not terminal.* That
happens when the buffer fills mid-hand. The value of the state after the
last recorded action is the one thing we do not have and cannot get
without another forward pass, so the tail bootstraps off its own critic
value: ``next_value`` is ``value`` itself, giving
``delta = reward - (1 - gamma) * value``. At the default ``gamma = 1``
that is exactly ``advantage = reward`` and ``ret = reward + value``, the
zero-TD-error reading; below 1 the tail carries the small correction
term instead, and the shorthand stops being literally true. Either way
it propagates honestly backwards. Treating truncation as termination
instead would tell the critic that a hand stopped mid-way scored
nothing, which for an end-loaded reward is a large, systematic lie.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple, Protocol

import numpy as np

__all__ = [
    "GAEBuffer",
    "Trajectories",
    "compute_gae",
    "compute_gae_reference",
    "group_trajectories",
    "num_rows",
]


class GAEBuffer(Protocol):
    """The slice of the rollout-buffer contract this module touches.

    Structural, so it matches the real ``RolloutBuffer`` and any stand-in
    a test builds without either having to import the other.
    """

    reward: np.ndarray  # f32[C]
    value: np.ndarray  # f32[C]
    done: np.ndarray  # bool[C]
    seat: np.ndarray  # i64[C]
    episode: np.ndarray  # i64[C]
    advantage: np.ndarray  # f32[C], written here
    ret: np.ndarray  # f32[C], written here


def num_rows(buffer: object) -> int:
    """How many leading rows of the buffer hold real transitions.

    The contract is a fixed capacity ``C`` with every field shaped
    ``[C, ...]``, but a buffer is not obliged to be full when the learner
    takes it -- a rollout that ends on a hand boundary stops short. A
    buffer says how far it filled with ``__len__``, or with an integer
    ``size``; with neither, the capacity is taken at face value.
    """
    capacity = int(np.asarray(buffer.reward).shape[0])  # type: ignore[attr-defined]
    n: int | None = None
    if hasattr(buffer, "__len__"):
        n = int(len(buffer))  # type: ignore[arg-type]
    else:
        size = getattr(buffer, "size", None)
        if isinstance(size, (int, np.integer)):
            n = int(size)
    if n is None:
        return capacity
    if not 0 <= n <= capacity:
        raise ValueError(f"buffer reports {n} rows of a {capacity}-row capacity")
    return n


class Trajectories(NamedTuple):
    """Buffer rows reshaped into a dense ``[groups, longest]`` block.

    ``index[g, t]`` is the buffer row of the ``t``-th decision of group
    ``g``, or ``-1`` past that group's end. Groups are
    ``(episode, seat)`` pairs and ``t`` follows buffer row order.
    """

    index: np.ndarray  # i64[G, L]
    length: np.ndarray  # i64[G]

    @property
    def valid(self) -> np.ndarray:
        """bool[G, L]: True where ``index`` names a real row."""
        return self.index >= 0


def group_trajectories(episode: np.ndarray, seat: np.ndarray) -> Trajectories:
    """Partition rows into ``(episode, seat)`` trajectories, in row order.

    ``lexsort`` with ``arange`` as the least significant key makes the
    within-group order explicitly the buffer order rather than relying on
    a sort's stability guarantee holding for whatever key encoding a
    future edit picks.
    """
    episode = np.asarray(episode, dtype=np.int64).reshape(-1)
    seat = np.asarray(seat, dtype=np.int64).reshape(-1)
    n = episode.shape[0]
    if n != seat.shape[0]:
        raise ValueError("episode and seat must be the same length")
    if n == 0:
        return Trajectories(
            index=np.zeros((0, 0), dtype=np.int64),
            length=np.zeros(0, dtype=np.int64),
        )

    order = np.lexsort((np.arange(n), seat, episode))
    ep_s, seat_s = episode[order], seat[order]
    starts_group = np.empty(n, dtype=bool)
    starts_group[0] = True
    np.logical_or(ep_s[1:] != ep_s[:-1], seat_s[1:] != seat_s[:-1], out=starts_group[1:])

    group = np.cumsum(starts_group) - 1  # 0..G-1, dense, in sorted order
    n_groups = int(group[-1]) + 1
    counts = np.bincount(group, minlength=n_groups)
    longest = int(counts.max())

    offsets = np.zeros(n_groups, dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    position = np.arange(n, dtype=np.int64) - offsets[group]

    index = np.full((n_groups, longest), -1, dtype=np.int64)
    index[group, position] = order
    return Trajectories(index=index, length=counts.astype(np.int64))


def compute_gae(buffer: GAEBuffer, gamma: float = 1.0, lam: float = 0.95) -> None:
    """Fill ``buffer.advantage`` and ``buffer.ret`` in place.

    Vectorised across trajectories: the recursion still runs backwards
    one step at a time, but each step handles every group at once, so the
    Python loop is as long as the *longest* hand (a couple of dozen
    steps) rather than as long as the buffer. The arithmetic is done in
    float64 and rounded once on the way into the float32 fields, which is
    what makes it bit-identical to ``compute_gae_reference``.
    """
    gamma = float(gamma)
    lam = float(lam)
    n = num_rows(buffer)

    # Rows past the fill point are not part of any trajectory. Zeroing
    # them keeps a previous update's numbers from being sampled if the
    # buffer is later read without honouring its length.
    buffer.advantage[n:] = 0.0
    buffer.ret[n:] = 0.0
    if n == 0:
        return

    traj = group_trajectories(np.asarray(buffer.episode)[:n], np.asarray(buffer.seat)[:n])
    index, valid = traj.index, traj.valid
    n_groups, longest = index.shape
    gather = np.where(valid, index, 0)

    # Padded cells are given reward 0, value 0 and done True. That makes
    # their delta exactly 0 and their non-terminal factor exactly 0, so
    # the running accumulator is back at 0.0 by the time the backwards
    # sweep reaches a group's real last step. No separate reset needed.
    reward = np.where(valid, np.asarray(buffer.reward, dtype=np.float64)[:n][gather], 0.0)
    value = np.where(valid, np.asarray(buffer.value, dtype=np.float64)[:n][gather], 0.0)
    done = np.where(valid, np.asarray(buffer.done, dtype=bool)[:n][gather], True)

    advantage = np.zeros((n_groups, longest), dtype=np.float64)
    gae = np.zeros(n_groups, dtype=np.float64)
    zero = np.zeros(n_groups, dtype=np.float64)

    for t in range(longest - 1, -1, -1):
        nonterminal = np.where(done[:, t], 0.0, 1.0)
        if t + 1 < longest:
            # Past the group's end there is no next state, so the tail
            # bootstraps off itself; see the module docstring.
            following = np.where(valid[:, t + 1], value[:, t + 1], value[:, t])
        else:
            following = value[:, t]
        next_value = np.where(nonterminal > 0.0, following, zero)
        delta = reward[:, t] + gamma * next_value * nonterminal - value[:, t]
        gae = delta + gamma * lam * nonterminal * gae
        advantage[:, t] = gae

    rows = index[valid]
    buffer.advantage[rows] = advantage[valid]
    buffer.ret[rows] = advantage[valid] + value[valid]


def compute_gae_reference(
    buffer: GAEBuffer, gamma: float = 1.0, lam: float = 0.95
) -> None:
    """The same thing, written the obvious way. Slow, and the arbiter.

    Groups with a dictionary and recurses with a Python loop, so it
    shares no grouping machinery with ``compute_gae`` -- a mistake in the
    ragged-to-dense index construction cannot hide by being made twice.
    Every intermediate is a Python float, i.e. IEEE double, and the
    expressions are written in the same order as the vectorised ones, so
    the two agree bit for bit rather than approximately.
    """
    gamma = float(gamma)
    lam = float(lam)
    n = num_rows(buffer)
    buffer.advantage[n:] = 0.0
    buffer.ret[n:] = 0.0
    if n == 0:
        return

    episode = np.asarray(buffer.episode)
    seat = np.asarray(buffer.seat)
    reward = np.asarray(buffer.reward)
    value = np.asarray(buffer.value)
    done = np.asarray(buffer.done)

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in range(n):
        groups[(int(episode[row]), int(seat[row]))].append(row)

    for rows in groups.values():
        length = len(rows)
        gae = 0.0
        for t in range(length - 1, -1, -1):
            row = rows[t]
            here = float(value[row])
            if done[row]:
                nonterminal, next_value = 0.0, 0.0
            elif t + 1 < length:
                nonterminal, next_value = 1.0, float(value[rows[t + 1]])
            else:
                nonterminal, next_value = 1.0, here
            delta = float(reward[row]) + gamma * next_value * nonterminal - here
            gae = delta + gamma * lam * nonterminal * gae
            buffer.advantage[row] = gae
            buffer.ret[row] = gae + here
