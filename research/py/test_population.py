"""Checks on population seating and the archive's sampling distribution.

Runs without torch, which is the point of keeping population.py free of it:
the rules that decide who plays whom are the part most likely to be subtly
wrong, and they should be checkable on a laptop.

    PYTHONPATH=py python py/test_population.py
"""

from __future__ import annotations

import collections
import sys

import numpy as np

from tractor_rl.population import (
    Archive,
    ModelRef,
    PARTNER_PAIRS,
    Archive as _A,  # noqa: F401  (kept so the import list reads as the API)
    choose_archive_refs,
    distinct_models,
    live_seat_counts,
    plan_seating,
    seating_table,
    slot_index,
)

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL {msg}")


def stocked_archive(n_agents: int, per_agent: int, rng) -> Archive:
    a = Archive(n_agents, capacity=per_agent, burn_in=0)
    for it in range(per_agent):
        for agent in range(n_agents):
            a.publish(agent, it, f"w{agent}@{it}", rng)
    return a


def test_seating_shape() -> None:
    print("seating shape and split")
    rng = np.random.default_rng(0)
    arch = stocked_archive(8, 6, rng)
    for trial in range(200):
        refs = choose_archive_refs(rng, arch, 4)
        seating = plan_seating(rng, n_agents=8, n_tables=8,
                               live_per_table=2, archive_refs=refs)
        check(len(seating) == 8, "wrong table count")
        for row in seating:
            check(len(row) == 4, "table does not have four seats")
            live = [r for r in row if r.is_live]
            check(len(live) == 2, f"expected 2 live seats, got {len(live)}")
            check(len({r.agent for r in live}) == 2,
                  f"same agent seated twice at one table: {[str(r) for r in row]}")
            frozen = [r for r in row if not r.is_live]
            check(len(set(frozen)) == len(frozen),
                  f"the same frozen policy twice at one table: "
                  f"{[str(r) for r in row]}")
            for r in frozen:
                check(r in refs, f"archive seat {r} was not one of this "
                                 "iteration's chosen refs")
        if trial == 0:
            print("   ", " | ".join(" ".join(str(r) for r in row) for row in seating[:3]))


def test_live_spread() -> None:
    print("live seats spread evenly")
    rng = np.random.default_rng(1)
    arch = stocked_archive(8, 6, rng)
    for _ in range(200):
        refs = choose_archive_refs(rng, arch, 4)
        seating = plan_seating(rng, 8, 8, 2, refs)
        counts = live_seat_counts(seating, 8)
        check(counts.sum() == 16, "live seat total is wrong")
        # 16 seats over 8 agents is exactly 2 each; nothing should deviate.
        check(counts.max() - counts.min() <= 1,
              f"live seats not evenly spread: {counts.tolist()}")
    # An uneven division must still be within one.
    for _ in range(200):
        seating = plan_seating(rng, 5, 8, 2, choose_archive_refs(rng, arch, 4))
        counts = live_seat_counts(seating, 5)
        check(counts.max() - counts.min() <= 1,
              f"uneven division not balanced: {counts.tolist()}")


def test_partner_and_opponent_mix() -> None:
    print("live pairs alternate between partners and opponents")
    rng = np.random.default_rng(2)
    arch = stocked_archive(8, 6, rng)
    partners = total = 0
    for _ in range(1500):
        seating = plan_seating(rng, 8, 4, 2, choose_archive_refs(rng, arch, 4))
        for row in seating:
            seats = tuple(sorted(i for i, r in enumerate(row) if r.is_live))
            total += 1
            partners += seats in PARTNER_PAIRS
    frac = partners / total
    # Uniform over the six seat pairs puts two of them on the same team.
    check(0.28 < frac < 0.39, f"partner fraction {frac:.3f} is not near 1/3")
    print(f"    live agents were partners {frac:.1%} of the time (expect 33%)")


def test_cold_start() -> None:
    print("empty archive falls back to all-live seating")
    rng = np.random.default_rng(3)
    seating = plan_seating(rng, 8, 8, 2, archive_refs=())
    for row in seating:
        check(all(r.is_live for r in row), "cold start seated an archive ref")
        check(len({r.agent for r in row}) == 4,
              "cold start seated the same agent twice at a table")
    counts = live_seat_counts(seating, 8)
    check(counts.sum() == 32, "cold start should fill every seat with a live agent")


def test_reservoir_is_uniform_over_history() -> None:
    print("reservoir sampling is uniform over the whole history")
    # The claim that matters: an agent published 400 times but only 20 slots
    # kept should still be sampled as though every iteration were available.
    # A last-N window would concentrate entirely in [380, 400).
    iters = 400
    cap = 20
    seen = collections.Counter()
    trials = 400
    for t in range(trials):
        rng = np.random.default_rng(1000 + t)
        arch = Archive(1, capacity=cap, burn_in=0)
        for it in range(iters):
            arch.publish(0, it, it, rng)
        for ref in arch.refs(0):
            seen[ref.iteration] += 1
        check(len(arch.refs(0)) == cap, "reservoir did not stay at capacity")

    counts = np.array([seen.get(i, 0) for i in range(iters)], dtype=float)
    expected = trials * cap / iters
    early = counts[:100].mean()
    late = counts[-100:].mean()
    print(f"    expected {expected:.1f} per iteration; "
          f"first 100 mean {early:.1f}, last 100 mean {late:.1f}")
    # Uniform means the early history is retained as often as the recent.
    check(abs(early - expected) < 0.25 * expected,
          f"early history under-represented: {early:.1f} vs {expected:.1f}")
    check(abs(late - expected) < 0.25 * expected,
          f"recent history over-represented: {late:.1f} vs {expected:.1f}")
    check(counts.min() > 0, "some iteration was never sampled at all")


def test_burn_in() -> None:
    print("burn-in publications are refused")
    rng = np.random.default_rng(4)
    arch = Archive(2, capacity=10, burn_in=3)
    for it in range(8):
        for agent in range(2):
            arch.publish(agent, it, it, rng)
    for agent in range(2):
        its = sorted(r.iteration for r in arch.refs(agent))
        check(its == [3, 4, 5, 6, 7], f"burn-in not applied: {its}")
        check(arch.offered(agent) == 5, "offered counter includes refused items")


def test_choose_archive_refs() -> None:
    print("archive refs are distinct and spread across agents")
    rng = np.random.default_rng(5)
    arch = stocked_archive(8, 6, rng)
    for _ in range(300):
        refs = choose_archive_refs(rng, arch, 4)
        check(len(refs) == 4, f"budget not met: {len(refs)}")
        check(len(set(refs)) == len(refs), "duplicate archive refs chosen")
        check(len({r.agent for r in refs}) == 4,
              f"budget of 4 did not cover 4 distinct agents: {[str(r) for r in refs]}")
    # A budget larger than the archive must not spin or duplicate.
    small = Archive(2, capacity=2, burn_in=0)
    small.publish(0, 0, "x", rng)
    small.publish(1, 0, "y", rng)
    refs = choose_archive_refs(rng, small, 10)
    check(len(refs) == 2, f"exhausted archive returned {len(refs)} refs")
    check(len(set(refs)) == 2, "exhausted archive duplicated a ref")
    check(choose_archive_refs(rng, Archive(4), 4) == [], "empty archive must yield []")


def test_routing_arrays() -> None:
    print("routing arrays match the seating")
    rng = np.random.default_rng(6)
    arch = stocked_archive(8, 6, rng)
    refs = choose_archive_refs(rng, arch, 4)
    seating = plan_seating(rng, 8, 8, 2, refs)
    agents, live = seating_table(seating, 8)
    check(agents.shape == (8, 4) and live.shape == (8, 4), "routing array shape")
    for t, row in enumerate(seating):
        for s, ref in enumerate(row):
            check(int(agents[t, s]) == ref.agent, "agent id mismatch")
            check(bool(live[t, s]) == ref.is_live, "live mask mismatch")
    models = distinct_models(seating)
    check(len(models) == len({m for row in seating for m in row}),
          "distinct_models lost or invented an entry")
    check(len(models) <= 8 + len(refs), "more distinct models than seats can hold")
    print(f"    {len(models)} distinct models on the board "
          f"= one forward pass each per round")


def test_routing_and_attribution() -> None:
    """The bug that no loss curve would show: a seat routed to the wrong net.

    Replays the trainer's indexing on synthetic decisions -- a flat batch of
    (table, seat) pairs, exactly what merge_batches produces -- and checks
    that every decision reaches the weights the seating assigned it, and that
    each agent would be handed only its own decisions to train on.
    """
    print("decisions route to the seated model and are attributed correctly")
    rng = np.random.default_rng(8)
    arch = stocked_archive(8, 6, rng)
    for _ in range(50):
        refs_avail = choose_archive_refs(rng, arch, 4)
        seating = plan_seating(rng, 8, 8, 2, refs_avail)
        refs, slot_table = slot_index(seating)

        n = 4000
        table_of = rng.integers(0, 8, n)
        seats = rng.integers(0, 4, n)
        slot_of = slot_table[table_of, seats]

        # 1. Every decision's slot names the weights the seating put there.
        for i in range(n):
            check(refs[slot_of[i]] == seating[table_of[i]][seats[i]],
                  "decision routed to the wrong model")

        # 2. The trainer's attribution: gather live rows per agent, then map
        #    back. Every stored decision must belong to the agent it is filed
        #    under, and nothing live may be dropped.
        live_by_agent = {}
        for k, ref in enumerate(refs):
            rows = np.nonzero(slot_of == k)[0]
            if ref.is_live and rows.shape[0]:
                live_by_agent[ref.agent] = rows
        live_rows = (np.sort(np.concatenate(list(live_by_agent.values())))
                     if live_by_agent else np.zeros(0, dtype=np.int64))
        owner = np.empty(live_rows.shape[0], dtype=np.int64)
        for agent, rows in live_by_agent.items():
            owner[np.searchsorted(live_rows, rows)] = agent

        expected_live = np.array(
            [seating[table_of[i]][seats[i]].is_live for i in range(n)])
        check(live_rows.shape[0] == int(expected_live.sum()),
              "live decision count does not match the seating")
        for pos, row in enumerate(live_rows):
            ref = seating[table_of[row]][seats[row]]
            check(ref.is_live, "a frozen seat was filed for training")
            check(int(owner[pos]) == ref.agent,
                  f"decision at table {table_of[row]} seat {seats[row]} filed "
                  f"under agent {owner[pos]} but seated {ref}")
    print("    4000 synthetic decisions x 50 seatings, all routed and filed correctly")


def test_rejects_bad_config() -> None:
    print("bad configuration is refused")
    rng = np.random.default_rng(7)
    for bad in (0, 5):
        try:
            plan_seating(rng, 8, 4, bad, [ModelRef(0, 1)])
        except ValueError:
            pass
        else:
            check(False, f"live_per_table={bad} should have raised")
    try:
        plan_seating(rng, 2, 4, 3, [ModelRef(0, 1)])
    except ValueError:
        pass
    else:
        check(False, "seating 3 distinct live agents from a population of 2 "
                     "should have raised")


def main() -> int:
    for fn in (test_seating_shape, test_live_spread, test_partner_and_opponent_mix,
               test_cold_start, test_reservoir_is_uniform_over_history, test_burn_in,
               test_choose_archive_refs, test_routing_arrays,
               test_routing_and_attribution, test_rejects_bad_config):
        fn()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed")
        return 1
    print("all population checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
