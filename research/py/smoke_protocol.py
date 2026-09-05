"""End-to-end check of the actor/learner boundary with a random policy.

Validates that frames parse, features come out the right shape, and that
the loop sustains a useful decision rate. Run before touching the learner:

    PYTHONPATH=py python py/smoke_protocol.py --workers 4 --games 48 --seconds 20
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from tractor_rl.features import (
    ACTION_DIM,
    STATE_DIM,
    action_features,
    merge_batches,
    parse_episodes,
    split_per_worker,
    state_features,
)
from tractor_rl.protocol import WorkerConfig, WorkerPool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--preplay", type=str, default="smart-v34")
    ap.add_argument("--baseline", type=str, default="smart-v34")
    ap.add_argument("--featurize", type=int, default=1)
    ap.add_argument("--warmup", type=float, default=5.0, help="seconds discarded before timing")
    args = ap.parse_args()

    pool = WorkerPool(
        args.workers,
        WorkerConfig(
            games=args.games,
            seed=17,
            preplay=args.preplay,
            baseline=args.baseline,
        ),
    )
    rng = np.random.default_rng(0)

    # Worker boot (node + tsx) and the first deal of every game are one-off
    # costs; timing them would understate steady-state throughput.
    warm_start = time.perf_counter()
    while time.perf_counter() - warm_start < args.warmup:
        batches = pool.read_round()
        merged = merge_batches(batches)
        picks = (rng.random(merged.cand_counts.shape[0]) * merged.cand_counts).astype(np.int64)
        pool.write_round(batches, split_per_worker(picks, merged.bounds))

    decisions = 0
    hands = 0
    rounds = 0
    attacker_points: list[int] = []
    t_read = 0.0
    t_feat = 0.0
    t_write = 0.0
    round_times: list[float] = []
    started = time.perf_counter()

    try:
        while time.perf_counter() - started < args.seconds:
            t0 = time.perf_counter()
            batches = pool.read_round()
            t1 = time.perf_counter()

            merged = merge_batches(batches)
            if args.featurize and merged.num_decisions:
                sf = state_features(merged.states)
                af = action_features(merged.cands, merged.states, merged.owner)
                assert sf.shape == (merged.num_decisions, STATE_DIM), sf.shape
                assert af.shape == (merged.owner.shape[0], ACTION_DIM), af.shape
                assert np.isfinite(sf).all() and np.isfinite(af).all()
                assert (merged.cand_counts > 0).all(), "decision with no candidates"
            picks_all = (
                rng.random(merged.cand_counts.shape[0]) * merged.cand_counts
            ).astype(np.int64)
            choices = split_per_worker(picks_all, merged.bounds)
            decisions += merged.num_decisions
            for batch in batches:
                for ep in parse_episodes(batch.episodes):
                    hands += 1
                    attacker_points.append(ep.attacker_points)
            t2 = time.perf_counter()
            pool.write_round(batches, choices)
            t3 = time.perf_counter()

            t_read += t1 - t0
            t_feat += t2 - t1
            t_write += t3 - t2
            round_times.append(t3 - t0)
            rounds += 1
        elapsed = time.perf_counter() - started
    finally:
        pool.close()
    per_round = decisions / max(rounds, 1)
    print(f"workers {args.workers} x {args.games} games, {elapsed:.1f}s")
    print(f"rounds            {rounds}  ({per_round:.0f} decisions/round)")
    print(f"decisions         {decisions}  -> {decisions / elapsed:,.0f}/s")
    print(f"hands             {hands}  -> {hands / elapsed:,.1f}/s")
    if attacker_points:
        print(f"attacker points   mean {np.mean(attacker_points):.1f}")
    print(f"time in read      {t_read / elapsed:5.1%}")
    print(f"time in features  {t_feat / elapsed:5.1%}")
    print(f"time in write     {t_write / elapsed:5.1%}")
    if round_times:
        rt = np.asarray(round_times) * 1000.0
        print(f"round ms          p50 {np.percentile(rt, 50):.1f}  p90 {np.percentile(rt, 90):.1f}  max {rt.max():.1f}")


if __name__ == "__main__":
    main()
