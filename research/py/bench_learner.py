"""Times one learner step, and what the critic cadence costs.

The end-to-end A/B was useless on a laptop: buffer warm-up ate the run and
left single-digit gradient steps. This fills a buffer with real states once
and times `step()` itself, which is what the cadence actually changes.

Targets are synthetic. Step *time* depends on tensor shapes and the graph,
not on the values, so this measures cost honestly even though it would
teach the network nothing.
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
import torch

from tractor_rl import layout as L
from tractor_rl.dmc2 import DMC2Config, Learner2, ReplayBuffer2
from tractor_rl.features2 import merge_batches
from tractor_rl.model2 import NetConfig, PrivilegedCritic, TractorNet
from tractor_rl.protocol import WorkerConfig, WorkerPool


def fill(buffer: ReplayBuffer2, target: int, workers: int, games: int) -> None:
    wc = WorkerConfig(games=games, seed=5, learner_seats=15, privileged=True)
    pool = WorkerPool(workers, wc, num_groups=1)
    rng = np.random.default_rng(0)
    try:
        while buffer.size < target:
            batches = pool.read_round()
            m = merge_batches(batches)
            if m.num_decisions:
                n = m.num_decisions
                buffer.add(
                    m.states,
                    m.cands[m.cand_offsets],           # first candidate; shape is what matters
                    rng.standard_normal(n).astype(np.float32),
                    rng.standard_normal(n).astype(np.float32),
                )
            pool.write_round(
                batches, [np.zeros(b.num_decisions, dtype=np.int64) for b in batches])
    finally:
        pool.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--games", type=int, default=64)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    base = DMC2Config()
    dev = torch.device("cpu")
    buffer = ReplayBuffer2(base.buffer_capacity, L.STATE_BYTES_PRIVILEGED)
    print(f"filling buffer to {base.min_buffer:,} ...")
    fill(buffer, base.min_buffer + 2000, args.workers, args.games)
    print(f"buffer {buffer.size:,}, batch {base.batch_size}\n")

    rng = np.random.default_rng(0)
    results = {}
    for label, every in [("critic every step", 1),
                         ("critic every 2  (current)", 2),
                         ("critic every 4", 4),
                         ("critic never", 10 ** 9)]:
        cfg = dataclasses.replace(base, critic_every=every)
        net = TractorNet(NetConfig()).to(dev)
        learner = Learner2(net, PrivilegedCritic().to(dev), cfg, dev)
        for _ in range(3):
            learner.step(buffer, rng)
        learner.steps = 0
        t0 = time.perf_counter()
        for _ in range(args.reps):
            learner.step(buffer, rng)
        dt = (time.perf_counter() - t0) / args.reps
        results[label] = dt
        print(f"  {label:<26} {dt*1000:7.1f} ms/step   {1/dt:5.2f} steps/s")

    never = results["critic never"]
    cur = results["critic every 2  (current)"]
    print(f"\n  critic share at the current cadence: "
          f"{100*(cur-never)/cur:.0f}% of learner time")
    print(f"  moving to every 4 would save {100*(cur-results['critic every 4'])/cur:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
