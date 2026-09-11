"""Where does featurisation time actually go?

Featurisation is 27% of training wall time, which is a lot for what is
mostly moving integers between arrays. This times each piece on a realistic
batch so the optimisation targets something measured.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from tractor_rl import layout as L
from tractor_rl.features2 import (
    action_features,
    belief_targets,
    history_features,
    merge_batches,
    privileged_features,
    state_features,
)
from tractor_rl.encoding import SCATTER, encode_planes
from tractor_rl.protocol import WorkerConfig, WorkerPool


def collect(rounds: int) -> object:
    cfg = WorkerConfig(games=64, seed=7, learner_seats=15, privileged=True)
    pool = WorkerPool(3, cfg, num_groups=1)
    batches = []
    try:
        for _ in range(rounds):
            bs = pool.read_round()
            m = merge_batches(bs)
            if m.num_decisions:
                batches.append(m)
            pool.write_round(bs, [np.zeros(b.num_decisions, dtype=np.int64) for b in bs])
    finally:
        pool.close()
    return batches


def timeit(label: str, fn, reps: int) -> float:
    fn()  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    dt = (time.perf_counter() - t0) / reps
    print(f"  {label:<34} {dt*1000:8.2f} ms")
    return dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    print("collecting a realistic batch...")
    batches = collect(args.rounds)
    m = max(batches, key=lambda b: b.num_decisions)
    states, cands, owner = m.states, m.cands, m.owner
    print(f"batch: {states.shape[0]} decisions, {cands.shape[0]} candidates, "
          f"{states.shape[1]} state bytes\n")

    total = 0.0
    total += timeit("state_features", lambda: state_features(states), args.reps)
    total += timeit("history_features", lambda: history_features(states), args.reps)
    total += timeit("action_features",
                    lambda: action_features(cands, states, owner), args.reps)
    total += timeit("belief_targets", lambda: belief_targets(states), args.reps)
    total += timeit("privileged_features",
                    lambda: privileged_features(states), args.reps)
    print(f"  {'-'*34} {'-'*11}")
    print(f"  {'total':<34} {total*1000:8.2f} ms\n")

    # The suspicious part: the chain permutation is done as a matmul against a
    # gathered [B, 54, 66] one-hot, when it is really a scatter-add.
    b = states.shape[0]
    raw = states[:, L.HEADER_BYTES:L.HEADER_BYTES + L.PLANES_BYTES]
    raw = raw.reshape(b, L.NUM_PLANES, L.NUM_SLOTS).astype(np.float32)
    rank = states[:, L.H_TRUMP_RANK].astype(np.int64)
    suit = states[:, L.H_TRUMP_SUIT].astype(np.int64)

    print("inside encode_planes:")
    timeit("gather SCATTER[rank, suit]", lambda: SCATTER[rank, suit], args.reps)
    timeit("full encode_planes", lambda: encode_planes(raw, rank, suit), args.reps)
    mats = SCATTER[rank, suit]
    timeit("einsum only", lambda: np.einsum("bps,bse->bpe", raw, mats, optimize=True),
           args.reps)
    print(f"\n  gathered one-hot is {mats.nbytes/1e6:.1f} MB per call "
          f"({mats.shape})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
