"""Original featuriser vs current, measured back to back in one process.

Separate runs are not comparable on a contended laptop -- an untouched
function appeared to regress 4x between two runs. Interleaving both
implementations over the same batch makes the machine's noise common-mode.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from tractor_rl import layout as L
from tractor_rl.encoding import ENC_OF_RAW, NUM_ENC_SLOTS, SCATTER
from tractor_rl.features2 import (
    ACTION_DIM, HISTORY_STEPS, action_features, history_features,
    merge_batches, state_features, _trump,
)
from tractor_rl.protocol import WorkerConfig, WorkerPool


def orig_encode_planes(planes, rank, suit):
    """The gather+einsum this started as."""
    mats = SCATTER[rank, suit]
    return np.einsum("bps,bse->bpe", planes, mats, optimize=True)


def orig_action_counts(cands, states, owner):
    n = cands.shape[0]
    slots = cands[:, L.CANDIDATE_HEADER_BYTES:].astype(np.int64)
    valid = slots != L.NO_CARD
    rows, _ = np.nonzero(valid)
    flat = rows * L.NUM_SLOTS + slots[valid]
    raw = np.bincount(flat, minlength=n * L.NUM_SLOTS).astype(np.float32)
    raw = raw.reshape(n, 1, L.NUM_SLOTS)
    rank, suit = _trump(states)
    return orig_encode_planes(raw, rank[owner], suit[owner])[:, 0]


def cur_action_counts(cands, states, owner):
    n = cands.shape[0]
    slots = cands[:, L.CANDIDATE_HEADER_BYTES:].astype(np.int64)
    valid = slots != L.NO_CARD
    rows, _ = np.nonzero(valid)
    rank, suit = _trump(states)
    enc_map = ENC_OF_RAW[rank[owner], suit[owner]]
    safe = np.where(valid, slots, 0)
    es = np.take_along_axis(enc_map, safe, axis=1)
    flat = rows * NUM_ENC_SLOTS + es[valid]
    return np.bincount(flat, minlength=n * NUM_ENC_SLOTS).astype(np.float32).reshape(n, NUM_ENC_SLOTS)


def orig_policy_planes(states):
    b = states.shape[0]
    raw = states[:, L.HEADER_BYTES:L.HEADER_BYTES + L.PLANES_BYTES]
    raw = raw.reshape(b, L.NUM_PLANES, L.NUM_SLOTS).astype(np.float32)
    rank, suit = _trump(states)
    return orig_encode_planes(raw, rank, suit)


def cur_policy_planes(states):
    from tractor_rl.features2 import _policy_planes
    return _policy_planes(states)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=25)
    args = ap.parse_args()

    pool = WorkerPool(3, WorkerConfig(games=64, seed=5, learner_seats=15,
                                      privileged=True), num_groups=1)
    states = cands = owner = None
    try:
        for _ in range(4):
            bs = pool.read_round()
            m = merge_batches(bs)
            if m.num_decisions:
                states, cands, owner = m.states, m.cands, m.owner
            pool.write_round(bs, [np.zeros(b.num_decisions, dtype=np.int64) for b in bs])
    finally:
        pool.close()
    print(f"batch: {states.shape[0]} decisions, {cands.shape[0]} candidates\n")

    def bench(label, old, new, check=True):
        if check:
            assert np.array_equal(old(), new()), f"{label}: outputs differ"
        old(); new()
        t0 = time.perf_counter()
        for _ in range(args.reps): old()
        t_old = (time.perf_counter() - t0) / args.reps
        t0 = time.perf_counter()
        for _ in range(args.reps): new()
        t_new = (time.perf_counter() - t0) / args.reps
        print(f"  {label:<28} {t_old*1000:7.2f} -> {t_new*1000:6.2f} ms   {t_old/t_new:5.2f}x")
        return t_old, t_new

    tot_o = tot_n = 0.0
    o, n = bench("plane encoding (state)",
                 lambda: orig_policy_planes(states), lambda: cur_policy_planes(states))
    tot_o += o; tot_n += n
    o, n = bench("candidate encoding (action)",
                 lambda: orig_action_counts(cands, states, owner),
                 lambda: cur_action_counts(cands, states, owner))
    tot_o += o; tot_n += n
    print(f"  {'-'*28} {'-'*24}")
    print(f"  {'encoding subtotal':<28} {tot_o*1000:7.2f} -> {tot_n*1000:6.2f} ms   {tot_o/tot_n:5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
