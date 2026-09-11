"""The torch featuriser must agree with the numpy one exactly.

Both derive the same quantities from the same bytes with integer-exact
arithmetic, so there is no tolerance to spend: a disagreement is a bug, and
one that would show up only as a slightly worse agent.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from tractor_rl import features2 as npf
from tractor_rl import features_torch as tf
from tractor_rl.features2 import merge_batches
from tractor_rl.protocol import WorkerConfig, WorkerPool


def load_fixture():
    """Real records captured by py/make_fixture.py.

    A benchmark box has no game engine, and installing Node on a GPU box to
    generate records it uses once is not worth it. Synthetic bytes would not
    do -- the featuriser indexes by real card slots and trump contexts.
    """
    import numpy as _np
    from pathlib import Path as _Path
    q = _Path(__file__).resolve().parents[1] / "fixtures" / "records.npz"
    if not q.exists():
        raise SystemExit(f"missing {q}; run py/make_fixture.py where the engine runs")
    return _np.load(q)


def collect(rounds: int, workers: int = 3, games: int = 64):
    pool = WorkerPool(workers, WorkerConfig(games=games, seed=17, learner_seats=15,
                                            privileged=True), num_groups=1)
    best = None
    try:
        for _ in range(rounds):
            bs = pool.read_round()
            m = merge_batches(bs)
            if m.num_decisions and (best is None or m.num_decisions > best.num_decisions):
                best = m
            pool.write_round(bs, [np.zeros(b.num_decisions, dtype=np.int64) for b in bs])
    finally:
        pool.close()
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--fixture", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device)

    if args.fixture:
        z = load_fixture()
        n = int(z["owner"].max()) + 1
        st_np, cd_np, ow_np = z["states"][:n], z["cands"], z["owner"]
        print(f"batch: {n} decisions, {cd_np.shape[0]} candidates (fixture), "
              f"device {args.device}")
    else:
        m = collect(args.rounds)
        if m is None:
            raise SystemExit("no decisions collected")
        st_np, cd_np, ow_np = m.states, m.cands, m.owner
        print(f"batch: {m.num_decisions} decisions, {m.cands.shape[0]} candidates, "
              f"device {args.device}")
    print()
    st_t = torch.from_numpy(np.ascontiguousarray(st_np)).to(dev)
    cd_t = torch.from_numpy(np.ascontiguousarray(cd_np)).to(dev)
    ow_t = torch.from_numpy(np.ascontiguousarray(ow_np)).to(dev)

    failures = 0

    def check(name, a: np.ndarray, b: torch.Tensor):
        nonlocal failures
        got = b.detach().cpu().numpy()
        if a.shape != got.shape:
            print(f"FAIL {name}: shape {a.shape} vs {got.shape}")
            failures += 1
            return
        bad = np.argwhere(a != got)
        if len(bad):
            failures += 1
            print(f"FAIL {name}: {len(bad)} of {a.size} differ")
            for idx in bad[:5]:
                k = tuple(int(i) for i in idx)
                print(f"     at {k}: numpy={a[k]!r} torch={got[k]!r}")
        else:
            print(f"ok   {name}: {a.size:,} values bit-identical")

    check("state_features", npf.state_features(st_np), tf.state_features(st_t))
    tok_np, msk_np = npf.history_features(st_np)
    tok_t, msk_t = tf.history_features(st_t)
    check("history tokens", tok_np, tok_t)
    check("history mask", msk_np, msk_t)
    check("action_features",
          npf.action_features(cd_np, st_np, ow_np),
          tf.action_features(cd_t, st_t, ow_t))
    priv_np = npf.privileged_features(st_np)
    priv_t = tf.privileged_features(st_t)
    check("privileged_features", priv_np, priv_t)
    check("belief_targets",
          npf.belief_targets(st_np, priv_np),
          tf.belief_targets(st_t, priv_t))

    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("torch featuriser matches numpy exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
