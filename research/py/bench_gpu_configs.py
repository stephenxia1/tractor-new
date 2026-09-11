"""Throughput and numerics for each device configuration.

Two questions, deliberately separated:

* **speed** -- what does one learner step cost with featurisation on the
  host vs the device, with TF32 on or off, in fp32 vs bf16
* **damage** -- does reduced precision move the numbers we actually steer
  by. `critic_residual_std` is a small difference between two similar
  quantities (~0.19 against a target spread of ~0.59), so it degrades
  before wall-clock gains would justify anything.

The numerics check holds weights and batch fixed and compares outputs
directly, rather than inferring precision loss from a training curve.
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
import torch

from tractor_rl import layout as L
from tractor_rl.dmc2 import DMC2Config, Learner2, RawBatch, ReplayBuffer2, featurise_batch
from tractor_rl.features2 import merge_batches
from tractor_rl.model2 import NetConfig, PrivilegedCritic, TractorNet
from tractor_rl.protocol import WorkerConfig, WorkerPool


def load_fixture():
    """Real records captured by py/make_fixture.py; see test_torch_features."""
    from pathlib import Path as _Path
    q = _Path(__file__).resolve().parents[1] / "fixtures" / "records.npz"
    if not q.exists():
        raise SystemExit(f"missing {q}; run py/make_fixture.py where the engine runs")
    return np.load(q)


def fill(buffer, target, workers, games):
    pool = WorkerPool(workers, WorkerConfig(games=games, seed=5, learner_seats=15,
                                            privileged=True), num_groups=1)
    rng = np.random.default_rng(0)
    try:
        while buffer.size < target:
            bs = pool.read_round()
            m = merge_batches(bs)
            if m.num_decisions:
                n = m.num_decisions
                # Targets shaped like the real ones (target_std ~0.59) so the
                # residual arithmetic sits in its usual range.
                buffer.add(m.states, m.cands[m.cand_offsets],
                           (rng.standard_normal(n) * 0.59).astype(np.float32),
                           rng.standard_normal(n).astype(np.float32))
            pool.write_round(bs, [np.zeros(b.num_decisions, dtype=np.int64) for b in bs])
    finally:
        pool.close()


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--fixture", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")

    cfg = dataclasses.replace(DMC2Config(), stats_every=1)
    buf = ReplayBuffer2(cfg.buffer_capacity, L.STATE_BYTES_PRIVILEGED)
    rng = np.random.default_rng(0)
    if args.fixture:
        z = load_fixture()
        states = z["states"]
        actions = np.repeat(z["cands"][:1], states.shape[0], axis=0)
        buf.add(states, actions,
                (rng.standard_normal(states.shape[0]) * 0.59).astype(np.float32),
                rng.standard_normal(states.shape[0]).astype(np.float32))
        print(f"buffer {buf.size:,} (fixture)")
    else:
        print(f"filling buffer to {cfg.min_buffer:,} ...")
        fill(buf, cfg.min_buffer + 1500, args.workers, args.games)
        print(f"buffer {buf.size:,}")
    print()
    sample = buf.sample(cfg.batch_size, rng)

    print(f"{'config':<40}{'ms/step':>10}{'steps/s':>10}")
    print("-" * 60)
    base_state = TractorNet(NetConfig()).state_dict()
    results = {}
    for label, on_device, tf32, amp in [
        ("host featurise, fp32", False, False, None),
        ("device featurise, fp32", True, False, None),
        ("device featurise, fp32 + TF32", True, True, None),
        ("device featurise, bf16", True, False, torch.bfloat16),
        ("device featurise, bf16 + TF32", True, True, torch.bfloat16),
    ]:
        if dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
        elif tf32 or on_device:
            continue
        net = TractorNet(NetConfig()).to(dev)
        net.load_state_dict(base_state)
        learner = Learner2(net, PrivilegedCritic().to(dev), cfg, dev)
        learner.device_featurise = on_device and dev.type == "cuda"
        learner.amp_dtype = amp

        def one():
            if learner.device_featurise:
                learner.step_batch(learner.featurise_on_device(RawBatch(*sample)))
            else:
                learner.step_batch(featurise_batch(*sample))

        for _ in range(3):
            one()
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(args.reps):
            one()
        sync(dev)
        dt = (time.perf_counter() - t0) / args.reps
        results[label] = (dt, dict(learner.stats))
        print(f"  {label:<38}{dt*1000:>10.2f}{1/dt:>10.1f}")

    print(f"\n{'config':<40}{'loss_q':>10}{'resid_std':>11}{'belief':>9}")
    print("-" * 70)
    for label, (_, st) in results.items():
        if not st:
            continue
        print(f"  {label:<38}{st['loss_q']:>10.5f}{st['critic_residual_std']:>11.5f}"
              f"{st['belief_acc']:>9.4f}")

    # Direct numerics: same weights, same batch, fp32 vs bf16 outputs.
    if dev.type == "cuda":
        net = TractorNet(NetConfig()).to(dev)
        net.load_state_dict(base_state)
        learner = Learner2(net, PrivilegedCritic().to(dev), cfg, dev)
        learner.device_featurise = True
        b = learner.featurise_on_device(RawBatch(*sample))
        owner = torch.arange(b.sf.shape[0], device=dev)
        with torch.no_grad():
            ref = net(b.sf, b.af, owner, b.roles, b.tokens, b.mask, want="all")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                got = net(b.sf, b.af, owner, b.roles, b.tokens, b.mask, want="all")
        print("\nfp32 vs bf16 on identical weights and batch:")
        for k in ("q", "v", "belief"):
            a, c = ref[k].float(), got[k].float()
            denom = a.abs().mean().clamp_min(1e-8)
            print(f"  {k:<8} max abs diff {(a-c).abs().max():.3e}"
                  f"   mean rel {((a-c).abs().mean()/denom):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
