"""Throughput measurements that must not share an app with training.

Two ``modal run``s of one app stop each other, so a benchmark started
under ``tractor-train`` while a detached run is live risks that run.
This is its own app, on the training image.

    modal run modal_perf.py::update --checkpoint /vol/v3-points/checkpoints/step_0002000.pt
"""

from __future__ import annotations

import modal

import modal_train

app = modal.App("tractor-perf")
image = modal_train.image.add_local_python_source("modal_train")
volume = modal_train.volume
VOLUME = modal_train.VOLUME


def _update_bench(checkpoint: str, hands: int, buckets: list) -> list:
    """One real round, then the same update timed at each bucket count.

    Every arm updates a fresh copy of the same weights on the same
    buffer, so the times differ only by how minibatches are drawn; the
    losses and KL say whether the drawing changed what was learned.
    """
    import copy
    import time

    import numpy as np
    import torch

    from tractor.telemetry import Spans
    from tractor.train.buffer import RolloutBuffer
    from tractor.train.gae import compute_gae
    from tractor.train.parallel import RolloutPool
    from tractor.train.ppo import PPOConfig, PPOLearner

    net, cfg = modal_train._load_net(checkpoint, "cuda")
    net.eval()
    pool = RolloutPool(cfg=cfg, workers=7, hands_per_round=hands)
    try:
        buf = RolloutBuffer(hands * 100, cfg)
        spans = Spans(enabled=True, device="cuda")
        t0 = time.perf_counter()
        buf, stats = pool.collect(
            net, np.random.default_rng(0), buf, n_hands=hands, device="cuda",
            level_weight=0.0, pool=512, amp=True, spans=spans,
        )
        collect_s = time.perf_counter() - t0
    finally:
        pool.close()
    compute_gae(buf, gamma=1.0, lam=0.95)
    fwd = spans.summary().get("forward", {}).get("total", 0.0)
    out = [{"collect_s": collect_s, "forward_s": fwd, "rows": len(buf),
            "mean_history": float(np.asarray(buf.history_len[: len(buf)]).mean())}]
    for b in buckets:
        arm = copy.deepcopy(net)
        learner = PPOLearner(arm, PPOConfig(epochs=1, length_buckets=b), "cuda")
        arm.train()
        torch.manual_seed(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        metrics = learner.update(buf)
        torch.cuda.synchronize()
        out.append({
            "buckets": b, "update_s": time.perf_counter() - t0,
            **{k: float(metrics[k]) for k in
               ("policy_loss", "value_loss", "approx_kl", "clip_fraction",
                "explained_variance") if k in metrics},
        })
        print(out[-1], flush=True)
    return out


@app.function(image=image, gpu="L40S", volumes={VOLUME: volume},
              timeout=3600, cpu=12.0, memory=65536)
def update_bench_l40s(checkpoint: str, hands: int, buckets: list) -> list:
    return _update_bench(checkpoint, hands, buckets)


@app.local_entrypoint()
def update(checkpoint: str = "", hands: int = 1024, buckets: str = "0,4,8") -> None:
    rows = update_bench_l40s.remote(
        checkpoint, hands, [int(b) for b in buckets.split(",")]
    )
    head = rows[0]
    print(f"\ncollect {head['collect_s']:.1f}s (forward {head['forward_s']:.1f}s), "
          f"{head['rows']} rows, mean history {head['mean_history']:.1f}")
    for r in rows[1:]:
        print(f"buckets {r['buckets']:2d}: update {r['update_s']:5.1f}s  "
              f"policy {r.get('policy_loss', float('nan')):+.4f}  "
              f"value {r.get('value_loss', float('nan')):.5f}  "
              f"kl {r.get('approx_kl', float('nan')):.4f}  "
              f"clip {r.get('clip_fraction', float('nan')):.3f}")
