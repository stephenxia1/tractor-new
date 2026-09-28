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


# --- where a population step goes ---------------------------------------------


def _pop_profile(checkpoint: str, arch: str, hands: int, declare_hands: int,
                 worker_counts: tuple = (7,), chunk_counts: tuple = (1,)) -> dict:
    """Break one population step into its parts, on the real sizes.

    1. A rollout round of ``hands`` hands through the eight agents at each
       worker count and forward split, with the parent's spans; what is
       left of a round's wall clock after them is the parent waiting on
       the workers.
    2. The forward's components -- history encoder, query block, scorer,
       heads, critic -- timed apart on real rows at a per-agent batch and
       at a whole lockstep batch.
    3. One agent's PPO update, minibatch by minibatch: host gather and
       copy, forward and loss, backward, optimiser; then a kernel table.
    4. One pre-play step at ``declare_hands`` hands, split into dealing
       and declaring, burying, critic scoring and the updates.
    """
    import time

    import numpy as np
    import torch

    from tractor.nn.arch import ARCHES
    from tractor.nn.features import Cache
    from tractor.sim.bury import BuryConfig
    from tractor.telemetry import Spans
    from tractor.train.buffer import RolloutBuffer
    from tractor.train.declare_loop import DeclareTrainConfig
    from tractor.train.gae import compute_gae
    from tractor.train.parallel import RolloutPool
    from tractor.train.population import (
        _PRE, PopulationConfig, _preplay_step, build_agents, hand_agents,
        load_agents, seating,
    )
    from tractor.train.ppo import _minibatches
    from tractor.train.starts import PrePlayConfig

    dev = torch.device("cuda")

    def sync() -> None:
        torch.cuda.synchronize()

    cfg = PopulationConfig(
        hands_per_step=hands, buffer_capacity=hands * 100, declare_phase_steps=1,
        bury_policy=True, ppo={"epochs": 1, "length_buckets": 4},
        declare={"hands_per_step": declare_hands}, **ARCHES[arch],
    )
    declare_cfg = DeclareTrainConfig(**{
        **cfg.declare,
        "pre": PrePlayConfig(random_burial_frac=0.0, bury=BuryConfig(), burial="policy"),
    })
    torch.manual_seed(0)
    agents = build_agents(cfg, dev, declare_cfg)
    if checkpoint:
        load_agents(agents, torch.load(checkpoint, map_location="cpu", weights_only=False))
    nets = [a.net for a in agents]
    mcfg = nets[0].cfg
    rng = np.random.default_rng(0)
    out: dict = {"arch": arch, "hands": hands}

    # -- 1. the rollout, per worker count and forward split -------------------
    seat_of = hand_agents(seating(0, 0, cfg.agents), hands)
    buf = RolloutBuffer(hands * 100, mcfg)
    out["collect"] = {}
    for workers in worker_counts:
        pool = RolloutPool(cfg=mcfg, workers=workers, hands_per_round=hands)
        try:
            pool.collect(nets, rng, buf, n_hands=256, device=dev, pool=cfg.pool, amp=True,
                         level_weight=0.0, agents=seat_of[:256])  # warm the device
            for chunks in chunk_counts:
                spans = Spans(enabled=True, device=dev)
                t0 = time.perf_counter()
                buf, st = pool.collect(nets, rng, buf, n_hands=hands, device=dev,
                                       pool=cfg.pool, amp=True, level_weight=0.0,
                                       spans=spans, agents=seat_of, chunks=chunks)
                collect_s = time.perf_counter() - t0
                parts = {k: v["total"] for k, v in spans.summary().items()}
                out["collect"][f"workers {workers}, chunks {chunks}"] = {
                    "seconds": collect_s, **parts,
                    "waiting_on_workers": collect_s - sum(parts.values()),
                    "lockstep_rounds": st["forwards"], "mean_batch": st["mean_batch"],
                    "rows": len(buf),
                    "mean_history": float(buf.history_len[: len(buf)].mean()),
                }
                print(f"  workers {workers} chunks {chunks}: {collect_s:.1f}s "
                      f"forward {parts.get('forward', 0):.1f}s", flush=True)
        finally:
            pool.close()
    compute_gae(buf, gamma=1.0, lam=0.95)
    n = len(buf)

    # -- 2. the forward, by component -----------------------------------------
    def timed(fn, reps: int = 20) -> float:
        fn()
        sync()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        return (time.perf_counter() - t) / reps * 1e3  # ms

    net = nets[0].eval()
    pick = np.random.default_rng(1)
    for b in (190, 1500):
        idx = np.sort(pick.choice(n, size=b, replace=False))
        obs, priv, _ = buf.as_torch(idx, dev)
        trick = obs.planes[:, 5:9]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc, keep = net.actor.history(obs.history, obs.history_len)
            summary = net.actor.query(obs, enc, keep)
            out[f"forward_ms_batch{b}"] = {
                "history": timed(lambda: net.actor.history(obs.history, obs.history_len)),
                "query": timed(lambda: net.actor.query(obs, enc, keep)),
                "scorer": timed(lambda: net.actor.scorer(summary, obs)),
                "heads": timed(lambda: net.actor.heads(summary, obs.cards)),
                "critic": timed(lambda: net.critic(enc, obs.history_len, priv,
                                                   context=obs.context, cards=obs.cards,
                                                   trick=trick)),
                "whole": timed(lambda: net(obs, priv)),
                "longest_history": int(obs.history_len.max()),
            }

    # -- 3. one agent's update ------------------------------------------------
    owner = seat_of[buf.episode[:n], buf.seat[:n]]
    rows = np.nonzero(owner == 0)[0]
    learner = agents[0].learner
    order = rows[torch.randperm(len(rows)).numpy()]
    batches = _minibatches(order, buf, learner.cfg.minibatch, learner.cfg.length_buckets)
    net.train()
    t = {"gather": 0.0, "forward": 0.0, "backward": 0.0, "optimiser": 0.0}
    for i, idx in enumerate(batches[:65]):
        sync()
        a = time.perf_counter()
        obs, priv, sc = buf.as_torch(idx, dev)
        sync()
        b_ = time.perf_counter()
        loss, _parts = learner.losses(obs, priv, sc)
        sync()
        c = time.perf_counter()
        learner.opt.zero_grad(set_to_none=True)
        loss.backward()
        sync()
        d = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(net.parameters(), learner.cfg.max_grad_norm)
        learner.opt.step()
        sync()
        e = time.perf_counter()
        if i >= 5:  # past the first few, which warm the kernels
            t["gather"] += b_ - a
            t["forward"] += c - b_
            t["backward"] += d - c
            t["optimiser"] += e - d
    out["update_ms_per_minibatch"] = {key: v / 60 * 1e3 for key, v in t.items()}
    out["update_minibatches_per_agent"] = len(batches)

    # The same rows resident on the device, for comparison with the gather.
    sync()
    a = time.perf_counter()
    whole = buf.as_torch(rows, dev)
    sync()
    out["update_rows_to_device_s"] = time.perf_counter() - a
    del whole

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for idx in batches[65:75]:
            obs, priv, sc = buf.as_torch(idx, dev)
            loss, _parts = learner.losses(obs, priv, sc)
            learner.opt.zero_grad(set_to_none=True)
            loss.backward()
            learner.opt.step()
        sync()
    out["update_kernels"] = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=18, max_name_column_width=60)

    # -- 4. one pre-play step -------------------------------------------------
    sync()
    a = time.perf_counter()
    st = _preplay_step(agents, seating(0, 0, cfg.agents, _PRE), declare_cfg, rng, dev, Cache())
    sync()
    out["preplay"] = {
        "seconds": time.perf_counter() - a,
        **{k: st[k] for k in ("time/declare_s", "time/bury_s", "time/critic_s", "time/update_s")},
    }
    return out


@app.function(image=image, gpu="L40S", volumes={VOLUME: volume}, timeout=3600,
              cpu=12.0, memory=65536)
def pop_profile_l40s(checkpoint: str, arch: str, hands: int, declare_hands: int,
                     worker_counts: list, chunk_counts: list) -> dict:
    return _pop_profile(checkpoint, arch, hands, declare_hands,
                        tuple(worker_counts), tuple(chunk_counts))


@app.local_entrypoint()
def pop_profile(checkpoint: str = "", arch: str = "v6", hands: int = 4096,
                declare_hands: int = 8192, workers: str = "7",
                chunks: str = "1") -> None:
    """Where one population step goes: ``modal run modal_perf.py::pop_profile``.

    ``--workers 7,10 --chunks 1,2,3`` times a rollout round at each pair.
    An empty ``--checkpoint`` profiles freshly initialised agents.
    """
    import json

    res = pop_profile_l40s.remote(
        checkpoint, arch, hands, declare_hands,
        [int(w) for w in workers.split(",")], [int(c) for c in chunks.split(",")])
    table = res.pop("update_kernels")
    print(json.dumps(res, indent=2))
    print(table)
    with open(f"runs/pop-profile-{arch}.json", "w") as fh:
        json.dump({**res, "update_kernels": table}, fh, indent=2)
