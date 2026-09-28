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
        _PRE, PopulationConfig, _preplay_step, build_agents, build_critic, hand_agents,
        load_agents, load_critic, seating,
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
    critic_learner = build_critic(cfg, dev)
    shared = None if critic_learner is None else critic_learner.critic
    if checkpoint:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        load_critic(critic_learner, blob)
        load_agents(agents, blob)
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
                         level_weight=0.0, agents=seat_of[:256],
                         critic=shared)  # warm the device
            for chunks in chunk_counts:
                spans = Spans(enabled=True, device=dev)
                t0 = time.perf_counter()
                buf, st = pool.collect(nets, rng, buf, n_hands=hands, device=dev,
                                       pool=cfg.pool, amp=True, level_weight=0.0,
                                       spans=spans, agents=seat_of, chunks=chunks,
                                       critic=shared)
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
    st = _preplay_step(agents, seating(0, 0, cfg.agents, _PRE), declare_cfg, rng, dev, Cache(),
                       shared)
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


# --- torch.compile, larger minibatches, stacked updates ------------------------


def _opt_bench(arch: str, hands: int, n_timed: int) -> dict:
    """Update and forward throughput under each candidate optimisation.

    All in one container, on one round of real rows, because host CPUs
    differ between containers by more than some of these effects.

    What the earlier compile attempt got wrong, and what this avoids
    (``PPOConfig.compile``'s notes):

    * Ragged minibatches -- the last of every epoch was 24 rows against
      256 -- recompiled the graph for each new size. Here every timed
      minibatch is exactly ``m`` rows (ragged tails are left out, and the
      fraction dropped reported).
    * The history encoder read ``history_len.max()`` on the host to trim
      each batch, a graph break and a new width per minibatch. Here the
      caller trims to a bucket -- a multiple of 16 -- and the model runs
      in ``static_shapes`` mode, trusting the width it is handed: at most
      eight widths, each compiled once.
    * Eight agents are eight module instances; whether their identical
      graphs are shared or recompiled per agent is measured, and the
      dynamo cache is widened so a recompile shows up as time rather
      than as a silent fall back to eager.
    """
    import contextlib
    import time
    from dataclasses import replace

    import numpy as np
    import torch
    import torch._dynamo

    from tractor.nn.arch import ARCHES
    from tractor.nn.model import (
        ActorCritic, ModelConfig, ObsBatch, shared_critic, static_shapes,
    )
    from tractor.train.buffer import RolloutBuffer
    from tractor.train.gae import compute_gae
    from tractor.train.parallel import RolloutPool
    from tractor.train.population import hand_agents, seating
    from tractor.train.ppo import PPOConfig, PPOLearner, _minibatches

    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    dev = torch.device("cuda")

    def sync() -> None:
        torch.cuda.synchronize()

    base = ModelConfig(**ARCHES[arch]["model"])
    actor_cfg = replace(base, critic_kind="none")
    torch.manual_seed(0)
    nets = [ActorCritic(actor_cfg).to(dev) for _ in range(8)]
    critic = shared_critic(base, 8).to(dev)
    rng = np.random.default_rng(0)
    seat_of = hand_agents(seating(0, 0, 8), hands)
    buf = RolloutBuffer(hands * 100, actor_cfg)
    out: dict = {"collect": {}}

    # -- 0. the rollout: lockstep against overlapped, at two pool widths ------
    from tractor.telemetry import Spans

    with RolloutPool(cfg=actor_cfg, workers=10, hands_per_round=hands) as pool:
        pool.collect(nets, rng, buf, n_hands=hands, device=dev, pool=2048, amp=True,
                     level_weight=0.0, agents=seat_of, critic=critic)  # warm
        for name, width, overlap in (("lockstep, pool 2048", 2048, False),
                                     ("overlap, pool 2048", 2048, True),
                                     ("overlap, pool 4096", 4096, True),
                                     ("lockstep, pool 4096", 4096, False)):
            spans = Spans(enabled=True, device=dev)
            t0 = time.perf_counter()
            buf, st = pool.collect(nets, rng, buf, n_hands=hands, device=dev, pool=width,
                                   amp=True, level_weight=0.0, agents=seat_of,
                                   critic=critic, spans=spans, overlap=overlap)
            secs = time.perf_counter() - t0
            parts = {k: round(v["total"], 2) for k, v in spans.summary().items()}
            out["collect"][name] = {"seconds": secs, "forwards": st["forwards"],
                                    "mean_batch": st["mean_batch"], **parts}
            print(f"  collect {name:22} {secs:6.1f}s  forwards {st['forwards']}  "
                  f"batch {st['mean_batch']:.0f}  {parts}", flush=True)
    compute_gae(buf, gamma=1.0, lam=0.95)
    n = len(buf)
    out.update({"rows": n, "mean_history": float(buf.history_len[:n].mean())})

    def bucket(width: int) -> int:
        return min(128, max(16, -(-width // 16) * 16))

    def batches(m: int, static: bool) -> list:
        order = rng.permutation(n)
        mbs = _minibatches(order, buf, m, 4)
        if static:
            full = [b for b in mbs if len(b) == m]
            out.setdefault("dropped_rows_frac", {})[m] = 1 - sum(map(len, full)) / n
            mbs = full
        return mbs

    def load(idx: np.ndarray, static: bool):
        obs, priv, sc = buf.as_torch(idx, dev)
        if static:
            w = bucket(int(buf.history_len[idx].max()))
            obs = obs._replace(history=obs.history[:, :w].contiguous())
        return obs, priv, sc

    # -- 1. the update, per minibatch ----------------------------------------
    def update_arm(name: str, m: int, static: bool, mode: str | None,
                   agent: int = 0) -> dict:
        net = nets[agent]
        learner = PPOLearner(net, PPOConfig(minibatch=m, length_buckets=4,
                                            check_actions=mode is None), dev)
        fn = learner.losses
        if mode is not None:
            fn = torch.compile(learner.losses, dynamic=False,
                               mode=None if mode == "default" else mode)
        mbs = batches(m, static)
        widths_seen: dict = {}
        first_calls = 0.0
        ctx = static_shapes() if static else contextlib.nullcontext()
        net.train()
        with ctx:
            # Warm: every width the timed minibatches will use, once.
            for idx in mbs:
                w = bucket(int(buf.history_len[idx].max())) if static else 0
                if w in widths_seen:
                    continue
                obs, priv, sc = load(idx, static)
                sync()
                t0 = time.perf_counter()
                loss, _p = fn(obs, priv, sc)
                learner.opt.zero_grad(set_to_none=True)
                loss.backward()
                learner.opt.step()
                sync()
                widths_seen[w] = time.perf_counter() - t0
                first_calls += widths_seen[w]
            timed = mbs[:n_timed]
            sync()
            t0 = time.perf_counter()
            for idx in timed:
                obs, priv, sc = load(idx, static)
                loss, _p = fn(obs, priv, sc)
                learner.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                learner.opt.step()
            sync()
            elapsed = time.perf_counter() - t0
        rows = sum(map(len, timed))
        res = {
            "minibatch": m, "static": static, "mode": mode or "eager",
            "ms_per_minibatch": elapsed / len(timed) * 1e3,
            "us_per_row": elapsed / rows * 1e6,
            "widths": len(widths_seen), "first_calls_s": first_calls,
        }
        print(f"  update {name:22} {res['ms_per_minibatch']:7.2f} ms/mb "
              f"{res['us_per_row']:6.1f} us/row  widths {len(widths_seen)}  "
              f"first calls {first_calls:6.1f}s", flush=True)
        return res

    out["update"] = {}
    arms = [
        ("eager 256", 256, False, None),
        ("eager 256 static", 256, True, None),
        ("eager 512 static", 512, True, None),
        ("eager 1024 static", 1024, True, None),
        ("compile 256", 256, True, "default"),
        ("compile 1024", 1024, True, "default"),
        ("cudagraphs 256", 256, True, "reduce-overhead"),
        ("cudagraphs 1024", 1024, True, "reduce-overhead"),
    ]
    for name, m, static, mode in arms:
        try:
            out["update"][name] = update_arm(name, m, static, mode)
        except Exception as exc:  # noqa: BLE001 - a failed arm is a finding
            out["update"][name] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            print(f"  update {name}: FAILED {out['update'][name]['error']}", flush=True)
        torch._dynamo.reset()

    # Does a second agent reuse the first's compiled graphs?
    try:
        from torch._dynamo.utils import counters

        counters.clear()
        a = update_arm("compile 256 agent 0", 256, True, "default", agent=0)
        graphs_after_first = counters["stats"]["unique_graphs"]
        b = update_arm("compile 256 agent 1", 256, True, "default", agent=1)
        out["cross_agent"] = {
            "graphs_after_agent0": graphs_after_first,
            "graphs_after_agent1": counters["stats"]["unique_graphs"],
            "agent1_first_calls_s": b["first_calls_s"],
            "agent0_first_calls_s": a["first_calls_s"],
        }
        print(f"  cross-agent {out['cross_agent']}", flush=True)
    except Exception as exc:  # noqa: BLE001
        out["cross_agent"] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    torch._dynamo.reset()

    # Compiled and eager agree on one minibatch.
    try:
        net = nets[2]
        learner = PPOLearner(net, PPOConfig(check_actions=False), dev)
        idx = batches(256, True)[0]
        obs, priv, sc = load(idx, True)
        with static_shapes():
            eager_loss, _p = learner.losses(obs, priv, sc)
            comp_loss, _p = torch.compile(learner.losses, dynamic=False)(obs, priv, sc)
        out["agreement"] = {"eager": float(eager_loss), "compiled": float(comp_loss)}
        # And the static path agrees with the host-trimmed one.
        full_obs, _pr, _sc = buf.as_torch(idx, dev)
        trimmed_loss, _p = learner.losses(full_obs, priv, sc)
        out["agreement"]["eager_trimmed"] = float(trimmed_loss)
        print(f"  agreement {out['agreement']}", flush=True)
    except Exception as exc:  # noqa: BLE001
        out["agreement"] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    torch._dynamo.reset()

    # -- 2. the rollout forward: one agent, ~190 rows; the critic, ~1500 --------
    def fwd_arm(name: str, module, rows: int, mode: str | None, critic_call: bool) -> dict:
        idx = np.sort(rng.choice(n, size=rows, replace=False))
        obs, priv, _sc = load(idx, True)
        seats = torch.zeros(rows, 4, dtype=torch.int64, device=dev)

        def call():
            if critic_call:
                return module(obs.history, obs.history_len, priv, context=obs.context,
                              cards=obs.cards, trick=obs.planes[:, 5:9], seat_agents=seats)
            return module(obs, priv)

        fn = call if mode is None else torch.compile(
            call, dynamic=False, mode=None if mode == "default" else mode)
        module.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), static_shapes():
            sync()
            t0 = time.perf_counter()
            fn()
            fn()
            sync()
            first = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(30):
                fn()
            sync()
            ms = (time.perf_counter() - t0) / 30 * 1e3
        print(f"  forward {name:26} {ms:7.2f} ms  first {first:6.1f}s", flush=True)
        return {"ms": ms, "first_s": first}

    out["forward"] = {}
    for name, module, rows, mode, is_critic in [
        ("actor 192 eager", nets[0], 192, None, False),
        ("actor 192 compile", nets[0], 192, "default", False),
        ("actor 192 cudagraphs", nets[0], 192, "reduce-overhead", False),
        ("critic 1536 eager", critic, 1536, None, True),
        ("critic 1536 compile", critic, 1536, "default", True),
        ("critic 1536 cudagraphs", critic, 1536, "reduce-overhead", True),
    ]:
        try:
            out["forward"][name] = fwd_arm(name, module, rows, mode, is_critic)
        except Exception as exc:  # noqa: BLE001
            out["forward"][name] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            print(f"  forward {name}: FAILED {out['forward'][name]['error']}", flush=True)
        torch._dynamo.reset()

    # -- 3. eight agents stacked: vmap over their parameters --------------------
    try:
        from torch.func import functional_call, stack_module_state, vmap

        m = 256
        mbs = batches(m, True)[:8]
        w = max(bucket(int(buf.history_len[i].max())) for i in mbs)
        loaded = [buf.as_torch(i, dev) for i in mbs]
        obs = ObsBatch(*(torch.stack([o[0][f][:, :w] if f == 0 else o[0][f]
                                      for o in loaded]) for f in range(len(ObsBatch._fields))))
        priv = torch.stack([o[1] for o in loaded])
        params, buffers = stack_module_state(nets)
        base_net = nets[0]

        def one(p, b, o, pr):
            out_, _v = functional_call(base_net, (p, b), (ObsBatch(*o), pr))
            return out_.log_probs.logsumexp(-1).mean() + out_.value.mean()

        def run_stacked():
            with static_shapes(), torch.autocast("cuda", dtype=torch.bfloat16):
                losses = vmap(one)(params, buffers, tuple(obs), priv)
            losses.sum().backward()

        def run_loop():
            with static_shapes(), torch.autocast("cuda", dtype=torch.bfloat16):
                total = 0.0
                for k, net in enumerate(nets):
                    o = ObsBatch(*(t[k] for t in obs))
                    out_, _v = net(o, priv[k])
                    total = total + out_.log_probs.logsumexp(-1).mean() + out_.value.mean()
            total.backward()

        for p in params.values():
            p.requires_grad_(True)
        timings = {}
        for name, fn in (("stacked (vmap)", run_stacked), ("eight in a loop", run_loop)):
            fn()
            sync()
            t0 = time.perf_counter()
            for _ in range(10):
                fn()
            sync()
            timings[name] = (time.perf_counter() - t0) / 10 * 1e3
            print(f"  eight agents fwd+bwd, {name:16} {timings[name]:7.1f} ms", flush=True)
        out["stacked"] = {"width": w, "ms": timings}
    except Exception as exc:  # noqa: BLE001
        out["stacked"] = {"error": f"{type(exc).__name__}: {exc}"[:800]}
        print(f"  stacked: FAILED {out['stacked']['error']}", flush=True)
    return out


@app.function(image=image, gpu="L40S", volumes={VOLUME: volume}, timeout=3 * 3600,
              cpu=12.0, memory=65536)
def opt_bench_l40s(arch: str, hands: int, n_timed: int) -> dict:
    """The result is written to ``/vol/bench/opt-<arch>.json`` from here:
    a local client that dies cancels a ``.remote`` call even detached,
    and the first run of this lost its compile arms that way."""
    import json
    import os

    res = _opt_bench(arch, hands, n_timed)
    os.makedirs(f"{VOLUME}/bench", exist_ok=True)
    with open(f"{VOLUME}/bench/opt-{arch}.json", "w") as fh:
        json.dump(res, fh, indent=2)
    volume.commit()
    return res


@app.local_entrypoint()
def opt_bench(arch: str = "v6", hands: int = 4096, n_timed: int = 60) -> None:
    """``modal run --detach modal_perf.py::opt_bench``: compile, minibatch
    size, stacking. Spawned, so this client may exit; the result lands in
    ``/vol/bench/opt-<arch>.json`` (``modal volume get tractor-rl
    bench/opt-v6.json``) and the lines stream to ``modal app logs``."""
    call = opt_bench_l40s.spawn(arch, hands, n_timed)
    print(f"spawned {call.object_id}; result -> /vol/bench/opt-{arch}.json")


# --- follow-up: the critic's compile failure, its training compile, stacking ----


def _opt_probe(arch: str, hands: int) -> dict:
    """What ``opt_bench`` left open.

    1. The shared critic's inference compile failed ("Expected a number
       but got Identity") where the actor's, through the same static mask,
       did not: the full traceback, then the same compile with the mask
       built without ``torch.arange``.
    2. The shared critic's training step, compiled -- the update arms only
       ran actors.
    3. Eight agents stacked by ``vmap``, all in one mode this time.
    """
    import time
    import traceback
    from dataclasses import replace

    import numpy as np
    import torch
    import torch._dynamo
    import torch.nn.functional as F

    import tractor.nn.model as M
    from tractor.nn.arch import ARCHES
    from tractor.train.buffer import RolloutBuffer
    from tractor.train.gae import compute_gae
    from tractor.train.parallel import RolloutPool
    from tractor.train.population import hand_agents, seating

    torch._dynamo.config.cache_size_limit = 256
    dev = torch.device("cuda")

    def sync() -> None:
        torch.cuda.synchronize()

    base = M.ModelConfig(**ARCHES[arch]["model"])
    actor_cfg = replace(base, critic_kind="none")
    torch.manual_seed(0)
    nets = [M.ActorCritic(actor_cfg).to(dev) for _ in range(8)]
    critic = M.shared_critic(base, 8).to(dev)
    rng = np.random.default_rng(0)
    seat_of = hand_agents(seating(0, 0, 8), hands)
    buf = RolloutBuffer(hands * 100, actor_cfg)
    with RolloutPool(cfg=actor_cfg, workers=10, hands_per_round=hands) as pool:
        buf, _st = pool.collect(nets, rng, buf, n_hands=hands, device=dev, pool=2048,
                                amp=True, level_weight=0.0, agents=seat_of, critic=critic)
    compute_gae(buf, gamma=1.0, lam=0.95)
    n = len(buf)
    out: dict = {"rows": n}

    def bucket(width: int) -> int:
        return min(128, max(16, -(-width // 16) * 16))

    def load(idx):
        obs, priv, sc = buf.as_torch(idx, dev)
        w = bucket(int(buf.history_len[idx].max()))
        return obs._replace(history=obs.history[:, :w].contiguous()), priv, sc

    idx = np.sort(rng.choice(n, size=1536, replace=False))
    obs, priv, sc = load(idx)
    seats = torch.from_numpy(M.relative_seat_agents(
        seat_of[buf.episode[idx]], buf.seat[idx])).to(dev)

    def critic_call():
        return critic(obs.history, obs.history_len, priv, context=obs.context,
                      cards=obs.cards, trick=obs.planes[:, 5:9], seat_agents=seats)

    def timed_inference(fn, label):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), M.static_shapes():
            sync()
            t0 = time.perf_counter()
            fn()
            fn()
            sync()
            first = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(30):
                fn()
            sync()
        ms = (time.perf_counter() - t0) / 30 * 1e3
        print(f"  probe {label:34} {ms:7.2f} ms  first {first:6.1f}s", flush=True)
        return {"ms": ms, "first_s": first}

    critic.eval()
    out["critic_eager"] = timed_inference(critic_call, "critic 1536 eager")

    # 1a. The failure, with its traceback.
    try:
        out["critic_compile"] = timed_inference(
            torch.compile(critic_call, dynamic=False), "critic 1536 compile")
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        out["critic_compile"] = {"error": tb[-6000:]}
        print("  probe critic compile FAILED; traceback tail:", flush=True)
        for line in tb.splitlines()[-60:]:
            print("    | " + line, flush=True)
    torch._dynamo.reset()

    # 1b. The mask without torch.arange: a cumulative sum of ones.
    original = M._length_mask

    def mask_no_arange(lengths, size):
        if M._static():
            idx_ = torch.ones(size, dtype=lengths.dtype, device=lengths.device).cumsum(0) - 1
            return idx_.unsqueeze(0) < lengths.unsqueeze(1)
        return original(lengths, size)

    M._length_mask = mask_no_arange
    try:
        out["critic_compile_no_arange"] = timed_inference(
            torch.compile(critic_call, dynamic=False), "critic 1536 compile, no arange")
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        out["critic_compile_no_arange"] = {"error": tb[-4000:]}
        print("  probe critic compile (no arange) FAILED:", flush=True)
        for line in tb.splitlines()[-30:]:
            print("    | " + line, flush=True)
    finally:
        M._length_mask = original
    torch._dynamo.reset()

    # 2. The critic's training step, eager and compiled.
    def train_step_fn(compiled: bool):
        opt = torch.optim.AdamW(critic.parameters(), lr=3e-4, fused=True)

        def step(o, p, s, target):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = critic(o.history, o.history_len, p, context=o.context, cards=o.cards,
                           trick=o.planes[:, 5:9], seat_agents=s)
            return F.mse_loss(v.float(), target)

        fn = torch.compile(step, dynamic=False) if compiled else step
        mb = [np.sort(rng.choice(n, size=256, replace=False)) for _ in range(40)]
        # Fixed widths: every minibatch trimmed to its bucket.
        loaded = [load(i) for i in mb]
        seats_mb = [torch.from_numpy(M.relative_seat_agents(
            seat_of[buf.episode[i]], buf.seat[i])).to(dev) for i in mb]
        critic.train()
        with M.static_shapes():
            sync()
            t0 = time.perf_counter()
            seen = set()
            for (o, p, s_), s in zip(loaded, seats_mb):
                if o.history.shape[1] in seen:
                    continue
                seen.add(o.history.shape[1])
                loss = fn(o, p, s, s_["ret"].float())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sync()
            first = time.perf_counter() - t0
            t0 = time.perf_counter()
            for (o, p, s_), s in zip(loaded, seats_mb):
                loss = fn(o, p, s, s_["ret"].float())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sync()
        ms = (time.perf_counter() - t0) / len(mb) * 1e3
        label = "compiled" if compiled else "eager"
        print(f"  probe critic train step {label:9} {ms:7.2f} ms/mb  first {first:6.1f}s  "
              f"widths {len(seen)}", flush=True)
        return {"ms_per_minibatch": ms, "first_s": first, "widths": len(seen)}

    for compiled in (False, True):
        key = "critic_train_" + ("compiled" if compiled else "eager")
        try:
            out[key] = train_step_fn(compiled)
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            out[key] = {"error": tb[-4000:]}
            print(f"  probe {key} FAILED:", flush=True)
            for line in tb.splitlines()[-30:]:
                print("    | " + line, flush=True)
        torch._dynamo.reset()

    # 3. Eight agents stacked, all in training mode.
    try:
        from torch.func import functional_call, stack_module_state, vmap

        for net in nets:
            net.train()
        mbs = [np.sort(rng.choice(n, size=256, replace=False)) for _ in range(8)]
        w = max(bucket(int(buf.history_len[i].max())) for i in mbs)
        loaded = [buf.as_torch(i, dev) for i in mbs]
        stacked_obs = M.ObsBatch(*(
            torch.stack([o[0][f][:, :w] if f == 0 else o[0][f] for o in loaded])
            for f in range(len(M.ObsBatch._fields))))
        stacked_priv = torch.stack([o[1] for o in loaded])
        params, buffers = stack_module_state(nets)
        for p in params.values():
            p.requires_grad_(True)
        base_net = nets[0]

        def one(p, b, o, pr):
            out_, _v = functional_call(base_net, (p, b), (M.ObsBatch(*o), pr))
            return out_.log_probs.logsumexp(-1).mean() + out_.value.mean()

        def run_stacked():
            with M.static_shapes(), torch.autocast("cuda", dtype=torch.bfloat16):
                losses = vmap(one, randomness="different")(
                    params, buffers, tuple(stacked_obs), stacked_priv)
            losses.sum().backward()

        def run_loop():
            with M.static_shapes(), torch.autocast("cuda", dtype=torch.bfloat16):
                total = 0.0
                for k, net in enumerate(nets):
                    o = M.ObsBatch(*(t[k] for t in stacked_obs))
                    out_, _v = net(o, stacked_priv[k])
                    total = total + out_.log_probs.logsumexp(-1).mean() + out_.value.mean()
            total.backward()

        timings = {}
        for name, fn in (("stacked (vmap)", run_stacked), ("eight in a loop", run_loop)):
            fn()
            sync()
            t0 = time.perf_counter()
            for _ in range(10):
                fn()
            sync()
            timings[name] = (time.perf_counter() - t0) / 10 * 1e3
            print(f"  probe eight agents fwd+bwd, {name:16} {timings[name]:7.1f} ms", flush=True)
        out["stacked"] = {"width": w, "ms": timings}
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        out["stacked"] = {"error": tb[-4000:]}
        print("  probe stacked FAILED:", flush=True)
        for line in tb.splitlines()[-30:]:
            print("    | " + line, flush=True)
    return out


@app.function(image=image, gpu="L40S", volumes={VOLUME: volume}, timeout=2 * 3600,
              cpu=12.0, memory=65536)
def opt_probe_l40s(arch: str, hands: int) -> dict:
    import json
    import os

    res = _opt_probe(arch, hands)
    os.makedirs(f"{VOLUME}/bench", exist_ok=True)
    with open(f"{VOLUME}/bench/opt-probe-{arch}.json", "w") as fh:
        json.dump(res, fh, indent=2)
    volume.commit()
    return res


@app.local_entrypoint()
def opt_probe(arch: str = "v6", hands: int = 1024) -> None:
    """``modal run --detach modal_perf.py::opt_probe``, spawned; the result
    lands in ``/vol/bench/opt-probe-<arch>.json``."""
    call = opt_probe_l40s.spawn(arch, hands)
    print(f"spawned {call.object_id}; result -> /vol/bench/opt-probe-{arch}.json")
