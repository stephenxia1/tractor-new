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


# --- round two: what is left of an 80 s step ---------------------------------


def _opt_bench2(hands: int, declare_hands: int, more_workers: int, out_path: str | None,
                reps: int = 150, base_workers: int = 10, n_rounds: int = 90,
                mb_rows: int = 256, model_overrides: dict | None = None) -> dict:
    """The costs left in a v6 population step, each against a candidate.

    One container, one round of real rows. Each section is timed on the
    code the run uses today and on the candidate beside it, and the
    result is written after every section so a failure keeps the rest.

    G. Collection at ``base_workers`` and at ``more_workers``: whether the
       rollout is bound by the workers or by the parent.
    S. How many candidate slots are ever filled (``max_candidates`` is
       128): per row, per training minibatch, per rollout round.
    A. The parent's round. Today each agent's rows are gathered on the
       host and uploaded apart, and each agent's forward ends in a sync
       (the sampler reads its log-probs back) before the next agent's
       gather starts. The candidate gathers each worker's contiguous run
       once, uploads once, splits by agent on the device, launches every
       forward and the critic's, and reads back once; then the eight
       agents stacked into one forward (``vmap``); then compiled.
    E. The actor update per minibatch. Today every minibatch is gathered
       on the host from pageable memory -- each upload a stream sync --
       and reads about fifteen scalars back. Against: the agent's rows
       resident on the device (one upload), with and without those
       reads; the host gather on a thread into pinned memory; and, in
       eager mode, a bound on packing the move lists: 16 slots.
    D. The compiled actor update's kernels, by device time.
    F. The shared critic's update, today against resident rows, no reads.
    C. The pre-play updates, per agent: float32 with four reads a
       minibatch (today), then without the reads, then bfloat16, then
       compiled.
    """
    import json
    import time
    import traceback
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch
    import torch._dynamo as dynamo
    import torch.nn.functional as F

    import tractor.nn.model as M
    from tractor.nn.arch import ARCHES
    from tractor.nn.features import PLANE_TRICK, Cache
    from tractor.sim.bury import BuryConfig
    from tractor.sim.state import NUM_SEATS
    from tractor.telemetry import Spans
    from tractor.train.buffer import RolloutBuffer, obs_batch_from_arrays
    from tractor.train.declare_loop import DeclareTrainConfig
    from tractor.train.gae import compute_gae
    from tractor.train.parallel import RolloutPool
    from tractor.train.population import (
        PopulationConfig, agent_model_config, build_agents, build_critic, hand_agents, seating,
    )
    from tractor.train.ppo import _LOSS_KEYS, _minibatches
    from tractor.train.rollout import routed_step
    from tractor.train.starts import PrePlayConfig, critic_values, make_starts

    dynamo.config.cache_size_limit = 256
    dynamo.config.accumulated_cache_size_limit = 4096
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    out: dict = {"hands": hands, "declare_hands": declare_hands, "torch": torch.__version__}

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    def err() -> str:
        return traceback.format_exc()[-3000:]

    def save() -> None:
        if out_path:
            import os

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump(out, fh, indent=2, default=float)
            if out_path.startswith(VOLUME):
                volume.commit()

    def section(name: str, fn) -> None:
        t0 = time.perf_counter()
        print(f"== {name}", flush=True)
        try:
            out[name] = fn()
        except Exception:  # noqa: BLE001
            out[name] = {"error": err()}
            print(f"  {name} FAILED\n{err()[-1500:]}", flush=True)
        out.setdefault("section_seconds", {})[name] = time.perf_counter() - t0
        dynamo.reset()
        if cuda:
            torch.cuda.empty_cache()
        save()

    def cand_bucket(c: int) -> int:
        return min(128, max(16, -(-int(c) // 16) * 16))

    cfg = PopulationConfig(
        hands_per_step=hands, buffer_capacity=hands * 100, declare_phase_steps=1,
        bury_policy=True, pool=4096, rollout_overlap=True,
        ppo={"epochs": 1, "length_buckets": 4, "compile": cuda, "bf16": cuda},
        declare={"hands_per_step": declare_hands},
        **{**ARCHES["v6"], "model": {**ARCHES["v6"]["model"], **(model_overrides or {})}},
    )
    declare_cfg = DeclareTrainConfig(**{
        **cfg.declare,
        "pre": PrePlayConfig(random_burial_frac=0.0, bury=BuryConfig(), burial="policy"),
    })
    torch.manual_seed(0)
    agents = build_agents(cfg, dev, declare_cfg)
    critic_learner = build_critic(cfg, dev)
    shared = critic_learner.critic
    nets = [a.net for a in agents]
    mcfg = agent_model_config(cfg)
    rng = np.random.default_rng(0)
    seat_of = hand_agents(seating(0, 0, 8), hands)
    state = {"buf": RolloutBuffer(hands * 100, mcfg)}

    # -- G. collection, by worker count ---------------------------------------
    def g_collect():
        res = {}
        for workers in (base_workers, more_workers) if more_workers else (base_workers,):
            with RolloutPool(cfg=mcfg, workers=workers, hands_per_round=hands) as pool:
                for _rep in ("warm", "timed"):
                    spans = Spans(enabled=True, device=dev if cuda else "cpu")
                    t0 = time.perf_counter()
                    buf, st = pool.collect(
                        nets, rng, state["buf"], n_hands=hands, device=dev, pool=4096,
                        amp=cuda, level_weight=0.0, agents=seat_of, critic=shared,
                        spans=spans, overlap=True)
                    secs = time.perf_counter() - t0
                    state["buf"] = buf
                parts = {k: round(v["total"], 2) for k, v in spans.summary().items()}
                res[f"{workers} workers"] = {"seconds": secs, "forwards": st["forwards"],
                                             "mean_batch": st["mean_batch"], **parts}
                print(f"  collect {workers:2d} workers {secs:6.1f}s  {parts}", flush=True)
        return res

    section("G_collect", g_collect)
    buf = state["buf"]
    compute_gae(buf, gamma=1.0, lam=0.95)
    n = len(buf)
    owner_all = seat_of[buf.episode[:n], buf.seat[:n]]
    sa_all = M.relative_seat_agents(seat_of[buf.episode[:n]], buf.seat[:n])
    counts = buf.candidate_mask[:n].sum(1)
    out["rows"] = int(n)

    # Emulated rounds: ten workers, each a contiguous run, ~1140 rows as the
    # overlapped rollout at pool 4096 forwards.
    def rounds(k: int, per_worker: int = 114) -> list[np.ndarray]:
        res = []
        for _ in range(k):
            starts = np.sort(rng.integers(0, n - per_worker, size=10))
            res.append(np.concatenate([np.arange(s, s + per_worker) for s in starts]))
        return res

    # -- S. candidate slots in use --------------------------------------------
    def s_slots():
        q = [50, 90, 99, 99.9, 100]
        mb_max = [int(counts[b].max()) for b in
                  _minibatches(rng.permutation(n), buf, 256, 4) if len(b) == 256]
        rd_max = [int(counts[r].max()) for r in rounds(200)]

        def share(xs):
            xs = np.asarray(xs)
            return {str(c): float((xs <= c).mean()) for c in (16, 32, 48, 64, 96, 128)}

        res = {
            "per_row_quantiles": {str(p): float(np.percentile(counts, p)) for p in q},
            "mean_per_row": float(counts.mean()),
            "minibatch_max_at_most": share(mb_max),
            "round_max_at_most": share(rd_max),
            "history_len_quantiles": {str(p): float(np.percentile(buf.history_len[:n], p))
                                      for p in q},
        }
        print(f"  {res}", flush=True)
        return res

    section("S_slots", s_slots)

    fields = ("history", "history_len", "context", "unseen", "planes", "cards",
              "candidates", "candidate_mask")
    non_float = ("history_len", "candidate_mask")

    # -- A. the parent's round -------------------------------------------------
    def a_parent():
        rr = rounds(n_rounds)
        warm = max(1, len(rr) // 3)

        def today(rows):
            def gather(idx):
                r = rows[idx]
                obs = obs_batch_from_arrays(*(getattr(buf, f)[r] for f in fields), device=dev)
                priv = torch.from_numpy(buf.privileged[r].astype(np.float32)).to(dev)
                return obs, priv

            owner = owner_all[rows]
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
                routed_step(nets, owner, gather, 1.0, False, rng, None,
                            lengths=buf.history_len[rows], chunks=1, critic=shared,
                            seat_agents=sa_all[rows])

        def batched(rows, trim: bool, actor_fn, critic_fn, stacked=None):
            runs = np.split(rows, np.nonzero(np.diff(rows) != 1)[0] + 1)

            def cat(arr, *trail):
                return np.concatenate([arr[(slice(r[0], r[-1] + 1),) + trail] for r in runs])

            lengths = cat(buf.history_len)
            width = M.history_bucket(int(lengths.max()))
            k = cand_bucket(counts[rows].max()) if trim else mcfg.max_candidates
            owner = owner_all[rows]
            perm = np.argsort(owner, kind="stable")
            bounds = np.searchsorted(owner[perm], np.arange(len(nets) + 1))
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda), \
                    M.static_shapes():
                obs = obs_batch_from_arrays(
                    cat(buf.history, slice(0, width)), lengths, cat(buf.context),
                    cat(buf.unseen), cat(buf.planes), cat(buf.cards),
                    cat(buf.candidates, slice(0, k)), cat(buf.candidate_mask, slice(0, k)),
                    device=dev)
                priv = torch.from_numpy(cat(buf.privileged)).to(dev).float()
                p = torch.from_numpy(perm).to(dev)
                sa = torch.from_numpy(sa_all[rows][perm]).to(dev)
                obs = M.ObsBatch(*(x[p] for x in obs))
                priv = priv[p]
                sorted_len = lengths[perm]
                lps = []
                if stacked is not None:
                    # Every agent in one forward: each agent's rows padded
                    # to the largest share by repeating its last row.
                    sizes = np.diff(bounds)
                    big = int(sizes.max())
                    pick = np.zeros((len(nets), big), dtype=np.int64)
                    for a in range(len(nets)):
                        if sizes[a]:
                            pick[a] = bounds[a] + np.minimum(np.arange(big), sizes[a] - 1)
                    keep = np.concatenate([a * big + np.arange(sizes[a])
                                           for a in range(len(nets))])
                    pick_d = torch.from_numpy(pick).to(dev)
                    keep_d = torch.from_numpy(keep).to(dev)
                    fn_v, params, buffers = stacked
                    lp_all = fn_v(params, buffers, M.ObsBatch(*(x[pick_d] for x in obs)))
                    lps.append(lp_all.reshape(-1, lp_all.shape[-1])[keep_d])
                for a in range(len(nets) if stacked is None else 0):
                    lo, hi = int(bounds[a]), int(bounds[a + 1])
                    if hi == lo:
                        continue
                    w = M.history_bucket(int(sorted_len[lo:hi].max()))
                    sub = M.ObsBatch(*(x[lo:hi] for x in obs))
                    sub = sub._replace(history=sub.history[:, :w].contiguous())
                    lps.append(actor_fn(nets[a], sub))
                v = critic_fn(obs, priv, sa)
                lp = torch.cat(lps).float().cpu().numpy()
                v = v.float().cpu().numpy()
            slots = (lp + rng.gumbel(size=lp.shape)).argmax(1)
            res_slots = np.empty_like(slots)
            res_slots[perm] = slots
            return res_slots, v

        def actor_eager(net, obs):
            return net.actor(obs).log_probs

        def critic_eager(obs, priv, sa):
            return shared(obs.history, obs.history_len, priv, context=obs.context,
                          cards=obs.cards,
                          trick=obs.planes[:, PLANE_TRICK: PLANE_TRICK + NUM_SEATS],
                          seat_agents=sa)

        def make_stacked():
            import copy

            from torch.func import functional_call, stack_module_state, vmap

            actors = [net.actor for net in nets]
            params, buffers = stack_module_state(actors)
            base = copy.deepcopy(actors[0]).to("meta")
            base.eval()

            def one(p_, b_, o_):
                return functional_call(base, (p_, b_), (o_,)).log_probs

            return vmap(one), params, buffers

        for net in nets:
            net.eval()
        shared.eval()
        t0 = time.perf_counter()
        stacked = make_stacked()
        sync()
        stack_s = time.perf_counter() - t0
        arms = [("today", today),
                ("one gather, one sync", lambda r: batched(r, False, actor_eager, critic_eager)),
                ("one gather, one sync, agents stacked (vmap)",
                 lambda r: batched(r, False, None, critic_eager, stacked))]
        if cuda:
            actor_c = torch.compile(actor_eager, dynamic=True)
            critic_c = torch.compile(critic_eager, dynamic=True)
            arms.append(("... and compiled, dynamic shapes",
                         lambda r: batched(r, False, actor_c, critic_c)))
        res = {"stack_weights_s": stack_s}
        for label, fn in arms:
            try:
                sync()
                t0 = time.perf_counter()
                for r in rr[:warm]:
                    fn(r)
                sync()
                first = time.perf_counter() - t0
                t0 = time.perf_counter()
                for r in rr[warm:]:
                    fn(r)
                sync()
                ms = (time.perf_counter() - t0) / len(rr[warm:]) * 1e3
                res[label] = {"ms_per_round": ms, "first_rounds_s": first, "first_rounds": warm}
                print(f"  A {label:40} {ms:7.1f} ms a round  (first {warm}: {first:6.1f}s)",
                      flush=True)
            except Exception:  # noqa: BLE001
                res[label] = {"error": err()}
                print(f"  A {label} FAILED\n{err()[-1500:]}", flush=True)
        return res

    section("A_parent", a_parent)

    # -- E, D. the actor update -------------------------------------------------
    scalar_keys = ("action", "logp", "value", "reward", "reward_points", "reward_level",
                   "done", "seat", "episode", "advantage", "ret")

    def e_actor():
        res = {}
        agent = agents[0]
        learner = agent.learner
        net = agent.net
        rows0 = np.nonzero(owner_all == 0)[0]
        res["agent0_rows"] = int(len(rows0))
        for rep in ("first (compiles)", "today"):
            sync()
            t0 = time.perf_counter()
            m = learner.update(buf, rows=rows0)
            sync()
            secs = time.perf_counter() - t0
            k = max(m["n_minibatches"], 1)
            res[f"update: {rep}"] = {"seconds": secs, "minibatches": m["n_minibatches"],
                                     "ms_per_minibatch": secs / k * 1e3}
            print(f"  E update {rep:18} {secs:6.1f}s  {secs / k * 1e3:6.2f} ms a minibatch",
                  flush=True)

        # The agent's rows on the device: one gather, one upload.
        allf = fields + ("privileged",) + scalar_keys
        t0 = time.perf_counter()
        host = {f: np.take(getattr(buf, f), rows0, axis=0) for f in allf}
        t_gather = time.perf_counter() - t0

        def threaded_take(src, idx, threads=8):
            dst = np.empty((len(idx),) + src.shape[1:], dtype=src.dtype)
            parts = [p for p in np.array_split(np.arange(len(idx)), threads) if len(p)]
            with ThreadPoolExecutor(threads) as ex:
                list(ex.map(lambda p: np.take(src, idx[p], axis=0, mode="clip",
                                              out=dst[p[0]: p[-1] + 1]), parts))
            return dst

        t0 = time.perf_counter()
        for f in allf:
            threaded_take(getattr(buf, f), rows0)
        t_gather8 = time.perf_counter() - t0
        sync()
        t0 = time.perf_counter()
        R = {f: torch.from_numpy(a).to(dev) for f, a in host.items()}
        sync()
        t_upload = time.perf_counter() - t0
        nbytes = sum(a.nbytes for a in host.values())
        del host
        res["resident"] = {"bytes": nbytes, "gather_s": t_gather,
                           "gather_8_threads_s": t_gather8, "upload_pageable_s": t_upload}
        print(f"  E resident: {nbytes / 2**30:.2f} GiB, gather {t_gather:.2f}s "
              f"(8 threads {t_gather8:.2f}s), upload {t_upload:.2f}s", flush=True)

        order = rows0[rng.permutation(len(rows0))]
        mbs = [b for b in _minibatches(order, buf, mb_rows, 4) if len(b) == mb_rows][:reps]
        widths = [M.history_bucket(int(buf.history_len[b].max())) for b in mbs]
        pos_all = torch.from_numpy(np.stack([np.searchsorted(rows0, b) for b in mbs])).to(dev)

        def resident_batch(i, slots=None):
            pos = pos_all[i]
            w = widths[i]
            k = slots or mcfg.max_candidates
            obs = M.ObsBatch(
                history=R["history"][pos, :w].float(),
                history_len=R["history_len"][pos],
                context=R["context"][pos].float(),
                unseen=R["unseen"][pos].float(),
                planes=R["planes"][pos].float(),
                cards=R["cards"][pos].float(),
                candidates=R["candidates"][pos, :k].float(),
                candidate_mask=R["candidate_mask"][pos, :k],
            )
            sc = {k_: R[k_][pos] for k_ in scalar_keys}
            if slots:
                # Timing only: not a policy, a bound on what packing the
                # move lists (median 4 of 128 slots filled) could save.
                sc["action"] = sc["action"].clamp(max=k - 1)
            return obs, R["privileged"][pos].float(), sc

        acc = []

        def train_step(obs, priv, sc, reads: bool, losses=None):
            with M.static_shapes():
                loss, parts = (losses or learner._losses)(obs, priv, sc)
            if reads and not bool(torch.isfinite(loss)):
                return
            learner.opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            if reads:
                bool(torch.isfinite(gn))
            learner.opt.step()
            if reads:
                for key in _LOSS_KEYS:
                    float(parts[key].detach())
                float(gn)
                int(parts["outcome_rows"])
                int(parts["ratio_clamped"])
            else:
                acc.append(parts["total_loss"].detach())

        def timed(label, body):
            net.train()
            body(range(min(8, len(mbs))))
            sync()
            t0 = time.perf_counter()
            body(range(len(mbs)))
            sync()
            ms = (time.perf_counter() - t0) / len(mbs) * 1e3
            acc.clear()
            res[label] = {"ms_per_minibatch": ms}
            print(f"  E {label:44} {ms:7.2f} ms a minibatch", flush=True)

        timed("resident, reads kept",
              lambda it: [train_step(*resident_batch(i), True) for i in it])
        timed("resident, no reads",
              lambda it: [train_step(*resident_batch(i), False) for i in it])

        # The host gather on a thread, into pinned memory, uploaded without
        # a stream sync.
        tdt = {np.dtype(np.float16): torch.float16, np.dtype(np.float32): torch.float32,
               np.dtype(np.int64): torch.int64, np.dtype(bool): torch.bool}

        def host_mb(i):
            idx = mbs[i]
            got = {}
            for f in allf:
                src = buf.history[:, : widths[i]] if f == "history" else getattr(buf, f)
                t = torch.empty((len(idx),) + src.shape[1:], dtype=tdt[src.dtype],
                                pin_memory=cuda)
                np.take(src, idx, axis=0, mode="clip", out=t.numpy())
                got[f] = t
            return got

        def from_pinned(got):
            d = {f: t.to(dev, non_blocking=True) for f, t in got.items()}
            obs = M.ObsBatch(*(d[f] if f in non_float else d[f].float() for f in fields))
            return obs, d["privileged"].float(), {k: d[k] for k in scalar_keys}

        def prefetched(it):
            it = list(it)
            with ThreadPoolExecutor(1) as ex:
                fut = ex.submit(host_mb, it[0])
                for j in range(len(it)):
                    got = fut.result()
                    if j + 1 < len(it):
                        fut = ex.submit(host_mb, it[j + 1])
                    train_step(*from_pinned(got), False)

        timed("host gather on a thread, pinned, no reads", prefetched)

        # How much of the update the 128 padded slots cost: eager at 16.
        eager = learner.losses
        timed("eager, resident, no reads, 128 slots",
              lambda it: [train_step(*resident_batch(i), False, eager) for i in it])
        timed("eager, resident, no reads, 16 slots (a bound)",
              lambda it: [train_step(*resident_batch(i, 16), False, eager) for i in it])

        if cuda:
            from torch.profiler import ProfilerActivity, profile

            net.train()
            sync()
            t0 = time.perf_counter()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for i in range(min(20, len(mbs))):
                    train_step(*resident_batch(i), False)
                sync()
            wall = time.perf_counter() - t0
            ka = prof.key_averages()
            gpu_us = sum(getattr(e, "self_device_time_total", 0) or
                         getattr(e, "self_cuda_time_total", 0) for e in ka)
            try:
                table = ka.table(sort_by="self_device_time_total", row_limit=30,
                                 max_name_column_width=80)
            except Exception:  # noqa: BLE001
                table = ka.table(sort_by="self_cuda_time_total", row_limit=30,
                                 max_name_column_width=80)
            res["D_profile"] = {"wall_s_20_minibatches": wall,
                                "device_busy_s_20_minibatches": gpu_us / 1e6, "table": table}
            print(table, flush=True)
            print(f"  D wall {wall:.2f}s, device busy {gpu_us / 1e6:.2f}s over 20 minibatches",
                  flush=True)
        del R
        return res

    section("E_actor_update", e_actor)

    # -- F. the shared critic's update ------------------------------------------
    def f_critic():
        res = {}
        for rep in ("first (compiles)", "today"):
            sync()
            t0 = time.perf_counter()
            critic_learner.update(buf, sa_all)
            sync()
            secs = time.perf_counter() - t0
            res[f"update: {rep}"] = {"seconds": secs,
                                     "ms_per_minibatch": secs / max(n // 256, 1) * 1e3}
            print(f"  F critic update {rep:18} {secs:6.1f}s", flush=True)
        keep = ("history", "history_len", "privileged", "context", "cards", "planes", "ret")
        sync()
        t0 = time.perf_counter()
        R = {f: torch.from_numpy(getattr(buf, f)[:n]).to(dev) for f in keep}
        sa = torch.from_numpy(sa_all).to(dev)
        sync()
        res["upload_s"] = time.perf_counter() - t0
        res["upload_bytes"] = int(sum(t.numel() * t.element_size() for t in R.values()))
        mbs = [b for b in _minibatches(rng.permutation(n), buf, mb_rows, 4)
               if len(b) == mb_rows][: 4 * reps]
        widths = [M.history_bucket(int(buf.history_len[b].max())) for b in mbs]
        pos_all = torch.from_numpy(np.stack(mbs)).to(dev)
        lrn = critic_learner
        acc = []

        def body(it):
            for i in it:
                pos = pos_all[i]
                inputs = dict(
                    history_enc=R["history"][pos, : widths[i]].float(),
                    history_len=R["history_len"][pos],
                    privileged=R["privileged"][pos].float(),
                    context=R["context"][pos].float(),
                    cards=R["cards"][pos].float(),
                    trick=R["planes"][pos, PLANE_TRICK: PLANE_TRICK + NUM_SEATS].float(),
                    seat_agents=sa[pos],
                )
                with M.static_shapes():
                    loss = lrn._step(inputs, R["ret"][pos].float())
                lrn.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lrn.critic.parameters(), 0.5)
                lrn.opt.step()
                acc.append(loss.detach())

        lrn.critic.train()
        body(range(min(8, len(mbs))))
        sync()
        t0 = time.perf_counter()
        body(range(len(mbs)))
        sync()
        ms = (time.perf_counter() - t0) / len(mbs) * 1e3
        res["resident, no reads"] = {"ms_per_minibatch": ms}
        print(f"  F resident, no reads {ms:6.2f} ms a minibatch "
              f"(upload {res['upload_s']:.2f}s, {res['upload_bytes'] / 2**30:.2f} GiB)",
              flush=True)
        lrn.critic.eval()
        del R
        return res

    section("F_critic_update", f_critic)

    # -- C. the pre-play updates ------------------------------------------------
    def c_preplay():
        res = {}
        for a in agents:
            a.net.eval()
            a.declare.net.eval()
            a.bury.net.eval()
        table = hand_agents(seating(0, 0, 8), declare_hands)
        t0 = time.perf_counter()
        starts, pre_rows, _st = make_starts(
            rng, declare_hands, nets, [a.declare.net for a in agents], dev,
            pre=declare_cfg.pre, record=True, cache=Cache(),
            bury_net=[a.bury.net for a in agents], agents=table)
        res["make_starts_s"] = time.perf_counter() - t0
        values = critic_values(nets, starts, dev, agents=table, shared=shared)
        landlord = np.array([s.landlord for s in starts])
        brow, drow = pre_rows.bury, pre_rows.declare
        b_owner = table[brow.hand, landlord[brow.hand]]
        b_ret = values[brow.hand, landlord[brow.hand]].astype(np.float32) * 10
        d_owner = table[drow.hand, drow.seat]
        d_ret = values[drow.hand, drow.seat].astype(np.float32) * 10
        res["rows_all_agents"] = {"bury": int(len(brow)), "declare": int(len(drow))}

        def t(x):
            return torch.from_numpy(np.ascontiguousarray(x)).to(dev)

        def pre_loss(net, ins, act, olp, adv, ret, mask, amp):
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                _l, logp_all, v = net(*ins)
            logp_all = logp_all.float()
            v = v.float()
            logp = logp_all.gather(1, act[:, None]).squeeze(1)
            ratio = torch.exp(logp - olp)
            pg = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            ent = -(logp_all.exp() * torch.where(mask, logp_all, torch.zeros_like(logp_all))
                    ).sum(1).mean()
            vl = F.mse_loss(v, ret)
            return pg + 0.5 * vl - 0.01 * ent, pg, vl, ent

        def ppo(net, inputs, action, old_logp, value, ret, mask, mb, amp, fn):
            opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
            adv = ret - value
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            k = len(action)
            net.train()
            acc = []
            steps = 0
            for _ in range(4):
                order = torch.randperm(k, device=dev)
                hi = k - mb + 1 if fn is not None else k
                for lo in range(0, max(hi, 1), mb):
                    idx = order[lo: lo + mb]
                    loss, pg, _vl, _ent = (fn or pre_loss)(
                        net, tuple(x[idx] for x in inputs), action[idx], old_logp[idx],
                        adv[idx], ret[idx], mask[idx], amp)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()
                    acc.append(pg.detach())
                    steps += 1
            float(torch.stack(acc).sum())
            net.eval()
            return steps

        compiled_loss = torch.compile(pre_loss, dynamic=False) if cuda else None
        for kind, rows, owner, ret, mb, learner in (
                ("bury", brow, b_owner, b_ret, 1024, agents[0].bury),
                ("declare", drow, d_owner, d_ret, 512, agents[0].declare)):
            sel = np.nonzero(owner == 0)[0]
            sub = rows.subset(sel)
            res[f"{kind}_rows_agent0"] = int(len(sel))
            for _rep in ("warm", "today"):
                sync()
                t0 = time.perf_counter()
                learner.update(sub, ret[sel])
                sync()
                secs = time.perf_counter() - t0
            res[f"{kind}: today (fp32, reads)"] = {"seconds": secs}
            print(f"  C {kind:7} {'today (fp32, reads)':28} {secs:6.2f}s one agent", flush=True)
            if kind == "bury":
                inputs = (t(sub.cards), t(sub.ctx), t(sub.mask), t(sub.privileged))
            else:
                inputs = (t(sub.obs), t(sub.mask), t(sub.privileged))
            args = (t(sub.action), t(sub.logp), t(sub.value), t(ret[sel]), t(sub.mask))
            arms = [("fp32, no reads", False, None, False),
                    ("bf16, no reads", True, None, False)]
            if compiled_loss is not None:
                arms += [("fp32, no reads, compiled", False, compiled_loss, False),
                         ("tf32, no reads, compiled", False, compiled_loss, True),
                         ("bf16, no reads, compiled", True, compiled_loss, False)]
            for label, amp, fn, tf32 in arms:
                was = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = tf32
                try:
                    if fn is not None:
                        t0 = time.perf_counter()
                        ppo(learner.net, inputs, *args, mb, amp, fn)
                        res[f"{kind}: {label}: first call s"] = time.perf_counter() - t0
                    sync()
                    t0 = time.perf_counter()
                    steps = ppo(learner.net, inputs, *args, mb, amp, fn)
                    sync()
                    secs = time.perf_counter() - t0
                    res[f"{kind}: {label}"] = {"seconds": secs, "minibatches": steps}
                    print(f"  C {kind:7} {label:28} {secs:6.2f}s one agent "
                          f"({steps} minibatches)", flush=True)
                except Exception:  # noqa: BLE001
                    res[f"{kind}: {label}"] = {"error": err()}
                    print(f"  C {kind} {label} FAILED\n{err()[-1500:]}", flush=True)
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = was
        return res

    section("C_preplay", c_preplay)
    return out


@app.function(image=image, gpu="L40S", volumes={VOLUME: volume}, timeout=3 * 3600,
              cpu=24.0, memory=98304)
def opt_bench2_l40s(hands: int, declare_hands: int, more_workers: int) -> dict:
    return _opt_bench2(hands, declare_hands, more_workers, f"{VOLUME}/bench/opt2-v6.json")


@app.local_entrypoint()
def opt_bench2(hands: int = 4096, declare_hands: int = 8192, more_workers: int = 20) -> None:
    """``modal run --detach modal_perf.py::opt_bench2``: spawned, so a
    client that dies cannot cancel it; the result is written after each
    section to ``/vol/bench/opt2-v6.json``."""
    call = opt_bench2_l40s.spawn(hands, declare_hands, more_workers)
    print(f"spawned {call.object_id}; result -> /vol/bench/opt2-v6.json")


# --- hardware: a bigger GPU, more CPUs --------------------------------------


def _hw_bench(hands: int, declare_hands: int, worker_counts: list, out_path: str | None,
              requested_cpus: float = 0.0, model_overrides: dict | None = None) -> dict:
    """A population step's parts on the code the run uses, on this container.

    Run on an H100 and an L40S with the same CPU count, the pair separates
    what the GPU buys from what the CPUs buy. On the L40S at 12 CPUs the
    rollout was bound by its parent, not its workers (``opt_bench2``: 10
    and 20 workers both 30.8 s); with the forward compiled the parent is
    busy ~18.5 s of a ~20.7 s collection, so whether the workers now
    bound it is a question for a sweep.

    * collection, compiled forwards, overlap, pool 4096, at each worker
      count: seconds, and the parent's busy time (its spans' sum);
    * the eight agents' updates on the round kept on the device, and the
      shared critic's -- the production path, compiled;
    * one pre-play step (``population._preplay_step``), after a warm one.
    """
    import json
    import os
    import time
    import traceback

    import numpy as np
    import torch

    import tractor.nn.model as M
    from tractor.nn.arch import ARCHES
    from tractor.nn.features import Cache
    from tractor.sim.bury import BuryConfig
    from tractor.telemetry import Spans
    from tractor.train.buffer import DeviceRows, RolloutBuffer
    from tractor.train.declare_loop import DeclareTrainConfig
    from tractor.train.gae import compute_gae
    from tractor.train.parallel import RolloutPool
    from tractor.train.population import (
        PopulationConfig, _preplay_step, agent_model_config, build_agents, build_critic,
        hand_agents, seating,
    )
    from tractor.train.starts import PrePlayConfig

    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count()
    out: dict = {
        "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
        "requested_cpus": requested_cpus, "cpu_affinity": affinity,
        "torch": torch.__version__, "hands": hands, "declare_hands": declare_hands,
    }
    print(out, flush=True)

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    def save() -> None:
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump(out, fh, indent=2, default=float)
            if out_path.startswith(VOLUME):
                volume.commit()

    def section(name: str, fn) -> None:
        t0 = time.perf_counter()
        print(f"== {name}", flush=True)
        try:
            out[name] = fn()
        except Exception:  # noqa: BLE001
            out[name] = {"error": traceback.format_exc()[-3000:]}
            print(f"  {name} FAILED\n{out[name]['error'][-1500:]}", flush=True)
        out.setdefault("section_seconds", {})[name] = time.perf_counter() - t0
        if cuda:
            out.setdefault("peak_gpu_gib", {})[name] = torch.cuda.max_memory_allocated() / 2**30
        save()

    cfg = PopulationConfig(
        hands_per_step=hands, buffer_capacity=hands * 100, declare_phase_steps=1,
        bury_policy=True, pool=4096, rollout_overlap=True,
        rollout_forwards="compiled" if cuda else "static",
        ppo={"epochs": 1, "length_buckets": 4, "compile": cuda, "bf16": cuda},
        declare={"hands_per_step": declare_hands},
        **{**ARCHES["v6"], "model": {**ARCHES["v6"]["model"], **(model_overrides or {})}},
    )
    declare_cfg = DeclareTrainConfig(**{
        **cfg.declare,
        "pre": PrePlayConfig(random_burial_frac=0.0, bury=BuryConfig(), burial="policy"),
    })
    torch.manual_seed(0)
    agents = build_agents(cfg, dev, declare_cfg)
    critic_learner = build_critic(cfg, dev)
    shared = critic_learner.critic
    nets = [a.net for a in agents]
    mcfg = agent_model_config(cfg)
    rng = np.random.default_rng(0)
    seat_of = hand_agents(seating(0, 0, 8), hands)
    state = {"buf": RolloutBuffer(hands * 100, mcfg)}
    parent_spans = ("gather", "forward", "sample", "critic", "writeback", "release")

    def collect_sweep():
        res = {}
        for workers in worker_counts:
            with RolloutPool(cfg=mcfg, workers=workers, hands_per_round=hands) as pool:
                runs = []
                for rep in range(3):  # the first warms (and, once, compiles)
                    spans = Spans(enabled=True, device=dev if cuda else "cpu")
                    t0 = time.perf_counter()
                    buf, st = pool.collect(
                        nets, rng, state["buf"], n_hands=hands, device=dev, pool=4096,
                        amp=cuda, level_weight=0.0, agents=seat_of, critic=shared,
                        spans=spans, overlap=True, forwards=cfg.rollout_forwards)
                    secs = time.perf_counter() - t0
                    state["buf"] = buf
                    parts = {k: v["total"] for k, v in spans.summary().items()}
                    busy = sum(parts.get(k, 0.0) for k in parent_spans)
                    runs.append({"seconds": secs, "parent_busy_s": busy,
                                 "decisions": st["decisions"], **parts})
                    print(f"  collect {workers:2d} workers rep {rep}: {secs:6.1f}s, parent busy "
                          f"{busy:5.1f}s  fwd {parts.get('forward', 0):5.1f}  critic "
                          f"{parts.get('critic', 0):4.1f}  gather {parts.get('gather', 0):4.1f}",
                          flush=True)
            timed = runs[1:]
            res[f"{workers} workers"] = {
                "seconds": float(np.mean([r["seconds"] for r in timed])),
                "parent_busy_s": float(np.mean([r["parent_busy_s"] for r in timed])),
                "runs": runs,
            }
        return res

    section("collect", collect_sweep)
    buf = state["buf"]
    compute_gae(buf, gamma=1.0, lam=0.95)
    n = len(buf)
    out["rows"] = int(n)
    owner = seat_of[buf.episode[:n], buf.seat[:n]]
    sa = M.relative_seat_agents(seat_of[buf.episode[:n]], buf.seat[:n])

    def updates():
        res = {}
        for rep in ("first (compiles)", "timed", "timed again"):
            for a in agents:
                a.net.train()
            sync()
            t0 = time.perf_counter()
            on_device = DeviceRows(buf, dev)
            sync()
            t_up = time.perf_counter() - t0
            t0 = time.perf_counter()
            mbs = 0
            for k, a in enumerate(agents):
                m = a.learner.update(buf, rows=np.nonzero(owner == k)[0], device_rows=on_device)
                mbs += int(m["n_minibatches"])
            sync()
            t_actor = time.perf_counter() - t0
            t0 = time.perf_counter()
            c = critic_learner.update(buf, sa, device_rows=on_device)
            sync()
            t_critic = time.perf_counter() - t0
            del on_device
            res[rep] = {"upload_s": t_up, "actors_s": t_actor, "critic_s": t_critic,
                        "actor_minibatches": mbs,
                        "actor_ms_per_minibatch": t_actor / max(mbs, 1) * 1e3,
                        "critic_value_loss": c["critic/value_loss"]}
            print(f"  update {rep:17} upload {t_up:4.1f}s  actors {t_actor:6.1f}s "
                  f"({t_actor / max(mbs, 1) * 1e3:5.1f} ms/mb)  critic {t_critic:5.1f}s",
                  flush=True)
        return res

    section("updates", updates)

    def preplay():
        res = {}
        table = seating(0, 0, 8)
        for rep in ("warm", "timed"):
            t0 = time.perf_counter()
            st = _preplay_step(agents, table, declare_cfg, rng, dev, Cache(), shared)
            secs = time.perf_counter() - t0
            res[rep] = {"seconds": secs,
                        **{k: v for k, v in st.items() if k.startswith("time/")}}
            print(f"  pre-play {rep:5} {secs:5.1f}s  "
                  f"{ {k: round(v, 2) for k, v in st.items() if k.startswith('time/')} }",
                  flush=True)
        return res

    section("preplay", preplay)
    return out


_HW = dict(volumes={VOLUME: volume}, timeout=3 * 3600, cpu=32.0, memory=131072)


@app.function(image=image, gpu="H100", **_HW)
def hw_bench_h100(hands: int, declare_hands: int, workers: list) -> dict:
    return _hw_bench(hands, declare_hands, workers, f"{VOLUME}/bench/hw-h100.json", 32.0)


@app.function(image=image, gpu="L40S", **_HW)
def hw_bench_l40s(hands: int, declare_hands: int, workers: list) -> dict:
    return _hw_bench(hands, declare_hands, workers, f"{VOLUME}/bench/hw-l40s.json", 32.0)


@app.local_entrypoint()
def hw_bench(hands: int = 4096, declare_hands: int = 8192, workers: str = "10,16,24,30",
             gpus: str = "h100,l40s") -> None:
    """``modal run --detach modal_perf.py::hw_bench``: the same measurement
    on each GPU, 32 CPUs apiece, spawned; results in
    ``/vol/bench/hw-<gpu>.json``."""
    counts = [int(w) for w in workers.split(",")]
    fns = {"h100": hw_bench_h100, "l40s": hw_bench_l40s}
    for g in gpus.split(","):
        call = fns[g].spawn(hands, declare_hands, counts)
        print(f"spawned {g}: {call.object_id}; result -> /vol/bench/hw-{g}.json")
