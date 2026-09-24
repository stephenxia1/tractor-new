"""Self-play training on Modal.

A thin wrapper: everything that is actually training lives in
``tractor.train.loop`` so the same code runs under pytest locally. This
file supplies a GPU, a volume for checkpoints, and the wandb secret.

    modal run modal_train.py::smoke                 # a few steps, cheap
    modal run modal_train.py::main --steps 2000
    modal run modal_train.py::launch --steps 2500 --eval-bots smart-v34

`--eval-bots` is a comma-separated list of engine bot names. It turns on
the duplicate-deal match of `tractor.eval.bots`, which is why this
image carries Node and the npm workspace as well as torch.
"""

from __future__ import annotations

import os
import pathlib

import modal

LOCAL_REPO = pathlib.Path(__file__).resolve().parent.parent
REPO = "/repo"
RESEARCH = f"{REPO}/research"
VOLUME = "/vol"

# Overridable because two `modal run`s of one app stop each other: a
# smoke run started beside a live training run needs a name of its own.
app = modal.App(os.environ.get("TRACTOR_TRAIN_APP", "tractor-train"))
volume = modal.Volume.from_name("tractor-rl", create_if_missing=True)

# Node as well as torch, because `eval_bots` measures the network against
# the engine's heuristic bots and those are TypeScript: `tractor.eval.bots`
# spawns `npx tsx research/src/evalhost.ts`, which needs the npm workspace
# installed. The layer order mirrors research/modal_app.py -- manifests,
# `npm ci`, then source -- so editing a source file rebuilds in seconds
# instead of re-running the install.
#
# A Python-only image would still train; it would fail at the first
# scheduled evaluation, hours in, which is the wrong place to find out.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y --no-install-recommends nodejs",
    )
    .pip_install("numpy>=1.26", "torch==2.6.0", "wandb>=0.17")
    # Every workspace member's package.json must be present for `npm ci`,
    # even the ones training never imports.
    .add_local_file(LOCAL_REPO / "package.json", f"{REPO}/package.json", copy=True)
    .add_local_file(LOCAL_REPO / "package-lock.json",
                    f"{REPO}/package-lock.json", copy=True)
    .add_local_file(LOCAL_REPO / "tsconfig.base.json",
                    f"{REPO}/tsconfig.base.json", copy=True)
    .add_local_file(LOCAL_REPO / "apps/ws/package.json",
                    f"{REPO}/apps/ws/package.json", copy=True)
    .add_local_file(LOCAL_REPO / "apps/web/package.json",
                    f"{REPO}/apps/web/package.json", copy=True)
    .add_local_file(LOCAL_REPO / "packages/engine/package.json",
                    f"{REPO}/packages/engine/package.json", copy=True)
    .add_local_file(LOCAL_REPO / "packages/protocol/package.json",
                    f"{REPO}/packages/protocol/package.json", copy=True)
    .run_commands(f"cd {REPO} && npm ci --include=dev")
    .add_local_dir(
        LOCAL_REPO / "packages/engine", f"{REPO}/packages/engine", copy=True,
        ignore=["**/node_modules", "**/dist", "**/__pycache__", "**/*.pt", "python"],
    )
    .add_local_dir(
        LOCAL_REPO / "packages/protocol", f"{REPO}/packages/protocol", copy=True,
        ignore=["**/node_modules", "**/dist"],
    )
    .add_local_dir(
        LOCAL_REPO / "research",
        RESEARCH,
        copy=True,
        ignore=["**/node_modules", "**/__pycache__", "**/*.pt",
                "**/fixtures/games.jsonl", "checkpoints", "runs", "wandb"],
    )
    .env({"PYTHONPATH": f"{RESEARCH}/py", "PYTHONUNBUFFERED": "1"})
    .workdir(RESEARCH)
    # Fail on a broken image in seconds rather than an hour into a run.
    # Only re-runs when a layer above it changes.
    .run_commands(f"cd {RESEARCH} && npx tsx src/dupcheck.ts --hands 3 --games 1")
)


def _bot_list(spec: str) -> tuple[str, ...]:
    """`--eval-bots smart-v34,smart-v18` -> a tuple. Empty means off."""
    return tuple(name.strip() for name in spec.split(",") if name.strip())

# `Secret.from_name` is lazy, so a wrong name raises at deploy time and
# takes down every function in the app, not only the ones that wanted
# it. Hydrating here turns that into a local check with a clear answer.
# Absent, the run still trains and still writes JSONL to the volume:
# `Run` degrades rather than failing, so a missing secret costs charts
# and not the run.
def _wandb_secret() -> list:
    for candidate in ("wandb-secret", "wandb"):
        try:
            secret = modal.Secret.from_name(candidate)
            secret.hydrate()
            return [secret]
        except Exception:  # pragma: no cover - depends on the workspace
            continue
    print("modal_train: no wandb secret found; metrics go to JSONL only")
    return []


WANDB = _wandb_secret()


def _train(steps: int, hands: int, name: str, gpu_label: str,
           wandb_on: bool, resume: str, overrides: dict) -> dict:
    import os

    import torch

    from dataclasses import asdict

    from tractor.telemetry import Run
    from tractor.train.loop import TrainConfig, train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = f"{VOLUME}/{name}/checkpoints"
    os.makedirs(ckpt, exist_ok=True)

    cfg = TrainConfig(
        total_steps=steps,
        hands_per_step=hands,
        checkpoint_dir=ckpt,
        **overrides,
    )
    enabled = wandb_on and bool(os.environ.get("WANDB_API_KEY"))
    with Run(
        project="tractor-rl",
        name=name,
        config={**asdict(cfg), "device": device, "gpu": gpu_label},
        enabled=enabled,
        jsonl=f"{VOLUME}/{name}/metrics.jsonl",
    ) as run:
        def _flush(path) -> None:
            # Modal volumes persist on an explicit commit, so a long
            # run that only committed on return would keep nothing if
            # it were interrupted at hour seventeen.
            try:
                volume.commit()
            except Exception as exc:  # pragma: no cover
                print(f"modal_train: volume commit failed ({exc})")

        out = train(cfg, device=device, run=run,
                    resume=resume or None, on_checkpoint=_flush)
    volume.commit()
    out["device"] = device
    return out


@app.function(image=image, gpu="A10G", volumes={VOLUME: volume},
              secrets=WANDB, timeout=86400, cpu=8.0, memory=32768)
def train_a10g(steps: int, hands: int, name: str, wandb_on: bool,
               resume: str, overrides: dict) -> dict:
    return _train(steps, hands, name, "A10G", wandb_on, resume, overrides)


@app.function(image=image, gpu="A100-40GB", volumes={VOLUME: volume},
              secrets=WANDB, timeout=86400, cpu=16.0, memory=65536)
def train_a100(steps: int, hands: int, name: str, wandb_on: bool,
               resume: str, overrides: dict) -> dict:
    return _train(steps, hands, name, "A100-40GB", wandb_on, resume, overrides)


@app.function(image=image, gpu="L40S", volumes={VOLUME: volume},
              secrets=WANDB, timeout=86400, cpu=12.0, memory=65536)
def train_l40s(steps: int, hands: int, name: str, wandb_on: bool,
               resume: str, overrides: dict) -> dict:
    """One step up from the A10G, and the step the bottleneck asks for.

    A training step is about 82% accelerator: of 49.5 s measured at
    1024 hands, the rollout's forward is 8.9 s and the update 31.8 s,
    against 8.8 s of worker packing and transport. Both of the large
    terms are the same network in bf16, where this card is nominally
    about 2.9x the A10G, so it is the term worth buying down. The
    twelve cores keep seven rollout workers fed beside it.
    """
    return _train(steps, hands, name, "L40S", wandb_on, resume, overrides)


@app.function(image=image, gpu="A10G", volumes={VOLUME: volume},
              secrets=WANDB, timeout=3600, cpu=8.0, memory=32768)
def train_smoke() -> dict:
    """Three steps on a handful of hands: proves the loop runs on a GPU."""
    return _train(3, 4, "smoke", "A10G", False, "",
                  {"buffer_capacity": 2048, "checkpoint_every": 0,
                   "ppo": {"epochs": 1, "minibatch": 64}})


@app.local_entrypoint()
def smoke() -> None:
    import json

    print(json.dumps(train_smoke.remote(), indent=2, default=str))


@app.local_entrypoint()
def launch(steps: int = 2500, hands: int = 1024, name: str = "run",
           gpu: str = "l40s", wandb_on: bool = True, resume: str = "",
           eval_every: int = 250, eval_deals: int = 96,
           eval_bots: str = "", eval_bot_deals: int = 24,
           eval_bot_workers: int = 8, eval_bot_budget_s: float = 600.0,
           checkpoint_every: int = 125, rollout_workers: int = 7,
           epochs: int = 1, buffer_capacity: int = 0,
           keep_last: int = 24, anneal_steps: int = 0,
           final_level_weight: float = 0.0, warmup_steps: int = 0,
           play_phase_steps: int = 250, declare_phase_steps: int = 0,
           declare_hands: int = 1024, random_burial_frac: float = 0.2,
           bury_rounds: int = 8, bury_consider: int = 0,
           length_buckets: int = 0, prefetch_starts: bool = False,
           bury_evaluator: str = "actor", bury_policy: bool = False,
           restart_schedule: bool = False) -> None:
    """Submit a run and return, rather than waiting on it.

    The defaults for `hands`, `epochs` and `rollout_workers` come from
    measurement rather than taste. All three were taken by holding a
    quarter of the *hands* out of the fit and scoring the critic on
    them: `modal run modal_train.py::critic_hands`.

    `epochs` is 1. Four passes over a round is where the critic stops
    generalising, and the effect is largest exactly where the reward
    used to end up. At a level weight of 1 the critic explained +0.94
    of the variance on the hands it was fitted to and -0.32 on hands it
    was not, which is worse than predicting the mean; two epochs gave
    -0.23 and one +0.03. On the points-only reward the penalty is
    smaller but has not gone away: paired on identical hands at 1024 a
    round, one epoch holds out at +0.473 against +0.424 for four, and
    is better calibrated (slope 0.95 against 0.79).

    `hands` is 1024 because that is where more hands stop paying. The
    critic's problem is independent targets -- with an end-loaded
    reward a hand is about one target per seat however many rows it
    writes -- and held-out explained variance runs +0.023, +0.404,
    +0.378, +0.473, +0.479 across 128, 256, 512, 1024 and 2048 hands.
    The rise stops at 1024, where the calibration slope also peaks at
    0.95, and 2048 buys +0.006 for twice the rollout.

    `rollout_workers` is 7 because the rollout is host-bound and the
    container reserves eight cores. Measured on an A10G, a persistent
    pool of 7 runs a round 2.28x faster than the single-process
    rollout at 128 hands and 2.40x at 512.

    `main` waits for the function to finish, which ties a run measured
    in hours to the lifetime of the terminal that started it: the
    client cancels its input when it is interrupted, and it does so
    even under `--detach`, which stops the app from being torn down but
    does not stop the client from asking. `spawn` hands the work to
    Modal and exits, so nothing local can end it. Watch it with
    `modal app logs`, or on the run's Weights and Biases page.
    """
    fn = {"a10g": train_a10g, "a100": train_a100,
          "l40s": train_l40s}[gpu.lower()]
    call = fn.spawn(
        steps,
        hands,
        name,
        wandb_on,
        resume,
        {
            "eval_every": eval_every,
            "eval_deals": eval_deals,
            "eval_bots": _bot_list(eval_bots),
            "eval_bot_deals": eval_bot_deals,
            "eval_bot_workers": eval_bot_workers,
            "eval_bot_budget_s": eval_bot_budget_s,
            "checkpoint_every": checkpoint_every,
            "keep_last": keep_last,
            "anneal_steps": anneal_steps,
            "final_level_weight": final_level_weight,
            "rollout_workers": rollout_workers,
            # `length_buckets` 4 took a 1024-hand update from 9.7 s to
            # 7.9 s on an L40S with the same losses and KL (modal_perf.py).
            "ppo": {"epochs": epochs, "length_buckets": length_buckets},
            # A hand writes at most 100 rows, and a round that cannot
            # fit every hand silently collects fewer of them.
            "buffer_capacity": buffer_capacity or hands * 100,
            # The full hand (``declare_phase_steps`` > 0): see TrainConfig.
            "warmup_steps": warmup_steps,
            "play_phase_steps": play_phase_steps,
            "declare_phase_steps": declare_phase_steps,
            "random_burial_frac": random_burial_frac,
            "bury_rounds": bury_rounds,
            "bury_consider": bury_consider,
            "declare": {"hands_per_step": declare_hands},
            "prefetch_starts": prefetch_starts,
            "bury_evaluator": bury_evaluator,
            "bury_policy": bury_policy,
            "restart_schedule": restart_schedule,
        },
    )
    print(f"submitted {name}: call {call.object_id}")
    print(f"  logs:  modal app logs {app.app_id}")
    print(f"  wandb: https://wandb.ai/xia-stephen-1/tractor-rl/runs/{name}")


@app.local_entrypoint()
def main(steps: int = 1000, hands: int = 64, name: str = "run",
         gpu: str = "a10g", wandb_on: bool = True, resume: str = "",
         eval_every: int = 0, eval_deals: int = 64, eval_bots: str = "",
         eval_bot_deals: int = 24, eval_bot_workers: int = 8,
         eval_bot_budget_s: float = 600.0,
         checkpoint_every: int = 50, keep_last: int = 3,
         rollout_workers: int = 1, epochs: int = 4,
         buffer_capacity: int = 0,
         anneal_steps: int = 0, final_level_weight: float = 0.0) -> None:
    import json

    fn = {"a10g": train_a10g, "a100": train_a100,
          "l40s": train_l40s}[gpu.lower()]
    overrides = {
        "eval_every": eval_every,
        "eval_deals": eval_deals,
        "eval_bots": _bot_list(eval_bots),
        "checkpoint_every": checkpoint_every,
        "keep_last": keep_last,
        "anneal_steps": anneal_steps,
        "final_level_weight": final_level_weight,
        "rollout_workers": rollout_workers,
        "ppo": {"epochs": epochs},
        "buffer_capacity": buffer_capacity or hands * 100,
    }
    print(json.dumps(
        fn.remote(steps, hands, name, wandb_on, resume, overrides),
        indent=2, default=str))


# --- critic generalisation ------------------------------------------------


def _load_net(checkpoint: str, device: str):
    """``(net, model_cfg)`` from a checkpoint path, or a fresh net."""
    import torch

    from tractor.nn.model import ActorCritic, ModelConfig, load_model

    if not checkpoint:
        cfg = ModelConfig()
        return ActorCritic(cfg).to(device), cfg
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**blob["model_config"])
    net = ActorCritic(cfg)
    load_model(net, blob["model"])
    return net.to(device), cfg


def _critic_sweep(checkpoint: str, hands: int, holdout: float, seed: int,
                  pool: int, arms: list) -> list:
    """Score one critic fit per arm, sharing rollouts wherever possible.

    Arms are grouped so that the expensive part is done once per group
    rather than once per arm, and so that arms within a group are
    *paired* -- they see the same hands, so a difference between them is
    the parameter and not the deal.

    ``level_weight`` changes the rewards the rollout stores, so it needs
    its own rollout. ``gamma`` enters only through ``compute_gae``,
    which rewrites the returns in place, so every arm that differs only
    in gamma shares one. Everything else is a property of the fit and
    shares both.
    """
    import json
    import time

    import torch

    from tractor.train.critic_diag import collect_round, fit_and_score
    from tractor.train.gae import compute_gae
    from tractor.train.ppo import PPOConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, model_cfg = _load_net(checkpoint, device)
    print(f"critic_sweep: device={device} checkpoint={checkpoint or '(fresh)'} "
          f"hands={hands} holdout={holdout} arms={len(arms)}", flush=True)

    by_lw: dict = {}
    for arm in arms:
        by_lw.setdefault(float(arm["level_weight"]), []).append(arm)

    rows = []
    for lw, group in sorted(by_lw.items()):
        t0 = time.perf_counter()
        round_ = collect_round(
            net, device=device, level_weight=lw, model_cfg=model_cfg,
            n_hands=hands, gamma=1.0, seed=seed, pool=pool,
        )
        print(f"  collected {round_.hands} hands at level_weight={lw:.2f} "
              f"({len(round_.buffer)} rows, {time.perf_counter() - t0:.0f}s)",
              flush=True)

        by_gamma: dict = {}
        for arm in group:
            by_gamma.setdefault(float(arm["gamma"]), []).append(arm)

        for gamma, sub in sorted(by_gamma.items()):
            # In place, so every arm below reads the returns for *this*
            # gamma and the label on the row comes from the same place.
            compute_gae(round_.buffer, gamma=gamma, lam=round_.lam)
            round_.gamma = gamma
            for arm in sub:
                cfg = PPOConfig(
                    epochs=int(arm["epochs"]),
                    detach_critic=bool(arm["detach"]),
                )
                t1 = time.perf_counter()
                gap = fit_and_score(
                    net, round_, device=device, ppo_cfg=cfg,
                    holdout=holdout, seed=seed,
                )
                row = gap.as_dict()
                row["seconds"] = time.perf_counter() - t1
                rows.append(row)
                print(
                    f"    lw={gap.level_weight:.2f} gamma={gap.gamma:.3f} "
                    f"epochs={gap.epochs} detach={int(gap.detached)} -> "
                    f"fit {gap.ev_fit:+.3f}  held {gap.ev_held:+.3f} "
                    f"[{gap.ev_held_lo:+.3f},{gap.ev_held_hi:+.3f}]  "
                    f"gap {gap.gap:+.3f}  slope {gap.slope_held:+.2f}  "
                    f"({row['seconds']:.0f}s)",
                    flush=True,
                )

    out = f"{VOLUME}/critic_sweep.json"
    try:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        volume.commit()
        print(f"critic_sweep: wrote {out}", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"critic_sweep: could not write {out} ({exc})", flush=True)
    return rows


@app.function(image=image, gpu="A10G", volumes={VOLUME: volume},
              timeout=14400, cpu=8.0, memory=32768)
def critic_sweep_a10g(checkpoint: str, hands: int, holdout: float, seed: int,
                      pool: int, arms: list) -> list:
    return _critic_sweep(checkpoint, hands, holdout, seed, pool, arms)


@app.local_entrypoint()
def critic(checkpoint: str = "", hands: int = 256, holdout: float = 0.25,
           seed: int = 4242, pool: int = 128) -> None:
    """Measure the critic's generalisation gap across the anneal and the fixes.

        modal run modal_train.py::critic --checkpoint /vol/<run>/checkpoints/final.pt

    Two blocks. The first sweeps ``level_weight`` at the settings every
    run so far used, which is the prediction worth testing: with
    ``gamma = 1`` the terminal level term is shared by every row of a
    hand, so as the anneal moves weight on to it the returns inside a
    hand converge and the effective sample size falls. If that is the
    mechanism, the gap widens from left to right.

    The second block holds ``level_weight`` at 1.0 -- the end of the
    anneal, where the problem should be worst -- and varies one thing at
    a time.
    """
    anneal = [
        {"level_weight": lw, "gamma": 1.0, "epochs": 4, "detach": False}
        for lw in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    fixes = [
        {"level_weight": 1.0, "gamma": g, "epochs": e, "detach": d}
        for (g, e, d) in (
            (0.99, 4, False),  # discount alone
            (1.0, 2, False),   # half the epochs
            (1.0, 1, False),   # one epoch
            (1.0, 4, True),    # sever the trunk alone
            (0.99, 1, True),   # everything at once
        )
    ]
    # The (1.0, 4, False) arm is already in `anneal` at lw=1.0 and is
    # the baseline both blocks are read against; asking for it twice
    # would only measure the seed.
    rows = critic_sweep_a10g.remote(
        checkpoint, hands, holdout, seed, pool, anneal + fixes
    )
    print()
    print(f"{'lw':>5} {'gamma':>6} {'ep':>3} {'det':>4} {'fit':>7} {'held':>7} "
          f"{'95% CI':>17} {'gap':>7} {'slope':>6} {'targets':>8}")
    for r in rows:
        ci = f"[{r['ev_held_lo']:+.3f},{r['ev_held_hi']:+.3f}]"
        print(f"{r['level_weight']:5.2f} {r['gamma']:6.3f} {int(r['epochs']):3d} "
              f"{int(r['detached']):4d} {r['ev_fit']:+7.3f} {r['ev_held']:+7.3f} "
              f"{ci:>17} {r['gap']:+7.3f} {r['slope_held']:+6.2f} "
              f"{int(r['targets_fit']):8d}")


# --- rollout throughput ---------------------------------------------------


def _rollout_bench(checkpoint: str, hands: int, pool: int, worker_counts: list,
                   seed: int, rounds: int) -> list:
    """Time ``collect`` against a persistent ``RolloutPool``.

    Both arms run under bf16 autocast, because that is what training
    uses and the forward is a third of a serial round: measuring the
    parallel arm against an fp32 serial baseline would flatter it.

    Each pool plays one round that is thrown away before the timed
    ones. That round pays the worker start-up -- about 9.5 s for one
    worker, 12.3 s for seven, nearly all of it ``import torch`` -- which
    a training run pays once and a per-round measurement would
    otherwise charge to every arm. The start-up is reported on its own
    rather than hidden.
    """
    import os
    import time

    import numpy as np
    import torch

    from tractor.telemetry import Spans
    from tractor.train.buffer import RolloutBuffer
    from tractor.train.parallel import RolloutPool
    from tractor.train.rollout import collect
    from tractor.train.shared import plan

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, cfg = _load_net(checkpoint, device)
    amp = device == "cuda" and torch.cuda.is_bf16_supported()

    def rss_gib() -> float:
        """This process and its children, from the kernel rather than torch."""
        try:
            total = 0
            for path in ["/proc/self/smaps_rollup"]:
                with open(path) as fh:
                    for line in fh:
                        if line.startswith("Rss:"):
                            total += int(line.split()[1]) * 1024
            return total / 2**30
        except Exception:
            return float("nan")

    print(f"rollout_bench: device={device} cores={os.cpu_count()} "
          f"hands={hands} pool={pool} amp={amp} rounds={rounds}", flush=True)
    print(f"  a {hands}-hand round needs a "
          f"{hands * 100 * 28527 / 2**30:.2f} GiB output buffer", flush=True)

    rows = []

    spans = Spans()
    buf = RolloutBuffer(hands * 100, cfg)
    best = None
    for r in range(rounds):
        buf.reset()
        t0 = time.perf_counter()
        stats = collect(net, np.random.default_rng(seed + r), buf,
                        n_hands=hands, device=device, cfg=cfg, pool=pool,
                        amp=amp, spans=spans if r == 0 else None)
        secs = time.perf_counter() - t0
        best = secs if best is None else min(best, secs)
    spans.flush()
    span_totals = {k: float(v["total"]) for k, v in spans.summary().items()}
    serial_s = best
    rows.append({"mode": "serial", "workers": 1, "seconds": serial_s,
                 "decisions": stats["decisions"],
                 "decisions_per_s": stats["decisions"] / serial_s,
                 "mean_batch": stats["mean_batch"], "startup_s": 0.0,
                 **span_totals})
    print(f"  serial      {serial_s:7.2f}s  "
          f"{stats['decisions'] / serial_s:8.0f} dec/s  "
          f"batch {stats['mean_batch']:.0f}", flush=True)
    print(f"              spans { {k: round(v, 2) for k, v in span_totals.items()} }",
          flush=True)

    for workers in worker_counts:
        workers = int(workers)
        t_start = time.perf_counter()
        worker_pool = RolloutPool(cfg=cfg, workers=workers,
                                  hands_per_round=hands)
        # Every worker's range must hold its widest possible share, so
        # the block grows with the *ceiling* of the split, not the
        # total: more workers on the same round costs more memory.
        block_rows = worker_pool.capacity
        try:
            # Thrown away: it pays the start-up and warms the caches.
            warm_t = time.perf_counter()
            worker_pool.collect(net, np.random.default_rng(seed), None,
                                n_hands=hands, device=device, pool=pool,
                                amp=amp)
            startup_s = time.perf_counter() - t_start
            del warm_t

            best_p, pstats = None, None
            pspans = Spans()
            for r in range(rounds):
                t0 = time.perf_counter()
                _out, pstats = worker_pool.collect(
                    net, np.random.default_rng(seed + r), None,
                    n_hands=hands, device=device, pool=pool, amp=amp,
                    spans=pspans if r == 0 else None,
                )
                secs = time.perf_counter() - t0
                best_p = secs if best_p is None else min(best_p, secs)
            pspans.flush()
            parent = {k: round(float(v["total"]), 2)
                      for k, v in pspans.summary().items()}
        finally:
            worker_pool.close()

        block_gib = plan(block_rows, cfg) / 2**30
        rows.append({"mode": "parallel", "workers": workers,
                     "seconds": best_p, "decisions": pstats["decisions"],
                     "decisions_per_s": pstats["decisions"] / best_p,
                     "mean_batch": pstats["mean_batch"],
                     "startup_s": startup_s,
                     "shared_gib": block_gib, "rss_gib": rss_gib(),
                     "speedup": serial_s / best_p,
                     **{f"parent_{k}": v for k, v in parent.items()}})
        print(f"  parallel x{workers:<2d} {best_p:7.2f}s  "
              f"{pstats['decisions'] / best_p:8.0f} dec/s  "
              f"batch {pstats['mean_batch']:.0f}  "
              f"{serial_s / best_p:.2f}x serial  "
              f"(start-up {startup_s:.1f}s; shared block "
              f"{block_gib:.2f} GiB)", flush=True)
        # What the parent did while its workers waited: this is the
        # term that does not fall with the worker count.
        print(f"              parent {parent}  "
              f"sum {sum(parent.values()):.2f}s", flush=True)
    return rows


@app.function(image=image, gpu="A10G", volumes={VOLUME: volume},
              timeout=7200, cpu=8.0, memory=32768)
def rollout_bench_a10g(checkpoint: str, hands: int, pool: int,
                       worker_counts: list, seed: int, rounds: int) -> list:
    return _rollout_bench(checkpoint, hands, pool, worker_counts, seed, rounds)


@app.function(image=image, gpu="A10G", volumes={VOLUME: volume},
              timeout=7200, cpu=16.0, memory=65536)
def rollout_bench_cpu16(checkpoint: str, hands: int, pool: int,
                        worker_counts: list, seed: int, rounds: int) -> list:
    """The same probe with twice the cores beside the same card.

    Modal's cpu reservation is a scheduling weight rather than a cap --
    a container sees the host's core count either way -- so whether the
    extra weight buys anything is a measurement, not a given. The
    rollout is host-bound, so this is the arm most likely to move.
    """
    return _rollout_bench(checkpoint, hands, pool, worker_counts, seed, rounds)


@app.local_entrypoint()
def rollout(checkpoint: str = "", hands: int = 128, pool: int = 128,
            workers: str = "2,4,6,7", seed: int = 77, rounds: int = 3,
            cpus: int = 8) -> None:
    """Serial vs parallel rollout on the GPU box.

        modal run modal_train.py::rollout --checkpoint /vol/<run>/.../final.pt
        modal run modal_train.py::rollout --cpus 16 --workers 7,11,15
    """
    counts = [int(w) for w in workers.split(",") if w.strip()]
    fn = rollout_bench_cpu16 if int(cpus) >= 16 else rollout_bench_a10g
    rows = fn.remote(checkpoint, hands, pool, counts, seed, rounds)
    print()
    print(f"{'mode':10s} {'workers':>7} {'seconds':>8} {'dec/s':>9} "
          f"{'batch':>6} {'speedup':>8} {'shared':>8} {'rss':>7}")
    for r in rows:
        print(f"{r['mode']:10s} {r['workers']:7d} {r['seconds']:8.2f} "
              f"{r['decisions_per_s']:9.0f} {r['mean_batch']:6.0f} "
              f"{r.get('speedup', 1.0):8.2f} "
              f"{r.get('shared_gib', 0.0):7.2f} {r.get('rss_gib', 0.0):6.2f}")


# --- does more data help the critic? --------------------------------------


def _critic_scale(checkpoint: str, hands_list: list, holdout: float,
                  seed: int, pool: int, epochs: int,
                  level_weight: float) -> list:
    """Hold the fit fixed and widen the round.

    The epoch sweep showed the critic memorising its own buffer; one
    epoch stops that but only reaches break-even, which is a statement
    about how many independent targets a round carries rather than
    about the optimiser. With ``gamma = 1`` and an end-loaded reward a
    hand is close to one target per seat however many rows it writes,
    so the lever is hands, and this is where it stops paying.

    ``holdout`` is a fraction, so the held-out set grows with the round
    too: the interval narrows for the same reason the fit improves, and
    the two must not be confused. ``hands_held`` is reported so the
    reader can see which is which.
    """
    import json
    import time

    import torch

    from tractor.train.critic_diag import collect_round, fit_and_score
    from tractor.train.ppo import PPOConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, model_cfg = _load_net(checkpoint, device)
    print(f"critic_scale: device={device} epochs={epochs} "
          f"level_weight={level_weight} holdout={holdout}", flush=True)

    rows = []
    for hands in hands_list:
        hands = int(hands)
        t0 = time.perf_counter()
        round_ = collect_round(
            net, device=device, level_weight=level_weight,
            model_cfg=model_cfg, n_hands=hands, gamma=1.0, seed=seed,
            pool=pool,
        )
        collect_s = time.perf_counter() - t0
        gap = fit_and_score(
            net, round_, device=device,
            ppo_cfg=PPOConfig(epochs=int(epochs)), holdout=holdout, seed=seed,
        )
        row = gap.as_dict()
        row["hands"] = float(hands)
        row["collect_s"] = collect_s
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        print(f"  {hands:5d} hands -> fit {gap.ev_fit:+.3f}  "
              f"held {gap.ev_held:+.3f} "
              f"[{gap.ev_held_lo:+.3f},{gap.ev_held_hi:+.3f}]  "
              f"gap {gap.gap:+.3f}  slope {gap.slope_held:+.2f}  "
              f"targets {gap.targets_fit} fit / {gap.targets_held} held  "
              f"({row['seconds']:.0f}s)", flush=True)
        # Freed before the next, wider round is allocated: two rounds
        # at 2048 hands is 12 GiB of buffer that never coexist.
        del round_

    try:
        with open(f"{VOLUME}/critic_scale.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        volume.commit()
    except Exception as exc:  # pragma: no cover
        print(f"critic_scale: could not write ({exc})", flush=True)
    return rows


@app.function(image=image, gpu="A10G", volumes={VOLUME: volume},
              timeout=14400, cpu=16.0, memory=131072)
def critic_scale_a10g(checkpoint: str, hands_list: list, holdout: float,
                      seed: int, pool: int, epochs: int,
                      level_weight: float) -> list:
    return _critic_scale(checkpoint, hands_list, holdout, seed, pool,
                         epochs, level_weight)


@app.local_entrypoint()
def critic_hands(checkpoint: str = "", hands: str = "128,256,512,1024,2048",
                 holdout: float = 0.25, seed: int = 4242, pool: int = 128,
                 epochs: int = 1, level_weight: float = 0.0) -> None:
    """Where does widening the round stop helping the critic?

        modal run modal_train.py::critic_hands --checkpoint /vol/.../final.pt
    """
    counts = [int(h) for h in hands.split(",") if h.strip()]
    rows = critic_scale_a10g.remote(checkpoint, counts, holdout, seed, pool,
                                    epochs, level_weight)
    print()
    print(f"{'hands':>6} {'targets':>8} {'fit':>7} {'held':>7} "
          f"{'95% CI':>17} {'gap':>7} {'slope':>6} {'collect':>8}")
    for r in rows:
        ci = f"[{r['ev_held_lo']:+.3f},{r['ev_held_hi']:+.3f}]"
        print(f"{int(r['hands']):6d} {int(r['targets_fit']):8d} "
              f"{r['ev_fit']:+7.3f} {r['ev_held']:+7.3f} {ci:>17} "
              f"{r['gap']:+7.3f} {r['slope_held']:+6.2f} "
              f"{r['collect_s']:7.1f}s")


# --- the full-hand pipeline -------------------------------------------------


def _pipeline_bench(checkpoint: str, hands: int, configs: list) -> list:
    """Time dealing, declaring, burying and scoring ``hands`` play starts.

    One row per burial configuration, then one declaration training
    step. ``bury/value_end - bury/value_start`` is what the climb bought,
    in critic units (x400 for points).
    """
    import time

    import numpy as np
    import torch

    from tractor.nn.declare import DeclareNet
    from tractor.sim.bury import BuryConfig
    from tractor.train.declare_loop import DeclareLearner, DeclareTrainConfig
    from tractor.train.starts import PrePlayConfig, critic_values, make_starts

    torch.backends.cuda.matmul.allow_tf32 = True
    net, _cfg = _load_net(checkpoint, "cuda")
    net.eval()
    dnet = DeclareNet().cuda().eval()
    out = []
    for c in configs:
        bury = BuryConfig(
            rounds=c["rounds"], consider=c["consider"],
            group=c.get("group", 16), chunk=c.get("chunk", 4096),
        )
        pre = PrePlayConfig(random_burial_frac=c.get("random", 0.0), bury=bury)
        torch.cuda.synchronize()
        starts, _rows, st = make_starts(
            np.random.default_rng(0), hands, net, dnet, "cuda", pre=pre
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        critic_values(net, starts, "cuda")
        torch.cuda.synchronize()
        out.append({**c, **st, "critic_s": time.perf_counter() - t0})
        print(out[-1], flush=True)
    learner = DeclareLearner(dnet, DeclareTrainConfig(
        hands_per_step=hands,
        pre=PrePlayConfig(random_burial_frac=0.0, bury=BuryConfig(
            rounds=configs[-1]["rounds"], consider=configs[-1]["consider"],
            group=configs[-1].get("group", 16))),
    ), "cuda")
    t0 = time.perf_counter()
    st = learner.step(net, np.random.default_rng(1))
    out.append({"declare_step_s": time.perf_counter() - t0, **st})
    return out


@app.function(image=image, gpu="L40S", volumes={VOLUME: volume},
              timeout=3600, cpu=12.0, memory=65536)
def pipeline_bench_l40s(checkpoint: str, hands: int, configs: list) -> list:
    return _pipeline_bench(checkpoint, hands, configs)


@app.local_entrypoint()
def pipeline(checkpoint: str = "", hands: int = 1024) -> None:
    configs = [
        {"rounds": 8, "consider": 0, "group": 256},
        {"rounds": 16, "consider": 0, "group": 256},
        {"rounds": 8, "consider": 10, "group": 256},
        {"rounds": 8, "consider": 0, "group": 1024},
    ]
    rows = pipeline_bench_l40s.remote(checkpoint, hands, configs)
    print()
    for r in rows:
        if "declare_step_s" in r:
            print(f"declare step: {r['declare_step_s']:.1f}s  rows {r.get('declare/rows', 0):.0f}")
            continue
        gain = (r["bury/value_end"] - r["bury/value_start"]) * 400
        print(f"rounds {r['rounds']} consider {r['consider']:2d} group {r.get('group', 16):3d}: "
              f"declare {r['time/declare_s']:5.1f}s  bury {r['time/bury_s']:5.1f}s  "
              f"critic {r['critic_s']:4.1f}s  climb +{gain:.1f} pts  "
              f"rounds used {r['bury/rounds']:.0f}")
