"""The research stack without Modal: training, search labels and
evaluation on a machine or a Slurm cluster with a shared filesystem.

    python -m tractor.cluster train --root RUNS --name zero2-dst3 --resume CKPT ... [launch's flags]
    python -m tractor.cluster gen   --checkpoint CKPT --out-dir DIR --shard 0 --hands 1250
    python -m tractor.cluster eval  --checkpoint CKPT --bot smart-v34 --out RECORDS.jsonl

Each subcommand is the body of the Modal function it stands in for, with
the volume replaced by ``--root`` (a run lives in ``ROOT/<name>/``:
``checkpoints/``, ``milestones/``, ``metrics.jsonl``):

* ``train`` -- ``modal_train.py::launch`` and ``::_train``: the same flags,
  mapped to the same ``TrainConfig`` overrides. Milestones (every
  ``--milestone-every`` steps) are copied as each checkpoint is written,
  in place of ``::keep_milestones``. Under Slurm the job id stands in for
  Modal's call id, so a requeued job resumes from the run's newest
  checkpoint rather than starting over.
* ``gen`` -- ``modal_perf.py::_distill_gen``: one shard of expert-iteration
  labels (``train/distill.py``).
* ``eval`` -- ``modal_eval.py::_play``: one checkpoint against one bot on
  ``--deals`` duplicate deals, appended to a JSONL record in the layout
  ``modal_eval.py`` writes (so the pairing scripts read both). A pair
  already in the record is skipped.

research/ycrc/ holds the Slurm scripts that call these.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

FAIR = "smart-v18,smart-v28,smart-v34"


# --- train ------------------------------------------------------------------


def _train_parser(sub: Any) -> None:
    p = sub.add_parser("train", help="a training run (modal_train.py::launch)")
    p.add_argument("--root", default=os.environ.get("TRACTOR_ROOT", "runs"))
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--hands", type=int, default=1024)
    p.add_argument("--resume", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--milestone-every", type=int, default=250)
    p.add_argument("--wandb", action="store_true")
    # Everything below is modal_train.py::launch's, with its defaults.
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-deals", type=int, default=96)
    p.add_argument("--eval-bots", default="")
    p.add_argument("--eval-bot-deals", type=int, default=24)
    p.add_argument("--eval-bot-workers", type=int, default=8)
    p.add_argument("--eval-bot-budget-s", type=float, default=600.0)
    p.add_argument("--checkpoint-every", type=int, default=125)
    p.add_argument("--rollout-workers", type=int, default=7)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--buffer-capacity", type=int, default=0)
    p.add_argument("--keep-last", type=int, default=24)
    p.add_argument("--anneal-steps", type=int, default=0)
    p.add_argument("--final-level-weight", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--play-phase-steps", type=int, default=250)
    p.add_argument("--declare-phase-steps", type=int, default=0)
    p.add_argument("--declare-hands", type=int, default=1024)
    p.add_argument("--random-burial-frac", type=float, default=0.2)
    p.add_argument("--bury-rounds", type=int, default=8)
    p.add_argument("--bury-consider", type=int, default=0)
    p.add_argument("--length-buckets", type=int, default=0)
    p.add_argument("--prefetch-starts", action="store_true")
    p.add_argument("--bury-evaluator", default="actor")
    p.add_argument("--bury-policy", action="store_true")
    p.add_argument("--restart-schedule", action="store_true")
    p.add_argument("--population", type=int, default=0)
    p.add_argument("--reshuffle-every", type=int, default=10)
    p.add_argument("--pool", type=int, default=0)
    p.add_argument("--arch", default="v5")
    p.add_argument("--forward-chunks", type=int, default=1)
    p.add_argument("--rollout-overlap", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--cuda-graphs", action="store_true")
    p.add_argument("--rollout-forwards", default="")
    p.add_argument("--fixed-shapes", action="store_true")
    p.add_argument("--target-kl", type=float, default=0.0)
    p.add_argument("--bury-samples", type=int, default=1)
    p.add_argument("--bury-lr", type=float, default=0.0)
    p.add_argument("--bury-epochs", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--minibatch", type=int, default=0)
    p.add_argument("--no-bury-scale-minibatch", dest="bury_scale_minibatch",
                   action="store_false", default=True)
    p.add_argument("--bury-reward", default="critic")
    p.add_argument("--bury-landlords", type=int, default=0)
    p.add_argument("--distill-paths", default="")
    p.add_argument("--distill-coef", type=float, default=1.0)
    p.add_argument("--distill-lr", type=float, default=3e-5)
    p.add_argument("--distill-steps", type=int, default=2)
    p.add_argument("--distill-minibatch", type=int, default=512)
    p.add_argument("--declare-reward", default="critic")


def _bot_list(spec: str) -> tuple[str, ...]:
    from .eval.bots import resolve_bots

    return resolve_bots([b.strip() for b in spec.split(",") if b.strip()]) if spec else ()


def train_overrides(a: argparse.Namespace) -> dict:
    """``TrainConfig`` overrides from launch's flags, exactly as
    ``modal_train.py::launch`` builds them."""
    from .nn.arch import ARCHES

    extra = dict(ARCHES[a.arch])
    eval_every = a.eval_every
    if a.population:
        eval_every = 0
        extra.update(agents=a.population, reshuffle_every=a.reshuffle_every)
    if a.pool:
        extra["pool"] = a.pool
    extra["forward_chunks"] = a.forward_chunks
    extra["rollout_overlap"] = a.rollout_overlap
    extra["rollout_forwards"] = a.rollout_forwards or ("compiled" if a.compile else "eager")
    return {
        **extra,
        "eval_every": eval_every,
        "eval_deals": a.eval_deals,
        "eval_bots": _bot_list(a.eval_bots),
        "eval_bot_deals": a.eval_bot_deals,
        "eval_bot_workers": a.eval_bot_workers,
        "eval_bot_budget_s": a.eval_bot_budget_s,
        "checkpoint_every": a.checkpoint_every,
        "keep_last": a.keep_last,
        "anneal_steps": a.anneal_steps,
        "final_level_weight": a.final_level_weight,
        "rollout_workers": a.rollout_workers,
        "ppo": {"epochs": a.epochs, "length_buckets": a.length_buckets,
                **({"lr": a.lr} if a.lr else {}),
                **({"minibatch": a.minibatch} if a.minibatch else {}),
                "compile": a.compile,
                "cuda_graphs": a.cuda_graphs,
                "fixed_shapes": a.fixed_shapes,
                "target_kl": a.target_kl or None},
        "buffer_capacity": a.buffer_capacity or a.hands * 100,
        "warmup_steps": a.warmup_steps,
        "play_phase_steps": a.play_phase_steps,
        "declare_phase_steps": a.declare_phase_steps,
        "random_burial_frac": a.random_burial_frac,
        "bury_rounds": a.bury_rounds,
        "bury_consider": a.bury_consider,
        "declare": {"hands_per_step": a.declare_hands, "bury_samples": a.bury_samples,
                    "bury_scale_minibatch": a.bury_scale_minibatch,
                    "bury_reward": a.bury_reward, "bury_landlords": a.bury_landlords,
                    "declare_reward": a.declare_reward},
        "bury_train": {k: v for k, v in (("lr", a.bury_lr), ("epochs", a.bury_epochs)) if v},
        "prefetch_starts": a.prefetch_starts,
        "bury_evaluator": a.bury_evaluator,
        "bury_policy": a.bury_policy,
        "restart_schedule": a.restart_schedule,
        "distill": ({"paths": a.distill_paths, "coef": a.distill_coef, "lr": a.distill_lr,
                     "steps": a.distill_steps, "minibatch": a.distill_minibatch}
                    if a.distill_paths else {}),
    }


def _keep_milestone(written: Any, milestones: Path, every: int) -> None:
    """Copy ``written`` to ``milestones/`` when its step is a multiple of
    ``every``. On a local or network filesystem the checkpoint is whole by
    the time the training loop reports it."""
    import re

    path = Path(str(written))
    m = re.fullmatch(r"step_(\d+)\.pt", path.name)
    if m is None or every <= 0 or int(m.group(1)) % every:
        return
    milestones.mkdir(parents=True, exist_ok=True)
    out = milestones / path.name
    if out.exists():
        return
    tmp = out.with_suffix(".tmp")
    shutil.copyfile(path, tmp)
    os.replace(tmp, out)
    print(f"kept milestone {path.name}", flush=True)


def run_train(a: argparse.Namespace) -> int:
    import torch

    from .telemetry import Run
    from .train.loop import TrainConfig, train
    from .train.population import PopulationConfig, train_population
    from .train.relaunch import resolve_resume

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(a.root) / a.name
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    # The Slurm job id plays the part of Modal's call id: a requeued job
    # (same id) resumes from the run's newest checkpoint.
    job = os.environ.get("SLURM_JOB_ID")
    resume = resolve_resume(run_dir, a.resume, f"slurm-{job}" if job else None,
                            overwrite=a.overwrite)
    if resume:
        print(f"resuming from {resume}", flush=True)
    overrides = train_overrides(a)
    Config, fit = ((PopulationConfig, train_population) if overrides.get("agents")
                   else (TrainConfig, train))
    cfg = Config(total_steps=a.steps, hands_per_step=a.hands,
                 checkpoint_dir=str(ckpt), **overrides)
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    enabled = a.wandb and bool(os.environ.get("WANDB_API_KEY"))
    with Run(project="tractor-rl", name=a.name,
             config={**asdict(cfg), "device": device, "gpu": gpu, "slurm_job": job},
             enabled=enabled, jsonl=str(run_dir / "metrics.jsonl")) as run:
        out = fit(cfg, device=device, run=run, resume=resume or None,
                  on_checkpoint=lambda p: _keep_milestone(p, run_dir / "milestones",
                                                          a.milestone_every))
    print(json.dumps({k: v for k, v in out.items() if isinstance(v, (int, float, str))}))
    return 0


# --- gen ----------------------------------------------------------------------


def _gen_parser(sub: Any) -> None:
    p = sub.add_parser("gen", help="one shard of search labels (modal_perf.py::_distill_gen)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard", type=int,
                   default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--hands", type=int, default=1250)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--worlds", type=int, default=8)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--confident", type=float, default=0.99)


def run_gen(a: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from .eval.search import SearchConfig
    from .nn.arch import bury_net_from, declare_net_from
    from .nn.features import Cache
    from .nn.model import ActorCritic, ModelConfig, load_model
    from .train import starts as S
    from .train.distill import generate

    out = Path(a.out_dir) / f"shard_{a.shard:03d}.npz"
    if out.exists():
        print(f"{out} exists; nothing to do", flush=True)
        return 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**blob["model_config"])
    net = ActorCritic(cfg).to(dev)
    load_model(net, blob["model"])
    net.eval()
    declare_net = declare_net_from(blob["declare"]).to(dev).eval()
    bury_net = bury_net_from(blob["bury"]).to(dev).eval()
    step = int(blob.get("step", 0))
    del blob
    rng = np.random.default_rng([a.seed, a.shard])
    cache = Cache()
    t0 = time.perf_counter()
    pre = S.PrePlayConfig(random_burial_frac=0.0, burial="policy")
    starts, _rows, _st = S.make_starts(rng, a.hands, net, declare_net, dev, pre=pre,
                                       cache=cache, bury_net=bury_net)
    rows, stats = generate(net, starts, SearchConfig(worlds=a.worlds, top_k=a.top_k,
                                                     confident=a.confident),
                           rng, dev, cfg, cache)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    rows.save(str(tmp))
    os.replace(tmp, out)
    print(json.dumps({"checkpoint": a.checkpoint, "step": step, "shard": a.shard,
                      "hands": a.hands, "rows": len(rows),
                      "seconds": time.perf_counter() - t0, **stats}), flush=True)
    return 0


# --- eval ---------------------------------------------------------------------


def _eval_parser(sub: Any) -> None:
    p = sub.add_parser("eval", help="one checkpoint against bots (modal_eval.py::_play)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bots", default=FAIR)
    p.add_argument("--deals", type=int, default=770)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    p.add_argument("--pre-play", default="agent-policy")
    p.add_argument("--search-worlds", type=int, default=0)
    p.add_argument("--search-top", type=int, default=3)
    p.add_argument("--search-extra-tricks", type=int, default=0)
    p.add_argument("--search-confident", type=float, default=0.9)
    p.add_argument("--candidates", default="host")
    p.add_argument("--out", required=True)


def _load_net(checkpoint: str, device: str) -> tuple[Any, Any]:
    from .nn.model import ActorCritic, ModelConfig, load_model
    from .train.population import read_checkpoint

    blob = read_checkpoint(checkpoint)
    cfg = ModelConfig(**blob["model_config"])
    net = ActorCritic(cfg)
    load_model(net, blob["model"])
    return net.to(device), cfg


def play_pair(checkpoint: str, bot: str, deals: int, seed: int, workers: int,
              pre_play: str, search: dict | None, device: str,
              candidates: str = "host") -> dict:
    """``modal_eval.py::_play`` without the volume: the same arguments,
    the same record."""
    import torch

    from .eval.bots import play_against_bot
    from .eval.search import SearchConfig
    from .sim.bury import BuryConfig

    torch.set_num_threads(1)
    net, cfg = _load_net(checkpoint, device)
    net.eval()
    host_pre = "agent" if pre_play.startswith("agent") else pre_play
    declare_net = bury_net = None
    if pre_play in ("agent", "agent-heur", "agent-critic", "agent-policy"):
        from .nn.arch import bury_net_from, declare_net_from
        from .train.population import read_checkpoint

        blob = read_checkpoint(checkpoint)
        if blob.get("declare") is not None:
            declare_net = declare_net_from(blob["declare"]).to(device).eval()
        if pre_play == "agent-policy":
            if blob.get("bury") is None:
                raise ValueError(f"{checkpoint} carries no burial policy")
            bury_net = bury_net_from(blob["bury"]).to(device).eval()
    started = time.perf_counter()
    stats = play_against_bot(
        net, bot, device=device, cfg=cfg, num_deals=deals, seed=seed, workers=workers,
        per_hand=True, pre_play=host_pre,
        search=SearchConfig(**search) if search else None,
        declare_net=declare_net, bury_net=bury_net,
        bury=(BuryConfig(evaluator="critic", worlds=8, rounds=8)
              if pre_play == "agent-critic" else
              BuryConfig(rounds=0 if pre_play == "agent-heur" else 8, consider=12)),
        candidates=candidates)
    stats.update(seconds=time.perf_counter() - started, checkpoint=checkpoint, bot=bot,
                 pre_play=pre_play, search=search, candidates=candidates,
                 declare_net=declare_net is not None, seed=seed)
    return stats


def run_eval(a: argparse.Namespace) -> int:
    import torch

    search = ({"worlds": a.search_worlds, "top_k": a.search_top,
               "extra_tricks": a.search_extra_tricks, "confident": a.search_confident}
              if a.search_worlds > 0 else None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    done: set[tuple[str, str]] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if (r["deals"] == a.deals and r.get("seed") == a.seed
                    and r.get("pre_play") == a.pre_play and r.get("search") == search
                    and r.get("candidates", "host") == a.candidates):
                done.add((r["checkpoint"], r["bot"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    for bot in [b.strip() for b in a.bots.split(",") if b.strip()]:
        if (a.checkpoint, bot) in done:
            print(f"{bot}: already measured", flush=True)
            continue
        r = play_pair(a.checkpoint, bot, a.deals, a.seed, a.workers, a.pre_play, search,
                      device, a.candidates)
        with out.open("a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(f"{bot:10} win {r['winrate']:.3f}  levels {r['levels']:+.3f}  "
              f"points {r['points']:+.2f}  {r['seconds'] / 60:.1f} min", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tractor.cluster", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _train_parser(sub)
    _gen_parser(sub)
    _eval_parser(sub)
    a = ap.parse_args(argv)
    return {"train": run_train, "gen": run_gen, "eval": run_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
