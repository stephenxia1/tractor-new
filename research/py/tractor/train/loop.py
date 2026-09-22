"""The outer training loop: collect, credit, update, checkpoint, log.

Kept free of Modal and of wandb specifics so that the same loop runs
under pytest on a laptop and on a GPU container without branching. The
only thing the caller supplies is a device and a telemetry sink.

Reward shaping is annealed rather than switched. Training begins on
trick points, which arrive several times a hand and give the critic a
target long before any hand finishes; the level reward, which is what
the game is actually scored on, is mixed in over the course of
training. `reward_mix` is the weight on the level term and runs from
0 to `final_level_weight` across `anneal_steps` learner steps.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..nn.model import ActorCritic, ModelConfig
from ..telemetry import Spans, memory_snapshot
from .buffer import RolloutBuffer
from .evaluate import evaluate, evaluate_against_bots
from .gae import compute_gae
from .ppo import PPOConfig, PPOLearner
from .rollout import collect


@dataclass
class TrainConfig:
    """Everything that changes between runs, in one serialisable place."""

    total_steps: int = 1000
    hands_per_step: int = 64
    buffer_capacity: int = 16384
    gamma: float = 1.0
    lam: float = 0.95
    seed: int = 20260921
    temperature: float = 1.0

    # Hands kept in flight so that every seat waiting to act can be
    # batched into one forward. Left to the rollout's own default the
    # batch tracks `hands_per_step`, and a small step size starves the
    # GPU: a 4-hand step measured a mean batch of 3.5, against 80 at a
    # pool of 128. It costs memory in proportion, since each hand in
    # flight holds its own state and encoder output.
    pool: int = 128

    # Reward anneal, described in the module docstring.
    anneal_steps: int = 400
    final_level_weight: float = 1.0

    checkpoint_every: int = 50

    # Playing strength against a fixed reference, measured every
    # `eval_every` iterations and at the end. Zero disables it. The
    # measurement is single-decision and so far below the device's
    # efficient batch; `eval_deals` trades its cost against the width
    # of the interval it returns.
    eval_every: int = 0
    eval_deals: int = 64
    eval_seed: int = 12345

    # Heuristic bots from the TypeScript engine to play on the same
    # schedule, e.g. ``("smart-v34",)``. Empty is off. Each one spawns a
    # Node subprocess and plays `2 * eval_deals` hands a decision at a
    # time, so it costs roughly what the reference evaluation does, per
    # bot -- and unlike the reference it needs Node in the image.
    eval_bots: tuple[str, ...] = ()
    # Hosts run at once when measuring against a bot. The heuristics
    # search -- a decision costs better than half a second from the
    # strong ones -- so a measurement worth taking needs several.
    eval_bot_workers: int = 8
    # Deals per bot, separate from `eval_deals`: a bot costs a hundred
    # times what the reference opponent does per decision, and the
    # interval on a duplicate-paired score closes quickly enough that a
    # smaller set still separates one policy from another.
    eval_bot_deals: int = 24
    # Wall-clock ceiling for one ladder measurement. The rungs run
    # cheapest first, so a budget that runs out leaves the expensive end
    # unmeasured rather than leaving a rung half done, and the row says
    # how many were reached. None runs every rung however long it takes.
    eval_bot_budget_s: float | None = 600.0

    checkpoint_dir: str = "checkpoints"
    keep_last: int = 3

    model: dict = field(default_factory=dict)
    ppo: dict = field(default_factory=dict)


def reward_mix(step: int, cfg: TrainConfig) -> float:
    """Weight on the level reward at this learner step."""
    if cfg.anneal_steps <= 0:
        return cfg.final_level_weight
    frac = min(1.0, max(0.0, step / cfg.anneal_steps))
    return cfg.final_level_weight * frac


def save_checkpoint(
    path: Path,
    net: ActorCritic,
    learner: PPOLearner,
    step: int,
    cfg: TrainConfig,
) -> Path:
    """Weights, optimiser state and step, so a run resumes exactly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": net.state_dict(),
            "optimizer": learner.opt.state_dict(),
            "config": asdict(cfg),
            "model_config": asdict(net.cfg),
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path, net: ActorCritic, learner: PPOLearner | None = None
) -> int:
    """-> the step the checkpoint was written at."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(blob["model"])
    if learner is not None and "optimizer" in blob:
        learner.opt.load_state_dict(blob["optimizer"])
    return int(blob.get("step", 0))


def _prune(directory: Path, keep: int) -> None:
    """Checkpoints are large; a long run would otherwise fill the volume."""
    if keep <= 0:
        return
    found = sorted(
        directory.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    for stale in found[:-keep]:
        stale.unlink(missing_ok=True)


def train(
    cfg: TrainConfig,
    device: torch.device | str = "cpu",
    run: Any = None,
    resume: str | Path | None = None,
    on_checkpoint: Any = None,
) -> dict:
    """Run `cfg.total_steps` learner steps. Returns a summary dict.

    `on_checkpoint` is called with the path just written, and is where
    a caller whose storage needs an explicit flush performs it. A run
    measured in hours that only persists when it returns has no
    checkpoints at all until it does, which is the wrong way round:
    they exist so that the hours are not lost.
    """
    device = torch.device(device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model_cfg = ModelConfig(**cfg.model) if cfg.model else ModelConfig()
    net = ActorCritic(model_cfg).to(device)
    ppo_cfg = PPOConfig(**cfg.ppo) if cfg.ppo else PPOConfig()
    # No sink is handed to the learner: the loop already folds its
    # metrics into a row that also carries the rollout, the spans, the
    # memory and the evaluation, and a learner logging the same
    # `ppo/*` keys on its own optimiser-step axis puts two rows per
    # iteration into the same run. `PPOLearner` keeps the argument for
    # use outside this loop.
    learner = PPOLearner(net, ppo_cfg, device)

    start = 0
    if resume is not None and Path(resume).exists():
        start = load_checkpoint(Path(resume), net, learner)

    buffer = RolloutBuffer(cfg.buffer_capacity, model_cfg)
    ckpt_dir = Path(cfg.checkpoint_dir)
    started = time.perf_counter()
    seen = 0
    last: dict = {}

    for step in range(start, cfg.total_steps):
        step_started = time.perf_counter()
        spans = Spans(enabled=True, device=device)
        buffer.reset()
        level_weight = reward_mix(step, cfg)

        with spans.span("collect"):
            net.eval()
            stats = collect(
                net,
                rng,
                buffer,
                n_hands=cfg.hands_per_step,
                device=device,
                cfg=model_cfg,
                temperature=cfg.temperature,
                level_weight=level_weight,
                pool=cfg.pool,
                # The behaviour log-probability must come from the
                # same precision the learner will recompute in, or
                # the importance ratio is not 1 even at identical
                # weights and the clip silences whichever rows
                # happened to round the wrong way.
                amp=learner.amp,
                spans=spans,
            )

        with spans.span("gae"):
            compute_gae(buffer, gamma=cfg.gamma, lam=cfg.lam)

        with spans.span("update"):
            net.train()
            metrics = learner.update(buffer)

        # Before the metrics row is assembled, so that the evaluation's
        # own span and its scores belong to the iteration that produced
        # them and the row is logged exactly once.
        scores: dict[str, float] = {}
        if cfg.eval_every and (step + 1) % cfg.eval_every == 0:
            with spans.span("evaluate"):
                scores = evaluate(
                    net,
                    device=device,
                    cfg=model_cfg,
                    num_deals=cfg.eval_deals,
                    # Fixed across iterations, so successive points differ
                    # by the policy and not by which deals were drawn.
                    seed=cfg.eval_seed,
                )
                # Merged into the same dict, and so into the same row as
                # the reference evaluation and the iteration that
                # produced it. A second row would put two points at one
                # step on every chart that reads this run.
                if cfg.eval_bots:
                    scores.update(
                        evaluate_against_bots(
                            net,
                            tuple(cfg.eval_bots),
                            device=device,
                            cfg=model_cfg,
                            num_deals=cfg.eval_bot_deals,
                            seed=cfg.eval_seed,
                            workers=cfg.eval_bot_workers,
                            budget_s=cfg.eval_bot_budget_s,
                        )
                    )

        now = time.perf_counter()
        elapsed = now - started
        step_s = now - step_started
        decisions = int(stats.get("decisions", len(buffer)))
        seen += decisions
        row = {
            "learner_steps": step,
            "level_weight": level_weight,
            "decisions": decisions,
            # This step's rate, and the run's. Dividing one step's
            # decisions by the run's elapsed time -- which an earlier
            # revision did -- reads as a throughput collapse that is
            # really just the denominator growing.
            "decisions_per_s": decisions / max(step_s, 1e-9),
            "decisions_per_s_mean": seen / max(elapsed, 1e-9),
            "hands_per_s": stats.get("hands", 0) / max(step_s, 1e-9),
            "step_s": step_s,
            "buffer_fill": len(buffer) / max(cfg.buffer_capacity, 1),
            "wall_s": elapsed,
            **{f"rollout/{k}": v for k, v in stats.items()},
            **{f"ppo/{k}": v for k, v in metrics.items()},
            **{f"span/{k}": v["total"] for k, v in spans.summary().items()},
            **{f"mem/{k}": v for k, v in memory_snapshot(device).items()},
            **{f"eval/{k}": v for k, v in scores.items()},
        }
        if run is not None:
            run.log(row)
        last = row

        if cfg.checkpoint_every and (step + 1) % cfg.checkpoint_every == 0:
            written = save_checkpoint(
                ckpt_dir / f"step_{step + 1:07d}.pt", net, learner, step + 1, cfg
            )
            _prune(ckpt_dir, cfg.keep_last)
            if on_checkpoint is not None:
                on_checkpoint(written)

    final = save_checkpoint(
        ckpt_dir / "final.pt", net, learner, cfg.total_steps, cfg
    )
    if on_checkpoint is not None:
        on_checkpoint(final)
    summary = {
        "steps": cfg.total_steps - start,
        "wall_s": time.perf_counter() - started,
        "final": last,
    }
    if run is not None:
        run.summary({k: v for k, v in summary.items() if k != "final"})
    return summary
