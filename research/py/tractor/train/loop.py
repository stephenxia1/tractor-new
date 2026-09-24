"""The outer training loop: collect, credit, update, checkpoint, log.

Kept free of Modal and of wandb specifics so that the same loop runs
under pytest on a laptop and on a GPU container without branching. The
only thing the caller supplies is a device and a telemetry sink.

The reward is trick points, and by default only trick points.
`reward_mix` is the weight on the terminal level term; it runs from 0
to `final_level_weight` across `anneal_steps` learner steps, and both
default to 0, which holds the weight at zero for the whole run.

Points rather than levels because of what the terminal term does to
the critic. Every row of a hand carries the same terminal value, so as
weight moves onto it the returns within a hand converge and an
iteration's rows collapse towards one target per hand-seat. Measured
on a holdout of whole hands (`train/critic_diag.py`), the critic's
explained variance on hands it was not fitted to runs +0.405 at a
level weight of 0 and -0.324 at 1: past the middle of the anneal the
value function is worse than predicting the mean, so it adds variance
to the policy gradient rather than removing it.

Two things the points-only reward cannot see, both worth knowing
before turning it on for a long run. The kitty bonus is folded into
the attackers' total by `PlayState._finish` and was never in a trick
record, so no dense credit exists for it -- and it is not small, at
twice the number of cards in the winning play, up to eight times over
300 sampled hands. And the game is scored on levels, not points: 119
points and 0 points are the same result, and the dense reward rates
them 119 apart. Points are a proxy, and the win rate the evaluation
reports against the ladder is where the proxy would show itself
failing.
"""

from __future__ import annotations

import contextlib
import copy
import json
from concurrent.futures import Future, ThreadPoolExecutor
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..nn.bury_net import BuryNet
from ..nn.declare import DeclareNet
from ..nn.features import Cache
from ..nn.model import ActorCritic, ModelConfig, load_model
from ..sim.bury import BuryConfig
from .bury_loop import BuryLearner
from .declare_loop import DeclareLearner, DeclareTrainConfig
from .starts import PrePlayConfig, make_starts
from ..telemetry import Spans, memory_snapshot
from .buffer import RolloutBuffer
from .evaluate import evaluate, evaluate_against_bots
from .gae import compute_gae
from ..gen.episode import deal_sizes
from ..sim.state import NUM_SEATS
from .parallel import RolloutPool
from .rollout import DECKS
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
    #
    # 512 rather than 128 because it was measured: at 1024 hands a
    # round it takes the mean batch from 114 to 385 and the round from
    # 15.84 s to 13.80 s. The forward barely notices -- 7.58 s to
    # 6.86 s for 3.4x the batch -- which is itself the finding: the
    # forward is proportional to decisions rather than to launches, so
    # it is arithmetic and not per-call overhead, and no batch size
    # will remove it.
    pool: int = 512

    # Reward anneal, described in the module docstring. Both zero:
    # the reward is trick points and the terminal level term is off.
    # Set `final_level_weight` to 1 (with an `anneal_steps` to reach
    # it over) to restore the annealed hand-outcome reward.
    anneal_steps: int = 0
    final_level_weight: float = 0.0

    # Worker processes for the rollout. 1 keeps the single-process
    # ``collect``; above 1 the simulator moves into that many children
    # and the parent keeps the network. The rollout is host-bound -- of
    # an 8.8 s ``collect``, 2.70 s went on enumeration, 1.94 s on
    # encoding and 1.42 s on the simulator against 1.38 s in the
    # network -- so this is the only lever that moves it, and a
    # container reserving eight cores was using one.
    #
    # The pool is built once, before the first step, and reused. A
    # worker takes about 9.5 seconds to reach its first instruction,
    # nearly all of it ``import torch``, which is more than a round
    # costs; a pool per step measured 1.08x on an A10G and all of the
    # gain was inside the start-up.
    rollout_workers: int = 1

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

    # The full hand: declaration and burial by the agent, trained in
    # alternation with play. Off when ``declare_phase_steps`` is 0, and
    # every play start is then ``gen.episode``'s uniform draw, as before.
    #
    # On: the first ``warmup_steps`` play steps still start from the
    # uniform draw. Then, at the start of every block of
    # ``play_phase_steps`` play steps, a declaration phase runs
    # ``declare_phase_steps`` steps of ``train.declare_loop`` against the
    # play network as it stands (read only, so it is frozen for the
    # phase), and the block's play starts come from that frozen
    # declaration policy and the value network's burial climb -- all but
    # ``random_burial_frac`` of them, which are buried at random so the
    # value head stays accurate on burials the climb would not propose.
    warmup_steps: int = 0
    play_phase_steps: int = 250
    declare_phase_steps: int = 0
    random_burial_frac: float = 0.2
    bury_rounds: int = 8
    bury_consider: int = 0
    # What scores a burial: "actor" or "critic"; see ``sim.bury.BuryConfig``.
    bury_evaluator: str = "actor"
    bury_worlds: int = 8
    # Bury by a learned policy of eight picks (``nn/bury_net.py``) in
    # place of the climb. It trains in the declaration phase, on the same
    # hands, rewarded by the critic's play-start value from the landlord's
    # seat (``train/bury_loop.py``).
    bury_policy: bool = False
    # Count declaration phases from zero again, for a run resumed from a
    # checkpoint written under a different schedule.
    restart_schedule: bool = False
    declare: dict = field(default_factory=dict)
    # Build the next step's play starts on a thread while this step's
    # update runs, from a frozen copy of the network (the update is
    # changing the live one) on a CUDA stream of its own. The starts are
    # then one update stale, as a behaviour policy one step old is.
    # Never across a block boundary, whose starts need the declaration
    # phase that opens it.
    #
    # Off, because it was measured and bought nothing: on an L40S at
    # 1024 hands the step went 28.2 s -> 26.4 s, and all of that was
    # ``ppo.length_buckets``, on in the same arm. Building starts is
    # Python-heavy (the deal, candidate generation), so on a thread it
    # holds the GIL and the update -- whose kernel launches are Python
    # too -- slowed by what the starts cost (9.4 s -> 18.5 s).
    prefetch_starts: bool = False

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
    extra: dict | None = None,
) -> Path:
    """Weights, optimiser state and step, so a run resumes exactly.

    ``extra`` carries the declaration agent and the schedule's position
    when the full hand is being trained.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "step": step,
        "model": net.state_dict(),
        "optimizer": learner.opt.state_dict(),
        "config": asdict(cfg),
        "model_config": asdict(net.cfg),
    }
    if extra:
        blob.update(extra)
    torch.save(blob, path)
    return path


def load_declare(path: Path) -> dict | None:
    """The declaration agent a checkpoint carries, if any."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob.get("declare")


def load_checkpoint(
    path: Path, net: ActorCritic, learner: PPOLearner | None = None
) -> int:
    """-> the step the checkpoint was written at."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    added = load_model(net, blob["model"])
    if learner is not None and "optimizer" in blob:
        state = blob["optimizer"]
        if added:
            # Parameters the checkpoint predates come last in
            # ``parameters()`` (see ``Critic.context_proj``), so the old
            # optimiser state is extended with fresh slots for them
            # rather than remapped; AdamW fills their moments lazily.
            state = copy.deepcopy(state)
            group = state["param_groups"][0]
            have = len(group["params"])
            want = sum(1 for p in net.parameters() if p.requires_grad)
            group["params"] = list(group["params"]) + list(range(have, want))
        learner.opt.load_state_dict(state)
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


def _prefetch_starts(
    executor: ThreadPoolExecutor,
    net: ActorCritic,
    declare_net: Any,
    device: torch.device,
    hands: int,
    pre: PrePlayConfig,
    rng: np.random.Generator,
    bury_net: Any = None,
) -> Future:
    """Start building the next step's play starts; see ``prefetch_starts``."""
    frozen = copy.deepcopy(net).eval()
    stream = torch.cuda.Stream(device) if device.type == "cuda" else None

    def work() -> tuple[list, dict]:
        ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        with ctx:
            starts, _rows, st = make_starts(
                rng, hands, frozen, declare_net, device, pre=pre, cache=Cache(),
                bury_net=bury_net,
            )
        if stream is not None:
            stream.synchronize()
        return starts, st

    return executor.submit(work)


def _declare_phase(
    learner: DeclareLearner,
    play_net: ActorCritic,
    rng: np.random.Generator,
    steps: int,
    block: int,
) -> dict[str, float]:
    """Train the declaration policy for ``steps`` steps on frozen play.

    Returns the phase's mean statistics under ``phase/*``, which ride on
    the first play row of the block rather than logging rows of their
    own -- a second row at one learner step puts two points on every
    chart.
    """
    totals: dict[str, float] = {}
    started = time.perf_counter()
    for i in range(steps):
        t0 = time.perf_counter()
        st = learner.step(play_net, rng)
        for k, v in st.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        bury = (
            f"  bury ret {st['bury/return_mean']:+.3f} ent {st['bury/entropy']:.3f}"
            f" pts {st.get('bury/kitty_points', 0.0):4.1f}"
            if "bury/return_mean" in st else ""
        )
        print(
            f"  [declare {block}:{i + 1}/{steps}] {time.perf_counter() - t0:5.1f}s"
            f"  rows {st.get('declare/rows', 0):5.0f}"
            f"  return {st.get('declare/return_mean', 0.0):+.3f}"
            f"  declared {st.get('declare/declared_frac', 0.0):.2f}"
            f"  nt {st.get('declare/no_trump_frac', 0.0):.2f}"
            f"  ent {st.get('declare/entropy', 0.0):.3f}" + bury,
            flush=True,
        )
    out = {f"phase/{k}": v / max(steps, 1) for k, v in totals.items()}
    out["phase/seconds"] = time.perf_counter() - started
    out["phase/block"] = float(block)
    return out


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

    full_hand = cfg.declare_phase_steps > 0
    bury_cfg = BuryConfig(
        rounds=cfg.bury_rounds, consider=cfg.bury_consider,
        evaluator=cfg.bury_evaluator, worlds=cfg.bury_worlds,
    )
    declare_net = declare_learner = None
    bury_net = bury_learner = None
    burial = "policy" if cfg.bury_policy else "climb"
    phases_done = 0
    if full_hand:
        blob = (
            torch.load(Path(resume), map_location="cpu", weights_only=False)
            if resume is not None and Path(resume).exists()
            else {}
        )
        if cfg.bury_policy:
            bury_net = BuryNet().to(device)
            bury_learner = BuryLearner(bury_net, device=device)
            if blob.get("bury") is not None:
                bury_net.load_state_dict(blob["bury"]["model"])
                bury_learner.opt.load_state_dict(blob["bury"]["optimizer"])
            bury_net.eval()
        declare_cfg = DeclareTrainConfig(**{
            **cfg.declare,
            "pre": PrePlayConfig(random_burial_frac=0.0, bury=bury_cfg, burial=burial),
        })
        declare_net = DeclareNet().to(device)
        declare_learner = DeclareLearner(
            declare_net, declare_cfg, device, bury=bury_learner
        )
        saved = blob.get("declare")
        if saved is not None:
            declare_net.load_state_dict(saved["model"])
            declare_learner.opt.load_state_dict(saved["optimizer"])
            phases_done = 0 if cfg.restart_schedule else int(saved["phases_done"])
        declare_net.eval()
        del blob
    play_pre = PrePlayConfig(
        random_burial_frac=cfg.random_burial_frac, bury=bury_cfg, burial=burial
    )

    def extra() -> dict | None:
        if not full_hand:
            return None
        out = {"declare": {
            "model": declare_net.state_dict(),
            "optimizer": declare_learner.opt.state_dict(),
            "phases_done": phases_done,
        }}
        if bury_net is not None:
            out["bury"] = {
                "model": bury_net.state_dict(),
                "optimizer": bury_learner.opt.state_dict(),
            }
        return out

    # A hand can take one deal's worth of rows per seat, and `collect`
    # refuses to start a hand it cannot guarantee room for -- so a
    # buffer too small for `hands_per_step` does not raise, it quietly
    # returns fewer hands, and the run trains on a round it never
    # notices was short. Checked here, once, against the same bound the
    # rollout checks itself against.
    per_seat, _kitty = deal_sizes(DECKS)
    needed = cfg.hands_per_step * per_seat * NUM_SEATS
    if cfg.buffer_capacity < needed:
        raise ValueError(
            f"buffer_capacity {cfg.buffer_capacity} cannot hold "
            f"{cfg.hands_per_step} hands; a hand can write up to "
            f"{per_seat * NUM_SEATS} rows, so set it to at least {needed}"
        )

    buffer = RolloutBuffer(cfg.buffer_capacity, model_cfg)
    # Started before the first step and closed in the `finally` below,
    # so worker start-up is paid once for the run and not once a step.
    rollout_pool = (
        RolloutPool(
            cfg=model_cfg,
            workers=cfg.rollout_workers,
            hands_per_round=cfg.hands_per_step,
        )
        if cfg.rollout_workers > 1
        else None
    )
    ckpt_dir = Path(cfg.checkpoint_dir)
    prefetched: Any = None
    executor = ThreadPoolExecutor(max_workers=1) if cfg.prefetch_starts else None
    started = time.perf_counter()
    seen = 0
    last: dict = {}

    try:
        for step in range(start, cfg.total_steps):
            step_started = time.perf_counter()
            spans = Spans(enabled=True, device=device)
            buffer.reset()
            level_weight = reward_mix(step, cfg)

            # The schedule. A declaration phase opens every block of play
            # steps after the warm-up, once; ``phases_done`` is saved, so
            # a resume at a block boundary does not run it twice.
            starts = None
            pre_stats: dict[str, float] = {}
            if full_hand and step >= cfg.warmup_steps:
                block = (step - cfg.warmup_steps) // cfg.play_phase_steps
                if block >= phases_done:
                    prefetched = None  # built before this phase; stale
                    with spans.span("declare_phase"):
                        net.eval()
                        pre_stats = _declare_phase(
                            declare_learner, net, rng, cfg.declare_phase_steps, block
                        )
                    phases_done = block + 1
                with spans.span("starts"):
                    if prefetched is not None:
                        starts, st = prefetched.result()
                        prefetched = None
                    else:
                        net.eval()
                        starts, _rows, st = make_starts(
                            rng, cfg.hands_per_step, net, declare_net, device,
                            pre=play_pre, cache=declare_learner.cache,
                            bury_net=bury_net,
                        )
                    pre_stats.update(st)

            with spans.span("collect"):
                net.eval()
                # The behaviour log-probability must come from the same
                # precision the learner will recompute in, or the
                # importance ratio is not 1 even at identical weights and
                # the clip silences whichever rows happened to round the
                # wrong way. Both paths take `amp` for that reason.
                if rollout_pool is not None:
                    buffer, stats = rollout_pool.collect(
                        net,
                        rng,
                        buffer,
                        n_hands=cfg.hands_per_step,
                        device=device,
                        temperature=cfg.temperature,
                        level_weight=level_weight,
                        pool=cfg.pool,
                        amp=learner.amp,
                        spans=spans,
                        starts=starts,
                    )
                else:
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
                        amp=learner.amp,
                        spans=spans,
                        starts=starts,
                    )

            with spans.span("gae"):
                compute_gae(buffer, gamma=cfg.gamma, lam=cfg.lam)

            nxt = step + 1
            if (
                cfg.prefetch_starts
                and full_hand
                and nxt < cfg.total_steps
                and nxt >= cfg.warmup_steps
                and (nxt - cfg.warmup_steps) // cfg.play_phase_steps < phases_done
            ):
                prefetched = _prefetch_starts(
                    executor, net, declare_net, device, cfg.hands_per_step, play_pre,
                    np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
                    bury_net,
                )

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
                **{f"pre/{k}": v for k, v in pre_stats.items()},
            }
            if run is not None:
                run.log(row)
            last = row

            # One line per step on stdout. A detached run's only other
            # channel is the JSONL on the volume, which becomes visible
            # at checkpoints -- over an hour apart at this round size --
            # and the question worth watching continuously is whether
            # the accelerator is still what the step waits on.
            #
            # `gpu` is the forward plus the update over the step: both
            # are the same network, and everything outside them is the
            # simulator, the encoder and moving bytes. It is occupancy
            # and not utilisation -- it says the device was busy, not
            # that it was busy doing useful arithmetic.
            gpu_s = row.get("span/forward", 0.0) + row.get("span/update", 0.0)
            print(
                f"[{step + 1}/{cfg.total_steps}] {step_s:5.1f}s"
                f"  collect {row.get('span/collect', 0.0):5.1f}"
                f" (fwd {row.get('span/forward', 0.0):4.1f})"
                f"  update {row.get('span/update', 0.0):5.1f}"
                f"  gpu {100.0 * gpu_s / max(step_s, 1e-9):4.1f}%"
                f"  {row.get('decisions_per_s', 0.0):5.0f} dec/s"
                f"  ev {metrics.get('explained_variance', float('nan')):+.3f}"
                f"  lw {level_weight:.2f}",
                flush=True,
            )

            if cfg.checkpoint_every and (step + 1) % cfg.checkpoint_every == 0:
                written = save_checkpoint(
                    ckpt_dir / f"step_{step + 1:07d}.pt", net, learner, step + 1,
                    cfg, extra(),
                )
                _prune(ckpt_dir, cfg.keep_last)
                if on_checkpoint is not None:
                    on_checkpoint(written)

        final = save_checkpoint(
            ckpt_dir / "final.pt", net, learner, cfg.total_steps, cfg, extra()
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
    finally:
        # The workers hold a shared-memory block, which on POSIX
        # outlives the process that forgot to unlink it.
        if rollout_pool is not None:
            rollout_pool.close()
        if executor is not None:
            executor.shutdown(wait=True)

    return summary
