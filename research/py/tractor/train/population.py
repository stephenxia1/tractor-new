"""A population of agents, one per seat, at two tables.

The single agent plays all four seats of every hand, so its partner is
always itself. A population breaks that: ``agents`` independently
initialised agents -- eight by default, the user's spec -- each hold one
seat at one of ``agents // 4`` tables, so a partner and both opponents
are other agents. Every ``reshuffle_every`` steps the agents are dealt
to the seats afresh, so over a run each partners and opposes the rest.
Landlord-set hands only, as everywhere else in this track.

Each agent is a whole single agent: its own play network with its own
privileged critic, declaration network and burial policy, each with its
own optimiser. Nothing is shared but the positions.

Routing. The simulator does not know who is who. A hand's index decides
its table (``i % tables``), the table and the seat decide the agent, and
that table of agents, int64[hands, 4], is handed to the rollout and to
``make_starts``: every decision goes through its seat's network, every
row's value is recorded by its seat's critic, every burial is made by
the landlord's agent, and every declaration return is the declaring
seat's own critic's value. After GAE -- which already groups by hand and
seat -- each agent's learner updates on exactly its own rows.

Sample size. An agent sits in one seat of half the hands, so it learns
from an eighth of the seat-trajectories a single agent would take from
the same round. The user chose 4096 hands a step, which gives each agent
2048 trajectories: half of what the single agent's 1024-hand round gave
it. The pre-play phase is widened the same way by the caller (8192
hands gives each agent the declarations and burials a 1024-hand phase
gave the single agent).

Seatings are a function of ``(seed, epoch)`` rather than of a live
generator, so a resumed run sits every agent where the original run
would have.

The critic is shared (``shared_critic``, the default, the user's call):
one ``CardCritic`` for the whole population, trained on every row of the
round by ``train.critic_learner``, reading the raw history and told
which agent sits in each seat. It records every row's value in the
rollout, and it values every seat's play start for the pre-play rewards.
Each agent is then an actor alone -- its policy, its own value, belief
and outcome heads -- updated on its own rows.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..gen.episode import deal_sizes
from ..nn.arch import check_saved
from ..nn.bury_net import BuryNet, BuryNetConfig
from ..nn.declare import DeclareConfig, DeclareNet
from ..nn.features import Cache
from ..nn.model import (
    ActorCritic,
    ModelConfig,
    load_model,
    relative_seat_agents,
    shared_critic,
)
from ..sim.bury import BuryConfig
from ..sim.state import NUM_SEATS
from ..telemetry import Spans, memory_snapshot
from .buffer import DeviceRows, RolloutBuffer
from .bury_loop import BuryLearner
from .critic_learner import CriticLearner, CriticTrainConfig
from .declare_loop import DeclareLearner, DeclareTrainConfig
from .gae import compute_gae
from .loop import TrainConfig, _prune, reward_mix
from .parallel import RolloutPool
from .ppo import PPOConfig, PPOLearner
from .rollout import DECKS, collect
from .starts import PrePlayConfig, critic_values, make_starts

__all__ = [
    "PopulationConfig",
    "seating",
    "hand_agents",
    "train_population",
    "read_checkpoint",
]


@dataclass
class PopulationConfig(TrainConfig):
    """``TrainConfig`` plus the population's own knobs.

    Every inherited field means what it means for the single agent,
    applied to each agent. ``pool`` is wider by default: the rollout's
    forward is split eight ways, and at the single agent's 512 each
    agent's share of a batch would be about 48 rows. 2048 is a starting
    point, not a measurement.
    """

    agents: int = 8
    reshuffle_every: int = 10
    pool: int = 2048
    # One critic for every agent (the user's call, 2026-09-28): trained on
    # all rows, told which agent sits in each seat, reading raw history.
    # The agents are then actors alone. False gives each agent its own.
    shared_critic: bool = True
    # With fixed-shape updates (``ppo.compile`` or ``ppo.fixed_shapes``),
    # copy each round to the device once and let every learner draw its
    # minibatches there (``buffer.DeviceRows``): about 10 GiB at 4096
    # hands, which an L40S holds.
    device_rows: bool = True


# Salts that keep the play and pre-play seating streams apart.
_PLAY, _PRE = 0x5EA7, 0x9E57


def seating(seed: int, epoch: int, agents: int, salt: int = _PLAY) -> np.ndarray:
    """int64[tables, 4]: the agent in each seat of each table for ``epoch``."""
    rng = np.random.default_rng([seed, salt, epoch])
    return rng.permutation(agents).reshape(agents // NUM_SEATS, NUM_SEATS).astype(np.int64)


def hand_agents(table: np.ndarray, n_hands: int) -> np.ndarray:
    """int64[n_hands, 4]: hand ``i`` is played at table ``i % tables``."""
    return table[np.arange(n_hands) % len(table)]


@dataclass
class _Agent:
    net: ActorCritic
    learner: PPOLearner
    declare: DeclareLearner | None = None
    bury: BuryLearner | None = None


def _state(agent: _Agent, phases_done: int) -> dict:
    """One agent in the single agent's checkpoint layout."""
    out: dict[str, Any] = {
        "model": agent.net.state_dict(),
        "optimizer": agent.learner.opt.state_dict(),
        "declare": None,
        "bury": None,
    }
    if agent.declare is not None:
        out["declare"] = {
            "model": agent.declare.net.state_dict(),
            "optimizer": agent.declare.opt.state_dict(),
            "phases_done": phases_done,
            "config": asdict(agent.declare.net.cfg),
        }
    if agent.bury is not None:
        out["bury"] = {
            "model": agent.bury.net.state_dict(),
            "optimizer": agent.bury.opt.state_dict(),
            "config": asdict(agent.bury.net.cfg),
        }
    return out


def save_population(
    path: Path, agents: list[_Agent], step: int, cfg: PopulationConfig,
    phases_done: int, critic: CriticLearner | None = None,
) -> Path:
    """All the agents, and the shared critic when there is one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "step": step,
        "config": asdict(cfg),
        "model_config": asdict(agents[0].net.cfg),
        "phases_done": phases_done,
        "agents": [_state(a, phases_done) for a in agents],
    }
    if critic is not None:
        blob["critic"] = {
            "model": critic.critic.state_dict(),
            "optimizer": critic.opt.state_dict(),
            "config": asdict(critic.critic.cfg),
            "agents": critic.critic.agents,
        }
    torch.save(blob, path)
    return path


def read_checkpoint(path: str | Path) -> dict:
    """A checkpoint in the single agent's layout, from either kind.

    ``run/final.pt`` is read as it is. ``run/final.pt#3`` is agent 3 of a
    population checkpoint, laid out as a single agent's, so everything
    that evaluates one checkpoint evaluates one agent unchanged.
    """
    text = str(path)
    base, _sep, index = text.partition("#")
    blob = torch.load(base, map_location="cpu", weights_only=False)
    if not index:
        if "agents" in blob:
            raise ValueError(f"{base} is a population; name an agent as {base}#k")
        return blob
    if "agents" not in blob:
        raise ValueError(f"{base} is not a population checkpoint")
    k = int(index)
    if not 0 <= k < len(blob["agents"]):
        raise ValueError(f"{base} holds {len(blob['agents'])} agents, not agent {k}")
    return {
        "step": blob["step"],
        "config": blob["config"],
        "model_config": blob["model_config"],
        **blob["agents"][k],
    }


def _mean_over(rows: list[dict[str, float]]) -> dict[str, float]:
    """Each key's mean over the agents that reported it."""
    keys = {k for r in rows for k in r}
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in sorted(keys)}


def _preplay_step(
    agents: list[_Agent],
    table: np.ndarray,
    cfg: DeclareTrainConfig,
    rng: np.random.Generator,
    device: torch.device,
    cache: Cache,
    shared: Any = None,
) -> dict[str, float]:
    """One pre-play step for every agent, on one shared deal.

    ``DeclareLearner.step`` for a population: the batch is dealt once
    with every seat declaring and every landlord burying by its own
    agent, each seat is valued by its own critic, and each agent's
    declaration and burial learners update on their own rows.
    """
    n = cfg.hands_per_step
    seat_of = hand_agents(table, n)
    nets = [a.net for a in agents]
    for a in agents:
        a.net.eval()
        a.declare.net.eval()
        if a.bury is not None:
            a.bury.net.eval()
    bury_nets = None if agents[0].bury is None else [a.bury.net for a in agents]
    starts, pre_rows, stats = make_starts(
        rng, n, nets, [a.declare.net for a in agents], device,
        pre=cfg.pre, record=True, cache=cache, bury_net=bury_nets, agents=seat_of,
    )
    t_critic = time.perf_counter()
    values = critic_values(nets, starts, device, agents=seat_of, shared=shared)
    landlord = np.array([s.landlord for s in starts], dtype=np.int64)
    t_update = time.perf_counter()

    per_agent: list[dict[str, float]] = [{} for _ in agents]
    bury = pre_rows.bury
    if bury_nets is not None and bury is not None and len(bury):
        owner = seat_of[bury.hand, landlord[bury.hand]]
        ret = values[bury.hand, landlord[bury.hand]].astype(np.float32) * cfg.reward_scale
        for k, a in enumerate(agents):
            idx = np.nonzero(owner == k)[0]
            if len(idx):
                per_agent[k].update(a.bury.update(bury.subset(idx), ret[idx]))
    rows = pre_rows.declare
    if len(rows):
        owner = seat_of[rows.hand, rows.seat]
        ret = values[rows.hand, rows.seat].astype(np.float32) * cfg.reward_scale
        for k, a in enumerate(agents):
            idx = np.nonzero(owner == k)[0]
            if len(idx):
                per_agent[k].update(a.declare.update(rows.subset(idx), ret[idx]))
    return {
        **stats,
        **_mean_over([p for p in per_agent if p]),
        "declare/critic_landlord_value": float(values[np.arange(n), landlord].mean()),
        "time/critic_s": t_update - t_critic,
        "time/update_s": time.perf_counter() - t_update,
    }


def _preplay_phase(
    agents: list[_Agent],
    cfg: PopulationConfig,
    declare_cfg: DeclareTrainConfig,
    rng: np.random.Generator,
    device: torch.device,
    cache: Cache,
    block: int,
    shared: Any = None,
) -> dict[str, float]:
    """``cfg.declare_phase_steps`` pre-play steps, reseating on schedule."""
    totals: dict[str, float] = {}
    started = time.perf_counter()
    steps = cfg.declare_phase_steps
    for i in range(steps):
        t0 = time.perf_counter()
        epoch = (block * steps + i) // cfg.reshuffle_every
        st = _preplay_step(
            agents, seating(cfg.seed, epoch, cfg.agents, _PRE), declare_cfg, rng,
            device, cache, shared,
        )
        for k, v in st.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        bury = (
            f"  bury ret {st['bury/return_mean']:+.3f} ent {st['bury/entropy']:.3f}"
            f" pts {st.get('bury/kitty_points', 0.0):4.1f}"
            if "bury/return_mean" in st else ""
        )
        print(
            f"  [pre-play {block}:{i + 1}/{steps}] {time.perf_counter() - t0:5.1f}s"
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


def agent_model_config(cfg: PopulationConfig) -> ModelConfig:
    """Each agent's network: an actor alone when the critic is shared."""
    model_cfg = ModelConfig(**cfg.model) if cfg.model else ModelConfig()
    return replace(model_cfg, critic_kind="none") if cfg.shared_critic else model_cfg


def build_critic(cfg: PopulationConfig, device: torch.device) -> CriticLearner | None:
    """The population's shared critic and its learner, or ``None``.

    Its widths are the model config's critic widths; its optimiser takes
    the actors' PPO settings, so the two halves of a step learn alike.
    """
    if not cfg.shared_critic:
        return None
    model_cfg = ModelConfig(**cfg.model) if cfg.model else ModelConfig()
    ppo_cfg = PPOConfig(**cfg.ppo) if cfg.ppo else PPOConfig()
    critic = shared_critic(model_cfg, cfg.agents).to(device)
    return CriticLearner(critic, CriticTrainConfig(
        lr=ppo_cfg.lr, minibatch=ppo_cfg.minibatch, epochs=ppo_cfg.epochs,
        length_buckets=ppo_cfg.length_buckets, max_grad_norm=ppo_cfg.max_grad_norm,
        bf16=ppo_cfg.bf16, compile=ppo_cfg.compile, fixed_shapes=ppo_cfg.fixed_shapes,
        cuda_graphs=ppo_cfg.cuda_graphs,
    ), device)


def load_critic(learner: CriticLearner | None, blob: dict) -> None:
    """Restore the shared critic a population checkpoint holds."""
    saved = blob.get("critic")
    if learner is None:
        if saved is not None:
            raise ValueError("the checkpoint has a shared critic and this run does not")
        return
    if saved is None:
        raise ValueError("this run shares a critic and the checkpoint has none")
    if int(saved["agents"]) != learner.critic.agents:
        raise ValueError(f"the checkpoint's critic knows {saved['agents']} agents")
    learner.critic.load_state_dict(saved["model"])
    learner.opt.load_state_dict(saved["optimizer"])


def build_agents(
    cfg: PopulationConfig, device: torch.device, declare_cfg: DeclareTrainConfig
) -> list[_Agent]:
    """``cfg.agents`` fresh agents, built one after another from the current
    torch seed: independent initialisations, reproducible as a set."""
    model_cfg = agent_model_config(cfg)
    ppo_cfg = PPOConfig(**cfg.ppo) if cfg.ppo else PPOConfig()
    full_hand = cfg.declare_phase_steps > 0
    agents: list[_Agent] = []
    for _ in range(cfg.agents):
        net = ActorCritic(model_cfg).to(device)
        agent = _Agent(net=net, learner=PPOLearner(net, ppo_cfg, device))
        if full_hand:
            agent.declare = DeclareLearner(
                DeclareNet(DeclareConfig(**cfg.declare_model)).to(device), declare_cfg, device
            )
            if cfg.bury_policy:
                agent.bury = BuryLearner(
                    BuryNet(BuryNetConfig(**cfg.bury_model)).to(device), device=device
                )
        agents.append(agent)
    return agents


def load_agents(agents: list[_Agent], blob: dict) -> int:
    """Restore a population checkpoint into ``agents``. -> its step."""
    if "agents" not in blob:
        raise ValueError("not a population checkpoint")
    if len(blob["agents"]) != len(agents):
        raise ValueError(f"the checkpoint holds {len(blob['agents'])} agents, not {len(agents)}")
    for agent, saved in zip(agents, blob["agents"]):
        load_model(agent.net, saved["model"])
        agent.learner.opt.load_state_dict(saved["optimizer"])
        if agent.declare is not None and saved.get("declare") is not None:
            check_saved("declaration", saved["declare"], agent.declare.net.cfg)
            agent.declare.net.load_state_dict(saved["declare"]["model"])
            agent.declare.opt.load_state_dict(saved["declare"]["optimizer"])
        if agent.bury is not None and saved.get("bury") is not None:
            check_saved("burial", saved["bury"], agent.bury.net.cfg)
            agent.bury.net.load_state_dict(saved["bury"]["model"])
            agent.bury.opt.load_state_dict(saved["bury"]["optimizer"])
    return int(blob["step"])


def train_population(
    cfg: PopulationConfig,
    device: torch.device | str = "cpu",
    run: Any = None,
    resume: str | Path | None = None,
    on_checkpoint: Any = None,
) -> dict:
    """``train.loop.train`` for a population. Returns a summary dict."""
    if cfg.agents < NUM_SEATS or cfg.agents % NUM_SEATS:
        raise ValueError(f"{cfg.agents} agents do not fill whole tables of {NUM_SEATS}")
    if cfg.reshuffle_every < 1:
        raise ValueError("reshuffle_every must be at least 1")
    tables = cfg.agents // NUM_SEATS
    if cfg.hands_per_step < tables:
        # A table with no hands leaves its agents with no rows to learn from.
        raise ValueError(f"{cfg.hands_per_step} hands a step cannot fill {tables} tables")
    if cfg.eval_every:
        raise ValueError(
            "in-loop evaluation measures one network; evaluate a population's "
            "agents from its checkpoints (modal_eval.py, path#k)"
        )
    if cfg.prefetch_starts:
        raise ValueError("prefetch_starts is not supported for a population")

    device = torch.device(device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model_cfg = agent_model_config(cfg)
    full_hand = cfg.declare_phase_steps > 0
    bury_cfg = BuryConfig(
        rounds=cfg.bury_rounds, consider=cfg.bury_consider,
        evaluator=cfg.bury_evaluator, worlds=cfg.bury_worlds,
    )
    burial = "policy" if cfg.bury_policy else "climb"
    declare_cfg = DeclareTrainConfig(**{
        **cfg.declare,
        "pre": PrePlayConfig(random_burial_frac=0.0, bury=bury_cfg, burial=burial),
    })

    agents = build_agents(cfg, device, declare_cfg)
    critic = build_critic(cfg, device)
    shared = None if critic is None else critic.critic
    start = 0
    phases_done = 0
    if resume is not None:
        if not Path(resume).exists():
            # Not a fresh start in its place: a mistyped path would train
            # a new population over the old one's name.
            raise FileNotFoundError(f"no checkpoint to resume at {resume}")
        blob = torch.load(Path(resume), map_location="cpu", weights_only=False)
        # The critic first: a shared-critic checkpoint's agents have no
        # critics, and loading them into agents that do would fail on
        # missing weights rather than on the reason.
        load_critic(critic, blob)
        start = load_agents(agents, blob)
        phases_done = 0 if cfg.restart_schedule else int(blob.get("phases_done", 0))
        del blob
        # A stream of its own from the step it resumes at: seeded as the
        # fresh run was, it would deal the hands the run began with again.
        rng = np.random.default_rng([cfg.seed, start])

    play_pre = PrePlayConfig(
        random_burial_frac=cfg.random_burial_frac, bury=bury_cfg, burial=burial
    )
    per_seat, _kitty = deal_sizes(DECKS)
    needed = cfg.hands_per_step * per_seat * NUM_SEATS
    if cfg.buffer_capacity < needed:
        raise ValueError(
            f"buffer_capacity {cfg.buffer_capacity} cannot hold "
            f"{cfg.hands_per_step} hands; set it to at least {needed}"
        )
    buffer = RolloutBuffer(cfg.buffer_capacity, model_cfg)
    rollout_pool = (
        RolloutPool(cfg=model_cfg, workers=cfg.rollout_workers,
                    hands_per_round=cfg.hands_per_step)
        if cfg.rollout_workers > 1 else None
    )
    cache = Cache()
    nets = [a.net for a in agents]
    amp = agents[0].learner.amp
    ckpt_dir = Path(cfg.checkpoint_dir)
    started = time.perf_counter()
    seen = 0
    last: dict = {}

    try:
        # Kept across steps and refilled in place: a captured update
        # (``PPOConfig.cuda_graphs``) reads its rows at fixed addresses.
        device_rows: DeviceRows | None = None
        for step in range(start, cfg.total_steps):
            step_started = time.perf_counter()
            spans = Spans(enabled=True, device=device)
            buffer.reset()
            level_weight = reward_mix(step, cfg)
            epoch = step // cfg.reshuffle_every
            table = seating(cfg.seed, epoch, cfg.agents)
            seat_of = hand_agents(table, cfg.hands_per_step)

            starts = None
            pre_stats: dict[str, float] = {}
            if full_hand and step >= cfg.warmup_steps:
                block = (step - cfg.warmup_steps) // cfg.play_phase_steps
                if block >= phases_done:
                    with spans.span("declare_phase"):
                        pre_stats = _preplay_phase(
                            agents, cfg, declare_cfg, rng, device, cache, block, shared
                        )
                    phases_done = block + 1
                with spans.span("starts"):
                    for a in agents:
                        a.net.eval()
                    starts, _rows, st = make_starts(
                        rng, cfg.hands_per_step, nets,
                        [a.declare.net for a in agents], device, pre=play_pre,
                        cache=cache,
                        bury_net=None if agents[0].bury is None else [a.bury.net for a in agents],
                        agents=seat_of,
                    )
                    pre_stats.update(st)

            with spans.span("collect"):
                kwargs = dict(
                    n_hands=cfg.hands_per_step, device=device,
                    temperature=cfg.temperature, level_weight=level_weight,
                    pool=cfg.pool, amp=amp, spans=spans, starts=starts,
                    agents=seat_of, chunks=cfg.forward_chunks, critic=shared,
                    forwards=cfg.rollout_forwards,
                )
                if rollout_pool is not None:
                    buffer, stats = rollout_pool.collect(
                        nets, rng, buffer, overlap=cfg.rollout_overlap, **kwargs)
                else:
                    stats = collect(nets, rng, buffer, cfg=model_cfg, **kwargs)

            with spans.span("gae"):
                compute_gae(buffer, gamma=cfg.gamma, lam=cfg.lam)

            n = len(buffer)
            owner = seat_of[buffer.episode[:n], buffer.seat[:n]]
            on_device = None
            if cfg.device_rows and agents[0].learner.on_device_ok:
                with spans.span("upload"):
                    if device_rows is None:
                        device_rows = DeviceRows(buffer, device)
                    else:
                        device_rows.load(buffer)
                on_device = device_rows
            per_agent: list[dict[str, float]] = []
            with spans.span("update"):
                for k, agent in enumerate(agents):
                    rows = np.nonzero(owner == k)[0]
                    agent.net.train()
                    per_agent.append(agent.learner.update(
                        buffer, rows=rows, device_rows=on_device))
            critic_stats: dict[str, float] = {}
            if critic is not None:
                with spans.span("critic_update"):
                    critic_stats = critic.update(buffer, relative_seat_agents(
                        seat_of[buffer.episode[:n]], buffer.seat[:n]), device_rows=on_device)
            del on_device

            now = time.perf_counter()
            elapsed = now - started
            step_s = now - step_started
            decisions = int(stats.get("decisions", n))
            seen += decisions
            ppo = _mean_over(per_agent)
            row = {
                "learner_steps": step,
                "level_weight": level_weight,
                "decisions": decisions,
                "decisions_per_s": decisions / max(step_s, 1e-9),
                "decisions_per_s_mean": seen / max(elapsed, 1e-9),
                "hands_per_s": stats.get("hands", 0) / max(step_s, 1e-9),
                "step_s": step_s,
                "buffer_fill": n / max(cfg.buffer_capacity, 1),
                "wall_s": elapsed,
                "seating_epoch": float(epoch),
                **{f"rollout/{k}": v for k, v in stats.items()},
                **{f"ppo/{k}": v for k, v in ppo.items()},
                **{
                    f"agent{k}/{key}": m[key]
                    for k, m in enumerate(per_agent)
                    for key in ("explained_variance", "entropy", "approx_kl", "rows",
                                "early_stop", "kl_at_stop", "n_minibatches")
                },
                **{f"span/{k}": v["total"] for k, v in spans.summary().items()},
                **{f"mem/{k}": v for k, v in memory_snapshot(device).items()},
                **{f"pre/{k}": v for k, v in pre_stats.items()},
                **critic_stats,
            }
            if run is not None:
                run.log(row)
            last = row

            gpu_s = row.get("span/forward", 0.0) + row.get("span/update", 0.0)
            evs = [m["explained_variance"] for m in per_agent]
            print(
                f"[{step + 1}/{cfg.total_steps}] {step_s:5.1f}s"
                f"  collect {row.get('span/collect', 0.0):5.1f}"
                f" (fwd {row.get('span/forward', 0.0):4.1f})"
                f"  update {row.get('span/update', 0.0):5.1f}"
                f"  gpu {100.0 * gpu_s / max(step_s, 1e-9):4.1f}%"
                f"  {row.get('decisions_per_s', 0.0):5.0f} dec/s"
                f"  ev {np.mean(evs):+.3f} [{min(evs):+.2f},{max(evs):+.2f}]"
                + (f"  critic {row.get('span/critic_update', 0.0):4.1f}s"
                   if critic is not None else "")
                + f"  seats e{epoch}",
                flush=True,
            )

            if cfg.checkpoint_every and (step + 1) % cfg.checkpoint_every == 0:
                written = save_population(
                    ckpt_dir / f"step_{step + 1:07d}.pt", agents, step + 1, cfg,
                    phases_done, critic,
                )
                _prune(ckpt_dir, cfg.keep_last)
                if on_checkpoint is not None:
                    on_checkpoint(written)

        final = save_population(
            ckpt_dir / "final.pt", agents, cfg.total_steps, cfg, phases_done, critic
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
        if rollout_pool is not None:
            rollout_pool.close()

    return summary
