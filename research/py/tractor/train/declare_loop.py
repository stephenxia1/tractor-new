"""The declaration phase: train the declaring policy on frozen play.

Each step deals a batch of hands, lets the declaration network make
every declaration decision (sampling, so it explores), has the landlord
bury by the frozen play network's value climb, and scores each seat's
play start with the frozen privileged critic. That score -- the seat's
team's expected point margin from there under the current play policy
-- is the return of every declaration decision the seat made in the
hand. No discount: a hand's declarations are a handful of decisions and
only the position they leave behind is rewarded.

The update is PPO with a clipped ratio and a learned baseline. The play
network is only ever read under ``no_grad``; ``tests/test_declare.py``
checks that a declaration step leaves every one of its parameters as
it found them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..nn.declare import DeclareNet
from ..nn.features import Cache
from .starts import (
    DeclareRows, PrePlayConfig, burial_samples, critic_values, leave_one_out, make_starts,
    playout_values,
)

__all__ = ["DeclareTrainConfig", "DeclareLearner"]


@dataclass(frozen=True)
class DeclareTrainConfig:
    hands_per_step: int = 1024
    epochs: int = 4
    minibatch: int = 512
    lr: float = 3e-4
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # The critic's units are margin / 400; a seat's return is scaled up
    # into something nearer unit variance before the baseline fits it.
    reward_scale: float = 10.0
    # No random burials in this phase: the reward should be the position
    # the real pipeline makes. ``pre.burial`` says what buries -- the
    # value climb, or the burial policy, which then trains here too.
    pre: PrePlayConfig = field(default_factory=lambda: PrePlayConfig(random_burial_frac=0.0))
    # Burials per landlord for the burial policy's update. 1: one, with the
    # burial network's value head as baseline (every run before
    # 2026-11-14). k > 1: k burials of the same landlord, each valued by
    # the critic, each judged against the mean of the other k - 1.
    bury_samples: int = 1
    # With k > 1, the burial minibatch grows k-fold, so an update takes the
    # optimiser steps one burial per landlord did, each on k times the rows.
    # E3 arm C without it: half of every update's rows hit the PPO clip.
    bury_scale_minibatch: bool = True
    # What a burial's return is, with bury_samples > 1. "critic": the
    # privileged critic's value of the play start (the design every run
    # so far used). "playout": the start played to the end by the frozen
    # play policy, the landlord's team's actual margin in the critic's
    # units -- unbiased where the critic is not (E3: burials the critic
    # rated above the heuristic's lost to it in play), noisier, and k
    # hands of play per landlord. bury_landlords > 0 caps the landlords a
    # step's burial update uses, to bound that cost.
    bury_reward: str = "critic"
    bury_landlords: int = 0
    # A declaration's return. "critic": the seat's privileged critic value
    # at play start (the design every run so far used). "playout": the
    # start played to the end once by the frozen play policy, the seat's
    # team's actual margin in the critic's units. Single agent only.
    declare_reward: str = "critic"


class DeclareLearner:
    def __init__(
        self, net: DeclareNet, cfg: DeclareTrainConfig | None = None, device: Any = "cpu",
        bury: Any = None,
    ) -> None:
        """``bury``, a ``train.bury_loop.BuryLearner``, trains the burial
        policy on the same hands in the same phase."""
        self.net = net
        self.cfg = cfg or DeclareTrainConfig()
        self.device = torch.device(device)
        self.opt = torch.optim.AdamW(net.parameters(), lr=self.cfg.lr)
        self.cache = Cache()
        self.bury = bury

    def step(self, play_net: Any, rng: np.random.Generator) -> dict[str, float]:
        cfg = self.cfg
        self.net.eval()
        bury_net = None if self.bury is None else self.bury.net
        if bury_net is not None:
            bury_net.eval()
        starts, pre_rows, stats = make_starts(
            rng, cfg.hands_per_step, play_net, self.net, self.device,
            pre=cfg.pre, record=True, cache=self.cache, bury_net=bury_net,
        )
        values = critic_values(play_net, starts, self.device)
        assert pre_rows is not None
        if self.bury is not None and pre_rows.bury is not None and len(pre_rows.bury):
            # The landlord's play-start value is the return of all eight
            # of its picks.
            landlord = np.array([s.landlord for s in starts], dtype=np.int64)
            if cfg.bury_samples > 1:
                stats.update(bury_update_loo(
                    self.bury, play_net, starts, pre_rows.bury, values, cfg, rng,
                    self.device, self.cache))
            else:
                hand = pre_rows.bury.hand
                ret = values[hand, landlord[hand]].astype(np.float32) * cfg.reward_scale
                stats.update(self.bury.update(pre_rows.bury, ret))
        rows = pre_rows.declare
        if len(rows) == 0:
            return {**stats, "declare/rows": 0.0}
        if cfg.declare_reward == "playout":
            margin = playout_values(play_net, starts, self.device, rng=rng)
            landlord = np.array([s.landlord for s in starts], dtype=np.int64)
            same = (np.arange(4)[None, :] % 2) == (landlord[:, None] % 2)
            seat_values = np.where(same, margin[:, None], -margin[:, None])
            ret = seat_values[rows.hand, rows.seat].astype(np.float32) * cfg.reward_scale
            stats["declare/playout_margin_mean"] = float(np.abs(margin).mean())
        elif cfg.declare_reward == "critic":
            ret = values[rows.hand, rows.seat].astype(np.float32) * cfg.reward_scale
        else:
            raise ValueError(f"declare_reward is critic or playout, not {cfg.declare_reward!r}")
        return {
            **stats,
            **self.update(rows, ret),
            "declare/critic_landlord_value": float(
                np.mean([values[i, s.landlord] for i, s in enumerate(starts)])
            ),
        }

    def update(self, rows: DeclareRows, ret: np.ndarray) -> dict[str, float]:
        """PPO on declaration rows; ``ret[i]`` is row ``i``'s scaled return.

        Apart from ``step`` so that a population, which deals one batch
        for all its agents, can hand each agent's learner its own rows.
        """
        cfg = self.cfg
        dev = self.device
        obs = torch.from_numpy(rows.obs).to(dev)
        priv = torch.from_numpy(rows.privileged).to(dev)
        mask = torch.from_numpy(rows.mask).to(dev)
        action = torch.from_numpy(rows.action).to(dev)
        old_logp = torch.from_numpy(rows.logp).to(dev)
        returns = torch.from_numpy(ret).to(dev)
        adv = returns - torch.from_numpy(rows.value).to(dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv

        self.net.train()
        n = len(rows)
        losses = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        count = 0
        for _ in range(cfg.epochs):
            order = torch.randperm(n, device=dev)
            for lo in range(0, n, cfg.minibatch):
                idx = order[lo : lo + cfg.minibatch]
                _logits, logp_all, v = self.net(obs[idx], mask[idx], priv[idx])
                logp = logp_all.gather(1, action[idx, None]).squeeze(1)
                ratio = torch.exp(logp - old_logp[idx])
                a = adv[idx]
                pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                probs = logp_all.exp()
                ent = -(probs * torch.where(mask[idx], logp_all, torch.zeros_like(logp_all))).sum(1).mean()
                vloss = F.mse_loss(v, returns[idx])
                loss = pg + cfg.vf_coef * vloss - cfg.ent_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                losses["policy"] += float(pg)
                losses["value"] += float(vloss)
                losses["entropy"] += float(ent)
                losses["clip_frac"] += float(((ratio - 1).abs() > cfg.clip).float().mean())
                count += 1
        self.net.eval()

        declared = rows.action != 0
        return {
            **{f"declare/{k}": v / max(count, 1) for k, v in losses.items()},
            "declare/rows": float(n),
            "declare/return_mean": float(ret.mean()),
            "declare/declare_rate": float(declared.mean()),
        }


def bury_update_loo(
    bury: Any, play_net: Any, starts: list, rows: Any, values: np.ndarray,
    cfg: DeclareTrainConfig, rng: np.random.Generator, device: Any, cache: Cache,
    agents: Any = None, shared: Any = None, learners: Any = None,
) -> dict[str, float]:
    """The burial update with ``cfg.bury_samples`` burials per landlord and
    a leave-one-out baseline. ``values`` are the critic's at ``starts``
    (sample 0). For a population, ``agents`` routes each landlord to its
    agent, ``bury`` and ``learners`` are per agent and ``shared`` is the
    shared critic. -> the update's statistics (averaged over agents)."""
    if cfg.bury_reward not in ("critic", "playout"):
        raise ValueError(f"bury_reward is critic or playout, not {cfg.bury_reward!r}")
    if cfg.bury_landlords and cfg.bury_landlords > 0:
        hands = np.unique(rows.hand)
        if len(hands) > cfg.bury_landlords:
            keep = rng.choice(hands, size=cfg.bury_landlords, replace=False)
            rows = rows.subset(np.nonzero(np.isin(rows.hand, keep))[0])
    nets = [b.net for b in learners] if learners is not None else bury.net
    merged, sample_of, hand_of, more = burial_samples(
        nets, starts, rows, cfg.bury_samples, rng, device, cache, agents=agents)
    n = len(more) // cfg.bury_samples
    landlord = np.array([s.landlord for s in more], dtype=np.int64)
    v = np.empty(len(more), dtype=np.float64)
    if cfg.bury_reward == "playout":
        v[:] = playout_values(play_net, more, device, agents=(
            None if agents is None else np.asarray(agents)[hand_of]), shared=shared, rng=rng)
    else:
        v_more = critic_values(play_net, more[n:], device, agents=(
            None if agents is None else np.asarray(agents)[hand_of[n:]]), shared=shared)
        v[:n] = values[hand_of[:n], landlord[:n]]
        v[n:] = v_more[np.arange(len(more) - n), landlord[n:]]
    table = v.reshape(cfg.bury_samples, n)
    adv = leave_one_out(table).reshape(-1)
    # Each row's landlord and sample -> its column in ``table``.
    col = {int(h): j for j, h in enumerate(hand_of[:n])}
    idx = sample_of * n + np.array([col[int(h)] for h in merged.hand])
    ret = (v[idx] * cfg.reward_scale).astype(np.float32)
    a = (adv[idx] * cfg.reward_scale).astype(np.float32)
    stats_extra = {"bury/loo_spread": float(table.std(axis=0).mean() * cfg.reward_scale),
                   "bury/samples": float(cfg.bury_samples)}
    mb = None
    if cfg.bury_scale_minibatch:
        base = (learners[0] if learners is not None else bury).cfg.minibatch
        mb = base * cfg.bury_samples
    if learners is None:
        return {**bury.update(merged, ret, a, minibatch=mb), **stats_extra}
    owner = np.asarray(agents)[merged.hand, [starts[int(h)].landlord for h in merged.hand]]
    per = []
    for k, learner in enumerate(learners):
        sel = np.nonzero(owner == k)[0]
        if len(sel):
            per.append(learner.update(merged.subset(sel), ret[sel], a[sel], minibatch=mb))
    keys = {key for p in per for key in p}
    return {**{key: float(np.mean([p[key] for p in per if key in p])) for key in keys},
            **stats_extra}
