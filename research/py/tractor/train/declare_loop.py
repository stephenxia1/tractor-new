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
from .starts import PrePlayConfig, critic_values, make_starts

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
            hand = pre_rows.bury.hand
            ret = values[hand, landlord[hand]].astype(np.float32) * cfg.reward_scale
            stats.update(self.bury.update(pre_rows.bury, ret))
        rows = pre_rows.declare
        if len(rows) == 0:
            return {**stats, "declare/rows": 0.0}
        ret = values[rows.hand, rows.seat].astype(np.float32) * cfg.reward_scale

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

        # How much a declaration was worth to the team that made it: the
        # declaring team's play-start value, against hands nobody declared.
        declared = rows.action != 0
        out = {
            **stats,
            **{f"declare/{k}": v / max(count, 1) for k, v in losses.items()},
            "declare/rows": float(n),
            "declare/return_mean": float(ret.mean()),
            "declare/declare_rate": float(declared.mean()),
            "declare/critic_landlord_value": float(
                np.mean([values[i, s.landlord] for i, s in enumerate(starts)])
            ),
        }
        return out
