"""Training the burial policy: PPO over the landlord's eight picks.

Run inside the pre-play phase (``train.declare_loop.DeclareLearner``),
on the same hands the declaration policy trains on and with play frozen.
Every pick of a burial shares one return: the privileged critic's value
at the play start the burial produced, from the landlord's seat, in the
critic's units times the phase's ``reward_scale``. The baseline is the
burial network's own value head, which reads the other hands.

The critic is the reward on the user's decision, and it is a learned
estimate, so a policy maximising it can learn the critic's mistakes --
what sank the value climb. Two things push back: the critic keeps
training on the hands actually played from these burials, so it is
corrected where the policy leans, and play starts keep a share of
random burials so it stays accurate off the policy's path. Whether that
is enough is for the play benchmarks to say, not this loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..nn.bury_net import BuryNet
from ..sim.bury import BuryRows

__all__ = ["BuryTrainConfig", "BuryLearner"]


@dataclass(frozen=True)
class BuryTrainConfig:
    epochs: int = 4
    minibatch: int = 1024
    lr: float = 3e-4
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class BuryLearner:
    def __init__(
        self, net: BuryNet, cfg: BuryTrainConfig | None = None, device: Any = "cpu"
    ) -> None:
        self.net = net
        self.cfg = cfg or BuryTrainConfig()
        self.device = torch.device(device)
        self.opt = torch.optim.AdamW(net.parameters(), lr=self.cfg.lr)

    def update(self, rows: BuryRows, returns: np.ndarray) -> dict[str, float]:
        """One PPO update. ``returns[i]`` is row ``i``'s return."""
        cfg = self.cfg
        dev = self.device
        cards = torch.from_numpy(rows.cards).to(dev)
        ctx = torch.from_numpy(rows.ctx).to(dev)
        mask = torch.from_numpy(rows.mask).to(dev)
        priv = torch.from_numpy(rows.privileged).to(dev)
        action = torch.from_numpy(rows.action).to(dev)
        old_logp = torch.from_numpy(rows.logp).to(dev)
        ret = torch.from_numpy(np.asarray(returns, dtype=np.float32)).to(dev)
        adv = ret - torch.from_numpy(rows.value).to(dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv

        self.net.train()
        n = len(rows)
        totals = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        count = 0
        for _ in range(cfg.epochs):
            order = torch.randperm(n, device=dev)
            for lo in range(0, n, cfg.minibatch):
                idx = order[lo : lo + cfg.minibatch]
                _logits, logp_all, v = self.net(cards[idx], ctx[idx], mask[idx], priv[idx])
                logp = logp_all.gather(1, action[idx, None]).squeeze(1)
                ratio = torch.exp(logp - old_logp[idx])
                a = adv[idx]
                pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                probs = logp_all.exp()
                ent = -(probs * torch.where(mask[idx], logp_all, torch.zeros_like(logp_all))).sum(1).mean()
                vloss = F.mse_loss(v, ret[idx])
                loss = pg + cfg.vf_coef * vloss - cfg.ent_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                totals["policy"] += float(pg)
                totals["value"] += float(vloss)
                totals["entropy"] += float(ent)
                totals["clip_frac"] += float(((ratio - 1).abs() > cfg.clip).float().mean())
                count += 1
        self.net.eval()
        return {
            **{f"bury/{k}": v / max(count, 1) for k, v in totals.items()},
            "bury/rows": float(n),
            "bury/return_mean": float(np.mean(returns)),
        }
