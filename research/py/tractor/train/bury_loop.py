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
from ..sim.bury import BuryRows, ImitationRows

__all__ = ["BuryTrainConfig", "BuryLearner", "imitate"]


@dataclass(frozen=True)
class BuryTrainConfig:
    epochs: int = 4
    minibatch: int = 1024
    lr: float = 3e-4
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # The forward under bfloat16 autocast on CUDA, as the play network's
    # and the shared critic's learners run; the losses stay float32.
    # Measured on an L40S (v6, one agent's 7920 rows, 4 epochs): 1.79 s
    # in float32, 0.92 s here -- about 7 s of a population's pre-play
    # step. Off on the CPU, where bfloat16 matmuls are emulated.
    bf16: bool = True


class BuryLearner:
    def __init__(
        self, net: BuryNet, cfg: BuryTrainConfig | None = None, device: Any = "cpu"
    ) -> None:
        self.net = net
        self.cfg = cfg or BuryTrainConfig()
        self.device = torch.device(device)
        self.opt = torch.optim.AdamW(net.parameters(), lr=self.cfg.lr)
        self.amp = bool(self.cfg.bf16) and self.device.type == "cuda"

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
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                    enabled=self.amp):
                    _logits, logp_all, v = self.net(cards[idx], ctx[idx], mask[idx], priv[idx])
                logp_all = logp_all.float()
                v = v.float()
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


def imitate(
    net: BuryNet,
    rows: ImitationRows,
    device: Any = "cpu",
    epochs: int = 4,
    minibatch: int = 1024,
    lr: float = 3e-4,
    max_grad_norm: float = 0.5,
    seed: int = 0,
) -> dict[str, float]:
    """Fit the burial policy to a teacher's burials (``imitation_rows``).

    A burial is a set, so at each pick any of the teacher's cards still
    unburied is right: the loss is -log of the probability the policy
    puts on that set, not on one card in an arbitrary order. A warm start
    only -- the value head is left alone, and the optimiser is the
    caller's to discard. -> the last epoch's loss and how often the
    policy's own top pick was one of the teacher's cards.
    """
    dev = torch.device(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    cards = torch.from_numpy(rows.cards).to(dev)
    ctx = torch.from_numpy(rows.ctx).to(dev)
    mask = torch.from_numpy(rows.mask).to(dev)
    priv = torch.from_numpy(rows.privileged).to(dev)
    target = torch.from_numpy(rows.target).to(dev)
    if bool((target & ~mask).any()):
        raise ValueError("a teacher's card is not among the held cards")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = len(rows)
    net.train()
    loss_sum = hit_sum = 0.0
    for _ in range(epochs):
        loss_sum = hit_sum = 0.0
        order = torch.randperm(n, generator=gen).to(dev)
        for lo in range(0, n, minibatch):
            idx = order[lo : lo + minibatch]
            _logits, logp_all, _v = net(cards[idx], ctx[idx], mask[idx], priv[idx])
            logp_all = logp_all.float()
            on_set = torch.where(target[idx], logp_all, torch.full_like(logp_all, -1e9))
            loss = -torch.logsumexp(on_set, dim=1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()
            top = logp_all.argmax(dim=1)
            hit = target[idx].gather(1, top[:, None]).float().mean()
            loss_sum += float(loss.detach()) * len(idx)
            hit_sum += float(hit) * len(idx)
    net.eval()
    return {"imitate/loss": loss_sum / max(n, 1), "imitate/top_pick_in_set": hit_sum / max(n, 1),
            "imitate/rows": float(n)}
