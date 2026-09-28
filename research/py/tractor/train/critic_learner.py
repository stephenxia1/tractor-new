"""Training a population's shared critic.

Each agent's PPO update trains its actor on its own rows. The critic they
share trains here, on every row of the round -- eight times what any one
agent's critic saw when each had its own -- against the same targets the
actors' advantages were built from: ``buffer.ret`` after GAE, whose
bootstrap values this critic recorded during the rollout.

It reads only what a critic reads -- the raw history, the context, the
card features, the trick in progress and the privileged hands -- so a
minibatch moves about half the bytes of an actor's, and it is told which
agent sits in each seat (``model.relative_seat_agents``), which the
buffer does not store and the caller supplies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..nn.features import PLANE_TRICK
from ..sim.state import NUM_SEATS
from .ppo import _bf16_supported, _minibatches, explained_variance

__all__ = ["CriticTrainConfig", "CriticLearner"]


@dataclass(frozen=True)
class CriticTrainConfig:
    """Defaults match the actors' PPO: one pass, 256 rows, AdamW 3e-4,
    history-length buckets, gradient norm 0.5, bf16."""

    lr: float = 3e-4
    minibatch: int = 256
    epochs: int = 1
    length_buckets: int = 4
    max_grad_norm: float = 0.5
    bf16: bool = True


class CriticLearner:
    def __init__(self, critic: Any, cfg: CriticTrainConfig | None = None,
                 device: Any = "cpu") -> None:
        self.critic = critic
        self.cfg = cfg or CriticTrainConfig()
        self.device = torch.device(device)
        self.amp = bool(self.cfg.bf16) and _bf16_supported(self.device)
        params = [p for p in critic.parameters() if p.requires_grad]
        try:
            self.opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=0.0, fused=True)
        except (RuntimeError, ValueError, TypeError):
            self.opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=0.0)

    def _batch(self, buffer: Any, idx: np.ndarray, seat_agents: torch.Tensor):
        dev = self.device

        def t(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a)).to(dev)

        return dict(
            history_enc=t(buffer.history[idx]).float(),
            history_len=t(buffer.history_len[idx]),
            privileged=t(buffer.privileged[idx]).float(),
            context=t(buffer.context[idx]).float(),
            cards=t(buffer.cards[idx]).float(),
            trick=t(buffer.planes[idx][:, PLANE_TRICK : PLANE_TRICK + NUM_SEATS]).float(),
            seat_agents=seat_agents[torch.from_numpy(idx).to(dev)],
        ), t(buffer.ret[idx]).float()

    def update(self, buffer: Any, seat_agents: np.ndarray) -> dict[str, float]:
        """One fit over every filled row. ``seat_agents`` is int64[n, 4],
        row ``i``'s agents by relative seat. -> metrics under ``critic/``."""
        cfg = self.cfg
        n = len(buffer)
        if n == 0:
            raise ValueError("update() called on an empty buffer")
        ev = explained_variance(np.asarray(buffer.value)[:n], np.asarray(buffer.ret)[:n])
        agents = torch.from_numpy(np.asarray(seat_agents, dtype=np.int64)).to(self.device)
        self.critic.train()
        total = 0.0
        grad = 0.0
        rows = 0
        skipped = 0
        for _ in range(max(1, int(cfg.epochs))):
            order = torch.randperm(n).numpy()
            for idx in _minibatches(order, buffer, int(cfg.minibatch), int(cfg.length_buckets)):
                inputs, target = self._batch(buffer, idx, agents)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                    enabled=self.amp):
                    value = self.critic(detach=False, **inputs)
                loss = F.mse_loss(value.float(), target)
                if not bool(torch.isfinite(loss)):
                    skipped += 1
                    continue
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                if not bool(torch.isfinite(norm)):
                    self.opt.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
                self.opt.step()
                k = len(idx)
                total += float(loss.detach()) * k
                grad += float(norm) * k
                rows += k
        self.critic.eval()
        return {
            "critic/value_loss": total / max(rows, 1),
            "critic/grad_norm": grad / max(rows, 1),
            "critic/explained_variance": ev,
            "critic/rows": float(n),
            "critic/nonfinite_minibatches": float(skipped),
        }
