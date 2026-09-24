"""The declaration policy: when to declare trump, and what.

Small on purpose, and separate from the play network so the two can be
frozen and trained in alternation. The actor reads ``sim.deal.declare_obs``
-- the seat's revealed cards, the deal's progress and the standing
declaration -- and scores the 11 fixed actions, illegal ones masked out.
The baseline reads the same view plus the whole deal
(``declare_privileged``); it only ever reduces the variance of a
gradient, so it may know what no seat knows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..sim.deal import DECLARE_OBS_DIM, DECLARE_PRIV_DIM, NUM_ACTIONS
from .model import MASK_LOGIT

__all__ = ["DeclareConfig", "DeclareNet"]


@dataclass(frozen=True)
class DeclareConfig:
    width: int = 256
    depth: int = 2


def _mlp(d_in: int, width: int, depth: int, d_out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.GELU()]
        d = width
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class DeclareNet(nn.Module):
    def __init__(self, cfg: DeclareConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or DeclareConfig()
        self.actor = _mlp(DECLARE_OBS_DIM, cfg.width, cfg.depth, NUM_ACTIONS)
        self.critic = _mlp(DECLARE_OBS_DIM + DECLARE_PRIV_DIM, cfg.width, cfg.depth, 1)
        # A near-uniform start over the legal actions, so early training
        # explores rather than committing to whatever init favoured.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def logits(self, obs: Tensor, mask: Tensor) -> Tensor:
        raw = self.actor(obs)
        return torch.where(mask, raw, torch.full_like(raw, MASK_LOGIT))

    def value(self, obs: Tensor, privileged: Tensor) -> Tensor:
        return self.critic(torch.cat((obs, privileged), dim=-1)).squeeze(-1)

    def forward(
        self, obs: Tensor, mask: Tensor, privileged: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """-> (logits, log_probs, value)."""
        logits = self.logits(obs, mask)
        return logits, F.log_softmax(logits, dim=-1), self.value(obs, privileged)
