"""The Q-network and action selection.

One network, not three. The earlier DMC attempt in packages/engine used a
separate net per role (landlord / defender / attacker); here the role is an
input feature instead, so all three roles share a representation and every
sample trains every parameter. All four seats play mechanically identical
tractor -- what differs is which points they want -- so there is little for
separate towers to specialise on and a lot of data to gain by sharing.

Scoring is per candidate: the state is embedded once, each legal move is
embedded, and the pair is scored. That sidesteps any fixed action-space
head, which a game with variable-length combos does not have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .features import ACTION_DIM, STATE_DIM


@dataclass(slots=True)
class ModelConfig:
    hidden: int = 512
    state_emb: int = 256
    action_hidden: int = 128
    head_hidden: tuple[int, int] = (256, 128)


class QNet(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.trunk = nn.Sequential(
            nn.Linear(STATE_DIM, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.state_emb),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_DIM, cfg.action_hidden),
            nn.ReLU(),
        )
        h1, h2 = cfg.head_hidden
        self.head = nn.Sequential(
            nn.Linear(cfg.state_emb + cfg.action_hidden, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(
        self,
        states: torch.Tensor,  # [B, STATE_DIM]
        actions: torch.Tensor,  # [N, ACTION_DIM]
        owner: torch.Tensor,  # [N] index into states
    ) -> torch.Tensor:  # [N]
        state_emb = self.trunk(states)
        pair = torch.cat([state_emb[owner], self.action_encoder(actions)], dim=1)
        return self.head(pair).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def group_argmax(
    values: np.ndarray,  # [N]
    owner: np.ndarray,  # [N]
    offsets: np.ndarray,  # [B] first row of each group
    counts: np.ndarray,  # [B]
) -> np.ndarray:
    """Best candidate index *within* each decision, vectorised.

    Candidates arrive grouped by decision, so scattering them into a padded
    [B, maxK] matrix and taking a row-wise argmax beats any per-decision
    loop.
    """
    b = counts.shape[0]
    if b == 0:
        return np.zeros(0, dtype=np.int64)
    max_k = int(counts.max())
    padded = np.full((b, max_k), -np.inf, dtype=np.float32)
    within = np.arange(values.shape[0], dtype=np.int64) - offsets[owner]
    padded[owner, within] = values
    return padded.argmax(axis=1)


def epsilon_greedy(
    best: np.ndarray,
    counts: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace a fraction of greedy picks with uniformly random legal moves."""
    if epsilon <= 0.0:
        return best
    explore = rng.random(counts.shape[0]) < epsilon
    if not explore.any():
        return best
    random_pick = (rng.random(counts.shape[0]) * counts).astype(np.int64)
    return np.where(explore, random_pick, best)
