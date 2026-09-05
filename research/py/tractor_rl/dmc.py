"""Deep Monte-Carlo: replay buffer, return computation, and the learner.

DMC in one line: play a hand to the end, look at what actually happened,
and regress Q(state, action) onto the discounted return that followed. No
bootstrapping, no importance weights, no policy distribution -- which is
why it copes with an action space that changes shape at every decision.

Reward stream, from the acting seat's point of view:

* per trick, the points the *attackers* captured, positive for attacking
  seats and negative for defending seats (defenders score nothing by
  capturing; their job is denial, and the sign captures exactly that);
* the kitty bonus (rules.md §8.2), attributed to the last trick;
* a terminal term for the level swing the hand produced (§8.3), which is
  what the game is actually played for -- the 80-point cliff means the
  raw point total is not a linear objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import layout as L
from .features import Episode, action_features, state_features


@dataclass(slots=True)
class DMCConfig:
    gamma: float = 0.995
    # Points are divided by this, so a whole 200-point deck is about 2.0.
    point_scale: float = 100.0
    # Weight on the terminal level swing (-3..+3 before weighting).
    terminal_weight: float = 1.0
    epsilon: float = 0.02
    buffer_capacity: int = 400_000
    batch_size: int = 512
    lr: float = 1e-4
    grad_clip: float = 5.0
    # Learner cadence: this many optimiser steps every `train_every` rounds.
    train_every: int = 2
    steps_per_train: int = 1
    min_buffer: int = 20_000


def hand_returns(episode: Episode, cfg: DMCConfig) -> np.ndarray:
    """Attacker-point-of-view discounted return at each trick of a hand.

    `out[t]` is the return for a decision made during trick `t`, i.e. it
    includes the points trick `t` itself is about to award.
    """
    n = episode.num_tricks
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    captured = np.where(episode.trick_attacker_won, episode.trick_points, 0)
    rewards = captured.astype(np.float64) / cfg.point_scale
    # Anything in the final attacker total beyond the tricks is the kitty
    # multiplier; it is won or lost on the last trick, so it belongs there.
    bonus = float(episode.attacker_points) - float(captured.sum())
    rewards[-1] += bonus / cfg.point_scale

    terminal = float(episode.level_delta) * (-1.0 if episode.defenders_won else 1.0)
    running = terminal * cfg.terminal_weight
    out = np.zeros(n, dtype=np.float32)
    for t in range(n - 1, -1, -1):
        running = rewards[t] + cfg.gamma * running
        out[t] = running
    return out


class ReplayBuffer:
    """Stores raw wire bytes, not features.

    Keeping records in their 588-byte wire form costs ~4x less memory than
    dense features and means a change to features.py does not invalidate
    everything already collected.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, L.STATE_BYTES), dtype=np.uint8)
        self.actions = np.zeros((capacity, L.CANDIDATE_BYTES), dtype=np.uint8)
        self.targets = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.cursor = 0

    def add(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> None:
        n = states.shape[0]
        if n == 0:
            return
        if n >= self.capacity:  # pragma: no cover - absurd batch
            states, actions, targets = states[-self.capacity :], actions[-self.capacity :], targets[-self.capacity :]
            n = self.capacity
        end = self.cursor + n
        if end <= self.capacity:
            sl = slice(self.cursor, end)
            self.states[sl] = states
            self.actions[sl] = actions
            self.targets[sl] = targets
        else:
            first = self.capacity - self.cursor
            self.states[self.cursor :] = states[:first]
            self.actions[self.cursor :] = actions[:first]
            self.targets[self.cursor :] = targets[:first]
            rest = n - first
            self.states[:rest] = states[first:]
            self.actions[:rest] = actions[first:]
            self.targets[:rest] = targets[first:]
        self.cursor = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return self.states[idx], self.actions[idx], self.targets[idx]


class Learner:
    def __init__(self, model: nn.Module, cfg: DMCConfig, device: torch.device) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr)
        self.steps = 0
        self.last_loss = float("nan")

    def step(self, buffer: ReplayBuffer, rng: np.random.Generator) -> float | None:
        if buffer.size < self.cfg.min_buffer:
            return None
        states, actions, targets = buffer.sample(self.cfg.batch_size, rng)
        sf = torch.from_numpy(state_features(states)).to(self.device)
        owner_np = np.arange(states.shape[0], dtype=np.int64)
        af = torch.from_numpy(action_features(actions, states, owner_np)).to(self.device)
        owner = torch.from_numpy(owner_np).to(self.device)
        tgt = torch.from_numpy(targets).to(self.device)

        q = self.model(sf, af, owner)
        loss = torch.nn.functional.mse_loss(q, tgt)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.steps += 1
        self.last_loss = float(loss.detach())
        return self.last_loss
