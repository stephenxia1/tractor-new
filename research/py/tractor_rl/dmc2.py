"""DMC v2: points-primary targets, privileged distillation, auxiliary heads.

Two changes to what is being regressed.

**Points, not levels.** The old target ended in the §8.3 level bracket,
which is a step function of points: 79 and 81 points produce different
integers from near-identical play, and the term fires once per hand where
points arrive every trick. The primary target is now the discounted point
return -- dense and smooth -- and the level outcome becomes an auxiliary
head so the 80-point cliff stays represented, since it is real strategy
rather than noise.

**A privileged teacher.** The Monte-Carlo return conditions on one
sampled world out of an enormous number, so its variance is dominated by
the deal rather than by the decision. A critic that sees every hand
predicts the same expectation with far less noise, and
E[Q_priv | policy view] = E[G | policy view], so distilling it into the
policy network is unbiased in expectation while being much quieter. The
blend weight `lambda_mc` keeps some raw return in the mix, because the
teacher is itself only as good as its own training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layout as L
from .encoding import NUM_ENC_SLOTS
from .features2 import (
    Episode,
    action_features,
    belief_targets,
    history_features,
    privileged_features,
    state_features,
)


@dataclass(slots=True)
class DMC2Config:
    gamma: float = 0.995
    point_scale: float = 100.0
    epsilon: float = 0.02
    buffer_capacity: int = 400_000
    batch_size: int = 512
    lr: float = 1e-4
    critic_lr: float = 3e-4
    grad_clip: float = 5.0
    train_every: int = 2
    steps_per_train: int = 1
    min_buffer: int = 20_000
    # How much raw Monte-Carlo return survives in the target. 1.0 is the
    # old behaviour (no teacher); 0.0 trusts the critic completely.
    lambda_mc: float = 0.3
    # Auxiliary loss weights.
    w_level: float = 0.3
    w_value: float = 0.3
    w_belief: float = 0.5
    # Warm-up: the teacher is useless until it has trained a little, so
    # ignore it for this many learner steps.
    critic_warmup_steps: int = 2000


def hand_returns(episode: Episode, cfg: DMC2Config) -> tuple[np.ndarray, float]:
    """(discounted point return per trick, signed level outcome).

    Both are from the attackers' point of view; the caller flips the sign
    for defending seats.
    """
    n = episode.num_tricks
    level = float(episode.level_delta) * (-1.0 if episode.defenders_won else 1.0)
    if n == 0:
        return np.zeros(0, dtype=np.float32), level
    captured = np.where(episode.trick_attacker_won, episode.trick_points, 0)
    rewards = captured.astype(np.float64) / cfg.point_scale
    # Whatever the attackers finished with beyond the tricks is the kitty
    # multiplier, won or lost on the final trick.
    rewards[-1] += (float(episode.attacker_points) - float(captured.sum())) / cfg.point_scale
    out = np.zeros(n, dtype=np.float32)
    running = 0.0
    for t in range(n - 1, -1, -1):
        running = rewards[t] + cfg.gamma * running
        out[t] = running
    return out, level


class ReplayBuffer2:
    """Raw wire bytes plus both targets.

    Records are kept in wire form (~1 KB with the privileged block) rather
    than as features: it is smaller, and a change to the featuriser does
    not invalidate everything already collected.
    """

    def __init__(self, capacity: int, state_bytes: int) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, state_bytes), dtype=np.uint8)
        self.actions = np.zeros((capacity, L.CANDIDATE_BYTES), dtype=np.uint8)
        self.g_points = np.zeros(capacity, dtype=np.float32)
        self.g_level = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.cursor = 0

    def add(self, states, actions, g_points, g_level) -> None:
        n = states.shape[0]
        if n == 0:
            return
        if n > self.capacity:
            states, actions = states[-self.capacity:], actions[-self.capacity:]
            g_points, g_level = g_points[-self.capacity:], g_level[-self.capacity:]
            n = self.capacity
        end = self.cursor + n
        if end <= self.capacity:
            sl = slice(self.cursor, end)
            self.states[sl], self.actions[sl] = states, actions
            self.g_points[sl], self.g_level[sl] = g_points, g_level
        else:
            first = self.capacity - self.cursor
            self.states[self.cursor:], self.actions[self.cursor:] = states[:first], actions[:first]
            self.g_points[self.cursor:], self.g_level[self.cursor:] = g_points[:first], g_level[:first]
            rest = n - first
            self.states[:rest], self.actions[:rest] = states[first:], actions[first:]
            self.g_points[:rest], self.g_level[:rest] = g_points[first:], g_level[first:]
        self.cursor = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return self.states[idx], self.actions[idx], self.g_points[idx], self.g_level[idx]


class Learner2:
    def __init__(
        self,
        net: nn.Module,
        critic: nn.Module,
        cfg: DMC2Config,
        device: torch.device,
    ) -> None:
        self.net = net
        self.critic = critic
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.RMSprop(net.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.RMSprop(critic.parameters(), lr=cfg.critic_lr)
        self.steps = 0
        self.stats: dict[str, float] = {}

    def step(self, buffer: ReplayBuffer2, rng: np.random.Generator) -> dict[str, float] | None:
        cfg = self.cfg
        if buffer.size < cfg.min_buffer:
            return None
        states, actions, g_points, g_level = buffer.sample(cfg.batch_size, rng)
        n = states.shape[0]
        owner_np = np.arange(n, dtype=np.int64)

        sf = torch.from_numpy(state_features(states)).to(self.device)
        af = torch.from_numpy(action_features(actions, states, owner_np)).to(self.device)
        owner = torch.from_numpy(owner_np).to(self.device)
        roles = torch.from_numpy(
            states[:, L.H_ROLE].astype(np.int64)).to(self.device)
        tokens_np, mask_np = history_features(states)
        tokens = torch.from_numpy(tokens_np).to(self.device)
        mask = torch.from_numpy(mask_np).to(self.device)
        tgt_points = torch.from_numpy(g_points).to(self.device)
        tgt_level = torch.from_numpy(g_level).to(self.device)

        # --- privileged teacher, trained on the raw return ---
        priv = torch.from_numpy(privileged_features(states)).to(self.device)
        q_priv = self.critic(sf, priv, af, owner)
        critic_loss = F.mse_loss(q_priv, tgt_points)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_opt.step()

        # --- policy network ---
        out = self.net(sf, af, owner, roles, tokens, mask, want="all")
        if self.steps < cfg.critic_warmup_steps:
            target = tgt_points
        else:
            target = cfg.lambda_mc * tgt_points + (1.0 - cfg.lambda_mc) * q_priv.detach()

        loss_q = F.mse_loss(out["q"], target)
        loss_level = F.mse_loss(out["q_level"], tgt_level)
        loss_v = F.mse_loss(out["v"], tgt_points)
        belief = torch.from_numpy(belief_targets(states)).to(self.device)
        loss_belief = F.binary_cross_entropy_with_logits(out["belief"], belief)
        loss = (
            loss_q
            + cfg.w_level * loss_level
            + cfg.w_value * loss_v
            + cfg.w_belief * loss_belief
        )
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
        self.opt.step()

        self.steps += 1
        with torch.no_grad():
            # How much quieter the teacher's target is than the raw return
            # is the whole point of it, so measure it rather than assume.
            spread_mc = float(tgt_points.std())
            spread_teacher = float((tgt_points - q_priv.detach()).std())
            belief_acc = float(
                ((out["belief"] > 0).float() == belief).float().mean())
        self.stats = {
            "loss": float(loss.detach()),
            "loss_q": float(loss_q.detach()),
            "loss_level": float(loss_level.detach()),
            "loss_value": float(loss_v.detach()),
            "loss_belief": float(loss_belief.detach()),
            "critic_loss": float(critic_loss.detach()),
            "target_std": spread_mc,
            "critic_residual_std": spread_teacher,
            "belief_acc": belief_acc,
        }
        return self.stats
