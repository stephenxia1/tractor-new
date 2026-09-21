"""DMC self-play trainer for Tractor.

v1 design — single process, single actor, single learner.

    main loop:
      1. Run one episode against itself via the TS subprocess.
      2. Compute Monte-Carlo returns per (role, decision).
      3. Append samples to per-role buffers.
      4. Every K episodes, sample batches and do gradient steps.
      5. Periodically checkpoint weights.

Usage from the engine package:
    python python/dmc/train.py
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# Ensure the parent dir is on sys.path so `from dmc.foo` works regardless
# of cwd. The script is intended to run from packages/engine/ but we
# accept other launchers too.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from dmc.constants import (  # noqa: E402
    ACTION_DIM,
    EPS_GREEDY,
    GAMMA,
    HISTORY_MAX_STEPS,
    PER_STEP_DIM,
    ROLES,
    SCALAR_DIM,
    STATIC_DIM,
)
from dmc.network import QNet, TripleQNet, num_parameters  # noqa: E402
from dmc.subprocess_driver import Decision, EpisodeResult, TrainHost  # noqa: E402

# Per-role replay sample. We keep encoded tensors as floats (so they can
# be stacked into batches without per-sample reconstruction work) but
# limit the buffer size to keep RAM bounded.
class Sample:
    __slots__ = ('static', 'scalars', 'history', 'history_len', 'action', 'target')
    def __init__(self, static, scalars, history, history_len, action, target):
        self.static = static
        self.scalars = scalars
        self.history = history
        self.history_len = history_len
        self.action = action
        self.target = target


def observation_to_tensors(obs: dict, device: torch.device):
    """Convert an Observation dict (from TS) into torch tensors."""
    static = torch.tensor(obs['staticChannels'], dtype=torch.float32, device=device)
    scalars = torch.tensor(obs['scalars'], dtype=torch.float32, device=device)
    history = torch.tensor(obs['history'], dtype=torch.float32, device=device).view(
        HISTORY_MAX_STEPS, PER_STEP_DIM
    )
    history_len = torch.tensor(obs['historyLen'], dtype=torch.int64, device=device)
    return static, scalars, history, history_len


def candidates_to_tensor(candidates: list, device: torch.device) -> torch.Tensor:
    """Stack the encodings of K candidates into a (K, ACTION_DIM) tensor."""
    return torch.tensor(
        [c['encoding'] for c in candidates],
        dtype=torch.float32,
        device=device,
    )


def compute_discounted_suffix(
    per_trick_attacker_captures: List[int],
    gamma: float,
) -> List[float]:
    """G[t] = sum_{k>=t} gamma^(k-t) * captures[k] (attacker POV).

    Defender POV: negate.
    """
    n = len(per_trick_attacker_captures)
    out = [0.0] * n
    running = 0.0
    for t in range(n - 1, -1, -1):
        running = per_trick_attacker_captures[t] + gamma * running
        out[t] = running
    return out


def team_sign_for_role(role: str) -> float:
    """Attackers maximize the attacker-POV signal; landlord+defender minimize it."""
    return 1.0 if role == 'attacker' else -1.0


class ActionPicker:
    """Encapsulates the actor-side action selection: encode obs, run Q,
    sample epsilon-greedy."""

    def __init__(self, nets: TripleQNet, device: torch.device, eps: float, rng: random.Random) -> None:
        self.nets = nets
        self.device = device
        self.eps = eps
        self.rng = rng

    @torch.no_grad()
    def pick(self, decision: Decision) -> int:
        # Epsilon-greedy: with prob eps, uniform random over candidates.
        k = len(decision.candidates)
        if self.rng.random() < self.eps:
            return self.rng.randrange(k)
        # Greedy: encode state once, score all candidates, argmax.
        net = self.nets.net_for(decision.role)
        static, scalars, history, history_len = observation_to_tensors(decision.observation, self.device)
        state_enc = net.encode_state(
            static.unsqueeze(0),
            scalars.unsqueeze(0),
            history.unsqueeze(0),
            history_len.unsqueeze(0),
        ).squeeze(0)
        actions = candidates_to_tensor(decision.candidates, self.device)
        q = net.forward_batch(state_enc, actions)
        return int(torch.argmax(q).item())


def episode_to_samples(
    episode: EpisodeResult,
    gamma: float,
    device: torch.device,
) -> Dict[str, List[Sample]]:
    """Compute MC targets and produce per-role samples for the replay buffer."""
    discounted = compute_discounted_suffix(episode.per_trick_attacker_captures, gamma)
    per_role: Dict[str, List[Sample]] = {r: [] for r in ROLES}
    for d in episode.decisions:
        if d.chosen_index < 0:
            continue
        chosen = d.candidates[d.chosen_index]
        # Defenders may make decisions during a trick the attackers go on
        # to win late — but the trick_idx here is the trick the decision
        # was for. If trick_idx >= len(discounted), the decision happened
        # in a phase we don't track (shouldn't occur in v1 — play only).
        if d.trick_idx >= len(discounted):
            continue
        sign = team_sign_for_role(d.role)
        target = sign * discounted[d.trick_idx]
        static, scalars, history, history_len = observation_to_tensors(d.observation, device)
        action = torch.tensor(chosen['encoding'], dtype=torch.float32, device=device)
        per_role[d.role].append(Sample(
            static=static.cpu(),
            scalars=scalars.cpu(),
            history=history.cpu(),
            history_len=history_len.cpu(),
            action=action.cpu(),
            target=torch.tensor(target, dtype=torch.float32),
        ))
    return per_role


def train_step(
    net: QNet,
    optimizer: optim.Optimizer,
    batch: List[Sample],
    device: torch.device,
) -> float:
    """One gradient step on a sampled batch. Returns the MSE loss value."""
    static = torch.stack([s.static for s in batch]).to(device)
    scalars = torch.stack([s.scalars for s in batch]).to(device)
    history = torch.stack([s.history for s in batch]).to(device)
    history_len = torch.stack([s.history_len for s in batch]).to(device)
    action = torch.stack([s.action for s in batch]).to(device)
    target = torch.stack([s.target for s in batch]).to(device)
    pred = net(static, scalars, history, history_len, action)
    loss = nn.functional.mse_loss(pred, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--train-every', type=int, default=5,
                        help='Episodes between gradient steps.')
    parser.add_argument('--steps-per-update', type=int, default=4,
                        help='Gradient steps per role per training round.')
    parser.add_argument('--buffer-size', type=int, default=20000,
                        help='Per-role replay buffer capacity (samples).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--checkpoint-dir', type=str, default='python/dmc/checkpoints')
    parser.add_argument('--host-script', type=str, default='src/rl/dmc/trainHost.ts')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    nets = TripleQNet().to(device)
    optimizers = {r: optim.RMSprop(nets.net_for(r).parameters(), lr=args.lr) for r in ROLES}
    print(f'Initialized 3 Q-nets ({num_parameters(nets)/1e6:.2f}M params total).')

    # Per-role replay buffers (deque for O(1) FIFO).
    buffers: Dict[str, Deque[Sample]] = {
        r: deque(maxlen=args.buffer_size) for r in ROLES
    }

    # Subprocess actor.
    host = TrainHost(
        host_script=args.host_script,
        cwd=str(_SCRIPT_DIR.parent.parent),  # packages/engine/
    )
    picker = ActionPicker(nets=nets, device=device, eps=EPS_GREEDY, rng=rng)

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rolling_attacker_pts: Deque[int] = deque(maxlen=100)
    rolling_loss: Dict[str, Deque[float]] = {r: deque(maxlen=50) for r in ROLES}

    try:
        for ep in range(1, args.episodes + 1):
            seed = rng.randint(0, 2_000_000_000)
            result = host.run_episode(seed=seed, choose=picker.pick)
            rolling_attacker_pts.append(result.attacker_points)
            new_samples = episode_to_samples(result, GAMMA, device)
            for r, samples in new_samples.items():
                buffers[r].extend(samples)

            if ep % args.train_every == 0:
                for r in ROLES:
                    if len(buffers[r]) < args.batch_size:
                        continue
                    net = nets.net_for(r)
                    opt = optimizers[r]
                    for _ in range(args.steps_per_update):
                        batch = rng.sample(buffers[r], args.batch_size)
                        loss = train_step(net, opt, batch, device)
                        rolling_loss[r].append(loss)

            if ep % args.log_every == 0:
                avg_pts = sum(rolling_attacker_pts) / max(1, len(rolling_attacker_pts))
                buf_sizes = ' '.join(f'{r[:3]}={len(buffers[r])}' for r in ROLES)
                losses = ' '.join(
                    f'{r[:3]}={(sum(rolling_loss[r])/max(1,len(rolling_loss[r]))):.3f}'
                    for r in ROLES
                )
                elapsed = time.time() - t0
                ep_per_sec = ep / max(1, elapsed)
                print(
                    f'ep={ep:>5d} '
                    f'rate={ep_per_sec:.2f}/s '
                    f'atkPts(100ep)={avg_pts:.1f} '
                    f'buf[{buf_sizes}] '
                    f'loss[{losses}]'
                )

            if ep % args.checkpoint_every == 0:
                ckpt_path = ckpt_dir / f'dmc_ep{ep}.pt'
                torch.save({
                    'episode': ep,
                    'state_dict': nets.state_dict(),
                    'optimizer_states': {r: optimizers[r].state_dict() for r in ROLES},
                }, ckpt_path)
                print(f'  saved {ckpt_path}')
    finally:
        host.close()


if __name__ == '__main__':
    main()
