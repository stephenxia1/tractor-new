"""A resume takes the run's learning rate, not the checkpoint's."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from ..nn.model import ActorCritic, ModelConfig
from ..train.loop import load_checkpoint
from ..train.ppo import PPOConfig, PPOLearner

SMALL = ModelConfig(d_model=32, n_layers_history=1, n_layers_query=1, n_heads=2, d_ff=64,
                    critic_d=16, critic_layers=1, critic_heads=2, critic_d_ff=32,
                    max_candidates=16)


def test_load_checkpoint_keeps_the_configured_rate() -> None:
    torch.manual_seed(0)
    net = ActorCritic(SMALL)
    old = PPOLearner(net, PPOConfig(lr=3e-4), "cpu")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ck.pt"
        torch.save({"model": net.state_dict(), "optimizer": old.opt.state_dict(), "step": 7},
                   path)
        net2 = ActorCritic(SMALL)
        new = PPOLearner(net2, PPOConfig(lr=1e-4), "cpu")
        assert load_checkpoint(path, net2, new) == 7
        assert all(g["lr"] == 1e-4 for g in new.opt.param_groups)
