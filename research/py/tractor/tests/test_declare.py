"""The declaration phase: it learns, and it leaves play untouched."""

from __future__ import annotations

import numpy as np
import torch

from ..nn.declare import DeclareNet
from ..nn.model import ActorCritic, ModelConfig
from ..sim.bury import BuryConfig
from ..sim.deal import NUM_ACTIONS
from ..train.declare_loop import DeclareLearner, DeclareTrainConfig
from ..train.starts import PrePlayConfig, critic_values, make_starts


def _tiny_cfg() -> DeclareTrainConfig:
    return DeclareTrainConfig(
        hands_per_step=6, epochs=2, minibatch=16,
        pre=PrePlayConfig(random_burial_frac=0.0, bury=BuryConfig(rounds=1, consider=4)),
    )


def test_a_declaration_step_freezes_play() -> None:
    torch.manual_seed(0)
    play = ActorCritic(ModelConfig()).eval()
    before = {k: v.clone() for k, v in play.state_dict().items()}
    dnet = DeclareNet()
    d_before = [p.clone() for p in dnet.parameters()]
    learner = DeclareLearner(dnet, _tiny_cfg())
    stats = learner.step(play, np.random.default_rng(0))
    for k, v in play.state_dict().items():
        assert torch.equal(v, before[k]), f"declaration step moved play parameter {k}"
    if stats["declare/rows"] > 0:
        assert any(not torch.equal(a, b) for a, b in zip(d_before, dnet.parameters()))
    assert np.isfinite(stats["declare/return_mean"])


def test_starts_are_legal_play_positions() -> None:
    torch.manual_seed(1)
    play = ActorCritic(ModelConfig()).eval()
    starts, rows, _stats = make_starts(
        np.random.default_rng(3), 5, play, DeclareNet().eval(),
        pre=PrePlayConfig(random_burial_frac=0.5, bury=BuryConfig(rounds=1, consider=4)),
        record=True,
    )
    for st in starts:
        state = st.state()
        assert all(int(state.hands[s].sum()) == 25 for s in range(4))
        assert int(state.kitty.sum()) == 8
        assert len(state.legal_moves().moves) > 0
    assert rows is not None and rows.mask.shape[1] == NUM_ACTIONS
    assert np.all(rows.mask[np.arange(len(rows)), rows.action])
    v = critic_values(play, starts)
    assert v.shape == (5, 4) and np.isfinite(v).all()
