"""Card-by-card play (E7, research/E7.md) through the rollout and the learner.

``test_picks.py`` checks the legality oracle on its own. These check what
is built on it: that a row's stored steps and masks describe the play
that was made, that the serial collector and the worker pool agree on
what a row means, and that the learner, scoring the stored picks at the
weights that chose them, recovers the behaviour log-probability exactly
(ratio 1) -- the property every PPO update rests on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tractor.nn.model import (
    MAX_PICKS,
    PICK_SUBMIT,
    ActorCritic,
    ModelConfig,
    pick_actions,
)
from tractor.sim.picks import PickRules
from tractor.train.buffer import RolloutBuffer
from tractor.train.parallel import collect_parallel
from tractor.train.ppo import PPOConfig, PPOLearner
from tractor.train.rollout import collect

CARDS = ModelConfig(
    d_model=32,
    n_layers_history=1,
    n_layers_query=1,
    n_heads=2,
    d_ff=64,
    critic_d=16,
    critic_layers=1,
    critic_heads=2,
    critic_d_ff=32,
    belief_d=16,
    scalar_d=8,
    head_d=16,
    ffn="swiglu",
    critic_kind="cards",
    action_space="cards",
)
DEVICE = torch.device("cpu")


def _net(seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(CARDS)


def _serial(net: ActorCritic, hands: int, seed: int = 1) -> RolloutBuffer:
    buf = RolloutBuffer(hands * 100, CARDS)
    stats = collect(net, np.random.default_rng(seed), buf, n_hands=hands,
                    device=DEVICE, cfg=CARDS, pool=12)
    assert stats["hands"] == hands
    return buf


def _check_rows(buf: RolloutBuffer) -> None:
    n = len(buf)
    buf._validate(n, "test")
    steps = buf.action[:n]
    masks = buf.candidate_mask[:n]
    assert ((steps >= 1) & (steps <= MAX_PICKS)).all()
    assert (steps == masks.any(axis=-1).sum(axis=-1)).all()
    prefix = torch.from_numpy(buf.candidates[:n].astype(np.float32))
    actions = pick_actions(prefix, torch.from_numpy(masks)).numpy()
    for i in range(n):
        k = int(steps[i])
        # The last step submits; every earlier one adds a legal card.
        assert actions[i, k - 1] == PICK_SUBMIT
        assert (actions[i, :k] >= 0).all()
        assert masks[i, np.arange(k), actions[i, :k]].all()
        assert (actions[i, :k - 1] != PICK_SUBMIT).all()
        assert (actions[i, k:] == -1).all()
        # Ascending: a card type never precedes a smaller one.
        cards = actions[i, : k - 1]
        assert (np.diff(cards) >= 0).all()


def _ratio(net: ActorCritic, buf: RolloutBuffer) -> float:
    n = len(buf)
    buf.advantage[:n] = np.random.default_rng(0).normal(size=n).astype(np.float32)
    buf.ret[:n] = 0.0
    learner = PPOLearner(net, PPOConfig(bf16=False), DEVICE)
    obs, privileged, scalars = buf.as_torch(np.arange(n), DEVICE)
    with torch.no_grad():
        _, parts = learner.losses(obs, privileged, scalars)
    return float((parts["ratio"] - 1.0).abs().max())


def test_serial_rows_describe_ascending_legal_plays() -> None:
    buf = _serial(_net(), hands=4)
    _check_rows(buf)


def test_learner_recovers_the_behaviour_log_probability() -> None:
    net = _net(1)
    buf = _serial(net, hands=4, seed=2)
    assert _ratio(net, buf) < 1e-5


@pytest.mark.parametrize("overlap", [False, True])
def test_pool_rows_match_the_serial_contract(overlap: bool) -> None:
    net = _net(2)
    buf, stats = collect_parallel(net, np.random.default_rng(3), None, n_hands=6,
                                  device=DEVICE, cfg=CARDS, pool=12, workers=2,
                                  overlap=overlap)
    assert stats["hands"] == 6
    assert stats["pick_rounds"] >= 1
    _check_rows(buf)
    assert _ratio(net, buf) < 1e-5


def test_an_update_moves_the_policy() -> None:
    net = _net(3)
    buf = _serial(net, hands=4, seed=4)
    n = len(buf)
    buf.advantage[:n] = np.random.default_rng(5).normal(size=n).astype(np.float32)
    buf.ret[:n] = 0.0
    learner = PPOLearner(net, PPOConfig(bf16=False, minibatch=64, epochs=1), DEVICE)
    metrics = learner.update(buf)
    assert np.isfinite(metrics["policy_loss"]) and metrics["approx_kl"] > 0.0


def test_buffer_refuses_a_play_that_does_not_submit() -> None:
    buf = RolloutBuffer(4, CARDS)
    rules_free = np.zeros((MAX_PICKS, PICK_SUBMIT + 1), dtype=bool)
    rules_free[0, 3] = True  # one step offering a card, no SUBMIT
    row = dict(
        history=np.zeros(buf.history.shape[1:], dtype=buf.history.dtype),
        history_len=0,
        context=np.zeros(buf.context.shape[1:], dtype=np.float32),
        unseen=np.zeros(buf.unseen.shape[1:], dtype=np.float32),
        planes=np.zeros(buf.planes.shape[1:], dtype=np.float32),
        cards=np.zeros(buf.cards.shape[1:], dtype=np.float32),
        candidates=np.zeros(buf.candidates.shape[1:], dtype=np.float16),
        candidate_mask=rules_free,
        privileged=np.zeros(buf.privileged.shape[1:], dtype=np.float32),
        logp=0.0, value=0.0, seat=0, episode=0,
    )
    with pytest.raises(ValueError):
        buf.add(action=1, **row)
    # A pool's placeholder row is let through; the parent fills it in.
    buf.add(action=0, pending_play=True, **row)


def test_greedy_evaluation_plays_legal_moves() -> None:
    from tractor.train.evaluate import evaluate

    scores = evaluate(_net(4).eval(), device=DEVICE, cfg=CARDS, num_deals=2, seed=7)
    assert scores["hands"] == 4.0


def test_pick_rules_for_a_rollout_state_offer_a_first_pick() -> None:
    from tractor.gen.episode import generate

    state, _info = generate(np.random.default_rng(8), 2)
    rules = PickRules.for_state(state)
    mask = rules.mask(np.zeros(54, dtype=np.int8), -1)
    assert mask[:PICK_SUBMIT].any() and not mask[PICK_SUBMIT]
