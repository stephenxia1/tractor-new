"""The v6 architecture, and the v5 one it must not disturb.

Two promises. Every checkpoint through v5 must keep loading and keep
computing exactly what it computed: the fixtures were written from the
code before SwiGLU and the card critic existed, and the legacy configs
must reproduce their parameter layouts and their outputs to the bit.
And the new pieces must be wired where they claim: the card critic reads
the cards the rollout and the play-start scorer hand it, ignores padded
history, and every new parameter is trained by the learner's loss.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from ..nn.arch import ARCHES, bury_net_from, check_saved, declare_net_from
from ..nn.bury_net import BURY_CARD_DIM, BURY_CTX_DIM, BURY_PRIV_DIM, BuryNet, BuryNetConfig
from ..nn.declare import DeclareConfig, DeclareNet
from ..nn.features import Cache, encode
from ..nn.model import (
    ActorCritic,
    CardCritic,
    ModelConfig,
    ObsBatch,
    analytic_flops,
    batch_observations,
    batch_privileged,
    dummy_batch,
    privileged_view,
)
from ..sim.deal import DECLARE_OBS_DIM, DECLARE_PRIV_DIM, NUM_ACTIONS
from ..sim.state import NUM_SEATS
from ..train.buffer import RolloutBuffer
from ..train.gae import compute_gae
from ..train.loop import TrainConfig, train
from ..train.ppo import PPOConfig, PPOLearner
from ..train.rollout import collect
from ..train.starts import critic_values, make_starts

FIXTURES = Path(__file__).parent / "fixtures"

# v6's shape at test size.
SMALL_V6 = ModelConfig(
    d_model=32, n_layers_history=1, n_layers_query=1, n_heads=2, d_ff=64,
    critic_d=32, critic_layers=2, critic_heads=2, critic_d_ff=64,
    belief_d=16, scalar_d=8, head_d=16, max_candidates=16,
    ffn="swiglu", critic_kind="cards",
)


# --- v5 is untouched ----------------------------------------------------------


def test_the_v5_networks_keep_their_parameter_layouts() -> None:
    layouts = json.loads((FIXTURES / "legacy_state_dicts.json").read_text())
    for name, net in (("ActorCritic", ActorCritic()), ("DeclareNet", DeclareNet()),
                      ("BuryNet", BuryNet())):
        got = {k: list(v.shape) for k, v in net.state_dict().items()}
        assert got == layouts[name], name


def test_the_v5_networks_compute_exactly_what_they_did() -> None:
    fx = torch.load(FIXTURES / "legacy_outputs.torch", weights_only=False)

    ac = fx["actor_critic"]
    net = ActorCritic(ModelConfig(**ac["config"])).eval()
    net.load_state_dict(ac["state"])
    obs = ObsBatch(*ac["obs"])
    with torch.no_grad():
        out, value = net(obs, ac["priv"])
        _o, value0 = net(obs._replace(history_len=torch.zeros_like(obs.history_len)), ac["priv"])
    assert torch.equal(out.logits, ac["logits"])
    assert torch.equal(value, ac["value"])
    assert torch.equal(out.value, ac["own_value"])
    assert torch.equal(out.outcome_logits, ac["outcome"])
    assert torch.equal(out.belief_logits, ac["belief"])
    assert torch.equal(value0, ac["value_empty"])

    for key, cls, cfg_cls in (("declare", DeclareNet, DeclareConfig),
                              ("bury", BuryNet, BuryNetConfig)):
        part = fx[key]
        small = cls(cfg_cls(**part["config"])).eval()
        small.load_state_dict(part["state"])
        with torch.no_grad():
            got = small(*part["inputs"])
        for a, b in zip(got, part["outputs"]):
            assert torch.equal(a, b), key


def test_a_checkpoint_without_saved_configs_rebuilds_as_v5() -> None:
    dn, bn = DeclareNet(), BuryNet()
    assert declare_net_from({"model": dn.state_dict()}).cfg == DeclareConfig()
    assert bury_net_from({"model": bn.state_dict()}).cfg == BuryNetConfig()


# --- v6 -----------------------------------------------------------------------


def test_v6_sizes() -> None:
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    net = ActorCritic(ModelConfig(**ARCHES["v6"]["model"]))
    assert isinstance(net.critic, CardCritic)
    assert n(net.actor) == 13_269_454
    assert n(net.critic) == 4_508_929
    assert n(DeclareNet(DeclareConfig(**ARCHES["v6"]["declare_model"]))) == 5_005_324
    assert n(BuryNet(BuryNetConfig(**ARCHES["v6"]["bury_model"]))) == 3_621_634


def test_swiglu_needs_the_card_critic_and_a_residual_declaration_net() -> None:
    with pytest.raises(ValueError, match="flat critic"):
        ModelConfig(ffn="swiglu")
    with pytest.raises(ValueError, match="residual"):
        DeclareConfig(ffn="swiglu")


def test_the_card_critic_ignores_padded_history() -> None:
    torch.manual_seed(0)
    net = ActorCritic(SMALL_V6).eval()
    obs, priv = dummy_batch(SMALL_V6, batch=5, history_len=40, seed=2)
    obs = obs._replace(history_len=torch.tensor([0, 3, 17, 40, 9]))
    with torch.no_grad():
        out, value = net(obs, priv)
        enc = out.history_enc.clone()
        # Garbage past each row's length, where the actor writes zeros.
        for i, n in enumerate(obs.history_len.tolist()):
            enc[i, n:] = torch.randn_like(enc[i, n:]) * 10
        trick = obs.planes[:, 5:9]
        again = net.critic(enc, obs.history_len, priv, context=obs.context,
                           cards=obs.cards, trick=trick)
        # One row at a time: no row's value depends on its batch.
        alone = torch.stack([
            net.critic(out.history_enc[i : i + 1], obs.history_len[i : i + 1],
                       priv[i : i + 1], context=obs.context[i : i + 1],
                       cards=obs.cards[i : i + 1], trick=trick[i : i + 1])[0]
            for i in range(5)
        ])
    torch.testing.assert_close(again, value, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(alone, value, rtol=1e-5, atol=1e-5)


def test_the_card_critic_reads_the_cards_and_the_trick() -> None:
    torch.manual_seed(1)
    net = ActorCritic(SMALL_V6).eval()
    obs, priv = dummy_batch(SMALL_V6, batch=3, history_len=10, seed=5)
    with torch.no_grad():
        _out, value = net(obs, priv)
        moved_trick = obs._replace(planes=obs.planes.clone())
        moved_trick.planes[:, 5:9] = torch.roll(obs.planes[:, 5:9], 1, dims=1)
        _o, v_trick = net(moved_trick, priv)
        other_trump = obs._replace(cards=torch.roll(obs.cards, 1, dims=1))
        _o, v_cards = net(other_trump, priv)
    assert not torch.allclose(value, v_trick)
    assert not torch.allclose(value, v_cards)


@pytest.mark.parametrize("cfg", [ModelConfig(), SMALL_V6], ids=["v5", "v6"])
def test_play_start_values_match_the_rollouts_critic(cfg: ModelConfig) -> None:
    """``critic_values`` builds its inputs itself; they must be the ones
    the full forward sees at the same position."""
    torch.manual_seed(3)
    net = ActorCritic(cfg).eval()
    starts, _rows, _st = make_starts(np.random.default_rng(8), 3, None, None)
    got = critic_values(net, starts)
    cache = Cache()
    for h, st in enumerate(starts):
        state = st.state()
        obs_rows, cand, priv = [], [], []
        for seat in range(NUM_SEATS):
            obs_rows.append(encode(state, seat, cache))
            cand.append([np.zeros(54, dtype=np.int8)])
            priv.append(privileged_view(state.hands, state.kitty, seat))
        obs = batch_observations(obs_rows, cand, cfg)
        with torch.no_grad():
            _out, value = net(obs, batch_privileged(priv))
        np.testing.assert_allclose(got[h], value.numpy(), rtol=1e-4, atol=1e-4)


def test_every_v6_parameter_is_trained_by_the_learners_loss() -> None:
    torch.manual_seed(4)
    net = ActorCritic(SMALL_V6)
    buf = RolloutBuffer(400, SMALL_V6)
    collect(net, np.random.default_rng(2), buf, n_hands=3, device="cpu", cfg=SMALL_V6, pool=3)
    compute_gae(buf, gamma=1.0, lam=0.95)
    learner = PPOLearner(net, PPOConfig(bf16=False), "cpu")
    obs, privileged, scalars = buf.as_torch(np.arange(len(buf)), "cpu")
    total, _parts = learner.losses(obs, privileged, scalars)
    learner.opt.zero_grad(set_to_none=True)
    total.backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing, missing[:8]
    dead = [n for n, p in net.named_parameters()
            if n.startswith("critic.") and float(p.grad.abs().sum()) == 0.0]
    assert not dead, dead[:8]


def test_analytic_flops_counts_v6_as_torch_does() -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode

    cfg = replace(SMALL_V6, max_candidates=16)
    torch.manual_seed(6)
    net = ActorCritic(cfg).eval()
    batch = 2
    obs, priv = dummy_batch(cfg, batch=batch, seed=7, history_len=cfg.max_history,
                            n_candidates=cfg.max_candidates)
    obs = obs._replace(
        history_len=torch.full((batch,), cfg.max_history, dtype=torch.int64),
        candidate_mask=torch.ones(batch, cfg.max_candidates, dtype=torch.bool),
    )
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        counter = FlopCounterMode(display=False)
        with counter:
            net(obs, priv)
    predicted = analytic_flops(cfg, batch, cfg.max_candidates, cfg.max_history,
                               matmul_only=True, causal_discount=False)["total"]
    assert counter.get_total_flops() == predicted


def test_the_v6_pre_play_networks_mask_and_start_uniform() -> None:
    torch.manual_seed(7)
    dn = DeclareNet(DeclareConfig(**ARCHES["v6"]["declare_model"])).eval()
    obs = torch.randn(4, DECLARE_OBS_DIM)
    mask = torch.rand(4, NUM_ACTIONS) > 0.5
    mask[:, 0] = True
    with torch.no_grad():
        logits, logp, value = dn(obs, mask, torch.randn(4, DECLARE_PRIV_DIM))
    # Zero-initialised output layer: uniform over the legal actions.
    probs = logp.exp()
    torch.testing.assert_close(probs.sum(1), torch.ones(4))
    assert torch.all(probs[~mask] == 0)
    legal = mask.sum(1, keepdim=True).float()
    torch.testing.assert_close(probs[mask], (1 / legal).expand_as(probs)[mask])

    bn = BuryNet(BuryNetConfig(**ARCHES["v6"]["bury_model"])).eval()
    held = torch.rand(3, 54) > 0.4
    with torch.no_grad():
        _l, lp, v = bn(torch.randn(3, 54, BURY_CARD_DIM), torch.randn(3, BURY_CTX_DIM),
                       held, torch.randn(3, BURY_PRIV_DIM))
    assert torch.all(lp.exp()[~held] == 0)
    assert torch.isfinite(v).all()


def test_a_run_resumes_only_into_its_own_architecture() -> None:
    small = {k: v for k, v in asdict(SMALL_V6).items()}
    v6 = dict(model=small, declare_model={"width": 32, "depth": 2, "residual": True,
                                          "ffn": "swiglu"},
              bury_model={"d": 32, "layers": 1, "heads": 2, "value_width": 32,
                          "ffn": "swiglu"})
    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainConfig(
            total_steps=2, hands_per_step=2, buffer_capacity=400, checkpoint_dir=tmp,
            checkpoint_every=0, warmup_steps=1, play_phase_steps=2, declare_phase_steps=1,
            bury_policy=True, ppo={"epochs": 1, "minibatch": 32, "bf16": False},
            declare={"hands_per_step": 4, "epochs": 1, "minibatch": 16}, **v6,
        )
        train(cfg)
        blob = torch.load(Path(tmp) / "final.pt", weights_only=False)
        assert DeclareConfig(**blob["declare"]["config"]) == DeclareConfig(**v6["declare_model"])
        assert BuryNetConfig(**blob["bury"]["config"]) == BuryNetConfig(**v6["bury_model"])
        # The saved networks rebuild from their own configs.
        declare_net_from(blob["declare"])
        bury_net_from(blob["bury"])
        # Resuming it under the v5 declaration network is refused.
        with pytest.raises(ValueError, match="declaration network"):
            train(replace(cfg, total_steps=3, declare_model={}, checkpoint_dir=tmp + "/b"),
                  resume=Path(tmp) / "final.pt")
    with pytest.raises(ValueError, match="burial network"):
        check_saved("burial", {"model": {}, "config": {"d": 64}}, BuryNetConfig())


def test_the_critic_bakeoff_runs_end_to_end() -> None:
    from ..train.critic_bakeoff import run

    small = ModelConfig(d_model=32, n_layers_history=1, n_layers_query=1, n_heads=2,
                        d_ff=64, critic_d=16, critic_layers=1, critic_heads=2,
                        critic_d_ff=32, belief_d=16, scalar_d=8, head_d=16,
                        max_candidates=16)
    torch.manual_seed(5)
    net = ActorCritic(small)
    blob = {"model": net.state_dict(), "model_config": asdict(small),
            "declare": {"model": DeclareNet().state_dict()},
            "bury": {"model": BuryNet().state_dict()}}
    out = run(blob, torch.device("cpu"), rounds=2, hands=4, epochs=1, workers=2, pool=4)
    assert out["hands_held"] == 4 and out["rows_held"] > 0
    assert set(out["arms"]) == {"v5", "flat", "flat-wide", "cards"}
    for arm, got in out["arms"].items():
        last = got["passes"][-1]
        assert np.isfinite(last["ev_held"]) and np.isfinite(last["ev_start"]), arm
    # Paired against the flat critic, pass for pass.
    assert set(out["paired"]) == {"v5", "flat-wide", "cards"}
    assert len(out["paired"]["cards"]) == 1
