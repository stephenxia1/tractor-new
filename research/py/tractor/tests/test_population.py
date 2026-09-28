"""The population: seating, routing, and the loop around them.

The failure worth fearing is silent: a row acted by one agent's network
and trained into another's, or a declaration scored by the wrong seat's
critic. Every such row still loads and trains. So the routing tests
recompute each recorded quantity from the network that should have made
it, and check that a different agent's network would not have.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ..nn.bury_net import BuryNet
from ..nn.declare import DeclareNet
from ..nn.model import MASK_LOGIT, ActorCritic, ModelConfig, load_model
from ..sim.state import NUM_SEATS
from ..train.buffer import RolloutBuffer
from ..train.parallel import collect_parallel
from ..train.population import (
    PopulationConfig,
    hand_agents,
    read_checkpoint,
    seating,
    train_population,
)
from ..train.rollout import collect
from ..train.starts import PrePlayConfig, critic_values, make_starts

SMALL = ModelConfig(
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
    max_candidates=16,
)
AGENTS = 8


def _nets(n: int = AGENTS) -> list[ActorCritic]:
    torch.manual_seed(7)
    return [ActorCritic(SMALL).eval() for _ in range(n)]


def _jitter(net: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Distinct weights, including layers that start at zero."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=gen))
    return net.eval()


def _table(epoch: int = 0) -> np.ndarray:
    return seating(123, epoch, AGENTS)


# --- seating ----------------------------------------------------------------


def test_a_seating_puts_every_agent_in_exactly_one_seat() -> None:
    for epoch in range(20):
        table = _table(epoch)
        assert table.shape == (AGENTS // NUM_SEATS, NUM_SEATS)
        assert sorted(table.reshape(-1).tolist()) == list(range(AGENTS))


def test_a_seating_is_fixed_by_seed_and_epoch_and_moves_between_epochs() -> None:
    assert np.array_equal(seating(5, 3, AGENTS), seating(5, 3, AGENTS))
    assert len({seating(5, e, AGENTS).tobytes() for e in range(10)}) > 1
    # The pre-play stream is a different one.
    assert not all(
        np.array_equal(seating(5, e, AGENTS), seating(5, e, AGENTS, salt=1))
        for e in range(10)
    )


def test_hands_alternate_between_the_tables() -> None:
    table = _table()
    seat_of = hand_agents(table, 5)
    assert np.array_equal(seat_of[0], table[0])
    assert np.array_equal(seat_of[1], table[1])
    assert np.array_equal(seat_of[4], table[0])


# --- the rollout ------------------------------------------------------------


def _recompute(net: ActorCritic, buf: RolloutBuffer, rows: np.ndarray):
    """(log-probability of the stored action, value) under ``net``."""
    obs, privileged, batch = buf.as_torch(rows)
    with torch.no_grad():
        out, value = net(obs, privileged)
        logits = torch.where(
            obs.candidate_mask, out.logits, torch.full_like(out.logits, MASK_LOGIT)
        )
        logp = F.log_softmax(logits, dim=-1).gather(1, batch["action"][:, None])
    return logp.squeeze(1).numpy(), value.numpy()


def _check_rows_belong_to_their_seats(nets, buf, seat_of) -> None:
    n = len(buf)
    owner = seat_of[buf.episode[:n], buf.seat[:n]]
    choice = buf.candidate_mask[:n].sum(1) > 1
    for k in range(len(nets)):
        rows = np.nonzero(owner == k)[0]
        assert len(rows), f"agent {k} never acted"
        logp, value = _recompute(nets[k], buf, rows)
        np.testing.assert_allclose(logp, buf.logp[rows], atol=1e-4)
        np.testing.assert_allclose(value, buf.value[rows], atol=1e-4)
        # The check has power: another agent's network disagrees.
        other = nets[(k + 1) % len(nets)]
        wrong_logp, wrong_value = _recompute(other, buf, rows)
        assert not np.allclose(wrong_value, buf.value[rows], atol=1e-4)
        chosen = choice[rows]
        assert not np.allclose(wrong_logp[chosen], buf.logp[rows][chosen], atol=1e-4)


def test_each_row_is_acted_and_valued_by_its_seats_network() -> None:
    nets = _nets()
    hands = 6
    seat_of = hand_agents(_table(), hands)
    buf = RolloutBuffer(hands * 100, SMALL)
    stats = collect(nets, np.random.default_rng(3), buf, n_hands=hands,
                    device="cpu", cfg=SMALL, pool=4, agents=seat_of)
    assert stats["hands"] == hands
    _check_rows_belong_to_their_seats(nets, buf, seat_of)


def test_the_parallel_rollout_routes_rows_the_same_way() -> None:
    nets = _nets()
    hands = 6
    seat_of = hand_agents(_table(1), hands)
    buf, stats = collect_parallel(
        nets, np.random.default_rng(3), None, n_hands=hands, device="cpu",
        cfg=SMALL, pool=4, workers=2, agents=seat_of,
    )
    assert stats["hands"] == hands
    _check_rows_belong_to_their_seats(nets, buf, seat_of)


def test_one_agent_in_every_seat_is_the_unrouted_rollout() -> None:
    net = _nets(1)[0]
    runs = []
    for routed in (False, True):
        buf = RolloutBuffer(300, SMALL)
        collect(
            [net] if routed else net, np.random.default_rng(9), buf, n_hands=3,
            device="cpu", cfg=SMALL, pool=3,
            agents=np.zeros((3, NUM_SEATS), dtype=np.int64) if routed else None,
        )
        runs.append(buf)
    n = len(runs[0])
    assert n == len(runs[1])
    for name in ("action", "logp", "value", "reward", "seat", "episode"):
        assert np.array_equal(getattr(runs[0], name)[:n], getattr(runs[1], name)[:n]), name


def test_a_population_without_a_seating_is_refused() -> None:
    with pytest.raises(ValueError, match="no seating"):
        collect(_nets(2), np.random.default_rng(0), RolloutBuffer(200, SMALL),
                n_hands=1, device="cpu", cfg=SMALL)


# --- pre-play ---------------------------------------------------------------


def test_declarations_and_burials_are_made_by_their_seats_agents() -> None:
    torch.manual_seed(0)
    declare = [_jitter(DeclareNet(), k) for k in range(AGENTS)]
    bury = [_jitter(BuryNet(), 100 + k) for k in range(AGENTS)]
    n = 16
    seat_of = hand_agents(_table(), n)
    starts, rows, _stats = make_starts(
        np.random.default_rng(4), n, _nets(), declare, "cpu",
        pre=PrePlayConfig(random_burial_frac=0.0, burial="policy"),
        record=True, bury_net=bury, agents=seat_of,
    )
    decl = rows.declare
    assert len(decl) > 0
    owner = seat_of[decl.hand, decl.seat]
    with torch.no_grad():
        for k in np.unique(owner):
            idx = np.nonzero(owner == k)[0]
            args = [torch.from_numpy(a[idx]) for a in (decl.obs, decl.mask, decl.privileged)]
            _l, lp, v = declare[k](*args)
            got = lp.gather(1, torch.from_numpy(decl.action[idx])[:, None]).squeeze(1)
            np.testing.assert_allclose(got.numpy(), decl.logp[idx], atol=1e-5)
            np.testing.assert_allclose(v.numpy(), decl.value[idx], rtol=1e-4, atol=1e-4)
            _l, _lp, v_other = declare[(k + 1) % AGENTS](*args)
            assert not np.allclose(v_other.numpy(), decl.value[idx], rtol=1e-4, atol=1e-4)

    b = rows.bury
    landlord = np.array([s.landlord for s in starts])
    assert len(b) == 8 * n
    owner = seat_of[b.hand, landlord[b.hand]]
    with torch.no_grad():
        for k in np.unique(owner):
            idx = np.nonzero(owner == k)[0]
            args = [torch.from_numpy(a[idx]) for a in (b.cards, b.ctx, b.mask, b.privileged)]
            _l, lp, v = bury[k](*args)
            got = lp.gather(1, torch.from_numpy(b.action[idx])[:, None]).squeeze(1)
            np.testing.assert_allclose(got.numpy(), b.logp[idx], atol=1e-4)
            _l, lp_other, _v = bury[(k + 1) % AGENTS](*args)
            other = lp_other.gather(1, torch.from_numpy(b.action[idx])[:, None]).squeeze(1)
            assert not np.allclose(other.numpy(), b.logp[idx], atol=1e-4)
    # Every burial is eight cards from the landlord's pool.
    for s in starts:
        assert int(s.kitty.sum()) == 8


def test_each_seat_is_valued_by_its_own_critic() -> None:
    nets = [_jitter(net, k) for k, net in enumerate(_nets())]
    n = 6
    seat_of = hand_agents(_table(), n)
    starts, _rows, _st = make_starts(np.random.default_rng(1), n, None, None)
    routed = critic_values(nets, starts, agents=seat_of)
    for h in range(n):
        for s in range(NUM_SEATS):
            alone = critic_values(nets[seat_of[h, s]], [starts[h]])
            assert routed[h, s] == pytest.approx(float(alone[0, s]), abs=1e-5)


# --- the loop ---------------------------------------------------------------


def _cfg(tmp: str, **kw) -> PopulationConfig:
    base = dict(
        agents=AGENTS,
        total_steps=2,
        hands_per_step=2,
        buffer_capacity=400,
        checkpoint_every=0,
        checkpoint_dir=tmp,
        pool=4,
        reshuffle_every=1,
        model=asdict(SMALL),
        ppo={"epochs": 1, "minibatch": 32, "bf16": False},
    )
    base.update(kw)
    return PopulationConfig(**base)


def test_every_agent_learns_and_a_checkpoint_hands_out_one_agent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = train_population(_cfg(tmp, checkpoint_every=1))
        row = out["final"]
        # Two hands, two tables: every agent had a seat and updated.
        for k in range(AGENTS):
            assert row[f"agent{k}/rows"] > 0
        assert row["seating_epoch"] == 1.0

        blob = torch.load(Path(tmp) / "final.pt", weights_only=False)
        assert len(blob["agents"]) == AGENTS and blob["step"] == 2
        first = torch.load(Path(tmp) / "step_0000001.pt", weights_only=False)
        for k in range(AGENTS):
            moved = any(
                not torch.equal(v, first["agents"][k]["model"][name])
                for name, v in blob["agents"][k]["model"].items()
            )
            assert moved, f"agent {k} did not change in its second step"

        one = read_checkpoint(f"{Path(tmp) / 'final.pt'}#3")
        net = ActorCritic(ModelConfig(**one["model_config"]))
        load_model(net, one["model"])
        for name, v in net.state_dict().items():
            assert torch.equal(v, blob["agents"][3]["model"][name]), name
        with pytest.raises(ValueError, match="population"):
            read_checkpoint(Path(tmp) / "final.pt")
        with pytest.raises(ValueError, match="agent 8"):
            read_checkpoint(f"{Path(tmp) / 'final.pt'}#8")


def test_a_population_trains_through_the_parallel_rollout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = train_population(_cfg(tmp, total_steps=1, rollout_workers=2))
    row = out["final"]
    assert row["rollout/workers"] == 2.0
    assert sum(row[f"agent{k}/rows"] for k in range(AGENTS)) == row["rollout/decisions"]


def test_the_full_hand_schedule_runs_and_resumes() -> None:
    """Warm-up, a pre-play phase for every agent, play from their starts;
    a resume inside a block does not rerun its phase."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(
            tmp, total_steps=3, warmup_steps=1, play_phase_steps=2,
            declare_phase_steps=1, bury_policy=True, checkpoint_every=1,
            keep_last=5,
            declare={"hands_per_step": 16, "epochs": 1, "minibatch": 16},
        )
        out = train_population(cfg)
        assert "pre/phase/declare/return_mean" in out["final"] or \
            "pre/bury/kitty_points" in out["final"]
        final = torch.load(Path(tmp) / "final.pt", weights_only=False)
        assert final["phases_done"] == 1
        assert all(a["bury"] is not None for a in final["agents"])

        mid = Path(tmp) / "step_0000002.pt"
        saved = torch.load(mid, weights_only=False)
        train_population(replace(cfg, checkpoint_dir=tmp + "/b"), resume=mid)
        again = torch.load(Path(tmp) / "b" / "final.pt", weights_only=False)
        assert again["phases_done"] == 1
        for k in range(AGENTS):
            for name, v in saved["agents"][k]["declare"]["model"].items():
                assert torch.equal(v, again["agents"][k]["declare"]["model"][name])

        # The next block's phase runs on resume.
        train_population(replace(cfg, total_steps=4, checkpoint_dir=tmp + "/c"),
                         resume=Path(tmp) / "final.pt")
        later = torch.load(Path(tmp) / "c" / "final.pt", weights_only=False)
        assert later["phases_done"] == 2


def test_a_table_left_empty_is_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="tables"):
            train_population(_cfg(tmp, hands_per_step=1))
