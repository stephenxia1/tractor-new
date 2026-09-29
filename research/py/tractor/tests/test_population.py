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


def test_a_split_forward_still_routes_every_row_to_its_seat() -> None:
    """``chunks`` splits each agent's rows by history length; every row
    must still be acted and valued by its own seat's network."""
    nets = _nets()
    hands = 6
    seat_of = hand_agents(_table(2), hands)
    buf = RolloutBuffer(hands * 100, SMALL)
    collect(nets, np.random.default_rng(4), buf, n_hands=hands, device="cpu",
            cfg=SMALL, pool=6, agents=seat_of, chunks=3)
    _check_rows_belong_to_their_seats(nets, buf, seat_of)


def test_copy_ranges_is_compaction_then_a_copy() -> None:
    from ..train.shared import compact_ranges, copy_ranges

    rng = np.random.default_rng(0)
    src = RolloutBuffer(90, SMALL)
    for name in src.fields:
        arr = getattr(src, name)
        arr[...] = rng.integers(0, 5, size=arr.shape).astype(arr.dtype)
    filled = [(0, 17), (30, 0), (60, 29)]  # (start, count), one run empty
    dst = RolloutBuffer(90, SMALL)
    assert copy_ranges(src, dst, filled, threads=4) == 46
    compact_ranges(src, [(s, s + c) for s, c in filled])
    for name in src.fields:
        assert np.array_equal(getattr(dst, name)[:46], getattr(src, name)[:46]), name


# --- the shared critic --------------------------------------------------------

from ..nn.model import privileged_view, relative_seat_agents, shared_critic  # noqa: E402

ACTOR_ONLY = replace(SMALL, critic_kind="none")


def _actors(n: int = AGENTS) -> list[ActorCritic]:
    torch.manual_seed(11)
    return [ActorCritic(ACTOR_ONLY).eval() for _ in range(n)]


def _shared() -> torch.nn.Module:
    torch.manual_seed(12)
    return _jitter(shared_critic(SMALL, AGENTS), 3)


def _shared_values(critic, buf: RolloutBuffer, rows: np.ndarray, seat_agents: np.ndarray):
    obs, privileged, _ = buf.as_torch(rows)
    with torch.no_grad():
        return critic(obs.history, obs.history_len, privileged, context=obs.context,
                      cards=obs.cards, trick=obs.planes[:, 5:9],
                      seat_agents=torch.from_numpy(seat_agents)).numpy()


def _check_shared_values(critic, buf: RolloutBuffer, seat_of: np.ndarray) -> None:
    n = len(buf)
    rows = np.arange(n)
    seats = relative_seat_agents(seat_of[buf.episode[:n]], buf.seat[:n])
    np.testing.assert_allclose(_shared_values(critic, buf, rows, seats), buf.value[:n],
                               rtol=1e-4, atol=1e-4)
    # The check has power: the same rows with the seats' agents rotated.
    wrong = _shared_values(critic, buf, rows, np.roll(seats, 1, axis=1))
    assert not np.allclose(wrong, buf.value[:n], rtol=1e-4, atol=1e-4)


def test_relative_seat_agents_reads_the_table_from_each_seat() -> None:
    table = np.array([[10, 11, 12, 13]] * 4)
    got = relative_seat_agents(table, np.array([0, 1, 2, 3]))
    assert got.tolist() == [[10, 11, 12, 13], [11, 12, 13, 10],
                            [12, 13, 10, 11], [13, 10, 11, 12]]


def test_the_shared_critic_values_every_row_knowing_who_sits_where() -> None:
    nets, critic = _actors(), _shared()
    hands = 6
    seat_of = hand_agents(_table(3), hands)
    buf = RolloutBuffer(hands * 100, SMALL)
    collect(nets, np.random.default_rng(5), buf, n_hands=hands, device="cpu",
            cfg=ACTOR_ONLY, pool=4, agents=seat_of, critic=critic)
    _check_shared_values(critic, buf, seat_of)


def test_the_parallel_rollout_uses_the_shared_critic_the_same_way() -> None:
    nets, critic = _actors(), _shared()
    hands = 6
    seat_of = hand_agents(_table(4), hands)
    buf, _stats = collect_parallel(nets, np.random.default_rng(5), None, n_hands=hands,
                                   device="cpu", cfg=ACTOR_ONLY, pool=4, workers=2,
                                   agents=seat_of, critic=critic)
    _check_shared_values(critic, buf, seat_of)


def test_an_actor_without_a_critic_is_refused_without_the_shared_one() -> None:
    seat_of = hand_agents(_table(), 2)
    with pytest.raises(ValueError, match="shared one"):
        collect(_actors(), np.random.default_rng(0), RolloutBuffer(200, SMALL),
                n_hands=2, device="cpu", cfg=ACTOR_ONLY, agents=seat_of)


def test_play_start_values_come_from_the_shared_critic_per_seat() -> None:
    critic = _shared()
    n = 4
    seat_of = hand_agents(_table(5), n)
    starts, _rows, _st = make_starts(np.random.default_rng(2), n, None, None)
    got = critic_values(_actors(), starts, agents=seat_of, shared=critic)
    cache_obs = []
    from ..nn.features import Cache, encode
    from ..nn.model import batch_observations, batch_privileged
    cache = Cache()
    for h, st in enumerate(starts):
        state = st.state()
        obs = batch_observations(
            [encode(state, s, cache) for s in range(NUM_SEATS)],
            [[np.zeros(54, dtype=np.int8)]] * NUM_SEATS, SMALL,
        )
        priv = batch_privileged([privileged_view(state.hands, state.kitty, s)
                                 for s in range(NUM_SEATS)])
        seats = relative_seat_agents(np.repeat(seat_of[h : h + 1], NUM_SEATS, 0),
                                     np.arange(NUM_SEATS))
        with torch.no_grad():
            want = critic(obs.history, obs.history_len, priv, context=obs.context,
                          cards=obs.cards, trick=obs.planes[:, 5:9],
                          seat_agents=torch.from_numpy(seats))
        np.testing.assert_allclose(got[h], want.numpy(), rtol=1e-4, atol=1e-4)
        cache_obs.append(h)
    assert len(cache_obs) == n


def test_the_critic_learner_fits_what_it_is_shown() -> None:
    from ..train.critic_learner import CriticLearner, CriticTrainConfig
    from ..train.gae import compute_gae

    nets, critic = _actors(), _shared()
    seat_of = hand_agents(_table(6), 4)
    buf = RolloutBuffer(400, SMALL)
    collect(nets, np.random.default_rng(7), buf, n_hands=4, device="cpu", cfg=ACTOR_ONLY,
            pool=4, agents=seat_of, critic=critic)
    compute_gae(buf, gamma=1.0, lam=0.95)
    n = len(buf)
    seats = relative_seat_agents(seat_of[buf.episode[:n]], buf.seat[:n])
    learner = CriticLearner(critic, CriticTrainConfig(minibatch=64, bf16=False, lr=1e-3))
    before = float(np.mean((_shared_values(critic, buf, np.arange(n), seats) - buf.ret[:n]) ** 2))
    for _ in range(6):
        stats = learner.update(buf, seats)
    after = float(np.mean((_shared_values(critic, buf, np.arange(n), seats) - buf.ret[:n]) ** 2))
    assert after < before
    assert stats["critic/rows"] == n


def test_a_shared_critic_population_trains_saves_and_resumes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp, checkpoint_every=1)
        assert cfg.shared_critic
        out = train_population(cfg)
        row = out["final"]
        assert row["critic/rows"] == row["rollout/decisions"]
        blob = torch.load(Path(tmp) / "final.pt", weights_only=False)
        assert blob["critic"]["agents"] == AGENTS
        # The agents are actors alone.
        assert not any(k.startswith("critic.") for k in blob["agents"][0]["model"])
        first = torch.load(Path(tmp) / "step_0000001.pt", weights_only=False)
        assert any(not torch.equal(v, first["critic"]["model"][k])
                   for k, v in blob["critic"]["model"].items())
        one = read_checkpoint(f"{Path(tmp) / 'final.pt'}#2")
        net = ActorCritic(ModelConfig(**one["model_config"]))
        assert net.critic is None
        load_model(net, one["model"])
        # Resuming restores the shared critic too.
        train_population(replace(cfg, total_steps=3, checkpoint_dir=tmp + "/b"),
                         resume=Path(tmp) / "final.pt")
        # A run with its own critics refuses a shared-critic checkpoint.
        with pytest.raises(ValueError, match="shared critic"):
            train_population(replace(cfg, total_steps=3, shared_critic=False,
                                     checkpoint_dir=tmp + "/c"),
                             resume=Path(tmp) / "final.pt")


def test_agents_can_still_keep_their_own_critics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = train_population(_cfg(tmp, shared_critic=False))
        assert "critic/rows" not in out["final"]
        blob = torch.load(Path(tmp) / "final.pt", weights_only=False)
        assert "critic" not in blob
        assert any(k.startswith("critic.") for k in blob["agents"][0]["model"])


# --- the overlapped rollout ---------------------------------------------------


def _check_round(buf: RolloutBuffer, stats: dict, hands: int) -> None:
    n = len(buf)
    assert stats["hands"] == hands and n == stats["decisions"] > 0
    assert buf.candidate_mask[np.arange(n), buf.action[:n]].all()
    assert set(np.unique(buf.episode[:n]).tolist()) == set(range(hands))
    for e in range(hands):
        rows = np.nonzero(buf.episode[:n] == e)[0]
        assert int(buf.done[rows].sum()) == NUM_SEATS, e
        assert (np.diff(buf.history_len[rows]) >= 0).all(), e


def test_the_overlapped_rollout_keeps_every_invariant_round_after_round() -> None:
    from ..train.parallel import RolloutPool

    nets = _nets()
    hands = 10
    seat_of = hand_agents(_table(7), hands)
    with RolloutPool(cfg=SMALL, workers=2, hands_per_round=hands) as pool:
        # A second round on the same workers: a go-ahead left in a
        # worker's queue by the first would be read as its next command.
        for r in range(2):
            buf, stats = pool.collect(nets, np.random.default_rng(r), None, n_hands=hands,
                                      device="cpu", pool=6, agents=seat_of, overlap=True)
            _check_round(buf, stats, hands)
            _check_rows_belong_to_their_seats(nets, buf, seat_of)


@pytest.mark.parametrize("forwards", ["eager", "static"])
def test_the_overlapped_rollout_with_the_shared_critic(forwards: str) -> None:
    nets, critic = _actors(), _shared()
    hands = 8
    seat_of = hand_agents(_table(8), hands)
    buf, stats = collect_parallel(nets, np.random.default_rng(9), None, n_hands=hands,
                                  device="cpu", cfg=ACTOR_ONLY, pool=6, workers=2,
                                  agents=seat_of, critic=critic, overlap=True,
                                  forwards=forwards)
    _check_round(buf, stats, hands)
    # Every value recomputed by the eager critic: the static path agrees.
    _check_shared_values(critic, buf, seat_of)


# --- fixed shapes: what torch.compile needs, exercised eagerly -----------------


def test_fixed_shapes_skip_only_ragged_minibatches_and_learn() -> None:
    from ..train.critic_learner import CriticLearner, CriticTrainConfig
    from ..train.gae import compute_gae
    from ..train.ppo import PPOConfig, PPOLearner

    nets, critic = _actors(), _shared()
    seat_of = hand_agents(_table(9), 4)
    buf = RolloutBuffer(400, SMALL)
    collect(nets, np.random.default_rng(3), buf, n_hands=4, device="cpu", cfg=ACTOR_ONLY,
            pool=4, agents=seat_of, critic=critic)
    compute_gae(buf, gamma=1.0, lam=0.95)
    n = len(buf)
    learner = PPOLearner(nets[0], PPOConfig(minibatch=32, length_buckets=4, epochs=1,
                                            fixed_shapes=True, bf16=False), "cpu")
    m = learner.update(buf)
    assert np.isfinite(m["policy_loss"])
    # Only minibatches short of 32 rows were skipped, at most one per bucket.
    assert 0 <= m["skipped_ragged_rows"] < 4 * 32
    assert m["skipped_ragged_rows"] + m["n_minibatches"] * 32 == n

    seats = relative_seat_agents(seat_of[buf.episode[:n]], buf.seat[:n])
    fitter = CriticLearner(critic, CriticTrainConfig(minibatch=32, fixed_shapes=True,
                                                     bf16=False))
    st = fitter.update(buf, seats)
    assert np.isfinite(st["critic/value_loss"])
    assert st["critic/skipped_ragged_rows"] < 4 * 32


def test_a_population_trains_on_fixed_shapes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp, hands_per_step=4,
                   ppo={"epochs": 1, "minibatch": 32, "bf16": False, "fixed_shapes": True})
        out = train_population(cfg)
        row = out["final"]
        assert np.isfinite(row["ppo/policy_loss"]) and np.isfinite(row["critic/value_loss"])


# --- the round on the device -----------------------------------------------------


def _round(hands: int = 4):
    from ..train.gae import compute_gae

    nets, critic = _actors(), _shared()
    seat_of = hand_agents(_table(9), hands)
    buf = RolloutBuffer(400, SMALL)
    collect(nets, np.random.default_rng(3), buf, n_hands=hands, device="cpu", cfg=ACTOR_ONLY,
            pool=4, agents=seat_of, critic=critic)
    compute_gae(buf, gamma=1.0, lam=0.95)
    return nets, critic, seat_of, buf


def test_device_rows_are_the_buffers_rows() -> None:
    from ..train.buffer import DeviceRows

    _n, _c, _s, buf = _round()
    rows = DeviceRows(buf, "cpu")
    idx = np.array([5, 0, 17, 3])
    obs, priv, sc = rows.rows(torch.from_numpy(idx), width=16)
    want, want_priv, want_sc = buf.as_torch(idx, "cpu")
    for name in obs._fields:
        got, exp = getattr(obs, name), getattr(want, name)
        if name == "history":
            exp = exp[:, :16]
        assert got.dtype == exp.dtype and torch.equal(got, exp), name
    assert torch.equal(priv, want_priv)
    assert sc.keys() == want_sc.keys()
    assert all(torch.equal(sc[k], want_sc[k]) for k in sc)


def test_the_update_on_the_device_is_the_update_on_the_host() -> None:
    import copy

    from ..train.buffer import DeviceRows
    from ..train.ppo import PPOConfig, PPOLearner

    nets, _c, _s, buf = _round()
    cfg = PPOConfig(minibatch=32, length_buckets=4, epochs=2, fixed_shapes=True, bf16=False)
    a, b = copy.deepcopy(nets[0]), copy.deepcopy(nets[0])
    host, dev = PPOLearner(a, cfg, "cpu"), PPOLearner(b, cfg, "cpu")
    assert dev.on_device_ok
    torch.manual_seed(11)
    m_host = host.update(buf, rows=np.arange(0, len(buf), 2))
    torch.manual_seed(11)
    m_dev = dev.update(buf, rows=np.arange(0, len(buf), 2), device_rows=DeviceRows(buf, "cpu"))
    # The same minibatches in the same order, so the same weights.
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-5, atol=1e-6)
    assert m_host.keys() == m_dev.keys()
    for key, v in m_host.items():
        np.testing.assert_allclose(m_dev[key], v, rtol=1e-4, atol=1e-6, err_msg=key)
    assert host.steps == dev.steps > 0
    assert not hasattr(dev.opt, "found_inf")


def test_a_nonfinite_step_is_refused_on_the_device_as_on_the_host() -> None:
    from ..train.buffer import DeviceRows
    from ..train.ppo import PPOConfig, PPOLearner

    nets, _c, _s, buf = _round()
    buf.ret[: len(buf)] = np.nan  # the actor's own value head then has a NaN loss
    net = nets[0]
    learner = PPOLearner(net, PPOConfig(minibatch=32, length_buckets=4, epochs=1,
                                        fixed_shapes=True, bf16=False), "cpu")
    before = [p.detach().clone() for p in net.parameters()]
    m = learner.update(buf, device_rows=DeviceRows(buf, "cpu"))
    assert m["n_minibatches"] == 0 and m["nonfinite_minibatches"] > 0
    assert np.isnan(m["policy_loss"])
    assert all(torch.equal(p, q) for p, q in zip(net.parameters(), before))
    # Not a step taken and undone: the optimiser never counted one.
    assert learner.steps == 0
    assert all(float(st["step"]) == 0.0 for st in learner.opt.state.values())


def test_the_critic_fits_the_same_on_the_device() -> None:
    import copy

    from ..train.buffer import DeviceRows
    from ..train.critic_learner import CriticLearner, CriticTrainConfig

    _n, critic, seat_of, buf = _round()
    n = len(buf)
    seats = relative_seat_agents(seat_of[buf.episode[:n]], buf.seat[:n])
    a, b = copy.deepcopy(critic), copy.deepcopy(critic)
    cfg = CriticTrainConfig(minibatch=32, fixed_shapes=True, bf16=False)
    host, dev = CriticLearner(a, cfg), CriticLearner(b, cfg)
    assert dev.on_device_ok
    torch.manual_seed(5)
    got_host = host.update(buf, seats)
    torch.manual_seed(5)
    got_dev = dev.update(buf, seats, device_rows=DeviceRows(buf, "cpu"))
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-5, atol=1e-6)
    for key, v in got_host.items():
        np.testing.assert_allclose(got_dev[key], v, rtol=1e-4, atol=1e-6, err_msg=key)


@pytest.mark.parametrize("population", [True, False])
def test_static_forwards_collect_the_same_round(population: bool) -> None:
    """Histories cut on the host to their width bucket, the model trusting
    the width (what the compiled rollout runs): the same actions, and the
    same log-probabilities and values to float rounding."""
    got = []
    for forwards in ("eager", "static"):
        if population:
            nets, critic = _actors(), _shared()
            kw = dict(cfg=ACTOR_ONLY, agents=hand_agents(_table(4), 6), critic=critic)
        else:
            torch.manual_seed(7)
            nets, kw = ActorCritic(SMALL).eval(), dict(cfg=SMALL)
        buf = RolloutBuffer(600, SMALL)
        collect(nets, np.random.default_rng(21), buf, n_hands=6, device="cpu", pool=6,
                forwards=forwards, **kw)
        got.append(buf)
    eager, static = got
    n = len(eager)
    assert n == len(static) and n > 0
    assert np.array_equal(eager.action[:n], static.action[:n])
    np.testing.assert_allclose(static.logp[:n], eager.logp[:n], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(static.value[:n], eager.value[:n], rtol=1e-4, atol=1e-5)


def test_an_unknown_forwards_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="forwards"):
        collect(_nets(1)[0], np.random.default_rng(0), RolloutBuffer(100, SMALL), n_hands=1,
                device="cpu", cfg=SMALL, forwards="fast")


def test_device_rows_refill_in_place() -> None:
    """Kept across steps, the rows keep their addresses -- what a captured
    update reads them through -- until a larger buffer forces new ones."""
    from ..train.buffer import DeviceRows

    _n, _c, _s, buf = _round()
    rows = DeviceRows(buf, "cpu")
    ptr, generation = rows.t["history"].data_ptr(), rows.generation
    nets, critic = _actors(), _shared()
    other = RolloutBuffer(400, SMALL)
    collect(nets, np.random.default_rng(99), other, n_hands=3, device="cpu", cfg=ACTOR_ONLY,
            pool=3, agents=hand_agents(_table(2), 3), critic=critic)
    rows.load(other)
    assert rows.t["history"].data_ptr() == ptr and rows.generation == generation
    assert rows.n == len(other)
    idx = np.arange(len(other))
    obs, _p, sc = rows.rows(torch.from_numpy(idx))
    want, _wp, want_sc = other.as_torch(idx, "cpu")
    assert torch.equal(obs.history, want.history) and torch.equal(sc["ret"], want_sc["ret"])
    rows.load(RolloutBuffer(800, SMALL))
    assert rows.generation == generation + 1 and rows.capacity == 800


def test_a_shared_critic_with_its_own_encoder_collects_and_trains() -> None:
    """The rollout values rows the same through the static path, and a
    fixed-shape fit on the device rows moves the encoder too."""
    from ..train.buffer import DeviceRows
    from ..train.critic_learner import CriticLearner, CriticTrainConfig
    from ..train.gae import compute_gae

    cfg = replace(SMALL, critic_kind="cards", ffn="swiglu", critic_history_d=16,
                  critic_history_layers=1)
    got = []
    for forwards in ("eager", "static"):
        nets = _actors()
        torch.manual_seed(12)
        critic = _jitter(shared_critic(cfg, AGENTS), 3)
        buf = RolloutBuffer(600, SMALL)
        seat_of = hand_agents(_table(4), 6)
        collect(nets, np.random.default_rng(21), buf, n_hands=6, device="cpu", pool=6,
                cfg=ACTOR_ONLY, agents=seat_of, critic=critic, forwards=forwards)
        got.append(buf)
    eager, static = got
    n = len(eager)
    assert np.array_equal(eager.action[:n], static.action[:n])
    np.testing.assert_allclose(static.value[:n], eager.value[:n], rtol=1e-4, atol=1e-5)

    compute_gae(eager, gamma=1.0, lam=0.95)
    before = [p.detach().clone() for p in critic.encoder.parameters()]
    learner = CriticLearner(critic, CriticTrainConfig(minibatch=32, fixed_shapes=True,
                                                      bf16=False))
    seats = relative_seat_agents(seat_of[eager.episode[:n]], eager.seat[:n])
    st = learner.update(eager, seats, device_rows=DeviceRows(eager, "cpu"))
    assert np.isfinite(st["critic/value_loss"])
    assert any(not torch.equal(p, q) for p, q in zip(critic.encoder.parameters(), before))


# --- the history cache ----------------------------------------------------------


def _decisions_by_slot(buf: RolloutBuffer) -> list[np.ndarray]:
    """Each (hand, seat)'s rows in the order they were decided."""
    n = len(buf)
    key = buf.episode[:n] * NUM_SEATS + buf.seat[:n]
    order = np.lexsort((buf.history_len[:n], key))
    splits = np.nonzero(np.diff(key[order]))[0] + 1
    return np.split(order, splits)


def test_the_history_store_encodes_what_the_encoder_does() -> None:
    """Round by round -- each slot's k-th decision batched with the others'
    -- the store's encoding matches the whole encoder's on real hands,
    where a closing trick rewrites the tokens of the plays in it."""
    from ..nn.model import HistoryStore

    _n, _c, _s, buf = _round()
    torch.manual_seed(2)
    enc = ActorCritic(SMALL).actor.history.eval()
    groups = _decisions_by_slot(buf)
    slot_of = {int(g[0]): i for i, g in enumerate(groups)}
    store = HistoryStore(enc, len(groups), "cpu")
    worst = 0.0
    for k in range(max(len(g) for g in groups)):
        rows = np.array([g[k] for g in groups if len(g) > k])
        slots = np.array([i for i, g in enumerate(groups) if len(g) > k])
        obs, _p, _sc = buf.as_torch(rows, "cpu")
        with torch.no_grad():
            want, keep = enc(obs.history, obs.history_len)
            got = store.advance(slots, obs.history, buf.history_len[rows])
        diff = (got - want).abs() * keep.unsqueeze(-1)
        worst = max(worst, float(diff.max()))
    assert worst < 1e-4, worst
    assert slot_of  # every slot seen


def test_without_the_provisional_tokens_the_cache_goes_stale() -> None:
    """The check has power: resuming exactly where each slot stopped,
    trusting the cached tokens, gives a different encoding."""
    from ..nn.model import HistoryStore

    _n, _c, _s, buf = _round()
    torch.manual_seed(2)
    enc = ActorCritic(SMALL).actor.history.eval()
    groups = _decisions_by_slot(buf)
    store = HistoryStore(enc, len(groups), "cpu")
    worst = 0.0
    for k in range(max(len(g) for g in groups)):
        rows = np.array([g[k] for g in groups if len(g) > k])
        slots = np.array([i for i, g in enumerate(groups) if len(g) > k])
        obs, _p, _sc = buf.as_torch(rows, "cpu")
        with torch.no_grad():
            want, keep = enc(obs.history, obs.history_len)
            got = store.advance(slots, obs.history, buf.history_len[rows], provisional=0)
        worst = max(worst, float(((got - want).abs() * keep.unsqueeze(-1)).max()))
    assert worst > 1e-2, worst


def _encoded_critic():
    cfg = replace(SMALL, critic_kind="cards", ffn="swiglu", critic_history_d=16,
                  critic_history_layers=1)
    torch.manual_seed(12)
    return _jitter(shared_critic(cfg, AGENTS), 3)


@pytest.mark.parametrize("forwards", ["eager", "static"])
@pytest.mark.parametrize("who", ["population", "one agent"])
def test_the_history_cache_collects_the_same_round(forwards: str, who: str) -> None:
    """Encoding each (hand, seat)'s history a trick at a time acts and
    values every row as encoding it whole does."""
    got = []
    for cached in (False, True):
        if who == "population":
            nets, critic = _actors(), _encoded_critic()
            kw = dict(cfg=ACTOR_ONLY, agents=hand_agents(_table(4), 6), critic=critic)
        else:
            torch.manual_seed(7)
            nets, kw = ActorCritic(SMALL).eval(), dict(cfg=SMALL)
        buf = RolloutBuffer(600, SMALL)
        collect(nets, np.random.default_rng(21), buf, n_hands=6, device="cpu", pool=6,
                forwards=forwards, history_cache=cached, **kw)
        got.append(buf)
    whole, cached = got
    n = len(whole)
    assert n == len(cached) and n > 0
    assert np.array_equal(whole.action[:n], cached.action[:n])
    np.testing.assert_allclose(cached.logp[:n], whole.logp[:n], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(cached.value[:n], whole.value[:n], rtol=1e-4, atol=1e-5)


def test_the_parallel_rollout_keeps_the_history_cache() -> None:
    nets, critic = _actors(), _encoded_critic()
    hands = 8
    seat_of = hand_agents(_table(8), hands)
    kw = dict(device="cpu", cfg=ACTOR_ONLY, pool=6, workers=2, agents=seat_of,
              critic=critic, overlap=True, forwards="static")
    whole, _st = collect_parallel(nets, np.random.default_rng(9), None, n_hands=hands, **kw)
    cached, stats = collect_parallel(nets, np.random.default_rng(9), None, n_hands=hands,
                                     history_cache=True, **kw)
    _check_round(cached, stats, hands)
    n = len(whole)
    assert np.array_equal(whole.action[:n], cached.action[:n])
    np.testing.assert_allclose(cached.value[:n], whole.value[:n], rtol=1e-4, atol=1e-5)
