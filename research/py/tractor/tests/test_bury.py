"""Value-guided burial: legal, climbing, and scored as the network sees it."""

from __future__ import annotations

import numpy as np
import torch

from ..gen.episode import generate
from ..nn.features import Cache
from ..nn.model import ActorCritic, ModelConfig
from ..rules.cards import NUM_TYPES
from ..sim.bury import (
    BuryConfig,
    _base,
    _landlord_view_state,
    _rows,
    heuristic_burial,
    play_start_values,
    random_burial,
    value_burials,
)
from ..train.buffer import obs_batch_from_arrays
from ..train.rollout import _InFlight, _pack


def _pools(n: int, seed: int):
    """``n`` landlords' 33-card holdings, with their trump contexts."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        state, _info = generate(rng)
        pool = state.hands[state.landlord] + state.kitty
        out.append((pool.astype(np.int8), state.ctx, state.landlord))
    return out


def _net(seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(ModelConfig()).eval()


def test_heuristic_and_random_burials_are_legal() -> None:
    rng = np.random.default_rng(0)
    for pool, ctx, _landlord in _pools(20, 1):
        for burial in (heuristic_burial(pool, ctx), random_burial(pool, rng)):
            assert int(burial.sum()) == 8
            assert np.all(burial <= pool) and np.all(burial >= 0)


def test_candidate_rows_match_the_encoder() -> None:
    """A vectorised candidate row is the encoder's own observation."""
    cache = Cache()
    rng = np.random.default_rng(2)
    for pool, ctx, landlord in _pools(6, 3):
        base = _base(ctx, pool, landlord, 2, cache)
        for _ in range(3):
            burial = random_burial(pool, rng)
            hand = pool - burial
            rows = _rows([base], np.array([0]), hand[None], burial[None])
            state = _landlord_view_state(ctx, pool, burial, landlord, 2)
            packed = _pack(_InFlight(state=state, episode=0), cache, ModelConfig())
            for name in ("context", "unseen", "cards", "planes"):
                want = getattr(packed, name).astype(np.float32)
                assert np.array_equal(rows[name][0], want), name


def test_empty_history_value_matches_the_full_forward() -> None:
    net = _net(1)
    cache = Cache()
    cfg = ModelConfig()
    rng = np.random.default_rng(4)
    for pool, ctx, landlord in _pools(4, 5):
        base = _base(ctx, pool, landlord, 2, cache)
        burial = random_burial(pool, rng)
        rows = _rows([base], np.array([0]), (pool - burial)[None], burial[None])
        fast = play_start_values(net, rows, "cpu")[0]
        state = _landlord_view_state(ctx, pool, burial, landlord, 2)
        p = _pack(_InFlight(state=state, episode=0), cache, cfg)
        obs = obs_batch_from_arrays(
            p.history[None], np.array([p.history_len]), p.context[None],
            p.unseen[None], p.planes[None], p.cards[None], p.candidates[None],
            p.candidate_mask[None], device="cpu",
        )
        with torch.no_grad():
            slow = net.actor(obs).value[0].item()
        assert abs(fast - slow) < 1e-5, (fast, slow)


def test_the_climb_only_improves_and_stays_legal() -> None:
    net = _net(2)
    pools = _pools(5, 7)
    burials, stats = value_burials(
        net, [p for p, _c, _l in pools], [c for _p, c, _l in pools],
        [l for _p, _c, l in pools], bury=BuryConfig(rounds=4),
    )
    assert stats["value_end"] >= stats["value_start"] - 1e-6
    for (pool, _ctx, _landlord), burial in zip(pools, burials):
        assert burial.shape == (NUM_TYPES,)
        assert int(burial.sum()) == 8
        assert np.all(burial <= pool) and np.all(burial >= 0)


def test_device_rows_score_as_the_host_rows_do() -> None:
    """The on-device candidate build is the host build, value for value."""
    from ..sim.bury import _DeviceBases, _swaps

    net = _net(3)
    cache = Cache()
    pools = _pools(3, 11)
    pool_arr = np.stack([p for p, _c, _l in pools])
    bases = [_base(c, p, l, 2, cache) for p, c, l in pools]
    dev = _DeviceBases(bases, pool_arr, "cpu")
    owner, kits = [], []
    for i, (pool, ctx, _l) in enumerate(pools):
        b = heuristic_burial(pool, ctx)
        k = np.concatenate((b[None], _swaps(b, pool, np.nonzero(pool - b)[0][:4])))
        owner.append(np.full(len(k), i))
        kits.append(k)
    owner = np.concatenate(owner)
    kits = np.concatenate(kits)
    fast = dev.values(net, owner, kits, chunk=7)
    rows = _rows(bases, owner, pool_arr[owner] - kits, kits)
    slow = play_start_values(net, rows, "cpu")
    assert np.allclose(fast, slow, atol=1e-6), np.abs(fast - slow).max()
    # Every swap moves exactly one card out and one in.
    assert np.all(np.abs(kits[1:5] - kits[0]).sum(1) == 2)


def test_policy_burials_are_legal_and_blind_to_hidden_hands() -> None:
    """Eight legal picks; the other hands feed only the value head, so
    changing them changes no pick."""
    from ..nn.bury_net import BuryNet
    from ..sim.bury import policy_burials

    torch.manual_seed(5)
    net = BuryNet().eval()
    with torch.no_grad():
        net.score.weight.normal_()  # a non-uniform policy
    pools = _pools(6, 21)
    ps = [p for p, _c, _l in pools]
    cs = [c for _p, c, _l in pools]
    standings = [(1, 0)] * 6
    a, rows = policy_burials(net, ps, cs, standings, np.random.default_rng(1),
                             others=np.zeros((6, 3, NUM_TYPES)), record=True)
    b, _ = policy_burials(net, ps, cs, standings, np.random.default_rng(1),
                          others=np.random.default_rng(2).integers(0, 3, (6, 3, NUM_TYPES)))
    for pool, x, y in zip(ps, a, b):
        assert int(x.sum()) == 8 and np.all(x <= pool)
        assert np.array_equal(x, y)
    assert rows is not None and len(rows) == 8 * 6
    assert np.all(rows.mask[np.arange(len(rows)), rows.action])
    g1, _ = policy_burials(net, ps, cs, standings, np.random.default_rng(3), greedy=True)
    g2, _ = policy_burials(net, ps, cs, standings, np.random.default_rng(4), greedy=True)
    assert all(np.array_equal(x, y) for x, y in zip(g1, g2))


def test_the_heuristics_picks_are_its_burial() -> None:
    """``heuristic_picks`` is ``heuristic_burial`` as a sequence."""
    from ..sim.bury import heuristic_burial, heuristic_picks

    for pool, ctx, _l in _pools(20, 31):
        picks = heuristic_picks(pool, ctx)
        assert len(picks) == 8
        assert np.array_equal(np.bincount(picks, minlength=NUM_TYPES), heuristic_burial(pool, ctx))


def test_recorded_burials_give_back_their_landlords() -> None:
    """``pools_from_rows`` recovers what ``policy_burials`` was called with."""
    from ..nn.bury_net import BuryNet
    from ..sim.bury import policy_burials, pools_from_rows

    torch.manual_seed(6)
    net = BuryNet().eval()
    pools = _pools(7, 41)
    ps = [p for p, _c, _l in pools]
    cs = [c for _p, c, _l in pools]
    standings = [(0, -1), (1, 0), (2, 1), (3, 2), (4, 3), (1, -1), (2, 2)]
    others = np.random.default_rng(3).integers(0, 3, (7, 3, NUM_TYPES)).astype(np.float32)
    _b, rows = policy_burials(net, ps, cs, standings, np.random.default_rng(1),
                              others=others, record=True)
    got_pools, got_ctxs, got_standings, got_others = pools_from_rows(rows)
    assert np.array_equal(got_pools, np.stack(ps))
    assert [(c.level_rank, c.trump_suit) for c in got_ctxs] == [
        (c.level_rank, c.trump_suit) for c in cs]
    assert got_standings == standings
    assert np.array_equal(got_others, others.reshape(7, -1))


def test_imitation_teaches_the_heuristics_burial() -> None:
    """Rows built as ``policy_burials`` builds its own, every target a
    held card; after imitating them the greedy policy buries nearly the
    heuristic's cards on the same landlords."""
    from ..nn.bury_net import BuryNet
    from ..sim.bury import heuristic_burial, imitation_rows, policy_burials
    from ..train.bury_loop import imitate

    pools = _pools(48, 51)
    ps = [p for p, _c, _l in pools]
    cs = [c for _p, c, _l in pools]
    standings = [(1, 0)] * len(ps)
    rows = imitation_rows(ps, cs, standings)
    assert len(rows) == 8 * len(ps)
    assert not np.any(rows.target & ~rows.mask)
    first = rows.target[: len(ps)].sum(axis=1)
    want = np.array([int((heuristic_burial(p, c) > 0).sum()) for p, c in zip(ps, cs)])
    assert np.array_equal(first, want)

    torch.manual_seed(7)
    net = BuryNet()
    stats = imitate(net, rows, epochs=60, minibatch=128, lr=1e-3)
    assert stats["imitate/top_pick_in_set"] > 0.9, stats
    greedy, _ = policy_burials(net.eval(), ps, cs, standings, np.random.default_rng(0),
                               greedy=True)
    # Cards in common with the heuristic's burial, out of 8. A whole burial
    # matches only when all eight picks do, so this is the stabler measure.
    overlap = np.mean([int(np.minimum(g, heuristic_burial(p, c)).sum()) / 8
                       for g, p, c in zip(greedy, ps, cs)])
    assert overlap > 0.85, overlap


def test_leave_one_out_subtracts_the_others_mean() -> None:
    from ..train.starts import leave_one_out

    v = np.array([[1.0, 4.0], [2.0, 0.0], [6.0, 2.0]])
    got = leave_one_out(v)
    assert np.allclose(got[:, 0], [1 - 4, 2 - 3.5, 6 - 1.5])
    # The k advantages of a landlord sum to zero.
    assert np.allclose(got.sum(axis=0), 0.0)
    # Adding a constant to a landlord's column changes nothing.
    assert np.allclose(leave_one_out(v + np.array([10.0, -3.0])), got)


def test_extra_burials_share_their_landlords_hand() -> None:
    """``burial_samples``: k burials per landlord, every one legal, every
    start the original one but for the landlord's hand and the kitty, and
    the rows of sample s pointing at the hand they bury for."""
    from ..nn.bury_net import BuryNet
    from ..sim.bury import policy_burials
    from ..train.starts import Start, burial_samples

    torch.manual_seed(8)
    net = BuryNet().eval()
    with torch.no_grad():
        net.score.weight.normal_()
    rng = np.random.default_rng(9)
    starts, pools, cs = [], [], []
    for _ in range(5):
        state, _info = generate(rng)
        pool = (state.hands[state.landlord] + state.kitty).astype(np.int8)
        pools.append(pool)
        cs.append(state.ctx)
    others = np.zeros((5, 3, NUM_TYPES), dtype=np.float32)
    burials, rows = policy_burials(net, pools, cs, [(1, 0)] * 5, rng, others=others,
                                   record=True)
    # Starts made from the pools: the landlord holds pool - burial.
    rng2 = np.random.default_rng(9)
    for i in range(5):
        state, _info = generate(rng2)
        hands = state.hands.astype(np.int8).copy()
        hands[state.landlord] = (pools[i] - burials[i]).astype(np.int8)
        starts.append(Start(hands=hands, kitty=burials[i].astype(np.int8),
                            landlord=state.landlord, level_rank=state.ctx.level_rank,
                            trump_suit=state.ctx.trump_suit))
    merged, sample_of, hand_of, more = burial_samples(net, starts, rows, 3, rng)
    assert len(more) == 3 * 5 and len(merged) == 3 * 8 * 5
    assert np.array_equal(np.bincount(sample_of), [40, 40, 40])
    for s_idx, st in enumerate(more):
        h = hand_of[s_idx]
        orig = starts[h]
        assert int(st.kitty.sum()) == 8
        assert np.array_equal(st.hands[st.landlord].astype(np.int64) + st.kitty,
                              pools[h].astype(np.int64))
        for seat in range(4):
            if seat != st.landlord:
                assert np.array_equal(st.hands[seat], orig.hands[seat])
    assert set(merged.hand.tolist()) == set(range(5))
    assert np.all(merged.mask[np.arange(len(merged)), merged.action])


def test_a_resumed_burial_learner_keeps_the_configured_rate() -> None:
    """A saved optimiser state carries its learning rate; ``set_lr`` puts
    the config's back, so a resume can change it."""
    from ..nn.bury_net import BuryNet
    from ..train.bury_loop import BuryLearner, BuryTrainConfig

    old = BuryLearner(BuryNet(), BuryTrainConfig(lr=3e-4))
    new = BuryLearner(BuryNet(), BuryTrainConfig(lr=1e-4, epochs=2))
    new.opt.load_state_dict(old.opt.state_dict())
    assert new.opt.param_groups[0]["lr"] == 3e-4
    new.set_lr()
    assert all(g["lr"] == 1e-4 for g in new.opt.param_groups)


def test_playout_values_follow_their_starts() -> None:
    """``playout_values``: the landlord team's margin for each start, kept
    with its start -- greedy play of a permuted list gives the permuted
    values -- and in range (|margin| / 400 can pass 1 only by the kitty)."""
    from ..sim.bury import heuristic_burial
    from ..train.starts import Start, playout_values

    rng = np.random.default_rng(12)
    starts = []
    for _ in range(6):
        state, _info = generate(rng)
        pool = (state.hands[state.landlord] + state.kitty).astype(np.int64)
        burial = heuristic_burial(pool, state.ctx).astype(np.int64)
        hands = state.hands.astype(np.int8).copy()
        hands[state.landlord] = (pool - burial).astype(np.int8)
        starts.append(Start(hands=hands, kitty=burial.astype(np.int8), landlord=state.landlord,
                            level_rank=state.ctx.level_rank, trump_suit=state.ctx.trump_suit))
    net = _net(3)
    a = playout_values(net, starts, greedy=True, rng=np.random.default_rng(0))
    perm = [3, 0, 5, 1, 4, 2]
    b = playout_values(net, [starts[i] for i in perm], greedy=True, rng=np.random.default_rng(1))
    assert np.allclose(b, a[perm])
    assert np.all(np.abs(a) < 3.0) and np.any(a != 0.0)


def test_playout_values_on_a_rollout_pool_match_the_serial_ones() -> None:
    """The same greedy play-outs on worker processes as in this one."""
    from ..sim.bury import heuristic_burial
    from ..train.parallel import RolloutPool
    from ..train.starts import Start, playout_values

    rng = np.random.default_rng(13)
    starts = []
    for _ in range(6):
        state, _info = generate(rng)
        pool = (state.hands[state.landlord] + state.kitty).astype(np.int64)
        burial = heuristic_burial(pool, state.ctx).astype(np.int64)
        hands = state.hands.astype(np.int8).copy()
        hands[state.landlord] = (pool - burial).astype(np.int8)
        starts.append(Start(hands=hands, kitty=burial.astype(np.int8), landlord=state.landlord,
                            level_rank=state.ctx.level_rank, trump_suit=state.ctx.trump_suit))
    net = _net(4)
    serial = playout_values(net, starts, greedy=True, rng=np.random.default_rng(0))
    workers = RolloutPool(cfg=net.cfg, workers=2, hands_per_round=4)
    try:
        parallel = playout_values(net, starts, greedy=True, rng=np.random.default_rng(0),
                                  pool=workers)
    finally:
        workers.close()
    assert np.allclose(parallel, serial, atol=1e-6), (parallel, serial)
