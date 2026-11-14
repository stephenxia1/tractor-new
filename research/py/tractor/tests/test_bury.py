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
