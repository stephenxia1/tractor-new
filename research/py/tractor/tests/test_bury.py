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
