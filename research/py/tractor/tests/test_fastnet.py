"""Search's CPU forward (``eval/fastnet.py``) against the training forward.

It leaves out work and shares prefixes; neither may change what it
computes. Checked on real positions across whole hands, on search-shaped
batches (one decision's worlds, which share a history prefix, and leaf
batches whose seats to act differ), and through ``search_move`` itself.
"""

from __future__ import annotations

import numpy as np
import torch

from ..eval.fastnet import SearchNet
from ..eval.search import CandidatePolicy, SearchConfig, _in_world, sample_world, search_move
from ..gen.episode import generate
from ..nn.features import Cache
from ..nn.model import ActorCritic, ModelConfig
from ..train.buffer import obs_batch_from_arrays
from ..train.rollout import _evaluate, _InFlight, _pack

CFG = ModelConfig(d_model=48, n_layers_history=2, n_layers_query=2, n_heads=4, d_ff=96,
                  critic_d=32, critic_layers=1, max_candidates=128)


def _net(seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(CFG).eval()


def _states(n_hands: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_hands):
        state, _ = generate(rng, 2)
        while not state.done:
            out.append(state.clone())
            moves = state.legal_moves(cap=CFG.max_candidates).moves
            state.step(moves[int(rng.integers(len(moves)))])
    return out


def _obs(sims: list, cache: Cache):
    packed = [_pack(_InFlight(state=s, episode=0), cache, CFG) for s in sims]
    obs = obs_batch_from_arrays(
        np.stack([p.history for p in packed]), np.array([p.history_len for p in packed]),
        np.stack([p.context for p in packed]), np.stack([p.unseen for p in packed]),
        np.stack([p.planes for p in packed]), np.stack([p.cards for p in packed]),
        np.stack([p.candidates for p in packed]), np.stack([p.candidate_mask for p in packed]),
        device="cpu",
    )
    priv = torch.from_numpy(np.stack([p.privileged for p in packed]).astype(np.float32))
    return obs, priv


def _compare(net: ActorCritic, fast: SearchNet, sims: list, cache: Cache) -> None:
    obs, priv = _obs(sims, cache)
    with torch.no_grad():
        out, value = _evaluate(net, obs, priv)
    logits = fast.logits(obs)
    valid = obs.candidate_mask
    assert torch.allclose(logits[valid], out.logits[valid], atol=1e-4, rtol=1e-4)
    assert bool((logits[~valid] == out.logits[~valid]).all())
    assert torch.equal(logits.argmax(1), out.logits.argmax(1))
    assert torch.allclose(fast.value(obs, priv), value, atol=1e-5, rtol=1e-5)


def test_single_positions_across_whole_hands() -> None:
    net, cache = _net(), Cache()
    fast = SearchNet(net)
    states = _states(2, seed=1)
    for i in range(0, len(states), 16):
        _compare(net, fast, states[i : i + 16], cache)


def test_search_batches_share_a_prefix_and_leaves_do_not() -> None:
    net, cache = _net(1), Cache()
    fast = SearchNet(net)
    rng = np.random.default_rng(2)
    states = _states(2, seed=3)
    for state in states[10::23]:
        seat = int(state.to_act)
        worlds = [_in_world(state, sample_world(state, seat, rng)) for _ in range(12)]
        _compare(net, fast, worlds, cache)  # one seat, one shared prefix
        leaves = []
        for sim in worlds:  # play each to the end of the trick: mixed seats
            sim = sim.clone()
            start = len(sim.tricks)
            while not sim.done and len(sim.tricks) == start:
                moves = sim.legal_moves(cap=CFG.max_candidates).moves
                sim.step(moves[int(rng.integers(len(moves)))])
            if not sim.done:
                leaves.append(sim)
        if len(leaves) > 1:
            _compare(net, fast, leaves, cache)


def test_search_chooses_as_the_training_forward_does() -> None:
    net, cache = _net(2), Cache()
    search = SearchConfig(worlds=4, top_k=3, confident=0.99)
    exact = CandidatePolicy(net, cache, CFG, "cpu", fast=False)
    quick = CandidatePolicy(net, cache, CFG, "cpu")
    states = _states(1, seed=4)
    checked = 0
    for state in states[::9]:
        moves = state.legal_moves(cap=CFG.max_candidates).moves
        if len(moves) < 2:
            continue
        prior = np.log(np.full(len(moves), 1.0 / len(moves)))
        a = search_move(net, state, list(moves), prior, search, np.random.default_rng(5),
                        cache, CFG, "cpu", policy=exact)
        b = search_move(net, state, list(moves), prior, search, np.random.default_rng(5),
                        cache, CFG, "cpu", policy=quick)
        assert a == b
        checked += 1
    assert checked >= 3


# --- card-by-card networks (research/E7.md) -----------------------------------


def test_card_mode_encode_and_value_match_the_training_forward() -> None:
    from .test_cards_mode import CARDS, _net as cards_net

    net, cache = cards_net(11).eval(), Cache()
    fast = SearchNet(net)
    rng = np.random.default_rng(12)
    states = _states(1, seed=13)
    batches = [states[i : i + 12] for i in range(0, len(states), 12)]
    for state in states[5::20]:  # one decision's worlds: a shared prefix
        seat = int(state.to_act)
        batches.append([_in_world(state, sample_world(state, seat, rng)) for _ in range(8)])
    for sims in batches:
        packed = [_pack(_InFlight(state=s, episode=0), cache, CARDS) for s in sims]
        obs = obs_batch_from_arrays(
            np.stack([p.history for p in packed]), np.array([p.history_len for p in packed]),
            np.stack([p.context for p in packed]), np.stack([p.unseen for p in packed]),
            np.stack([p.planes for p in packed]), np.stack([p.cards for p in packed]),
            np.stack([p.candidates for p in packed]), np.stack([p.candidate_mask for p in packed]),
            device="cpu",
        )
        priv = torch.from_numpy(np.stack([p.privileged for p in packed]).astype(np.float32))
        with torch.no_grad():
            summary, card_states, value = net.encode(obs, priv)
        f_summary, f_cards = fast.encode(obs)
        assert torch.allclose(f_summary, summary, atol=1e-4, rtol=1e-4)
        assert torch.allclose(f_cards, card_states, atol=1e-4, rtol=1e-4)
        assert torch.allclose(fast.value(obs, priv), value, atol=1e-5, rtol=1e-5)


def test_card_mode_search_is_the_same_on_the_fast_path() -> None:
    from ..eval.search_cards import PickPolicy, beam_plays
    from .test_cards_mode import CARDS, _net as cards_net

    net, cache = cards_net(14).eval(), Cache()
    states = _states(1, seed=15)[3::11]
    quick, exact = PickPolicy(net, cache, CARDS, "cpu"), PickPolicy(net, cache, CARDS, "cpu", fast=False)
    for a, b in zip(quick.play(states), exact.play(states)):
        assert np.array_equal(a, b)
    assert np.allclose(quick.value(states), exact.value(states), atol=1e-5)
    for state in states:
        p1, l1 = beam_plays(net, state, 3, cache, CARDS, "cpu")
        p2, l2 = beam_plays(net, state, 3, cache, CARDS, "cpu", fast=False)
        assert all(np.array_equal(x, y) for x, y in zip(p1, p2))
        assert np.allclose(l1, l2, atol=1e-4)
