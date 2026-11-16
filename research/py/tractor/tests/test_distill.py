"""Expert iteration: search-labelled positions and the distillation step."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from ..eval.search import SearchConfig
from ..gen.episode import generate as deal
from ..nn.model import ActorCritic, ModelConfig
from ..sim.bury import heuristic_burial
from ..train.distill import Distiller, generate, load_rows
from ..train.starts import Start

SMALL = ModelConfig(d_model=32, n_layers_history=1, n_layers_query=1, n_heads=2, d_ff=64,
                    critic_d=16, critic_layers=1, critic_heads=2, critic_d_ff=32,
                    max_candidates=16)


def _starts(n: int, seed: int) -> list:
    # The heuristic only makes test positions here; nothing trains on it
    # beyond this test.
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        state, _info = deal(rng)
        pool = (state.hands[state.landlord] + state.kitty).astype(np.int64)
        burial = heuristic_burial(pool, state.ctx).astype(np.int64)
        hands = state.hands.astype(np.int8).copy()
        hands[state.landlord] = (pool - burial).astype(np.int8)
        out.append(Start(hands=hands, kitty=burial.astype(np.int8), landlord=state.landlord,
                         level_rank=state.ctx.level_rank, trump_suit=state.ctx.trump_suit))
    return out


def test_generated_rows_are_searched_positions_and_round_trip() -> None:
    torch.manual_seed(0)
    net = ActorCritic(SMALL).eval()
    search = SearchConfig(worlds=2, top_k=2, confident=0.999)
    rows, stats = generate(net, _starts(2, 1), search, np.random.default_rng(2), "cpu", SMALL)
    assert len(rows) == stats["searched"] > 0
    assert 0 < stats["searched_frac"] <= 1
    assert np.all(rows.candidate_mask[np.arange(len(rows)), rows.target])
    assert np.all(rows.candidate_mask.sum(axis=1) > 1)
    with tempfile.TemporaryDirectory() as tmp:
        rows.save(os.path.join(tmp, "a.npz"))
        rows.save(os.path.join(tmp, "b.npz"))
        both = load_rows(os.path.join(tmp, "*.npz"))
    assert len(both) == 2 * len(rows)
    assert np.array_equal(both.target[: len(rows)], rows.target)


def test_distillation_moves_the_actor_toward_the_targets() -> None:
    torch.manual_seed(0)
    net = ActorCritic(SMALL).eval()
    search = SearchConfig(worlds=2, top_k=2, confident=0.999)
    rows, _ = generate(net, _starts(2, 3), search, np.random.default_rng(4), "cpu", SMALL)
    critic_before = {k: v.clone() for k, v in net.critic.state_dict().items()}
    d = Distiller(net, rows, "cpu", lr=3e-3, minibatch=64)
    first = d.step(1, np.random.default_rng(5))
    for _ in range(20):
        last = d.step(1, np.random.default_rng(6))
    assert last["distill/ce"] < first["distill/ce"]
    # The actor alone: the critic is not this step's to change.
    assert all(torch.equal(v, net.critic.state_dict()[k]) for k, v in critic_before.items())
