"""Search for a card-by-card network (``eval/search_cards.py``): the beam's
plays are legal, best first, and carry exactly the probability the
learner assigns them; the shared search runs on them."""

from __future__ import annotations

import numpy as np
import torch

from ..eval.search import SearchConfig
from ..eval.search_cards import beam_plays, search_cards_move
from ..gen.episode import generate
from ..nn.features import Cache
from ..nn.model import ObsBatch, pick_actions, play_log_prob
from ..sim.picks import PickRules, unpack_masks
from ..train.rollout import _InFlight, _pack
from .test_cards_mode import CARDS, _net


def _positions(n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    state, _ = generate(rng, 2)
    out = []
    while not state.done and len(out) < n:
        out.append(state.clone())
        moves = state.legal_moves(cap=128).moves
        state.step(moves[int(rng.integers(len(moves)))])
    return out


def _teacher_forced(net, state, play: np.ndarray, cache: Cache) -> float:
    """The learner's log-probability of ``play`` (``ppo._loss``'s path)."""
    rules = PickRules.for_state(state)
    picks, masks, n = rules.encode(play)
    prefix = np.zeros((len(picks), 54), dtype=np.float32)
    running = np.zeros(54, dtype=np.float32)
    for i in range(n):
        prefix[i] = running
        if picks[i] >= 0 and picks[i] < 54:
            running[picks[i]] += 1
    mask = unpack_masks(masks)
    mask[n:] = False
    p = _pack(_InFlight(state=state, episode=0), cache, CARDS)
    obs = ObsBatch(*[torch.as_tensor(x)[None] for x in (
        p.history.astype(np.float32), np.int64(p.history_len), p.context.astype(np.float32),
        p.unseen.astype(np.float32), p.planes.astype(np.float32), p.cards.astype(np.float32),
        prefix, mask)])
    with torch.no_grad():
        out = net.actor(obs)
    actions = pick_actions(obs.candidates, obs.candidate_mask)
    return float(play_log_prob(out.log_probs, actions)[0])


def test_beam_plays_are_legal_best_first_and_exactly_scored() -> None:
    net, cache = _net(7).eval(), Cache()
    for state in _positions(40, seed=8)[::6]:
        plays, logp = beam_plays(net, state, 3, cache, CARDS, "cpu", width=32)
        assert 1 <= len(plays) <= 3
        assert (np.diff(logp) <= 1e-9).all()
        rules = PickRules.for_state(state)
        for play, lp in zip(plays, logp):
            rules.trace(play)  # raises if the play is not reachable
            assert abs(_teacher_forced(net, state, play, cache) - lp) < 1e-4


def test_search_runs_on_the_beams_plays() -> None:
    net, cache = _net(9).eval(), Cache()
    search = SearchConfig(worlds=3, top_k=3, confident=0.99)
    for state in _positions(30, seed=10)[::10]:
        play, _searched = search_cards_move(net, state, search, np.random.default_rng(1),
                                            cache, CARDS, "cpu")
        plays, _ = beam_plays(net, state, 3, cache, CARDS, "cpu")
        assert any(np.array_equal(play, p) for p in plays)
