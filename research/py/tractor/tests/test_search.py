"""Inference-time search: fair worlds, and no reading of hidden cards."""

from __future__ import annotations

import numpy as np
import torch

from ..eval.search import SearchConfig, known_voids, sample_world, search_move
from ..gen.episode import generate
from ..nn.features import Cache
from ..nn.model import ActorCritic, ModelConfig
from ..rules.cards import full_deck
from ..sim.state import NUM_SEATS, PlayState


def _midhand(seed: int, plies: int) -> PlayState:
    """A position reached by random legal play, so voids have had a chance."""
    rng = np.random.default_rng(seed)
    state, _ = generate(rng)
    for _ in range(plies):
        moves = state.legal_moves(cap=128).moves
        state.step(moves[int(rng.integers(len(moves)))])
    return state


def _redeal_hidden(state: PlayState, seat: int, seed: int) -> PlayState:
    """The same position with every card ``seat`` cannot see moved around.

    Sizes are kept and nothing else is: the new holdings may even break
    the voids public play implies, which is the point -- a search that
    only reads what the seat can see cannot tell the difference.
    """
    rng = np.random.default_rng(seed)
    out = state.clone()
    others = [s for s in range(NUM_SEATS) if s != seat]
    places = list(others)
    pool = sum((state.hands[s].astype(np.int64) for s in others), np.zeros(54, np.int64))
    if seat != state.landlord:
        pool = pool + state.kitty
        places.append(-1)
    cards = np.repeat(np.arange(54), pool)
    rng.shuffle(cards)
    at = 0
    for s in places:
        size = int(state.kitty.sum()) if s == -1 else int(state.hands[s].sum())
        counts = np.bincount(cards[at : at + size], minlength=54).astype(np.int8)
        at += size
        if s == -1:
            out.kitty = counts
        else:
            out.hands[s] = counts
    return out


def test_voids_are_true_of_the_real_hands() -> None:
    """Every void inferred from public play holds in the actual deal."""
    for seed in range(20):
        state = _midhand(seed, plies=60)
        voids = known_voids(state)
        for s in range(NUM_SEATS):
            for suit in np.nonzero(voids[s])[0]:
                held = int((state.hands[s] * state.ctx.suit_mask[suit]).sum())
                assert held == 0, (seed, s, suit)


def test_a_world_conserves_the_deck_and_the_constraints() -> None:
    for seed in range(20):
        state = _midhand(seed, plies=int(np.random.default_rng(seed).integers(1, 80)))
        seat = int(state.to_act)
        world = sample_world(state, seat, np.random.default_rng(seed))
        # The seat's own hand is untouched, and every other hand is the
        # size it really is.
        assert np.array_equal(world.hands[seat], state.hands[seat])
        for s in range(NUM_SEATS):
            assert int(world.hands[s].sum()) == int(state.hands[s].sum())
        assert int(world.kitty.sum()) == int(state.kitty.sum())
        if seat == state.landlord:
            assert np.array_equal(world.kitty, state.kitty)
        # Hands, kitty and the table add up to the whole deck.
        total = world.hands.sum(axis=0).astype(np.int64) + world.kitty + state.played_total
        assert np.array_equal(total, full_deck(state.num_decks))
        # Nobody is dealt a suit they have shown themselves out of.
        if not world.relaxed:
            voids = known_voids(state)
            for s in range(NUM_SEATS):
                for suit in np.nonzero(voids[s])[0]:
                    assert int((world.hands[s] * state.ctx.suit_mask[suit]).sum()) == 0


def test_worlds_do_not_depend_on_the_hidden_cards() -> None:
    """Re-dealing what the seat cannot see changes nothing it samples."""
    for seed in range(20):
        state = _midhand(seed, plies=40)
        seat = int(state.to_act)
        other = _redeal_hidden(state, seat, seed + 1000)
        a = sample_world(state, seat, np.random.default_rng(7))
        b = sample_world(other, seat, np.random.default_rng(7))
        assert np.array_equal(a.hands, b.hands)
        assert np.array_equal(a.kitty, b.kitty)


def test_the_search_does_not_read_the_hidden_cards() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig()
    net = ActorCritic(cfg).eval()
    search = SearchConfig(worlds=3, top_k=3, confident=1.1)
    chosen = 0
    for seed in range(6):
        state = _midhand(seed, plies=30)
        moves = list(state.legal_moves(cap=cfg.max_candidates).moves)
        if len(moves) < 2:
            continue
        # A prior that is flat, so the search -- not the prior -- decides.
        prior = np.full(len(moves), -np.log(len(moves)))
        seat = int(state.to_act)
        other = _redeal_hidden(state, seat, seed + 99)
        picks = [
            search_move(net, s, moves, prior, search, np.random.default_rng(3),
                        Cache(), cfg, "cpu")
            for s in (state, other)
        ]
        assert picks[0] == picks[1], seed
        assert picks[0][1] is True
        assert 0 <= picks[0][0] < len(moves)
        chosen += 1
    assert chosen >= 3


def test_the_search_leaves_easy_decisions_to_the_policy() -> None:
    cfg = ModelConfig()
    net = ActorCritic(cfg).eval()
    state = _midhand(0, plies=30)
    moves = list(state.legal_moves(cap=cfg.max_candidates).moves)
    rng = np.random.default_rng(0)
    # One legal move: nothing to search.
    assert search_move(net, state, moves[:1], np.zeros(1), SearchConfig(),
                       rng, Cache(), cfg, "cpu") == (0, False)
    # A confident prior: its choice stands.
    if len(moves) >= 2:
        prior = np.full(len(moves), np.log(0.01 / (len(moves) - 1)))
        prior[1] = np.log(0.99)
        assert search_move(net, state, moves, prior, SearchConfig(confident=0.9),
                           rng, Cache(), cfg, "cpu") == (1, False)
