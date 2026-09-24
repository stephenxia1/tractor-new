"""Determinized search at inference time, around the trained policy.

The strongest fair bots (``smart-v18`` and its descendants) do one thing
the network does not: before they move, they sample deals consistent
with what their seat can see, try their candidate moves in each, and
keep the one that does best on average. The network moves in a single
forward pass. This wraps it in the same loop, with every hand-tuned
piece replaced by a learned one:

1. **Worlds.** ``sample_world`` deals the cards this seat has not seen
   to the other three hands (and to the kitty, unless this seat is the
   landlord and buried it), respecting the two things public play
   reveals: how many cards each seat still holds, and which suits a seat
   has shown itself out of.

2. **Candidates.** The policy's own top ``top_k`` moves by prior. A
   decision with one legal move, or where the policy already puts
   ``confident`` of its mass on one move, is not searched.

3. **Simulation.** In each world, each candidate is played and the
   policy then plays every seat -- each from its own view of that world
   -- until ``extra_tricks`` more tricks have closed after the current
   one.

4. **Leaf value.** The points those tricks settled, signed for the
   searching team, plus the privileged critic's estimate of the points
   still to come *in that world* -- the one place a value network that
   reads all four hands is exactly what is wanted. A hand that ends
   inside the simulation is scored on what it actually settled,
   including the kitty bonus, which the points-only reward never sees.

Every candidate is scored against the same worlds, so the comparison
between candidates carries no sampling noise from the deal.

Nothing here reads a hidden card. The sampler takes the searching
seat's hand, the public record and the hand *sizes* from the state; the
other seats' actual holdings are overwritten before the simulation
starts. ``tests/test_search.py`` holds that to account by re-dealing the
hidden cards and demanding identical worlds and an identical choice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from ..nn.features import POINTS_PER_DECK, Cache
from ..nn.model import ModelConfig
from ..rules.cards import NUM_SUITS, NUM_TYPES
from ..rules.combos import play_suit
from ..sim.state import NUM_SEATS, PlayState, team_of
from ..train.buffer import obs_batch_from_arrays
from ..train.rollout import _evaluate, _InFlight, _pack

__all__ = ["SearchConfig", "known_voids", "sample_world", "search_move"]

# The critic is trained on per-seat trick credit of +/- p / (2 * deck
# points), so its value times this is the future point margin of the
# seat's team. See the reward section of train/rollout.py.
def _value_to_points(num_decks: int) -> float:
    return 2.0 * POINTS_PER_DECK * num_decks

# Retries before a sampler that keeps painting itself into a corner
# gives up on the void constraints. A dead end needs a seat's capacity
# to be exhausted by the cards only it may hold, which the most-
# constrained-first order makes rare.
MAX_SAMPLE_ATTEMPTS = 32

# Effective suits: the four plain suits, then trump.
NUM_EFF_SUITS = NUM_SUITS + 1


@dataclass(frozen=True)
class SearchConfig:
    """How much search to spend per decision."""

    worlds: int = 8
    top_k: int = 3
    # Tricks simulated past the current one before the critic judges
    # the position. 0 stops when the current trick closes.
    extra_tricks: int = 0
    # Skip the search when the policy's top move already has this much
    # probability. 1.0 or more searches every decision with a choice.
    confident: float = 0.9

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class World:
    hands: np.ndarray  # int8[4, 54]; the searching seat's row is its real hand
    kitty: np.ndarray  # int8[54]
    relaxed: bool  # a void constraint had to be broken to place every card


def known_voids(state: PlayState) -> np.ndarray:
    """bool[4, 5]: seats that have shown themselves out of a suit.

    A follower that answered a lead of ``n`` cards in suit ``s`` with
    fewer than ``n`` cards of ``s`` played every ``s`` it held (the
    follow rules require it), so it holds none from then on. Read from
    the public record only: the closed tricks and the one in progress.
    """
    voids = np.zeros((NUM_SEATS, NUM_EFF_SUITS), dtype=bool)
    tricks = [(t.leader, t.plays) for t in state.tricks]
    if state.trick_plays:
        tricks.append((state.trick_leader, state.trick_plays))
    for leader, plays in tricks:
        if not plays:
            continue
        lead = plays[0]
        suit = play_suit(lead, state.ctx)
        if suit is None:
            continue
        need = int(lead.sum())
        mask = state.ctx.suit_mask[suit]
        for i in range(1, len(plays)):
            if int((plays[i] * mask).sum()) < need:
                voids[(leader + i) % NUM_SEATS, suit] = True
    return voids


def sample_world(state: PlayState, seat: int, rng: np.random.Generator) -> World:
    """Deal every card ``seat`` has not seen, consistently with public play.

    Locations are the three other hands, sized as they stand, and the
    kitty unless ``seat`` is the landlord (who buried it and so knows
    it). Cards are placed most-constrained first, each into an allowed
    location with probability proportional to the room left there.
    """
    unseen = state.unseen_from(seat).astype(np.int64)
    others = [s for s in range(NUM_SEATS) if s != seat]
    sizes = [int(state.hands[s].sum()) for s in others]
    knows_kitty = seat == state.landlord
    if not knows_kitty:
        sizes.append(int(state.kitty.sum()))
    if int(unseen.sum()) != sum(sizes):
        raise RuntimeError(
            f"seat {seat} has {int(unseen.sum())} unseen cards for "
            f"{sum(sizes)} hidden places"
        )

    voids = known_voids(state)
    eff = state.ctx.eff
    cards = np.repeat(np.arange(NUM_TYPES), unseen)
    # allowed[i, loc]: may card i go to location loc.
    allowed = np.ones((cards.size, len(sizes)), dtype=bool)
    for j, s in enumerate(others):
        allowed[:, j] = ~voids[s, eff[cards]]

    relaxed = False
    for attempt in range(MAX_SAMPLE_ATTEMPTS + 1):
        if attempt == MAX_SAMPLE_ATTEMPTS:
            allowed[:] = True
            relaxed = True
        room = np.array(sizes, dtype=np.int64)
        # Most constrained first; random within a tie, so no card type
        # is systematically placed before another.
        order = np.lexsort((rng.random(cards.size), allowed.sum(axis=1)))
        where = np.empty(cards.size, dtype=np.int64)
        ok = True
        for i in order:
            weight = room * allowed[i]
            total = int(weight.sum())
            if total == 0:
                ok = False
                break
            loc = int(np.searchsorted(np.cumsum(weight), rng.random() * total, side="right"))
            where[i] = loc
            room[loc] -= 1
        if ok:
            break

    hands = state.hands.copy()
    for j, s in enumerate(others):
        hands[s] = np.bincount(cards[where == j], minlength=NUM_TYPES).astype(np.int8)
    if knows_kitty:
        kitty = state.kitty.copy()
    else:
        kitty = np.bincount(cards[where == len(others)], minlength=NUM_TYPES).astype(np.int8)
    return World(hands=hands, kitty=kitty, relaxed=relaxed)


def _in_world(state: PlayState, world: World) -> PlayState:
    sim = state.clone()
    sim.hands = world.hands.copy()
    sim.kitty = world.kitty.copy()
    return sim


def _forward(
    net: Any, sims: list[PlayState], cache: Cache, cfg: ModelConfig, device: Any
) -> tuple[list[Any], Any, torch.Tensor]:
    """One batched forward over the seat to act in each state."""
    packed = [_pack(_InFlight(state=s, episode=0), cache, cfg) for s in sims]
    obs = obs_batch_from_arrays(
        np.stack([p.history for p in packed]),
        np.array([p.history_len for p in packed], dtype=np.int64),
        np.stack([p.context for p in packed]),
        np.stack([p.unseen for p in packed]),
        np.stack([p.planes for p in packed]),
        np.stack([p.cards for p in packed]),
        np.stack([p.candidates for p in packed]),
        np.stack([p.candidate_mask for p in packed]),
        device=device,
    )
    privileged = torch.from_numpy(
        np.stack([p.privileged for p in packed]).astype(np.float32)
    ).to(device)
    with torch.no_grad():
        out, value = _evaluate(net, obs, privileged)
    return packed, out, value


def search_move(
    net: Any,
    state: PlayState,
    moves: list[np.ndarray],
    log_prior: np.ndarray,
    search: SearchConfig,
    rng: np.random.Generator,
    cache: Cache,
    cfg: ModelConfig,
    device: Any,
) -> tuple[int, bool]:
    """``(index into moves, searched)`` for the seat to act in ``state``."""
    order = np.argsort(-log_prior, kind="stable")
    best = int(order[0])
    if len(moves) <= 1 or float(np.exp(log_prior[best])) >= search.confident:
        return best, False
    top = [int(i) for i in order[: search.top_k]]
    if len(top) == 1:
        return best, False

    seat = int(state.to_act)
    team = team_of(seat)
    start = len(state.tricks)
    stop = start + 1 + search.extra_tricks

    worlds = [sample_world(state, seat, rng) for _ in range(search.worlds)]
    sims: list[PlayState] = []
    for world in worlds:
        for i in top:
            sim = _in_world(state, world)
            sim.step(moves[i])
            sims.append(sim)

    # Play every simulation forward in lockstep: one batched forward per
    # ply over the simulations that still have a real choice to make.
    while True:
        live = [s for s in sims if not s.done and len(s.tricks) < stop]
        if not live:
            break
        choose: list[PlayState] = []
        for s in live:
            options = s.legal_moves(cap=cfg.max_candidates)
            if len(options.moves) == 1:
                s.step(options.moves[0])
            else:
                choose.append(s)
        if not choose:
            continue
        packed, out, _value = _forward(net, choose, cache, cfg, device)
        # Greedy, as the evaluation plays; padded slots already sit at
        # MASK_LOGIT and cannot win the argmax.
        slots = out.logits.float().argmax(dim=1).cpu().numpy()
        for s, p, slot in zip(choose, packed, slots):
            s.step(p.moves.moves[int(slot)])

    # Leaf values: the critic's future margin for the seat to act, signed
    # for the searching team. Finished hands have nothing to come.
    future = np.zeros(len(sims), dtype=np.float64)
    open_rows = [k for k, s in enumerate(sims) if not s.done]
    if open_rows:
        _packed, _out, value = _forward(
            net, [sims[k] for k in open_rows], cache, cfg, device
        )
        value = value.float().cpu().numpy()
        for k, v in zip(open_rows, value):
            sign = 1.0 if team_of(int(sims[k].to_act)) == team else -1.0
            future[k] = sign * float(v) * _value_to_points(state.num_decks)

    scores = np.zeros(len(sims), dtype=np.float64)
    for k, s in enumerate(sims):
        settled = 0.0
        for t in s.tricks[start:]:
            settled += t.points if team_of(t.winner) == team else -t.points
        if s.done:
            bonus = s.attacker_points - s.trick_points_attacker
            attackers = team != s.defenders
            settled += bonus if attackers else -bonus
        scores[k] = settled + future[k]

    mean = scores.reshape(len(worlds), len(top)).mean(axis=0)
    # Ties go to the policy's preference, which is `top`'s order.
    return top[int(np.argmax(mean))], True
