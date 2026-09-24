"""Self-play rollout: many hands in flight, one forward pass per step.

Why a pool of hands. The simulator answers a decision in tens of
microseconds and the network answers one in milliseconds, and at batch
one almost all of that millisecond is launch overhead rather than
arithmetic. A loop that plays one hand at a time therefore runs the
accelerator at a single-digit percentage of what it can do, and buying a
bigger one changes nothing. This actor instead keeps a pool of hands in
flight, encodes every seat that is currently to act, and puts all of them
through a single forward. Hands are stepped in lockstep and a slot is
refilled the moment its hand ends, so the batch stays at the pool size
for the whole round rather than decaying as hands finish at different
lengths.

All four seats are the same network, with no shared state between them
beyond the position itself -- a zero-style, partner-agnostic agent. There
is no scripted bot in this file, and none behind it either: the
declaration and the burial come from ``gen/episode.py``, which draws both
at random, so what is being learned is the play phase and only the play
phase.

Sampling reads the observation *as stored*. The observation arrays are
cast to float16 before the batch is built, and the buffer keeps those
same arrays, so the log-probability recorded beside a row is the
log-probability of exactly the tensor a learner will later read back. The
alternative -- sample from float32, store float16 -- puts a small
systematic error into every importance ratio, in the one direction that
looks like a slightly stale behaviour policy and is therefore invisible.

Rewards
-------
Dense, zero-sum, and always from the acting seat's team's point of view.
Teams are seats ``{0, 2}`` and ``{1, 3}``; ``team_of`` is the authority.

*Trick credit* (``reward_points``). A trick has exactly four decisions,
one per seat. When it closes carrying ``p`` points -- for the trick that
ends the hand, its points plus the kitty bonus the attackers won with it
-- each of those four decisions is credited

    + p / (2 * 200)   when the deciding seat's team took the trick
    - p / (2 * 200)   otherwise

200 is the points in the two-deck deck, and is derived from the deck
count rather than written in. The 2 is the load-bearing part: each team
makes exactly two of a trick's four decisions, so dividing by it makes a
*team's* credit for the trick exactly ``p / 200``, and a team's trick
credits over a whole hand then sum to exactly the fraction of the deck's
points that team captured. Without the 2 that sum is double the quantity
it is supposed to be, which is a factor that survives a training run
disguised as a learning rate.

*Terminal* (``reward_level``). ``hand_result`` turns the attackers' point
total into level gains ``(defenders, attackers)``, of which at most one
is non-zero and neither exceeds three. Each seat's last decision of the
hand -- the row carrying ``done`` -- is credited

    + (attacker_gain - defender_gain) / 3   for an attacking seat
    - (attacker_gain - defender_gain) / 3   for a defending seat

so the terminal term lands in [-1, 1].

*The kitty bonus.* When the attackers take the last trick,
``PlayState._finish`` folds the kitty's points into their total
multiplied by *twice the number of cards in the winning play*
(rules.md §8.2), not by a flat two: a single-card last trick doubles
them, a pair quadruples them, and the largest seen over 300 sampled
hands was eight times. Those points are in no trick record, so until
2026-09-24 the dense term could not see them at all. They are now
credited with the last trick, whose four decisions exist as rows even
when forced. Earlier decisions learn of the bonus through the critic,
which reads the kitty in its privileged view, and the landlord's own
value head sees what it buried -- which is what value-guided burial
needs, since a reward blind to the bonus would rate burying every
point card as free.

The two terms are stored apart and combined as
``reward = reward_points + level_weight * reward_level``. A run that
starts on the dense point signal and anneals towards the hand outcome
moves ``level_weight`` from 0 to 1; at the default of 1 the combined
reward is simply their sum.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..gen.episode import deal_sizes, generate
from ..nn.features import POINTS_PER_DECK, Cache, encode
from ..nn.model import MASK_LOGIT, ModelConfig, privileged_view
from ..rules.cards import NUM_TYPES
from ..sim.actions import MoveSet
from ..sim.state import NUM_SEATS, PlayState, hand_result, team_of
from .buffer import RolloutBuffer, obs_batch_from_arrays

__all__ = ["collect", "DEFAULT_POOL", "DECKS", "MAX_LEVEL_GAIN"]

# Two decks, matching ``gen.episode.generate``'s default and the only
# deal the feature encoder's normalisers are calibrated for.
DECKS = 2

# Hands in flight. 256 sits in the middle of the 128-512 band the network
# is efficient over, and costs 256 live ``PlayState`` objects -- about
# 100 KB -- which is nothing next to one batch of observations.
DEFAULT_POOL = 256

# Each team makes two of a trick's four decisions; see the module
# docstring for why the trick credit is divided by this.
SEATS_PER_TEAM_PER_TRICK = 2

# The largest level swing ``hand_result`` can report, used to normalise
# the terminal reward into [-1, 1].
MAX_LEVEL_GAIN = 3.0

_NULL_SPAN = contextlib.nullcontext()


def _span(spans: Any, name: str) -> Any:
    """``spans.span(name)``, or a no-op when no recorder was passed."""
    return _NULL_SPAN if spans is None else spans.span(name)


@dataclass(slots=True)
class _Decision:
    """One seat's packed inputs for the next forward pass.

    Float16 already, because these arrays go straight into the buffer
    after the forward that reads them.
    """

    seat: int
    history: np.ndarray
    history_len: int
    context: np.ndarray
    unseen: np.ndarray
    planes: np.ndarray
    cards: np.ndarray
    candidates: np.ndarray
    candidate_mask: np.ndarray
    privileged: np.ndarray
    moves: MoveSet


@dataclass(slots=True)
class _InFlight:
    """A hand the pool is playing, and where its rows have landed."""

    state: PlayState
    episode: int
    # Rows of the trick in progress, credited and cleared when it closes.
    trick_rows: list[int] = field(default_factory=list)
    # Each seat's most recent row, so the terminal reward and the done
    # flag can be written to the last decision each seat made.
    last_row: np.ndarray = field(
        default_factory=lambda: np.full(NUM_SEATS, -1, dtype=np.int64)
    )


def _pack(
    hand: _InFlight,
    cache: Cache,
    cfg: ModelConfig,
    spans: Any = None,
    moves: MoveSet | None = None,
) -> _Decision:
    """Encode the seat to act and enumerate what it may play.

    The two are timed apart. They are both per-decision host work that
    the batched forward exists to amortise, but they are different
    work with different remedies, and one span over the pair cannot
    say which of them is worth attention.

    ``moves`` replaces the enumeration for a caller that already holds
    the action set. ``eval/bots.py`` is the one: there the TypeScript
    environment owns the game and its candidate list is the authority,
    so re-deriving one here would offer the network moves the other side
    is not prepared to play. It routes through this function anyway,
    because a position becoming network input in two places is how an
    evaluation ends up measuring a policy training never produced.
    """
    state = hand.state
    seat = int(state.to_act)
    with _span(spans, "encode"):
        obs = encode(state, seat, cache)
    # Capped at the tensor width: anything the enumerator produced past
    # this would be dropped when the batch is padded, and a candidate the
    # policy cannot see is worse than one that was never offered.
    if moves is None:
        with _span(spans, "enumerate"):
            moves = state.legal_moves(cap=cfg.max_candidates)
    kept = min(len(moves.moves), cfg.max_candidates)
    if kept == 0:
        raise RuntimeError(f"seat {seat} has no legal move")

    candidates = np.zeros((cfg.max_candidates, NUM_TYPES), dtype=np.float16)
    candidates[:kept] = np.asarray(moves.moves[:kept], dtype=np.float16)
    mask = np.zeros(cfg.max_candidates, dtype=bool)
    mask[:kept] = True

    return _Decision(
        seat=seat,
        history=obs.history.astype(np.float16),
        history_len=int(obs.history_len),
        context=obs.context.astype(np.float16),
        unseen=obs.unseen.astype(np.float16),
        planes=obs.planes.astype(np.float16),
        cards=obs.cards.astype(np.float16),
        candidates=candidates,
        candidate_mask=mask,
        privileged=privileged_view(state.hands, state.kitty, seat).astype(
            np.float16
        ),
        moves=moves,
    )


def _evaluate(net: Any, obs: Any, privileged: torch.Tensor) -> tuple[Any, Any]:
    """``(ActorOutput, value)`` from an ActorCritic, or from a bare actor.

    An ``ActorCritic`` is the intended argument and gives the privileged
    value the critic exists for. A bare ``TractorNet`` is accepted so that
    a smoke test or an evaluation run does not have to build a critic it
    will not use; its own imperfect-information value head answers
    instead.
    """
    has_pair = (
        getattr(net, "critic", None) is not None
        and getattr(net, "actor", None) is not None
    )
    if has_pair:
        return net(obs, privileged)
    out = net(obs)
    return out, out.value


def _choose(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    greedy: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a slot per row, returning the slots and their log-probs.

    The distribution is re-derived from the logits rather than taken from
    ``ActorOutput.log_probs`` so that the temperature enters in one place
    and the recorded log-probability is always the behaviour policy's --
    the quantity an importance ratio needs, not the tempered-down
    policy's evaluation of itself.

    Sampling is Gumbel-max over the whole padded slot axis, drawn from the
    caller's ``Generator`` so that a rollout is reproducible from one
    seed. Invalid slots sit at ``MASK_LOGIT`` after the log-softmax, about
    -1e9, so no Gumbel draw can lift one past a legal slot; they are
    excluded exactly rather than by being unlikely.
    """
    scaled = logits if temperature == 1.0 else logits / temperature
    scaled = torch.where(mask, scaled, torch.full_like(scaled, MASK_LOGIT))
    # `.float()` before numpy, not instead of the autocast: the value
    # is the one bfloat16 produced, which is the point -- it is what
    # the learner will recompute -- and float32 merely stores it
    # losslessly, numpy having no bfloat16.
    log_probs = (
        F.log_softmax(scaled, dim=-1).detach().float().cpu().numpy()
    )

    if greedy:
        slots = log_probs.argmax(axis=1)
        # A greedy behaviour policy is a point mass on the slot it
        # took, so its log-probability is zero. Returning the
        # softmax value instead would hand the learner a ratio of
        # exp(new_logp) against a policy that never sampled, which
        # is not an importance weight for anything.
        chosen = np.zeros(log_probs.shape[0], dtype=np.float32)
    else:
        noise = rng.gumbel(size=log_probs.shape)
        slots = (log_probs + noise).argmax(axis=1)
        chosen = log_probs[np.arange(log_probs.shape[0]), slots]
    return slots.astype(np.int64), chosen.astype(np.float32)


def _combine(buffer: RolloutBuffer, row: int, level_weight: np.float32) -> None:
    """Rewrite ``reward`` from its two terms.

    Derived rather than accumulated in parallel: computing the combined
    reward from the same two float32 values the buffer stores is what
    makes ``reward == reward_points + level_weight * reward_level`` hold
    to the bit, which is the invariant a learner that re-weights the
    terms depends on.
    """
    buffer.reward[row] = (
        buffer.reward_points[row] + level_weight * buffer.reward_level[row]
    )


def _credit_trick(
    buffer: RolloutBuffer,
    hand: _InFlight,
    point_norm: float,
    level_weight: np.float32,
) -> None:
    """Split the closed trick's points between the two teams.

    The trick that ends the hand also carries the kitty bonus, which
    ``PlayState._finish`` has folded into ``attacker_points`` by the
    time this runs. It is nonzero only when the attackers took that
    trick, so it goes to the trick's winners like the trick's own
    points, and a team's credits over a hand sum to the margin the game
    actually scores.
    """
    state = hand.state
    trick = state.tricks[-1]
    points = trick.points
    if state.done:
        points += state.attacker_points - state.trick_points_attacker
    rows = hand.trick_rows
    if len(rows) != NUM_SEATS:
        raise RuntimeError(
            f"trick closed with {len(rows)} recorded decisions, expected "
            f"{NUM_SEATS}"
        )
    share = np.float32(points / (SEATS_PER_TEAM_PER_TRICK * point_norm))
    winners = team_of(trick.winner)
    for row in rows:
        sign = np.float32(1.0 if team_of(int(buffer.seat[row])) == winners else -1.0)
        buffer.reward_points[row] += sign * share
        _combine(buffer, row, level_weight)
    hand.trick_rows = []


def _credit_terminal(
    buffer: RolloutBuffer, hand: _InFlight, level_weight: np.float32
) -> None:
    """Score the finished hand onto each seat's last decision."""
    state = hand.state
    defender_gain, attacker_gain = hand_result(state.attacker_points)
    swing = np.float32((attacker_gain - defender_gain) / MAX_LEVEL_GAIN)
    for seat in range(NUM_SEATS):
        row = int(hand.last_row[seat])
        if row < 0:
            raise RuntimeError(
                f"seat {seat} made no decision in hand {hand.episode}"
            )
        sign = np.float32(-1.0 if team_of(seat) == state.defenders else 1.0)
        buffer.reward_level[row] += sign * swing
        _combine(buffer, row, level_weight)
        buffer.done[row] = True


def _room_for_another(
    buffer: RolloutBuffer, active: list[_InFlight], budget: int
) -> bool:
    """Can a new hand start without the buffer filling mid-hand?

    Every decision plays at least one card, so the cards still in a hand's
    four hands are a hard ceiling on the rows it has left to write. Adding
    a new hand is safe exactly when the free rows cover every hand in
    flight plus a whole fresh one. This is what keeps ``add`` from raising
    halfway through a trick and leaving a half-credited hand in the
    buffer, which would be data no consumer could detect as broken.
    """
    free = buffer.capacity - len(buffer)
    reserved = sum(int(hand.state.hands.sum()) for hand in active)
    return free - reserved >= budget


def collect(
    net: Any,
    rng: np.random.Generator,
    buffer: RolloutBuffer,
    n_hands: int,
    device: torch.device | str,
    cfg: ModelConfig | None = None,
    temperature: float = 1.0,
    greedy: bool = False,
    spans: Any = None,
    level_weight: float = 1.0,
    pool: int | None = None,
    amp: bool = False,
    starts: Any = None,
) -> dict[str, float]:
    """Play ``n_hands`` self-play hands into ``buffer``.

    ``starts``, when given, are the play starts (``train.starts.Start``),
    one per hand, in place of ``gen.episode.generate``'s uniform draw.

    ``net`` is an ``ActorCritic`` -- the actor picks the move and the
    critic, reading the four hands and the kitty through
    ``privileged_view``, records the value. A bare ``TractorNet`` also
    works and falls back to its own value head.

    ``temperature`` scales the logits before sampling and ``greedy`` takes
    the argmax instead. ``level_weight`` weighs the terminal term against
    the dense one; see the module docstring for the credit rule.
    ``spans`` records "encode", "forward", "sample" and "step" when a
    ``telemetry.Spans`` is passed. ``pool`` overrides how many hands run
    concurrently, which is also the forward batch size.

    Returns a stats dictionary. ``hands`` is how many hands actually
    completed: the collector stops starting hands once the buffer cannot
    guarantee room for another whole one, so a caller that sizes the
    buffer too small gets fewer hands rather than a half-written one.
    """
    cfg = cfg or getattr(net, "cfg", None) or ModelConfig()
    if cfg.max_candidates != buffer.candidates.shape[1]:
        raise ValueError(
            f"cfg.max_candidates {cfg.max_candidates} does not match the "
            f"buffer's {buffer.candidates.shape[1]}"
        )
    if not temperature > 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    n_hands = int(n_hands)
    if n_hands < 1:
        raise ValueError(f"n_hands must be positive, got {n_hands}")

    hand_size, _kitty = deal_sizes(DECKS)
    # A hard ceiling on the decisions in one hand: every decision sheds at
    # least one card from the acting seat's hand.
    max_decisions = hand_size * NUM_SEATS
    if buffer.capacity < max_decisions:
        raise ValueError(
            f"a hand can take up to {max_decisions} rows; a buffer of "
            f"{buffer.capacity} cannot hold one"
        )

    point_norm = float(POINTS_PER_DECK * DECKS)
    weight = np.float32(level_weight)
    device = torch.device(device)
    pool_size = DEFAULT_POOL if pool is None else int(pool)
    pool_size = max(1, min(pool_size, n_hands))

    # One cache for the whole round rather than one per hand: it is keyed
    # by trump context, of which there are 65, so a shared cache serves
    # every hand and a per-hand one would rebuild the static card block
    # each time a context recurred.
    cache = Cache()

    # Episode ids continue past whatever is already in the buffer, so two
    # collections into one buffer cannot produce two hands that group
    # together.
    filled = len(buffer)
    next_episode = 0 if filled == 0 else int(buffer.episode[:filled].max()) + 1

    active: list[_InFlight] = []
    started = 0
    hands_done = 0
    decisions = 0
    candidate_total = 0
    forced = 0
    truncated = 0
    unenumerated = 0
    forwards = 0
    batch_total = 0

    was_training = getattr(net, "training", False)
    if hasattr(net, "eval"):
        net.eval()
    started_at = time.perf_counter()
    try:
        with torch.no_grad(), torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=amp,
        ):
            while True:
                while (
                    len(active) < pool_size
                    and started < n_hands
                    and _room_for_another(buffer, active, max_decisions)
                ):
                    if starts is not None:
                        state = starts[started].state(DECKS)
                    else:
                        state, _info = generate(rng, DECKS)
                    active.append(_InFlight(state=state, episode=next_episode))
                    next_episode += 1
                    started += 1
                if not active:
                    break

                packed = [_pack(hand, cache, cfg, spans) for hand in active]

                obs = obs_batch_from_arrays(
                    np.stack([d.history for d in packed]),
                    np.array([d.history_len for d in packed], dtype=np.int64),
                    np.stack([d.context for d in packed]),
                    np.stack([d.unseen for d in packed]),
                    np.stack([d.planes for d in packed]),
                    np.stack([d.cards for d in packed]),
                    np.stack([d.candidates for d in packed]),
                    np.stack([d.candidate_mask for d in packed]),
                    device=device,
                )
                privileged = torch.from_numpy(
                    np.stack([d.privileged for d in packed]).astype(np.float32)
                ).to(device)

                with _span(spans, "forward"):
                    out, values = _evaluate(net, obs, privileged)

                with _span(spans, "sample"):
                    slots, chosen_logp = _choose(
                        out.logits, obs.candidate_mask, temperature, greedy, rng
                    )
                    value_np = (
                values.detach().float().cpu().numpy().astype(np.float32)
            )

                with _span(spans, "step"):
                    finished: list[int] = []
                    for i, (hand, decision) in enumerate(zip(active, packed)):
                        slot = int(slots[i])
                        row = buffer.add(
                            history=decision.history,
                            history_len=decision.history_len,
                            context=decision.context,
                            unseen=decision.unseen,
                            planes=decision.planes,
                            cards=decision.cards,
                            candidates=decision.candidates,
                            candidate_mask=decision.candidate_mask,
                            privileged=decision.privileged,
                            action=slot,
                            logp=float(chosen_logp[i]),
                            value=float(value_np[i]),
                            seat=decision.seat,
                            episode=hand.episode,
                        )
                        hand.trick_rows.append(row)
                        hand.last_row[decision.seat] = row

                        decisions += 1
                        moves = decision.moves
                        candidate_total += len(moves.moves)
                        forced += int(len(moves.moves) == 1)
                        truncated += int(moves.truncated)
                        unenumerated += int(moves.unenumerated)

                        # float16 holds card counts 0..2 exactly, so the
                        # slot read back is the move the enumerator built.
                        cards = decision.candidates[slot].astype(np.int8)
                        before = len(hand.state.tricks)
                        hand.state.step(cards)
                        if len(hand.state.tricks) > before:
                            _credit_trick(buffer, hand, point_norm, weight)
                        if hand.state.done:
                            _credit_terminal(buffer, hand, weight)
                            hands_done += 1
                            finished.append(i)

                forwards += 1
                batch_total += len(packed)
                for i in reversed(finished):
                    active.pop(i)
    finally:
        if was_training and hasattr(net, "train"):
            net.train()
    elapsed = time.perf_counter() - started_at

    return {
        "hands": hands_done,
        "decisions": decisions,
        "candidates_per_decision": candidate_total / max(decisions, 1),
        "forced_fraction": forced / max(decisions, 1),
        "truncated": truncated,
        "unenumerated": unenumerated,
        "forwards": forwards,
        "mean_batch": batch_total / max(forwards, 1),
        "seconds": elapsed,
    }
