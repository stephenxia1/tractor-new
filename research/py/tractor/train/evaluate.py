"""Playing strength, measured during training rather than after it.

The duplicate-deal harness of ``tractor.eval.duplicate`` scores one
policy against another over paired deals. This module supplies the
adapter that lets the network be one of those policies, and a helper
the training loop calls on a schedule.

The adapter deliberately routes through the same ``_pack``, ``_evaluate``
and ``_choose`` the rollout uses, rather than re-deriving the encoding.
An evaluation that built its observations even slightly differently
from training would measure a policy the training loop never produced,
and the discrepancy would be invisible: both paths would run, both
would give plausible numbers, and only the comparison between them
would be wrong.

Evaluation plays one decision at a time, because the duplicate harness
owns the game loop and hands out positions singly. That is far below
the device's efficient batch, which is why it runs on a schedule and
over a few hundred deals rather than continuously.

`evaluate_against_bots` answers the same question against the
TypeScript engine's heuristic bots instead, through
``tractor.eval.bots``. The reference opponent here has no judgement in
it; those do, and they are what the shipped game plays with.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import torch

from ..eval.bots import LADDER, play_against_bot, resolve_bots
from ..eval.duplicate import (
    DECK_POINTS,
    Policy,
    PolicyFactory,
    compare,
    uniform_random_policy,
)
from ..nn.features import Cache
from ..nn.model import ModelConfig
from ..sim.state import PlayState
from .buffer import obs_batch_from_arrays
from .rollout import _choose, _evaluate, _InFlight, _pack


def net_policy(
    net: Any,
    device: torch.device | str = "cpu",
    cfg: ModelConfig | None = None,
    *,
    greedy: bool = True,
    temperature: float = 1.0,
) -> PolicyFactory:
    """A policy factory that plays with ``net``.

    Greedy by default: a measurement wants the policy's considered move,
    not a sample from it, and sampling would add variance the duplicate
    pairing cannot cancel because it is not a property of the deal.
    """
    cfg = cfg or ModelConfig()
    device = torch.device(device)

    def make(rng: np.random.Generator) -> Policy:
        # One cache per seat pair, as in the rollout. It holds the static
        # per-trump-context block, which is shared across hands, so it is
        # correct to carry it across the deals of one comparison.
        cache = Cache()

        def policy(state: PlayState, seat: int) -> np.ndarray:
            if int(state.to_act) != seat:
                raise ValueError(
                    f"asked seat {seat} to act, but {int(state.to_act)} is to act"
                )
            packed = _pack(_InFlight(state=state, episode=0), cache, cfg)
            obs = obs_batch_from_arrays(
                packed.history[None],
                np.array([packed.history_len], dtype=np.int64),
                packed.context[None],
                packed.unseen[None],
                packed.planes[None],
                packed.cards[None],
                packed.candidates[None],
                packed.candidate_mask[None],
                device=device,
            )
            privileged = torch.from_numpy(
                packed.privileged[None].astype(np.float32)
            ).to(device)
            with torch.no_grad():
                out, _value = _evaluate(net, obs, privileged)
            slots, _logp = _choose(
                out.logits, obs.candidate_mask, temperature, greedy, rng
            )
            return packed.moves.moves[int(slots[0])]

        return policy

    return make


def evaluate(
    net: Any,
    device: torch.device | str = "cpu",
    cfg: ModelConfig | None = None,
    *,
    num_deals: int = 64,
    seed: int = 0,
    opponent: PolicyFactory | None = None,
    greedy: bool = True,
) -> dict[str, float]:
    """Score ``net`` against ``opponent`` over paired deals.

    The default opponent is uniform random play, which is the reference
    with no judgement in it. It is a low bar and stays informative only
    early; the same function takes any policy factory, so a frozen
    checkpoint can be passed once there is one worth comparing against.

    Returns a flat dict so the caller can log it directly. ``levels`` is
    the signed level change per deal and ``points`` the attacking-point
    margin, both positive when the network is ahead.
    """
    was_training = getattr(net, "training", False)
    if hasattr(net, "eval"):
        net.eval()
    try:
        result = compare(
            net_policy(net, device, cfg, greedy=greedy),
            opponent if opponent is not None else uniform_random_policy,
            num_deals=num_deals,
            seed=seed,
        )
    finally:
        if was_training and hasattr(net, "train"):
            net.train()

    return {
        "deals": float(result.num_deals),
        "levels": result.mean,
        "levels_stderr": result.stderr,
        "levels_ci_low": result.ci_low,
        "levels_ci_high": result.ci_high,
        "points": result.point_mean,
        "points_stderr": result.point_stderr,
        # The hand's own totals, which a margin hides: a margin of zero
        # is 100 against 100 as readily as 0 against 0. Reported by the
        # side the network sat on, both as the attacking total, so both
        # read against the same 80-point threshold.
        "points_as_attacker": _mean(result.taken[result.attacked]),
        "points_conceded": _mean(result.taken[~result.attacked]),
        # A hand is won by taking the landlord title or by keeping it,
        # which is the 80-point threshold from either side. Over hands
        # played, not deals paired.
        "winrate": float(result.wins.mean()),
        "hands": float(result.wins.size),
        "decisions": float(result.decisions),
        # Positive only when the whole interval is, which is the
        # question a training chart is actually being asked.
        "ahead": float(result.ci_low > 0.0),
    }



def _mean(values: np.ndarray) -> float:
    """The mean, or NaN when there is nothing to average.

    A short measurement can put the network on one side of the table
    for every hand, and a zero would read as a hand it lost badly
    rather than as one it never played.
    """
    return float(values.mean()) if values.size else float("nan")

def evaluate_against_bots(
    net: Any,
    bots: Sequence[str],
    device: torch.device | str = "cpu",
    cfg: ModelConfig | None = None,
    *,
    num_deals: int = 64,
    seed: int = 0,
    max_candidates: int = 128,
    node: str | Sequence[str] | None = None,
    workers: int = 1,
    budget_s: float | None = None,
) -> dict[str, float]:
    """Score ``net`` against each of the engine's heuristic ``bots``.

    The reference opponent `evaluate` uses has no judgement in it, which
    is what makes it a reference and also what makes it uninformative
    past the first hour of training. These are the bots the shipped game
    plays with, so a margin against them is the quantity the project is
    actually asking about.

    Keys are namespaced by bot -- ``vs_smart-v34/levels`` and so on --
    so several bots merge into one metrics row without colliding, and a
    chart keeps them apart.

    Each bot costs a subprocess and ``2 * num_deals`` hands played one
    decision at a time, so this belongs on the same schedule as
    `evaluate` rather than beside it every step.
    """
    rungs = resolve_bots(bots)
    out: dict[str, float] = {}
    started = time.perf_counter()
    run = 0
    for bot in rungs:
        # Checked before each rung rather than during one: a rung is the
        # unit that produces a number, and half of one is not a cheaper
        # measurement, it is no measurement.
        if budget_s is not None and time.perf_counter() - started >= budget_s:
            break
        scores = play_against_bot(
            net,
            bot,
            device=device,
            cfg=cfg,
            num_deals=num_deals,
            seed=seed,
            max_candidates=max_candidates,
            node=node,
            workers=workers,
        )
        out.update({f"vs_{bot}/{key}": value for key, value in scores.items()})
        run += 1
    # Which rungs were reached is part of the measurement: a ladder that
    # silently stopped at the cheap end reads as a strong result.
    out["ladder/rungs_run"] = float(run)
    out["ladder/rungs_skipped"] = float(len(rungs) - run)
    out["ladder/seconds"] = time.perf_counter() - started
    return out
