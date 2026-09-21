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
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..eval.duplicate import Policy, PolicyFactory, compare, uniform_random_policy
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
        "decisions": float(result.decisions),
        # Positive only when the whole interval is, which is the
        # question a training chart is actually being asked.
        "ahead": float(result.ci_low > 0.0),
    }
