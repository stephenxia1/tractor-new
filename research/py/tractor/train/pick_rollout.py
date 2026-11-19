"""Card-by-card play in the rollout (E7, research/E7.md).

A decision runs the encoders once (``ActorCritic.encode``) and then the
pick head once per step, until every play in the batch has submitted.
The masks come from ``sim/picks.py`` and are computed on the host for
each step, from the cards picked so far; a step with a single legal
action is taken without a forward and contributes zero to the play's
log-probability, exactly as the learner will score it.

What a row stores (``ModelConfig.action_space``): ``candidates`` the
cards picked before each step, ``candidate_mask`` each step's legal
actions, ``action`` the number of steps, ``logp`` the play's summed
log-probability under the behaviour policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..nn.model import (
    MASK_LOGIT,
    MAX_PICKS,
    NUM_PICK_ACTIONS,
    PICK_SUBMIT,
)
from ..rules.cards import NUM_TYPES

__all__ = ["PickBatch", "sample_plays", "actor_encode", "score_step", "greedy_play"]


@dataclass
class PickBatch:
    """The plays one batch of decisions chose, row-aligned with the batch."""

    prefix: np.ndarray  # float16[n, MAX_PICKS, 54]
    mask: np.ndarray  # bool[n, MAX_PICKS, 55]
    steps: np.ndarray  # int64[n]
    logp: np.ndarray  # float32[n]
    value: np.ndarray  # float32[n]
    play: np.ndarray  # int8[n, 54]
    forced_steps: int = 0
    scored_steps: int = 0


def actor_encode(net: Any, obs: Any, privileged: torch.Tensor):
    """``(summary, card states, value)`` from an ActorCritic or a bare
    actor (whose own value head answers)."""
    actor = getattr(net, "actor", None)
    if actor is not None:
        if net.critic is None:
            raise ValueError("card-mode rollout with a shared critic is not supported yet")
        return net.encode(obs, privileged)
    summary, cards, _enc, _keep = net.encode(obs)
    value = net.heads.value(summary).squeeze(-1)
    return summary, cards, value


def _picker(net: Any) -> Any:
    return getattr(net, "actor", net).picker


def score_step(
    picker: Any,
    summary: torch.Tensor,
    card_states: torch.Tensor,
    prefix: np.ndarray,
    mask: np.ndarray,
    step: np.ndarray,
    temperature: float,
    greedy: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """One pick for each row: ``summary``/``card_states`` already selected
    to these rows; ``prefix`` int[n, 54] the cards picked so far, ``mask``
    bool[n, 55] the legal actions, ``step`` int[n] each row's step index.
    -> ``(actions int64[n], log-probabilities float32[n])``; greedy picks
    record a log-probability of zero, as ``rollout._choose`` does."""
    device = summary.device
    prefix_t = torch.from_numpy(prefix.astype(np.float32)).to(device).unsqueeze(1)
    step_t = torch.from_numpy(step.astype(np.int64)).to(device)
    logits = picker(summary, card_states, prefix_t, start=step_t)[:, 0].float()
    if temperature != 1.0:
        logits = logits / temperature
    m = torch.from_numpy(mask).to(device)
    logits = torch.where(m, logits, torch.full_like(logits, MASK_LOGIT))
    logp = F.log_softmax(logits, dim=-1).detach().cpu().numpy()
    if greedy:
        a = logp.argmax(axis=1)
        return a.astype(np.int64), np.zeros(len(a), dtype=np.float32)
    a = (logp + rng.gumbel(size=logp.shape)).argmax(axis=1)
    return a.astype(np.int64), logp[np.arange(len(a)), a].astype(np.float32)


def sample_plays(
    nets: Sequence[Any],
    owner: np.ndarray,
    gather: Callable[[np.ndarray], tuple[Any, torch.Tensor]],
    rules: Sequence[Any],
    temperature: float,
    greedy: bool,
    rng: np.random.Generator,
    encode: Callable[[Any, Any, torch.Tensor], tuple] | None = None,
) -> PickBatch:
    """Each row's play, picked card by card through its owner's network.

    ``rules[i]`` is row ``i``'s ``sim.picks.PickRules``. Owners are
    visited in index order and draw from ``rng`` in turn, as
    ``rollout.routed_step`` does for whole moves. ``encode(net, obs,
    privileged)`` replaces ``actor_encode`` (search passes its CPU
    forward, ``eval/fastnet.py``); a ``None`` value leaves the values 0.
    """
    encode = encode or actor_encode
    n = len(owner)
    out = PickBatch(
        prefix=np.zeros((n, MAX_PICKS, NUM_TYPES), dtype=np.float16),
        mask=np.zeros((n, MAX_PICKS, NUM_PICK_ACTIONS), dtype=bool),
        steps=np.zeros(n, dtype=np.int64),
        logp=np.zeros(n, dtype=np.float32),
        value=np.zeros(n, dtype=np.float32),
        play=np.zeros((n, NUM_TYPES), dtype=np.int8),
    )
    for agent in np.unique(owner):
        idx = np.nonzero(owner == agent)[0]
        net = nets[int(agent)]
        obs, privileged = gather(idx)
        summary, card_states, value = encode(net, obs, privileged)
        if value is not None:
            out.value[idx] = value.detach().float().cpu().numpy()
        picker = _picker(net)
        picked = np.zeros((len(idx), NUM_TYPES), dtype=np.int8)
        last = np.full(len(idx), -1, dtype=np.int64)
        done = np.zeros(len(idx), dtype=bool)
        for k in range(MAX_PICKS):
            live = np.nonzero(~done)[0]
            if len(live) == 0:
                break
            masks = np.stack([rules[idx[j]].mask(picked[j], int(last[j])) for j in live])
            out.prefix[idx[live], k] = picked[live]
            out.mask[idx[live], k] = masks
            legal_n = masks.sum(axis=1)
            if (legal_n == 0).any():
                bad = int(idx[live[np.argmax(legal_n == 0)]])
                raise RuntimeError(f"row {bad}: no legal pick at step {k}")
            action = np.zeros(len(live), dtype=np.int64)
            forced = legal_n == 1
            action[forced] = masks[forced].argmax(axis=1)
            out.forced_steps += int(forced.sum())
            choose = np.nonzero(~forced)[0]
            if len(choose):
                rows = live[choose]
                sel = torch.from_numpy(rows).to(summary.device)
                a, lp = score_step(
                    picker, summary[sel], card_states[sel], picked[rows], masks[choose],
                    np.full(len(rows), k), temperature, greedy, rng,
                )
                out.logp[idx[rows]] += lp
                action[choose] = a
                out.scored_steps += len(rows)
            for pos, j in enumerate(live):
                a = int(action[pos])
                out.steps[idx[j]] = k + 1
                if a == PICK_SUBMIT:
                    done[j] = True
                else:
                    picked[j, a] += 1
                    last[j] = a
        if not done.all():
            raise RuntimeError("a play did not submit within MAX_PICKS steps")
        out.play[idx] = picked
    return out


def greedy_play(net: Any, packed: Any, device: torch.device | str = "cpu") -> np.ndarray:
    """int8[54]: one decision's play, picked greedily, from a card-mode
    ``rollout._pack`` result -- the evaluators' move, under exactly the
    rollout's masks and forwards."""
    from .buffer import obs_batch_from_arrays

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
    privileged = torch.from_numpy(packed.privileged[None].astype(np.float32)).to(device)
    with torch.no_grad():
        picks = sample_plays(
            [net], np.zeros(1, dtype=np.int64), lambda _idx: (obs, privileged),
            [packed.rules], 1.0, True, np.random.default_rng(0),
        )
    return picks.play[0]
