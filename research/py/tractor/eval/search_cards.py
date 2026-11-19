"""Search for a card-by-card network (research/E7.md).

``eval/search.py::search_plays`` does not know how a network chooses a
play: it needs the root's candidate plays with their prior, and a policy
that plays and values batches of states. This supplies both for a
network that picks a play card by card:

* ``beam_plays`` -- the root's top plays by probability, from a beam
  over picks. A play's probability is exact (one pick sequence per play:
  ascending types, then SUBMIT), and a prefix's log-probability bounds
  every play that extends it, so the beam stops as soon as ``k`` finished
  plays beat every live prefix; ``width`` caps the beam.
* ``PickPolicy`` -- greedy plays pick by pick (``pick_rollout.
  sample_plays``, as the evaluators play) and the privileged critic's
  value, for the simulations.

Not yet wired into the evaluators or label generation: E7 is on hold.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from ..nn.features import Cache
from ..nn.model import MASK_LOGIT, PICK_SUBMIT, ModelConfig
from ..rules.cards import NUM_TYPES
from ..sim.state import PlayState
from ..train.buffer import obs_batch_from_arrays
from ..train.pick_rollout import _picker, actor_encode, sample_plays
from ..train.rollout import _InFlight, _pack
from .search import SearchConfig, search_plays

__all__ = ["PickPolicy", "beam_plays", "search_cards_move"]


def _batch(states: Sequence[PlayState], cache: Cache, cfg: ModelConfig, device: Any):
    packed = [_pack(_InFlight(state=s, episode=0), cache, cfg) for s in states]
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
    return packed, obs, privileged


class PickPolicy:
    """A card-by-card ``ActorCritic`` as ``search_plays`` uses it."""

    def __init__(self, net: Any, cache: Cache, cfg: ModelConfig, device: Any) -> None:
        if not cfg.cards_mode:
            raise ValueError("PickPolicy is for card-by-card networks")
        self.net, self.cache, self.cfg = net, cache, cfg
        self.device = torch.device(device)

    def play(self, states: Sequence[PlayState]) -> list[np.ndarray]:
        packed, obs, privileged = _batch(states, self.cache, self.cfg, self.device)
        with torch.no_grad():
            picks = sample_plays(
                [self.net], np.zeros(len(states), dtype=np.int64),
                lambda idx: (obs, privileged), [p.rules for p in packed],
                1.0, True, np.random.default_rng(0),
            )
        return [picks.play[i].copy() for i in range(len(states))]

    def value(self, states: Sequence[PlayState]) -> np.ndarray:
        if not states:
            return np.zeros(0, dtype=np.float64)
        _packed, obs, privileged = _batch(states, self.cache, self.cfg, self.device)
        with torch.no_grad():
            _summary, _cards, value = actor_encode(self.net, obs, privileged)
        return value.float().cpu().numpy().astype(np.float64)


def beam_plays(net: Any, state: PlayState, k: int, cache: Cache, cfg: ModelConfig,
               device: Any, width: int = 16) -> tuple[list[np.ndarray], np.ndarray]:
    """``(plays, log-probabilities)``: the seat to act's ``k`` most probable
    plays (fewer if fewer exist), best first."""
    packed, obs, privileged = _batch([state], cache, cfg, torch.device(device))
    rules = packed[0].rules
    with torch.no_grad():
        summary, card_states, _value = actor_encode(net, obs, privileged)
    picker = _picker(net)
    # Live prefixes: (picked counts, last type, log-probability, step).
    live: list[tuple[np.ndarray, int, float, int]] = [
        (np.zeros(NUM_TYPES, dtype=np.int8), -1, 0.0, 0)
    ]
    done: list[tuple[np.ndarray, float]] = []
    while live:
        masks = np.stack([rules.mask(p, last) for p, last, _lp, _k in live])
        prefix = torch.from_numpy(np.stack([p for p, *_ in live]).astype(np.float32))
        steps = torch.from_numpy(np.array([k_ for *_, k_ in live], dtype=np.int64))
        n = len(live)
        with torch.no_grad():
            logits = picker(summary.expand(n, -1), card_states.expand(n, -1, -1),
                            prefix.unsqueeze(1).to(summary.device),
                            start=steps.to(summary.device))[:, 0].float()
            m = torch.from_numpy(masks).to(logits.device)
            logits = torch.where(m, logits, torch.full_like(logits, MASK_LOGIT))
            logp = torch.log_softmax(logits, dim=-1).cpu().numpy()
        grown: list[tuple[np.ndarray, int, float, int]] = []
        for j, (p, _last, lp, step) in enumerate(live):
            for a in np.nonzero(masks[j])[0]:
                total = lp + float(logp[j, a])
                if a == PICK_SUBMIT:
                    done.append((p.copy(), total))
                else:
                    q = p.copy()
                    q[a] += 1
                    grown.append((q, int(a), total, step + 1))
        grown.sort(key=lambda t: -t[2])
        live = grown[:width]
        done.sort(key=lambda t: -t[1])
        # A prefix's log-probability bounds every play extending it.
        if len(done) >= k and (not live or done[k - 1][1] >= live[0][2]):
            break
    done = done[:k]
    return [p for p, _ in done], np.array([lp for _, lp in done], dtype=np.float64)


def search_cards_move(net: Any, state: PlayState, search: SearchConfig,
                      rng: np.random.Generator, cache: Cache, cfg: ModelConfig,
                      device: Any, width: int = 16) -> tuple[np.ndarray, bool]:
    """``(play, searched)``: search over the beam's top ``search.top_k``
    plays, with the card-by-card network playing out the simulations."""
    plays, log_prior = beam_plays(net, state, max(search.top_k, 1), cache, cfg, device, width)
    index, searched = search_plays(PickPolicy(net, cache, cfg, device), state, plays,
                                   log_prior, search, rng)
    return plays[index], searched
