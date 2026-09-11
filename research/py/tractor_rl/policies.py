"""Loads a checkpoint of either architecture and picks moves with it.

The arena has to compare v1 and v2 checkpoints against the same ladder, so
the difference between them lives here rather than leaking into the
evaluation code. Architecture is read from the checkpoint itself; v1 files
predate the tag and are recognised by their state-dict keys.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from . import layout as L


class Policy(Protocol):
    def pick(self, merged) -> np.ndarray:
        """Greedy candidate index per decision."""


def _group_argmax(q: np.ndarray, owner: np.ndarray, offsets: np.ndarray, counts: np.ndarray):
    b = counts.shape[0]
    if b == 0:
        return np.zeros(0, dtype=np.int64)
    max_k = int(counts.max())
    padded = np.full((b, max_k), -np.inf, dtype=np.float32)
    within = np.arange(q.shape[0], dtype=np.int64) - offsets[owner]
    padded[owner, within] = q
    return padded.argmax(axis=1)


class PolicyV1:
    """Original network: single head, raw-slot features, no history."""

    def __init__(self, state: dict, device: torch.device) -> None:
        from .features import action_features, state_features
        from .model import ModelConfig, QNet

        self._state_features = state_features
        self._action_features = action_features
        self.device = device
        self.net = QNet(ModelConfig()).to(device)
        self.net.load_state_dict(state["model"])
        self.net.eval()

    def pick(self, merged) -> np.ndarray:
        if merged.num_decisions == 0:
            return np.zeros(0, dtype=np.int64)
        sf = torch.from_numpy(self._state_features(merged.states)).to(self.device)
        af = torch.from_numpy(
            self._action_features(merged.cands, merged.states, merged.owner)).to(self.device)
        owner = torch.from_numpy(merged.owner).to(self.device)
        with torch.inference_mode():
            q = self.net(sf, af, owner).cpu().numpy()
        return _group_argmax(q, merged.owner, merged.cand_offsets, merged.cand_counts)


class PolicyV2:
    """v2 network: chain-position features, per-role heads, move history."""

    def __init__(self, state: dict, device: torch.device) -> None:
        from .features2 import action_features, history_features, state_features
        from .model2 import NetConfig, TractorNet

        self._state_features = state_features
        self._action_features = action_features
        self._history_features = history_features
        self.device = device
        cfg = NetConfig(**state.get("net_config", {}))
        self.cfg = cfg
        self.net = TractorNet(cfg).to(device)
        self.net.load_state_dict(state["model"])
        self.net.eval()

    def state_values(self, states: np.ndarray) -> np.ndarray:
        """V(s) for a batch of positions -- what search evaluates leaves with."""
        if states.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        sf = torch.from_numpy(self._state_features(states)).to(self.device)
        if self.cfg.history == "attn":
            tok, msk = self._history_features(states)
            tokens = torch.from_numpy(tok).to(self.device)
            mask = torch.from_numpy(msk).to(self.device)
        else:
            tokens = mask = None
        empty_actions = torch.zeros((0, 0), device=self.device)
        with torch.inference_mode():
            z = self.net.encode_state(sf, tokens, mask)
            v = self.net.v_head(z).squeeze(-1)
        del empty_actions
        return v.cpu().numpy()

    def pick(self, merged) -> np.ndarray:
        if merged.num_decisions == 0:
            return np.zeros(0, dtype=np.int64)
        sf = torch.from_numpy(self._state_features(merged.states)).to(self.device)
        af = torch.from_numpy(
            self._action_features(merged.cands, merged.states, merged.owner)).to(self.device)
        owner = torch.from_numpy(merged.owner).to(self.device)
        roles = torch.from_numpy(merged.states[:, L.H_ROLE].astype(np.int64)).to(self.device)
        if self.cfg.history == "attn":
            tok, msk = self._history_features(merged.states)
            tokens = torch.from_numpy(tok).to(self.device)
            mask = torch.from_numpy(msk).to(self.device)
        else:
            tokens = mask = None
        with torch.inference_mode():
            out = self.net(sf, af, owner, roles, tokens, mask, want="q")
        q = out["q"].cpu().numpy()
        return _group_argmax(q, merged.owner, merged.cand_offsets, merged.cand_counts)


def load_policy(path: str, device: torch.device) -> tuple[Policy, str]:
    """Returns (policy, description). Architecture comes from the file."""
    state = torch.load(path, map_location=device, weights_only=False)
    arch = state.get("arch")
    if arch is None:
        # v1 predates the tag; its trunk is a plain Sequential.
        arch = "v2" if any(k.startswith("blocks.") for k in state["model"]) else "v1"
    policy = PolicyV2(state, device) if arch == "v2" else PolicyV1(state, device)
    desc = (
        f"{path} [{arch}] steps={state.get('steps', '?')} hands={state.get('hands', '?')}"
    )
    return policy, desc


class SearchPolicy:
    """Value-guided determinized search over the leaves the worker sent.

    The worker has already done the hard part: sampled worlds, played each
    candidate in each, and resolved the trick. What arrives here is a set
    of leaf positions tagged with the candidate they inform and the points
    that trick settled. Scoring a candidate is then

        mean over leaves of ( settled points / scale + gamma * V(leaf) )

    which is the same quantity the value head was trained to predict, so
    the immediate reward and the continuation are in the same units.

    Decisions with no leaves -- a forced move, or search disabled for that
    seat -- fall back to the network's own Q ranking.
    """

    def __init__(self, base: PolicyV2, gamma: float = 0.995, point_scale: float = 100.0) -> None:
        self.base = base
        self.gamma = gamma
        self.point_scale = point_scale
        self.searched_decisions = 0
        self.fallback_decisions = 0
        self.leaves_seen = 0

    def split(self, merged) -> tuple[np.ndarray, np.ndarray]:
        """Row indices of (real decisions, leaves)."""
        leaf_of = merged.states[:, L.H_LEAF_OF].astype(np.int64)
        is_leaf = leaf_of != L.NOT_A_LEAF
        return np.nonzero(~is_leaf)[0], np.nonzero(is_leaf)[0]

    def pick(self, merged) -> np.ndarray:
        """Picks for the *real* decisions only, in their row order."""
        real_rows, leaf_rows = self.split(merged)
        if real_rows.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)

        # Q ranking over the real decisions, used as the prior and as the
        # fallback where search produced nothing.
        sub = _subset_view(merged, real_rows)
        picks = self.base.pick(sub)

        if leaf_rows.shape[0] == 0:
            self.fallback_decisions += real_rows.shape[0]
            return picks

        # Value every leaf in one pass.
        leaf_states = merged.states[leaf_rows]
        values = self.base.state_values(leaf_states)
        self.leaves_seen += leaf_rows.shape[0]
        points = _int16(leaf_states, L.H_LEAF_POINTS).astype(np.float32) / self.point_scale
        scores = points + self.gamma * values
        cand_of = leaf_states[:, L.H_LEAF_OF].astype(np.int64)

        # Leaves belong to the decision from the same game slot in the same
        # frame, so (worker, game) identifies the parent.
        worker_of_leaf = _worker_of(merged, leaf_rows)
        worker_of_real = _worker_of(merged, real_rows)
        game_leaf = _u16(leaf_states, L.H_GAME_ID)
        game_real = _u16(merged.states[real_rows], L.H_GAME_ID)
        key_leaf = worker_of_leaf * 100000 + game_leaf
        key_real = worker_of_real * 100000 + game_real
        order = {int(k): i for i, k in enumerate(key_real)}

        totals: dict[tuple[int, int], list[float]] = {}
        for i in range(leaf_rows.shape[0]):
            parent = order.get(int(key_leaf[i]))
            if parent is None:
                continue
            totals.setdefault((parent, int(cand_of[i])), []).append(float(scores[i]))

        by_parent: dict[int, dict[int, float]] = {}
        for (parent, cand), vals in totals.items():
            by_parent.setdefault(parent, {})[cand] = float(np.mean(vals))
        overridden = 0
        for parent, cand_scores in by_parent.items():
            n_cands = int(merged.cand_counts[real_rows[parent]])
            # Only trust search where it evaluated every candidate: a
            # partial set would be compared against nothing, and the
            # network's own ranking already covers the rest.
            if len(cand_scores) < n_cands:
                continue
            best = max(cand_scores.items(), key=lambda kv: kv[1])[0]
            if best < n_cands:
                picks[parent] = best
                overridden += 1
        self.searched_decisions += overridden
        self.fallback_decisions += real_rows.shape[0] - overridden
        return picks


def _int16(states: np.ndarray, offset: int) -> np.ndarray:
    raw = states[:, offset].astype(np.int32) + states[:, offset + 1].astype(np.int32) * 256
    return np.where(raw >= 32768, raw - 65536, raw)


def _u16(states: np.ndarray, offset: int) -> np.ndarray:
    return states[:, offset].astype(np.int64) + states[:, offset + 1].astype(np.int64) * 256


def _worker_of(merged, rows: np.ndarray) -> np.ndarray:
    return np.searchsorted(merged.bounds[1:], rows, side="right")


def _subset_view(merged, rows: np.ndarray):
    """A Merged-shaped view holding only `rows`, with candidates remapped."""
    from .features2 import Merged

    keep = np.zeros(merged.num_decisions, dtype=bool)
    keep[rows] = True
    cand_mask = keep[merged.owner]
    pos = np.full(merged.num_decisions, -1, dtype=np.int64)
    pos[rows] = np.arange(rows.shape[0])
    counts = merged.cand_counts[rows]
    return Merged(
        states=merged.states[rows],
        cands=merged.cands[cand_mask],
        owner=pos[merged.owner[cand_mask]],
        cand_counts=counts,
        cand_offsets=np.cumsum(counts) - counts,
        bounds=np.array([0, rows.shape[0]], dtype=np.int64),
    )
