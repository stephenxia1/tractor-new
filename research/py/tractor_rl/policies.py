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
        with torch.no_grad():
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
        with torch.no_grad():
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
