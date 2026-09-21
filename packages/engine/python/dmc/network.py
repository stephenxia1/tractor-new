"""PyTorch Q-networks for the DMC Tractor agent.

Three networks (landlord / defender / attacker), each with identical
architecture but independent parameters:

    static (7x108=756) ── FC ── h_static (256)
    history (100, 113)  ── LSTM ── h_hist (128)        ── concat ── head ── Q(s,a)
    action  (115)       ── FC  ── h_act    (64)

The (s, a) factorization lets us encode the state once and re-run only
the action encoder + head per candidate at decision time.

Per-decision batched eval API: `forward_batch(obs, actions)` where
`obs` is one observation and `actions` is (K, ACTION_DIM) — returns
(K,) Q-values.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .constants import (
    ACTION_DIM,
    HISTORY_MAX_STEPS,
    PER_STEP_DIM,
    ROLES,
    SCALAR_DIM,
    STATIC_DIM,
)

HIDDEN_STATIC = 256
HIDDEN_HIST = 128
HIDDEN_ACT = 64
HIDDEN_HEAD_1 = 256
HIDDEN_HEAD_2 = 128


class QNet(nn.Module):
    """Single Q-network for one role."""

    def __init__(self) -> None:
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(STATIC_DIM + SCALAR_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, HIDDEN_STATIC),
            nn.ReLU(),
        )
        self.history_lstm = nn.LSTM(
            input_size=PER_STEP_DIM,
            hidden_size=HIDDEN_HIST,
            num_layers=1,
            batch_first=True,
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_DIM, HIDDEN_ACT),
            nn.ReLU(),
        )
        head_in = HIDDEN_STATIC + HIDDEN_HIST + HIDDEN_ACT
        self.head = nn.Sequential(
            nn.Linear(head_in, HIDDEN_HEAD_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_HEAD_1, HIDDEN_HEAD_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_HEAD_2, 1),
        )

    def encode_state(
        self,
        static: torch.Tensor,    # (B, STATIC_DIM)
        scalars: torch.Tensor,   # (B, SCALAR_DIM)
        history: torch.Tensor,   # (B, HISTORY_MAX_STEPS, PER_STEP_DIM)
        history_len: torch.Tensor,  # (B,) int64
    ) -> torch.Tensor:
        """Encode the state portion of (s, a). Returns (B, HIDDEN_STATIC + HIDDEN_HIST)."""
        b = static.shape[0]
        # Static + scalars FC.
        h_static = self.static_encoder(torch.cat([static, scalars], dim=-1))
        # History LSTM. We pack to mask out padded steps. Sequences with
        # length 0 (e.g., the first decision of a hand) get a zero h_hist.
        h_hist = torch.zeros(b, HIDDEN_HIST, device=static.device, dtype=static.dtype)
        has_history = history_len > 0
        if has_history.any():
            idx = has_history.nonzero(as_tuple=True)[0]
            seqs = history[idx]                   # (b', T, PER_STEP_DIM)
            lens = history_len[idx].cpu()         # required by pack_padded_sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                seqs, lens, batch_first=True, enforce_sorted=False,
            )
            _, (h_n, _) = self.history_lstm(packed)
            # h_n: (num_layers=1, b', HIDDEN_HIST)
            h_hist[idx] = h_n[0]
        return torch.cat([h_static, h_hist], dim=-1)

    def forward_batch(
        self,
        state_enc: torch.Tensor,  # (HIDDEN_STATIC + HIDDEN_HIST,) — single state
        actions: torch.Tensor,    # (K, ACTION_DIM)
    ) -> torch.Tensor:
        """Score K candidate actions against one (already-encoded) state.

        Returns (K,) Q-values.
        """
        k = actions.shape[0]
        h_act = self.action_encoder(actions)
        # Tile the encoded state across K candidates.
        h_state = state_enc.unsqueeze(0).expand(k, -1)
        h = torch.cat([h_state, h_act], dim=-1)
        q = self.head(h).squeeze(-1)
        return q

    def forward(
        self,
        static: torch.Tensor,
        scalars: torch.Tensor,
        history: torch.Tensor,
        history_len: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Standard batched forward for training. All tensors batched along dim 0.

        Returns (B,) Q-values.
        """
        h_state = self.encode_state(static, scalars, history, history_len)
        h_act = self.action_encoder(action)
        h = torch.cat([h_state, h_act], dim=-1)
        return self.head(h).squeeze(-1)


class TripleQNet(nn.Module):
    """Holds the three role-specific Q-networks. Dispatches by role string."""

    def __init__(self) -> None:
        super().__init__()
        self.nets = nn.ModuleDict({r: QNet() for r in ROLES})

    def net_for(self, role: str) -> QNet:
        if role not in self.nets:
            raise ValueError(f'unknown role: {role}')
        return self.nets[role]


def num_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
