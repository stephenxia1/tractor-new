"""Network v2: shared trunk, per-role heads, causal history, value + belief.

Shape of the thing:

    policy state ──► residual trunk ─┐
    move history ──► causal attn ────┴─► z ──┬─► per-role Q(points)   [per candidate]
                                             ├─► Q(level)  auxiliary  [per candidate]
                                             ├─► V(state)             [for search]
                                             └─► belief: opponents' hands

Why each piece is here:

* **per-role heads** — landlord, defender and attacker want different
  things from the same board, but the board reads the same to all three,
  so the trunk is shared and only the head is specialised.
* **causal attention over history** — the planes say what has been played
  but not by whom, in what order, or when. Attention can look straight at
  "who discarded hearts on trick 3" instead of squeezing the whole hand
  through one recurrent state.
* **V(s)** — search needs to evaluate positions it has not chosen an
  action in, which Q(s,a) cannot do.
* **belief** — predicting opponents' hands shapes the trunk toward card
  counting, and gives the search a better world-sampler than uniform.

The privileged critic is a separate network (`PrivilegedCritic`) so there
is no path by which full-information features can reach this one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import NUM_ENC_SLOTS
from .features2 import (
    ACTION_DIM,
    HISTORY_STEPS,
    HISTORY_TOKEN_DIM,
    PRIVILEGED_DIM,
    STATE_DIM,
)
from . import layout as L


@dataclass(slots=True)
class NetConfig:
    width: int = 512
    blocks: int = 2
    action_dim: int = 128
    head_hidden: int = 256
    history: str = "attn"  # "attn" | "none"
    hist_model: int = 96
    hist_layers: int = 2
    hist_heads: int = 4


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.relu(self.fc1(h))
        return x + self.fc2(h)


class CausalHistoryEncoder(nn.Module):
    """Causal self-attention over the recent move sequence.

    Causal because a move may only attend to what preceded it, which keeps
    the representation of a prefix stable as the trick fills in.
    """

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(HISTORY_TOKEN_DIM, cfg.hist_model)
        self.pos = nn.Parameter(torch.zeros(HISTORY_STEPS, cfg.hist_model))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(cfg.hist_model),
                "attn": nn.MultiheadAttention(
                    cfg.hist_model, cfg.hist_heads, batch_first=True),
                "norm2": nn.LayerNorm(cfg.hist_model),
                "ff": nn.Sequential(
                    nn.Linear(cfg.hist_model, 2 * cfg.hist_model),
                    nn.ReLU(),
                    nn.Linear(2 * cfg.hist_model, cfg.hist_model),
                ),
            })
            for _ in range(cfg.hist_layers)
        ])
        self.heads = cfg.hist_heads
        self.head_dim = cfg.hist_model // cfg.hist_heads
        self.out_dim = cfg.hist_model

    def _split(self, attn: nn.MultiheadAttention):
        """q, k, v weights and biases out of the packed projection.

        The parameters stay `nn.MultiheadAttention`'s, so a checkpoint
        written before this rewrite still loads: only the arithmetic is
        reorganised, not the state dict.
        """
        d = self.out_dim
        wq, wk, wv = attn.in_proj_weight.split(d, 0)
        bq, bk, bv = attn.in_proj_bias.split(d, 0)
        return (wq, bq), (wk, bk), (wv, bv)

    def _heads(self, t: torch.Tensor, n: int) -> torch.Tensor:
        return t.view(t.shape[0], n, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens [B, W, D], mask [B, W] with 1 for real steps.
        #
        # Two things make this cheaper than the obvious loop over layers,
        # and neither changes the result. First, history is left-aligned
        # (`encode.ts` fills slots 0..L-1 and zeroes the rest), the output
        # is read at L-1, and the causal mask already stops that position
        # attending past itself -- so a key-padding mask cannot affect
        # anything downstream of it, and omitting it lets the non-final
        # layers use plain `is_causal` attention. Second, only one output
        # position is ever read, so the final layer needs keys and values
        # everywhere but a query, output projection and feed-forward at
        # just that position: 15/16 of the most expensive layer's
        # position-wise work is otherwise computed and discarded.
        b, w, _ = tokens.shape
        x = self.proj(tokens) + self.pos
        real = mask > 0.5
        empty = ~real.any(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1).long() - 1
        rows = torch.arange(b, device=x.device)
        last = len(self.layers) - 1

        for i, layer in enumerate(self.layers):
            attn = layer["attn"]
            (wq, bq), (wk, bk), (wv, bv) = self._split(attn)
            h = layer["norm1"](x)
            if i < last:
                q = self._heads(F.linear(h, wq, bq), w)
                k = self._heads(F.linear(h, wk, bk), w)
                v = self._heads(F.linear(h, wv, bv), w)
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                o = o.transpose(1, 2).reshape(b, w, self.out_dim)
                x = x + attn.out_proj(o)
                x = x + layer["ff"](layer["norm2"](x))
            else:
                # One query, at the read position. `is_causal` is wrong
                # here: with a single query PyTorch aligns it to the *last*
                # key, which would let a short row see the whole window, so
                # the reachable keys are named explicitly instead.
                hl = h[rows, lengths]
                q = F.linear(hl, wq, bq).view(b, self.heads, 1, self.head_dim)
                k = self._heads(F.linear(h, wk, bk), w)
                v = self._heads(F.linear(h, wv, bv), w)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=real[:, None, None, :])
                o = o.transpose(1, 2).reshape(b, self.out_dim)
                summary = x[rows, lengths] + attn.out_proj(o)
                summary = summary + layer["ff"](layer["norm2"](summary))
                # A row with no history attends to nothing meaningful; the
                # clamp above kept the gather in range, and this discards
                # whatever it produced.
                return torch.where(empty[:, None], torch.zeros_like(summary), summary)

        raise AssertionError("encoder configured with zero layers")


class TractorNet(nn.Module):
    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        cfg = config or NetConfig()
        self.cfg = cfg
        self.stem = nn.Linear(STATE_DIM, cfg.width)
        self.blocks = nn.ModuleList([ResidualBlock(cfg.width) for _ in range(cfg.blocks)])
        self.history = CausalHistoryEncoder(cfg) if cfg.history == "attn" else None
        fuse_in = cfg.width + (self.history.out_dim if self.history is not None else 0)
        self.fuse = nn.Sequential(nn.LayerNorm(fuse_in), nn.Linear(fuse_in, cfg.width), nn.ReLU())

        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_DIM, cfg.action_dim), nn.ReLU(),
            nn.Linear(cfg.action_dim, cfg.action_dim), nn.ReLU(),
        )

        def head() -> nn.Module:
            return nn.Sequential(
                nn.Linear(cfg.width + cfg.action_dim, cfg.head_hidden), nn.ReLU(),
                nn.Linear(cfg.head_hidden, cfg.head_hidden // 2), nn.ReLU(),
                nn.Linear(cfg.head_hidden // 2, 1),
            )

        # One points head per role; the level head is shared, being an
        # auxiliary signal rather than the thing actions are chosen by.
        self.q_points = nn.ModuleList([head() for _ in range(L.NUM_ROLES)])
        self.q_level = head()
        self.v_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.head_hidden), nn.ReLU(), nn.Linear(cfg.head_hidden, 1))
        self.belief_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.head_hidden), nn.ReLU(),
            nn.Linear(cfg.head_hidden, 3 * NUM_ENC_SLOTS))

    def encode_state(
        self, states: torch.Tensor, tokens: torch.Tensor | None, mask: torch.Tensor | None
    ) -> torch.Tensor:
        x = self.stem(states)
        for block in self.blocks:
            x = block(x)
        if self.history is not None:
            assert tokens is not None and mask is not None
            x = torch.cat([x, self.history(tokens, mask)], dim=1)
        return self.fuse(x)

    def forward(
        self,
        states: torch.Tensor,  # [B, STATE_DIM]
        actions: torch.Tensor,  # [N, ACTION_DIM]
        owner: torch.Tensor,  # [N]
        roles: torch.Tensor,  # [B] in 0..2
        tokens: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        want: str = "q",
    ) -> dict[str, torch.Tensor]:
        z = self.encode_state(states, tokens, mask)
        out: dict[str, torch.Tensor] = {}
        if want in ("state", "all"):
            out["v"] = self.v_head(z).squeeze(-1)
            out["belief"] = self.belief_head(z)
        if want in ("q", "all"):
            pair = torch.cat([z[owner], self.action_encoder(actions)], dim=1)
            cand_roles = roles[owner]
            if pair.is_cuda:
                # On GPU, run every head and gather. Boolean masking has a
                # data-dependent output shape, so `pair[sel]` and `sel.any()`
                # both have to read a count back to the host -- three
                # synchronisations per forward pass, which stalls the queue
                # far more than the wasted flops cost.
                stacked = torch.stack(
                    [head(pair).squeeze(-1) for head in self.q_points], dim=0)
                out["q"] = stacked.gather(
                    0, cand_roles.long().unsqueeze(0)).squeeze(0)
            else:
                # On CPU the masked form wins clearly (measured 3.2x): each
                # head only touches its own rows, and there is no sync to pay.
                q = torch.zeros(pair.shape[0], device=pair.device, dtype=pair.dtype)
                for role in range(L.NUM_ROLES):
                    sel = cand_roles == role
                    if sel.any():
                        q[sel] = self.q_points[role](pair[sel]).squeeze(-1)
                out["q"] = q
            if want == "all":
                out["q_level"] = self.q_level(pair).squeeze(-1)
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PrivilegedCritic(nn.Module):
    """Q(full state, action), trained on returns and used only as a teacher.

    Conditioning on every hand removes most of the variance that makes the
    imperfect-information target so noisy: given the actual deal, the
    return is close to determined. Its predictions then stand in for the
    raw Monte-Carlo return when training the policy network, which is
    unbiased in expectation -- E[Q_priv | policy view] is the very
    quantity the policy net is trying to learn -- but far quieter.
    """

    def __init__(self, width: int = 512, action_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(STATE_DIM + PRIVILEGED_DIM, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(nn.Linear(ACTION_DIM, action_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(width + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1),
        )

    def forward(
        self, states: torch.Tensor, privileged: torch.Tensor,
        actions: torch.Tensor, owner: torch.Tensor,
    ) -> torch.Tensor:
        z = self.stem(torch.cat([states, privileged], dim=1))
        pair = torch.cat([z[owner], self.action_encoder(actions)], dim=1)
        return self.head(pair).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
