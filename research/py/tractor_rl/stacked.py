"""N agents as one module: every Linear becomes a bmm over the agent axis.

Population training runs N independent networks at once, and stepping them
in a Python loop wastes most of a GPU -- each agent's batch is a fraction
of the device's width, so the kernels are launch-bound rather than
compute-bound. Holding the weights as `[N, in, out]` and using `baddbmm`
turns N small GEMMs into one batched GEMM. Measured on a T4 at batch 512
per agent, eight agents stacked cost 5.69x one agent against 8x for eight
separate steps, which is the difference between N=8 fitting on one box and
not.

Two things make this more than a mechanical rewrite.

**Attention had to come too.** The earlier benchmark covered the MLP path
and said so. Measured since: the causal history encoder is **62% of a
learner step** while holding 6.6% of the parameters, because it runs over
16 tokens per row -- 8192 tokens where the trunk sees 512. A per-agent loop
around it would have eaten the entire win. It is not slow because
`nn.MultiheadAttention` is slow (the isolated call is 110-170 ms and a
hand-rolled SDPA version measured *slower*); it is slow because it is
genuinely that much arithmetic. So the projections here are per-agent bmms
and the attention itself folds the agent axis into the batch, since
attention has no parameters of its own.

**The per-role heads are computed in full and gathered.** `TractorNet` on
CPU masks instead, running each role head only on its own rows, which was
measured 3.2x faster there. That form needs a data-dependent row count per
(agent, role) and bmm needs equal batches, so this follows the CUDA path
and pays ~3x on the head in exchange for one batched call. The heads are
about a third of the forward, so this is a real cost and the reason the
stacked net is a GPU thing rather than a universal improvement.

Parity with N separate `TractorNet`s is the whole safety property here --
a silent mismatch would look exactly like a training result. See
`py/test_stacked.py`, which checks outputs *and* gradients.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import NUM_ENC_SLOTS
from .features2 import ACTION_DIM, HISTORY_STEPS, HISTORY_TOKEN_DIM, STATE_DIM
from .model2 import NetConfig, TractorNet
from . import layout as L


class StackedLinear(nn.Module):
    """N independent `nn.Linear` layers as one `[N, in, out]` weight.

    Input `[N, B, in]` -> output `[N, B, out]`. Note the transpose against
    torch's convention: `nn.Linear` stores `[out, in]` and applies
    `x @ W.T`, while bmm wants `[in, out]` so it can apply `x @ W`.
    """

    def __init__(self, n: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match nn.Linear's init per slice, so an unloaded stacked net has
        # the same distribution as the nets it stands in for.
        for i in range(self.n):
            w = torch.empty(self.out_features, self.in_features)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            self.weight.data[i] = w.t()
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias.data[i], -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.baddbmm(self.bias, x, self.weight)

    def load_slice(self, i: int, linear: nn.Linear) -> None:
        self.weight.data[i] = linear.weight.data.t()
        self.bias.data[i, 0] = linear.bias.data

    def export_slice(self, i: int) -> dict[str, torch.Tensor]:
        return {"weight": self.weight.data[i].t().clone(),
                "bias": self.bias.data[i, 0].clone()}


class StackedLayerNorm(nn.Module):
    """N LayerNorms sharing a shape. Normalisation is per-row either way;
    only the affine parameters differ between agents."""

    def __init__(self, n: int, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.n, self.dim, self.eps = n, dim, eps
        self.weight = nn.Parameter(torch.ones(n, 1, dim))
        self.bias = nn.Parameter(torch.zeros(n, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [N, B, D] or [N, B, S, D]; broadcast the affine accordingly.
        normed = F.layer_norm(x, (self.dim,), eps=self.eps)
        if x.dim() == 4:
            return normed * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)
        return normed * self.weight + self.bias

    def load_slice(self, i: int, norm: nn.LayerNorm) -> None:
        self.weight.data[i, 0] = norm.weight.data
        self.bias.data[i, 0] = norm.bias.data

    def export_slice(self, i: int) -> dict[str, torch.Tensor]:
        return {"weight": self.weight.data[i, 0].clone(),
                "bias": self.bias.data[i, 0].clone()}


class StackedResidualBlock(nn.Module):
    def __init__(self, n: int, width: int) -> None:
        super().__init__()
        self.norm = StackedLayerNorm(n, width)
        self.fc1 = StackedLinear(n, width, width)
        self.fc2 = StackedLinear(n, width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.relu(self.fc1(h))
        return x + self.fc2(h)


class StackedHistoryEncoder(nn.Module):
    """Causal self-attention over the move sequence, N agents at once.

    The projections carry per-agent weights so they are bmms; the attention
    itself has no parameters, so the agent axis folds into the batch for a
    single `scaled_dot_product_attention` call over `[N*B, heads, S, dh]`.
    """

    def __init__(self, n: int, cfg: NetConfig) -> None:
        super().__init__()
        self.n = n
        self.cfg = cfg
        self.heads = cfg.hist_heads
        self.model = cfg.hist_model
        self.head_dim = cfg.hist_model // cfg.hist_heads
        self.proj = StackedLinear(n, HISTORY_TOKEN_DIM, cfg.hist_model)
        self.pos = nn.Parameter(torch.zeros(n, HISTORY_STEPS, cfg.hist_model))
        self.norm1 = nn.ModuleList(
            [StackedLayerNorm(n, cfg.hist_model) for _ in range(cfg.hist_layers)])
        self.norm2 = nn.ModuleList(
            [StackedLayerNorm(n, cfg.hist_model) for _ in range(cfg.hist_layers)])
        # nn.MultiheadAttention packs q, k and v into one [3E, E] weight;
        # keeping that layout makes loading a slice a straight copy.
        self.in_proj = nn.ModuleList(
            [StackedLinear(n, cfg.hist_model, 3 * cfg.hist_model)
             for _ in range(cfg.hist_layers)])
        self.out_proj = nn.ModuleList(
            [StackedLinear(n, cfg.hist_model, cfg.hist_model)
             for _ in range(cfg.hist_layers)])
        self.ff1 = nn.ModuleList(
            [StackedLinear(n, cfg.hist_model, 2 * cfg.hist_model)
             for _ in range(cfg.hist_layers)])
        self.ff2 = nn.ModuleList(
            [StackedLinear(n, 2 * cfg.hist_model, cfg.hist_model)
             for _ in range(cfg.hist_layers)])
        causal = torch.triu(
            torch.ones(HISTORY_STEPS, HISTORY_STEPS, dtype=torch.bool), 1)
        self.register_buffer("causal_mask", causal, persistent=False)
        self.out_dim = cfg.hist_model

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens [N, B, S, D], mask [N, B, S] with 1 for real steps.
        n, b, s, _ = tokens.shape
        x = self.proj(tokens.reshape(n, b * s, -1)).reshape(n, b, s, self.model)
        x = x + self.pos.unsqueeze(1)

        pad = mask < 0.5
        # A row with no history makes every key padded, which leaves
        # attention undefined; let it attend to itself and zero the result
        # after, exactly as the unstacked encoder does.
        empty = pad.all(dim=2)
        pad = torch.where(empty.unsqueeze(-1), torch.zeros_like(pad), pad)
        # SDPA's boolean mask is "True may attend", the opposite of
        # MultiheadAttention's convention. [N*B, 1, S, S].
        keep = (~self.causal_mask)[None, None] & (~pad).reshape(n * b, 1, 1, s)

        for i in range(self.cfg.hist_layers):
            h = self.norm1[i](x)
            qkv = self.in_proj[i](h.reshape(n, b * s, self.model))
            qkv = qkv.reshape(n * b, s, 3, self.heads, self.head_dim)
            q, k, v = (qkv[:, :, j].transpose(1, 2) for j in range(3))
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
            attn = attn.transpose(1, 2).reshape(n, b * s, self.model)
            x = x + self.out_proj[i](attn).reshape(n, b, s, self.model)
            h = self.norm2[i](x).reshape(n, b * s, self.model)
            ff = self.ff2[i](F.relu(self.ff1[i](h))).reshape(n, b, s, self.model)
            x = x + ff

        lengths = mask.sum(dim=2).clamp(min=1).long() - 1
        summary = torch.gather(
            x, 2, lengths[..., None, None].expand(n, b, 1, self.model)).squeeze(2)
        return torch.where(empty.unsqueeze(-1), torch.zeros_like(summary), summary)

    def load_slice(self, i: int, enc: nn.Module) -> None:
        self.proj.load_slice(i, enc.proj)
        self.pos.data[i] = enc.pos.data
        for j, layer in enumerate(enc.layers):
            self.norm1[j].load_slice(i, layer["norm1"])
            self.norm2[j].load_slice(i, layer["norm2"])
            mha = layer["attn"]
            self.in_proj[j].weight.data[i] = mha.in_proj_weight.data.t()
            self.in_proj[j].bias.data[i, 0] = mha.in_proj_bias.data
            self.out_proj[j].load_slice(i, mha.out_proj)
            self.ff1[j].load_slice(i, layer["ff"][0])
            self.ff2[j].load_slice(i, layer["ff"][2])


class StackedMLP(nn.Module):
    """Linear/ReLU chain with per-agent weights.

    `final_relu` is not a stylistic flag. `TractorNet.action_encoder` ends
    in a ReLU where every head ends in a bare Linear, and assuming the head
    shape for both is exactly the bug this module shipped with first: `v`
    and `belief` matched to 1e-16 while `q` was off by 1.5e-2, because only
    the candidate path runs through the action encoder.
    """

    def __init__(self, n: int, dims: list[int], final_relu: bool = False) -> None:
        super().__init__()
        self.final_relu = final_relu
        self.layers = nn.ModuleList(
            [StackedLinear(n, dims[k], dims[k + 1]) for k in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = len(self.layers) - 1
        for k, layer in enumerate(self.layers):
            x = layer(x)
            if k < last or self.final_relu:
                x = F.relu(x)
        return x

    def load_slice(self, i: int, seq: nn.Sequential) -> None:
        linears = [m for m in seq if isinstance(m, nn.Linear)]
        for layer, lin in zip(self.layers, linears):
            layer.load_slice(i, lin)


class StackedTractorNet(nn.Module):
    """N `TractorNet`s that step together.

    Every input carries a leading agent axis: states `[N, B, STATE_DIM]`,
    actions `[N, B, ACTION_DIM]`, roles `[N, B]`, tokens `[N, B, S, D]`.
    The learner path is what this exists for -- one candidate per decision,
    so there is no `owner` gather; the acting path still uses `TractorNet`,
    where each agent has a different, data-dependent number of decisions.
    """

    def __init__(self, n: int, config: NetConfig | None = None) -> None:
        super().__init__()
        cfg = config or NetConfig()
        self.n = n
        self.cfg = cfg
        self.stem = StackedLinear(n, STATE_DIM, cfg.width)
        self.blocks = nn.ModuleList(
            [StackedResidualBlock(n, cfg.width) for _ in range(cfg.blocks)])
        self.history = (StackedHistoryEncoder(n, cfg)
                        if cfg.history == "attn" else None)
        fuse_in = cfg.width + (self.history.out_dim if self.history is not None else 0)
        self.fuse_norm = StackedLayerNorm(n, fuse_in)
        self.fuse_lin = StackedLinear(n, fuse_in, cfg.width)
        self.action_encoder = StackedMLP(
            n, [ACTION_DIM, cfg.action_dim, cfg.action_dim], final_relu=True)
        head_dims = [cfg.width + cfg.action_dim, cfg.head_hidden,
                     cfg.head_hidden // 2, 1]
        self.q_points = nn.ModuleList(
            [StackedMLP(n, head_dims) for _ in range(L.NUM_ROLES)])
        self.q_level = StackedMLP(n, head_dims)
        self.v_head = StackedMLP(n, [cfg.width, cfg.head_hidden, 1])
        self.belief_head = StackedMLP(
            n, [cfg.width, cfg.head_hidden, 3 * NUM_ENC_SLOTS])

    def encode_state(self, states, tokens, mask):
        x = self.stem(states)
        for block in self.blocks:
            x = block(x)
        if self.history is not None:
            assert tokens is not None and mask is not None
            x = torch.cat([x, self.history(tokens, mask)], dim=2)
        return F.relu(self.fuse_lin(self.fuse_norm(x)))

    def forward(self, states, actions, roles, tokens=None, mask=None, want="all"):
        z = self.encode_state(states, tokens, mask)
        out: dict[str, torch.Tensor] = {}
        if want in ("state", "all"):
            out["v"] = self.v_head(z).squeeze(-1)
            out["belief"] = self.belief_head(z)
        if want in ("q", "all"):
            pair = torch.cat([z, self.action_encoder(actions)], dim=2)
            # Every role head on every row, then gather. The masked form the
            # CPU path uses needs a data-dependent row count per (agent,
            # role), which bmm cannot express.
            stacked = torch.stack(
                [h(pair).squeeze(-1) for h in self.q_points], dim=0)
            out["q"] = stacked.gather(0, roles.long().unsqueeze(0)).squeeze(0)
            if want == "all":
                out["q_level"] = self.q_level(pair).squeeze(-1)
        return out

    # ---- interop with the unstacked net ----

    def load_from(self, nets: list[TractorNet]) -> None:
        """Copy N `TractorNet`s in, slice by slice."""
        if len(nets) != self.n:
            raise ValueError(f"expected {self.n} nets, got {len(nets)}")
        for i, net in enumerate(nets):
            self.stem.load_slice(i, net.stem)
            for blk, src in zip(self.blocks, net.blocks):
                blk.norm.load_slice(i, src.norm)
                blk.fc1.load_slice(i, src.fc1)
                blk.fc2.load_slice(i, src.fc2)
            if self.history is not None:
                self.history.load_slice(i, net.history)
            self.fuse_norm.load_slice(i, net.fuse[0])
            self.fuse_lin.load_slice(i, net.fuse[1])
            self.action_encoder.load_slice(i, net.action_encoder)
            for head, src in zip(self.q_points, net.q_points):
                head.load_slice(i, src)
            self.q_level.load_slice(i, net.q_level)
            self.v_head.load_slice(i, net.v_head)
            self.belief_head.load_slice(i, net.belief_head)

    @torch.no_grad()
    def write_into(self, i: int, net: TractorNet) -> None:
        """Copy agent `i`'s weights back out into a plain `TractorNet`.

        This is how a population member reaches `arena.py`, the exporter and
        every other tool that expects one network.
        """
        def copy_lin(dst: nn.Linear, src: StackedLinear) -> None:
            dst.weight.data.copy_(src.weight.data[i].t())
            dst.bias.data.copy_(src.bias.data[i, 0])

        def copy_norm(dst: nn.LayerNorm, src: StackedLayerNorm) -> None:
            dst.weight.data.copy_(src.weight.data[i, 0])
            dst.bias.data.copy_(src.bias.data[i, 0])

        def copy_mlp(dst: nn.Sequential, src: StackedMLP) -> None:
            for lin, layer in zip([m for m in dst if isinstance(m, nn.Linear)],
                                  src.layers):
                copy_lin(lin, layer)

        copy_lin(net.stem, self.stem)
        for blk, dst in zip(self.blocks, net.blocks):
            copy_norm(dst.norm, blk.norm)
            copy_lin(dst.fc1, blk.fc1)
            copy_lin(dst.fc2, blk.fc2)
        if self.history is not None:
            h = self.history
            copy_lin(net.history.proj, h.proj)
            net.history.pos.data.copy_(h.pos.data[i])
            for j, layer in enumerate(net.history.layers):
                copy_norm(layer["norm1"], h.norm1[j])
                copy_norm(layer["norm2"], h.norm2[j])
                layer["attn"].in_proj_weight.data.copy_(h.in_proj[j].weight.data[i].t())
                layer["attn"].in_proj_bias.data.copy_(h.in_proj[j].bias.data[i, 0])
                copy_lin(layer["attn"].out_proj, h.out_proj[j])
                copy_lin(layer["ff"][0], h.ff1[j])
                copy_lin(layer["ff"][2], h.ff2[j])
        copy_norm(net.fuse[0], self.fuse_norm)
        copy_lin(net.fuse[1], self.fuse_lin)
        copy_mlp(net.action_encoder, self.action_encoder)
        for dst, src in zip(net.q_points, self.q_points):
            copy_mlp(dst, src)
        copy_mlp(net.q_level, self.q_level)
        copy_mlp(net.v_head, self.v_head)
        copy_mlp(net.belief_head, self.belief_head)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
