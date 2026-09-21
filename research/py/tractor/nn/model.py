"""The policy-value network: one forward pass per decision.

The shape of this file follows the shape of the game. A hand is a
sequence of plays that every seat watches, so the history is encoded
once, causally, and is the only part of the network whose cost grows
with how far the hand has run. A *decision* is a small, fixed bundle of
things -- the scalar context, what is unseen, what each seat has put
down, what I hold -- so the query block is a short token sequence that
reads the history by cross-attention and collapses to a single summary
vector ``h``. Everything else hangs off ``h``.

Three properties are load-bearing and are enforced by
``tests/test_model.py`` rather than left to good intentions.

*Static shapes.* History is always ``cfg.max_history`` tokens and
candidates are always ``cfg.max_candidates`` slots, padded and masked. A
265,032-decision audit put the observed maximum history at 99 tokens and
the candidate count at mean 12.1, p99 105, p99.9 375; 128 slots covers
the p99 and matches the enumerator's own cap. Nothing here branches on a
tensor value, calls ``.item()``, or changes shape with the batch, so
``torch.compile`` sees a single graph.

*Masking is arithmetic, not etiquette.* Padded history positions are
zeroed after every block and excluded as attention keys; because the
history attention is causal and the padding is a suffix, a real position
can never read one. Invalid candidate slots have their logits
*overwritten* with ``MASK_LOGIT``, so their contents cannot reach any
output even in the low bits. 14.8% of decisions have exactly one legal
move, and that case has to come out as probability exactly 1.0.

*The critic is a separate object.* It reads the actor's encoded history
and the privileged hands, and takes a ``detach`` flag. With ``detach``
set there is no gradient path from a critic loss to an actor parameter,
which is what lets the two be trained with different losses.

Conventions
-----------
Seats are relative throughout, matching ``features.py``: index 0 is me,
1 the seat after me, 2 my partner, 3 the seat before me.

FLOPs are counted as 2 per multiply-accumulate, the usual ML
convention. ``analytic_flops`` derives the count from the configuration
so a benchmark has something to divide measured time by.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..rules.cards import NUM_TYPES, TRUMP
from .features import (
    CONTEXT_DIM,
    D_CARD,
    HISTORY_DIM,
    MAX_HISTORY,
    NUM_PLANES,
    NUM_STRENGTH,
    Observation,
    PLANE_HAND,
    PLANE_KITTY,
    PLANE_PLAYED,
    PLANE_TRICK,
    STATIC_POINTS,
    STATIC_STRENGTH,
    STATIC_SUIT,
)

# --- fixed sizes ----------------------------------------------------------

NUM_REL_SEATS = 4

# Query tokens: context, unseen, the seat's own buried kitty, one per
# relative seat, one per card type. Fixed, so the sequence length never
# moves. The kitty token is zero for every seat but the landlord, and
# carries the only view of those eight cards the observation holds.
NUM_QUERY_TOKENS = 3 + NUM_REL_SEATS + NUM_TYPES  # 61
QUERY_CONTEXT = 0
QUERY_UNSEEN = 1
QUERY_KITTY = 2
QUERY_SEAT = 3
QUERY_HAND = QUERY_SEAT + NUM_REL_SEATS

# ``sim.state.hand_result`` partitions attacker points into seven
# buckets -- 0, <40, <80, <120, <160, <200, 200+ -- and those are the
# outcome classes.
NUM_OUTCOMES = 7

# Where a card the acting seat cannot account for might be: the three
# other seats in relative order, the kitty, or already played.
NUM_BELIEF_LOCATIONS = 5
BELIEF_SEAT_1 = 0
BELIEF_SEAT_2 = 1
BELIEF_SEAT_3 = 2
BELIEF_KITTY = 3
BELIEF_PLAYED = 4

# Privileged rows for the critic: the four hands in relative order, then
# the kitty.
NUM_PRIVILEGED_ROWS = NUM_REL_SEATS + 1

# Scalars derived per candidate; see ``CandidateScorer.forward``.
NUM_CAND_SCALARS = 6
# Scalars pulled out of each card's static block: point value, is-trump,
# strength.
NUM_CARD_SCALARS = 3

# Large and finite. ``-inf`` would make ``log_softmax`` hand back
# ``-inf`` for invalid slots, and a later ``0 * -inf`` inside a masked
# loss is NaN. ``exp(-1e9 - m)`` is exactly 0.0 in float32, so an
# invalid slot has probability exactly zero while every tensor stays
# finite.
MASK_LOGIT = -1e9

# Per-element costs for the non-matmul work, so ``analytic_flops`` can
# be audited rather than believed. They are small next to the matmuls.
FLOPS_PER_SOFTMAX_ELEM = 5  # max, subtract, exp, accumulate, divide
FLOPS_PER_GELU = 8  # erf approximation, conservatively
FLOPS_PER_LAYERNORM_ELEM = 7  # mean, two for variance, rsqrt, sub, mul, add


# --- configuration --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Widths and depths. Frozen so a config can key a benchmark table.

    The defaults put the head dimension at 64 in both towers, the size
    the fused attention kernels are tuned for.
    """

    d_model: int = 384
    n_layers_history: int = 3
    n_layers_query: int = 3
    n_heads: int = 6  # 384 / 6 = 64
    d_ff: int = 1024
    critic_d: int = 128
    critic_layers: int = 2
    max_history: int = MAX_HISTORY
    max_candidates: int = 128

    # Derived widths. In the config so a sweep can move them, but they
    # are consequences of the choices above rather than choices.
    critic_heads: int = 2  # 128 / 2 = 64
    critic_d_ff: int = 256
    belief_d: int = 192
    scalar_d: int = 64
    head_d: int = 192

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must divide by n_heads")
        if self.critic_d % self.critic_heads:
            raise ValueError("critic_d must divide by critic_heads")
        if self.max_history < 1 or self.max_candidates < 1:
            raise ValueError("padded extents must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def critic_head_dim(self) -> int:
        return self.critic_d // self.critic_heads


# --- batched inputs -------------------------------------------------------


class ObsBatch(NamedTuple):
    """A batch of observations, padded to the configured extents.

    A ``NamedTuple`` rather than a dataclass so that it is a pytree node:
    ``torch.compile`` flattens it without a custom registration and
    ``.to(device)`` stays a one-liner.

    ``history``        float32[B, max_history, HISTORY_DIM]
    ``history_len``    int64[B], tokens actually written
    ``context``        float32[B, CONTEXT_DIM]
    ``unseen``         float32[B, 54], raw counts
    ``planes``         float32[B, NUM_PLANES, 54], raw counts
    ``cards``          float32[B, 54, D_CARD]
    ``candidates``     float32[B, max_candidates, 54], raw counts
    ``candidate_mask`` bool[B, max_candidates], True where legal
    """

    history: Tensor
    history_len: Tensor
    context: Tensor
    unseen: Tensor
    planes: Tensor
    cards: Tensor
    candidates: Tensor
    candidate_mask: Tensor

    def to(self, device: torch.device | str) -> "ObsBatch":
        return ObsBatch(*(t.to(device) for t in self))

    @property
    def batch(self) -> int:
        return int(self.history.shape[0])


class ActorOutput(NamedTuple):
    """Everything the actor produces. Every tensor is float32.

    ``logits``           [B, N], invalid slots hold exactly MASK_LOGIT
    ``log_probs``        [B, N], log-softmax over slots, finite
    ``value``            [B], imperfect-information value of the position
    ``outcome_logits``   [B, NUM_OUTCOMES]
    ``belief_logits``    [B, 54, NUM_BELIEF_LOCATIONS]
    ``belief_log_probs`` [B, 54, NUM_BELIEF_LOCATIONS]
    ``summary``          [B, d_model], the state summary h
    ``history_enc``      [B, max_history, d_model], cacheable; padded
                         positions are exactly zero
    """

    logits: Tensor
    log_probs: Tensor
    value: Tensor
    outcome_logits: Tensor
    belief_logits: Tensor
    belief_log_probs: Tensor
    summary: Tensor
    history_enc: Tensor


# --- shared pieces --------------------------------------------------------


# Position indices for the length masks, one per (size, device). Held
# rather than rebuilt because ``torch.arange`` inside a traced forward is
# what stops the inference graph compiling: under ``no_grad`` the
# backend propagates a symbolic extent into the arange's index
# arithmetic and fails converting it to a number. A tensor closed over
# is a constant, and there is no arange left to lower.
_MASK_INDEX: dict[tuple[int, str, int], Tensor] = {}


def _length_mask(lengths: Tensor, size: int) -> Tensor:
    """bool[B, size]: True for the first ``lengths[b]`` positions."""
    device = lengths.device
    key = (size, device.type, -1 if device.index is None else device.index)
    idx = _MASK_INDEX.get(key)
    if idx is None:
        idx = torch.arange(size, device=device)
        _MASK_INDEX[key] = idx
    return idx.unsqueeze(0) < lengths.unsqueeze(1).to(idx.dtype)


def _sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    n_heads: int,
    attn_mask: Tensor | None = None,
    is_causal: bool = False,
) -> Tensor:
    """Multi-head attention through ``scaled_dot_product_attention``.

    ``nn.MultiheadAttention`` is deliberately avoided: it hides the mask
    semantics, will not take ``is_causal`` and a key mask apart, and its
    fast path drops out under ``torch.compile``. Splitting heads by hand
    costs two transposes and keeps the kernel choice visible.

    ``attn_mask`` is boolean, broadcastable to [B, heads, Lq, Lk], True
    meaning *attend*. PyTorch forbids passing it together with
    ``is_causal``; see ``HistoryEncoder`` for why causality alone is
    enough there.
    """
    b, lq, d = q.shape
    lk = k.shape[1]
    dh = d // n_heads
    q = q.view(b, lq, n_heads, dh).transpose(1, 2)
    k = k.view(b, lk, n_heads, dh).transpose(1, 2)
    v = v.view(b, lk, n_heads, dh).transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal
    )
    return out.transpose(1, 2).reshape(b, lq, d)


class _SelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.proj(_sdpa(q, k, v, self.n_heads, attn_mask, is_causal))

    def forward_cached(
        self, x: Tensor, k_cache: Tensor, v_cache: Tensor, start: int
    ) -> Tensor:
        """Attend `x` against every key written so far, then extend.

        `x` holds the tokens at absolute positions `start ..
        start + x.shape[1] - 1`. Their keys and values are written into
        the caches in place, and the queries attend over positions
        ``0 .. start + i`` for the i-th new token, which is exactly the
        causal mask the batched path applies.
        """
        n_new = x.shape[1]
        end = start + n_new
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        k_cache[:, start:end] = k
        v_cache[:, start:end] = v
        keys = k_cache[:, :end]
        vals = v_cache[:, :end]
        rows = torch.arange(start, end, device=x.device).unsqueeze(1)
        cols = torch.arange(end, device=x.device).unsqueeze(0)
        mask = (cols <= rows).view(1, 1, n_new, end)
        return self.proj(_sdpa(q, keys, vals, self.n_heads, mask))


class _CrossAttention(nn.Module):
    """Queries from the decision tokens, keys and values from history.

    The key/value projection is per layer, and it is the expensive half
    of the query block. It is also the half that depends only on the
    history, so a later caching phase can hoist ``kv(memory)`` out of
    the per-decision path and share it across the four seats of a trick.
    """

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x: Tensor, mem: Tensor, attn_mask: Tensor) -> Tensor:
        k, v = self.kv(mem).chunk(2, dim=-1)
        return self.proj(_sdpa(self.q(x), k, v, self.n_heads, attn_mask))


class _FeedForward(nn.Module):
    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _CausalBlock(nn.Module):
    """Pre-LayerNorm block with causal self-attention."""

    def __init__(self, d: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = _SelfAttention(d, n_heads)
        self.norm2 = nn.LayerNorm(d)
        self.ff = _FeedForward(d, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), is_causal=True)
        return x + self.ff(self.norm2(x))

    def forward_cached(
        self, x: Tensor, k_cache: Tensor, v_cache: Tensor, start: int
    ) -> Tensor:
        x = x + self.attn.forward_cached(self.norm1(x), k_cache, v_cache, start)
        return x + self.ff(self.norm2(x))


class _MaskedBlock(nn.Module):
    """Pre-LayerNorm block with key-masked, non-causal self-attention."""

    def __init__(self, d: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = _SelfAttention(d, n_heads)
        self.norm2 = nn.LayerNorm(d)
        self.ff = _FeedForward(d, d_ff)

    def forward(self, x: Tensor, key_mask: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=key_mask)
        return x + self.ff(self.norm2(x))


class _QueryBlock(nn.Module):
    """Self-attention over the decision tokens, then cross to history."""

    def __init__(self, d: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm_sa = nn.LayerNorm(d)
        self.sa = _SelfAttention(d, n_heads)
        self.norm_ca = nn.LayerNorm(d)
        self.ca = _CrossAttention(d, n_heads)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = _FeedForward(d, d_ff)

    def forward(
        self, x: Tensor, token_mask: Tensor, mem: Tensor, mem_mask: Tensor
    ) -> Tensor:
        x = x + self.sa(self.norm_sa(x), attn_mask=token_mask)
        x = x + self.ca(self.norm_ca(x), mem, attn_mask=mem_mask)
        return x + self.ff(self.norm_ff(x))


def _init_module(module: nn.Module) -> None:
    """Small-variance init. Residual output projections are scaled down
    by the caller so the residual stream does not grow with depth."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _scale_residual_outputs(blocks: nn.ModuleList) -> None:
    """Divide every residual-branch output projection by sqrt(2L)."""
    n = max(len(blocks), 1)
    scale = 1.0 / math.sqrt(2.0 * n)
    for block in blocks:
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear) and (
                name.endswith("proj") or name.endswith("fc2")
            ):
                with torch.no_grad():
                    mod.weight.mul_(scale)


# --- 1. history encoder ---------------------------------------------------


@dataclass
class HistoryCache:
    """Per-layer keys and values for one hand's plays, plus the output.

    One cache serves every seat at a table: the play stream is shared,
    and relative seat identity enters through the history token's own
    features rather than through the encoder's parameters.
    """

    keys: Tensor  # [layers, B, max_history, d_model]
    values: Tensor
    encoded: Tensor  # [B, max_history, d_model]
    tokens: Tensor  # [B, max_history, HISTORY_DIM] as last encoded
    length: int

    def reset(self) -> None:
        """Clear the cache for a new hand.

        The buffers are zeroed rather than only the length, so that a
        reused cache cannot serve a previous hand's activations through
        a slot that is later read before being written.
        """
        self.keys.zero_()
        self.values.zero_()
        self.encoded.zero_()
        self.tokens.zero_()
        self.length = 0


class HistoryEncoder(nn.Module):
    """Causal transformer over the plays of the hand.

    Returns the whole ``[B, max_history, d_model]`` sequence rather than
    a pooled vector. Two reasons: the query block cross-attends to all
    of it, and it is the exact thing a later phase caches -- within one
    hand the token at position t never changes once written, so
    encoding a new decision is an append, not a recompute.

    On masking. ``scaled_dot_product_attention`` will not take an
    explicit ``attn_mask`` together with ``is_causal=True``, and the
    brief calls for ``is_causal=True``. That is not a compromise here,
    because padding is a *suffix*: position t attends only to positions
    <= t, and every position <= t is real whenever t itself is. So no
    real position can read a padded one. What the per-row length mask
    still has to do is stop padded positions leaking *out*, and that is
    done twice over -- their outputs are zeroed after every block, and
    the query block excludes them as cross-attention keys.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(HISTORY_DIM, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(cfg.max_history, cfg.d_model))
        self.blocks = nn.ModuleList(
            _CausalBlock(cfg.d_model, cfg.n_heads, cfg.d_ff)
            for _ in range(cfg.n_layers_history)
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.apply(_init_module)
        nn.init.normal_(self.pos, std=0.02)
        _scale_residual_outputs(self.blocks)

    def forward(self, history: Tensor, history_len: Tensor) -> tuple[Tensor, Tensor]:
        """[B, L, HISTORY_DIM] -> ([B, L, d_model], bool[B, L] keep mask)."""
        keep = _length_mask(history_len, self.cfg.max_history)
        gate = keep.unsqueeze(-1).to(history.dtype)
        x = (self.in_proj(history) + self.pos) * gate
        for block in self.blocks:
            x = block(x) * gate
        return self.norm(x) * gate, keep

    # -- incremental path --------------------------------------------------

    def new_cache(
        self,
        batch: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "HistoryCache":
        d = self.cfg.d_model
        n = self.cfg.max_history
        layers = len(self.blocks)
        zeros = lambda: torch.zeros(layers, batch, n, d, device=device, dtype=dtype)
        return HistoryCache(
            keys=zeros(),
            values=zeros(),
            encoded=torch.zeros(batch, n, d, device=device, dtype=dtype),
            tokens=torch.zeros(batch, n, HISTORY_DIM, device=device, dtype=dtype),
            length=0,
        )

    @torch.no_grad()
    def advance(
        self,
        cache: "HistoryCache",
        history: Tensor,
        length: int,
        recompute_from: int | None = None,
    ) -> Tensor:
        """Bring the cache up to `length` plays of `history` and return it.

        `history` is the whole [B, max_history, HISTORY_DIM] block as
        the feature encoder produced it, not only the new rows. The
        cached prefix is compared against it and re-encoding starts at
        the first position that differs.

        The comparison is not defensive bookkeeping. A history token is
        not final when it is first written: the flag recording whether
        a play took its trick is zero while that trick is open and one
        once it closes, so a token written mid-trick changes. Encoding
        positions in the order they first appear, without the check,
        leaves the cache holding stale keys and the error does not decay
        -- on a real hand it reaches 1.9 against activations of scale
        3.7, and it never recovers.

        Everything after the first differing position is discarded and
        recomputed, which for the common case of a trick closing is
        three tokens.

        Finding that position costs a device synchronisation: the index
        has to reach Python before it can slice, and `nonzero` plus the
        scalar read drain the queue. A caller that already knows how far
        back the tokens can have moved -- a self-play loop knows, since
        only the open trick's tokens are provisional -- passes
        `recompute_from` and skips both the comparison and the sync.
        The caller then owns correctness: naming a position later than
        the first genuine change leaves stale keys in the cache, which
        is the failure this method exists to prevent.
        """
        if length > self.cfg.max_history:
            raise ValueError(
                f"history of {length} exceeds the {self.cfg.max_history}-slot cache"
            )
        if history.shape[0] != cache.encoded.shape[0]:
            raise ValueError(
                f"cache holds batch {cache.encoded.shape[0]}, got {history.shape[0]}"
            )

        if recompute_from is not None:
            start = min(max(int(recompute_from), 0), length)
        else:
            start = min(cache.length, length)
            if start > 0:
                same = (history[:, :start] == cache.tokens[:, :start]).all(
                    dim=-1
                )
                fresh = same.all(dim=0)
                moved = (~fresh).nonzero()
                if moved.numel():
                    start = int(moved[0])
        if start >= length:
            cache.length = length
            return cache.encoded[:, :length]

        x = self.in_proj(history[:, start:length])
        x = x + self.pos[start:length].unsqueeze(0)
        for i, block in enumerate(self.blocks):
            x = block.forward_cached(x, cache.keys[i], cache.values[i], start)
        cache.encoded[:, start:length] = self.norm(x)
        cache.tokens[:, start:length] = history[:, start:length]
        cache.length = length
        return cache.encoded[:, :length]

    @torch.no_grad()
    def append(self, cache: "HistoryCache", tokens: Tensor) -> Tensor:
        """Encode `tokens` as the next plays, assuming the prefix is final.

        Correct only where the caller guarantees that no already-cached
        token can still change. `advance` carries no such precondition
        and is what a training or evaluation loop should call.
        """
        start = cache.length
        end = start + tokens.shape[1]
        if end > self.cfg.max_history:
            raise ValueError(
                f"history of {end} exceeds the {self.cfg.max_history}-slot cache"
            )
        x = self.in_proj(tokens) + self.pos[start:end].unsqueeze(0)
        for i, block in enumerate(self.blocks):
            x = block.forward_cached(x, cache.keys[i], cache.values[i], start)
        cache.encoded[:, start:end] = self.norm(x)
        cache.tokens[:, start:end] = tokens
        cache.length = end
        return cache.encoded[:, :end]


# --- 2. query block -------------------------------------------------------


class QueryEncoder(nn.Module):
    """Builds the decision tokens and collapses them to a summary.

    Sixty tokens, all present in every row so the shape is static:

    * 0  the context vector,
    * 1  the unseen-count vector,
    * 2..5  one per relative seat, from that seat's completed-trick
      plane concatenated with its current-trick plane,
    * 6..59 one per card type, from the card-feature row, masked to the
      cards actually held.

    Token 0 is read out as the state summary. It is the token with no
    masking of its own and the one every block's self-attention can
    reach, which makes it the natural aggregator.

    A learned sink token is prepended to the cross-attention memory. On
    the opening lead the history is empty, and without a sink every key
    would be masked, giving softmax a row of ``-inf`` and a NaN. The
    sink costs one token and lets "attend to nothing" be expressible.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.ctx_proj = nn.Linear(CONTEXT_DIM, d)
        self.unseen_proj = nn.Linear(NUM_TYPES, d)
        # Shared across seats, with a learned per-seat offset, so the
        # relative-seat symmetry lives in the weights rather than in
        # four copies that have to learn it separately.
        self.seat_proj = nn.Linear(2 * NUM_TYPES, d)
        self.kitty_proj = nn.Linear(NUM_TYPES, d)
        self.seat_embed = nn.Parameter(torch.zeros(NUM_REL_SEATS, d))
        self.hand_proj = nn.Linear(D_CARD, d)
        self.card_embed = nn.Parameter(torch.zeros(NUM_TYPES, d))

        self.mem_sink = nn.Parameter(torch.zeros(1, d))
        self.mem_norm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(
            _QueryBlock(d, cfg.n_heads, cfg.d_ff) for _ in range(cfg.n_layers_query)
        )
        self.out_norm = nn.LayerNorm(d)

        self.apply(_init_module)
        for p in (self.seat_embed, self.card_embed, self.mem_sink):
            nn.init.normal_(p, std=0.02)
        _scale_residual_outputs(self.blocks)

    def build_tokens(self, obs: ObsBatch) -> tuple[Tensor, Tensor]:
        """[B, 60, d_model] tokens and their bool[B, 60] validity mask."""
        b = obs.context.shape[0]
        d = self.cfg.d_model

        ctx_tok = self.ctx_proj(obs.context).unsqueeze(1)  # [B, 1, d]
        unseen_tok = self.unseen_proj(obs.unseen).unsqueeze(1)  # [B, 1, d]

        played = obs.planes[:, PLANE_PLAYED : PLANE_PLAYED + NUM_REL_SEATS]
        trick = obs.planes[:, PLANE_TRICK : PLANE_TRICK + NUM_REL_SEATS]
        seat_tok = self.seat_proj(torch.cat((played, trick), dim=-1))
        seat_tok = seat_tok + self.seat_embed  # [B, 4, d]

        hand_tok = self.hand_proj(obs.cards) + self.card_embed  # [B, 54, d]

        # The landlord's own burial. Zero for every other seat, and the
        # only place those eight cards appear in the observation: they
        # are subtracted out of `unseen` for the seat that buried them.
        kitty_tok = self.kitty_proj(obs.planes[:, PLANE_KITTY]).unsqueeze(1)

        tokens = torch.cat(
            (ctx_tok, unseen_tok, kitty_tok, seat_tok, hand_tok), dim=1
        )

        held = obs.planes[:, PLANE_HAND] > 0.5  # [B, 54]
        fixed = torch.ones(
            b, QUERY_HAND, dtype=torch.bool, device=tokens.device
        )
        mask = torch.cat((fixed, held), dim=1)  # [B, 60]
        return tokens * mask.unsqueeze(-1).to(tokens.dtype), mask

    def forward(
        self, obs: ObsBatch, history_enc: Tensor, history_keep: Tensor
    ) -> Tensor:
        """-> [B, d_model], the state summary h."""
        tokens, token_mask = self.build_tokens(obs)
        gate = token_mask.unsqueeze(-1).to(tokens.dtype)

        b = tokens.shape[0]
        sink = self.mem_sink.unsqueeze(0).expand(b, 1, self.cfg.d_model)
        mem = self.mem_norm(torch.cat((sink, history_enc), dim=1))
        mem_keep = torch.cat(
            (
                torch.ones(b, 1, dtype=torch.bool, device=tokens.device),
                history_keep,
            ),
            dim=1,
        )

        # [B, 1, 1, L] broadcasts over heads and query positions.
        token_attn = token_mask[:, None, None, :]
        mem_attn = mem_keep[:, None, None, :]

        x = tokens
        for block in self.blocks:
            x = block(x, token_attn, mem, mem_attn) * gate
        return self.out_norm(x[:, QUERY_CONTEXT])


# --- 3. candidate scorer --------------------------------------------------


def _card_scalar_basis() -> Tensor:
    """[D_CARD, 3]: read point value, trump-ness and strength straight
    out of the static one-hot block.

    The static block already holds a one-hot point value over {0, 5, 10},
    a one-hot effective suit whose last slot is trump, and a one-hot
    strength. A single fixed matrix turns all three back into scalars,
    so the scorer needs no card table of its own and cannot drift out of
    step with ``features.py``.
    """
    basis = torch.zeros(D_CARD, NUM_CARD_SCALARS)
    basis[STATIC_POINTS + 1, 0] = 0.5  # 5 points, normalised by 10
    basis[STATIC_POINTS + 2, 0] = 1.0  # 10 points
    basis[STATIC_SUIT + TRUMP, 1] = 1.0
    for k in range(NUM_STRENGTH):
        basis[STATIC_STRENGTH + k, 2] = k / (NUM_STRENGTH - 1)
    return basis


class CandidateScorer(nn.Module):
    """Scores every padded candidate slot against the state summary.

    A candidate is an ``int8[54]`` count vector -- a multiset, not a
    sequence -- so it is embedded as a weighted sum of projected card
    features. That is one batched matmul for all slots and is exactly
    permutation-invariant, which a per-card RNN or a sorted encoding
    would only approximate.

    Slots never interact: there is no attention across candidates. That
    is what makes the masking exact rather than approximate -- the
    contents of an invalid slot cannot influence a valid one through any
    path, so the test that scrambles them and demands bit-identical
    output is checking a structural property, not a numerical accident.

    The count vector alone loses things a linear map over card features
    cannot recover, mostly because it is summed: how many cards, whether
    they are uniformly trump, how the points concentrate. Six scalars
    are derived and folded in through a small MLP.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # No bias: this is a basis for a multiset sum, and a bias would
        # scale with the number of cards, which scalar 0 already says.
        self.card_proj = nn.Linear(D_CARD, d, bias=False)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(NUM_CAND_SCALARS, cfg.scalar_d),
            nn.GELU(),
            nn.Linear(cfg.scalar_d, d),
        )
        self.norm = nn.LayerNorm(d)
        self.score_q = nn.Linear(d, d)
        self.cand_bias = nn.Linear(d, 1)
        self.register_buffer("scalar_basis", _card_scalar_basis(), persistent=False)
        self.apply(_init_module)

    def candidate_scalars(self, candidates: Tensor, cards: Tensor) -> Tensor:
        """[B, N, NUM_CAND_SCALARS] from the counts and the card block."""
        per_card = cards @ self.scalar_basis  # [B, 54, 3]
        agg = torch.bmm(candidates, per_card)  # [B, N, 3]
        points = agg[..., 0:1]
        trump = agg[..., 1:2]
        strength = agg[..., 2:3]

        n = candidates.sum(dim=-1, keepdim=True)  # [B, N, 1]
        nz = n.clamp(min=1.0)
        nonempty = (n > 0.5).to(candidates.dtype)
        return torch.cat(
            (
                n / 8.0,  # cards played, over the largest legal lead
                points,  # total points, already over 10
                # All trump, and not merely empty: an invalid slot holds
                # no cards and must not read as a pure-trump play.
                ((trump - n).abs() < 0.5).to(candidates.dtype) * nonempty,
                trump / nz,  # trump fraction
                strength / nz,  # mean strength
                ((n - 1.0).abs() < 0.5).to(candidates.dtype),  # a single card
            ),
            dim=-1,
        )

    def forward(
        self, summary: Tensor, obs: ObsBatch
    ) -> tuple[Tensor, Tensor]:
        """-> (logits[B, N], log_probs[B, N]), both float32."""
        basis = self.card_proj(obs.cards)  # [B, 54, d]
        emb = torch.bmm(obs.candidates, basis)  # [B, N, d]
        emb = self.norm(
            emb + self.scalar_mlp(self.candidate_scalars(obs.candidates, obs.cards))
        )

        # Bilinear interaction with the state summary, plus a term that
        # depends on the candidate alone -- a prior over shapes that
        # does not have to be relearned inside the bilinear form.
        q = self.score_q(summary).unsqueeze(-1)  # [B, d, 1]
        scale = 1.0 / math.sqrt(self.cfg.d_model)
        logits = torch.bmm(emb, q).squeeze(-1) * scale
        logits = logits + self.cand_bias(emb).squeeze(-1)

        # Promote before masking. MASK_LOGIT does not fit in float16,
        # so filling in the autocast dtype either overflows or lands on
        # -inf, and -inf multiplied by a zero probability is the NaN the
        # masking exists to avoid. Promoting first also makes the
        # ActorOutput contract -- every tensor float32 -- hold under
        # autocast.
        logits = logits.float()
        # Overwrite, not add: the value at an invalid slot is now
        # independent of anything that slot contained.
        logits = torch.where(
            obs.candidate_mask,
            logits,
            torch.full_like(logits, MASK_LOGIT),
        )
        log_probs = F.log_softmax(logits, dim=-1)
        return logits, log_probs


# --- 4. auxiliary heads ---------------------------------------------------


class AuxHeads(nn.Module):
    """Value, outcome class, and per-card location beliefs off ``h``.

    The belief head is the one with a shape problem. It predicts a
    distribution over five locations for each of 54 card types, so the
    naive form is an MLP evaluated 54 times per row. The first layer is
    factored instead: it reads ``concat(h, card_features)``, and for a
    block matrix ``W = [W_a | W_b]``,

        W @ [a; b] = W_a @ a + W_b @ b

    so ``W_a @ h`` is one ``[B, d]`` product and ``W_b @ cards`` is one
    ``[B, 54, d]`` product, broadcast-added. That is two matmuls instead
    of 54 MLP calls, and it is exact rather than an approximation.

    ``W_b @ cards`` is further cacheable: 28 of the 30 card-feature
    columns are the static block, constant for a whole hand.
    ``belief_card_term`` is exposed so a caching phase can hoist it.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, hd = cfg.d_model, cfg.head_d
        self.value = nn.Sequential(nn.Linear(d, hd), nn.GELU(), nn.Linear(hd, 1))
        self.outcome = nn.Sequential(
            nn.Linear(d, hd), nn.GELU(), nn.Linear(hd, NUM_OUTCOMES)
        )
        self.belief_h = nn.Linear(d, cfg.belief_d)  # W_a, carries the bias
        self.belief_c = nn.Linear(D_CARD, cfg.belief_d, bias=False)  # W_b
        self.belief_act = nn.GELU()
        self.belief_out = nn.Linear(cfg.belief_d, NUM_BELIEF_LOCATIONS)
        self.apply(_init_module)

    def belief_card_term(self, cards: Tensor) -> Tensor:
        """``W_b @ card_features`` -> [B, 54, belief_d]. Cacheable."""
        return self.belief_c(cards)

    def forward(
        self, summary: Tensor, cards: Tensor, card_term: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        value = self.value(summary).squeeze(-1)
        outcome_logits = self.outcome(summary)

        if card_term is None:
            card_term = self.belief_card_term(cards)
        z = self.belief_act(self.belief_h(summary).unsqueeze(1) + card_term)
        belief_logits = self.belief_out(z)  # [B, 54, 5]
        belief_log_probs = F.log_softmax(belief_logits.float(), dim=-1)
        return value, outcome_logits, belief_logits, belief_log_probs


# --- the actor ------------------------------------------------------------


class TractorNet(nn.Module):
    """History encoder, query block, candidate scorer, auxiliary heads.

    This is the whole actor. The critic is a separate object on purpose;
    see ``Critic``.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.history = HistoryEncoder(self.cfg)
        self.query = QueryEncoder(self.cfg)
        self.scorer = CandidateScorer(self.cfg)
        self.heads = AuxHeads(self.cfg)

    def forward(self, obs: ObsBatch) -> ActorOutput:
        history_enc, history_keep = self.history(obs.history, obs.history_len)
        summary = self.query(obs, history_enc, history_keep)
        logits, log_probs = self.scorer(summary, obs)
        value, outcome_logits, belief_logits, belief_log_probs = self.heads(
            summary, obs.cards
        )
        return ActorOutput(
            logits=logits,
            log_probs=log_probs,
            value=value,
            outcome_logits=outcome_logits,
            belief_logits=belief_logits,
            belief_log_probs=belief_log_probs,
            summary=summary,
            history_enc=history_enc,
        )


# --- 5. critic ------------------------------------------------------------


class Critic(nn.Module):
    """Privileged value head. A separate module, trained separately.

    It reads two things: the actor's encoded history, and the four hands
    plus the kitty. The second is information no seat has, which is the
    point -- a privileged critic gives a lower-variance baseline than
    one restricted to the actor's view.

    The gradient question is settled by the ``detach`` flag rather than
    by convention. With ``detach=True`` the history tensor is cut before
    it enters, so a critic loss reaches no actor parameter; the critic's
    own parameters are disjoint from the actor's by construction. With
    ``detach=False`` the critic's gradient does flow back through the
    history encoder, which is a different training scheme (shared trunk)
    and has to be chosen deliberately.

    A learned CLS token leads the sequence. Attention here is *not*
    causal -- the critic judges the position as of now, so every real
    token should be visible to every other -- and the CLS token also
    guarantees at least one unmasked key when the history is empty.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        cd = cfg.critic_d
        self.in_proj = nn.Linear(cfg.d_model, cd)
        self.cls = nn.Parameter(torch.zeros(1, cd))
        self.pos = nn.Parameter(torch.zeros(cfg.max_history, cd))
        self.blocks = nn.ModuleList(
            _MaskedBlock(cd, cfg.critic_heads, cfg.critic_d_ff)
            for _ in range(cfg.critic_layers)
        )
        self.norm = nn.LayerNorm(cd)
        self.privileged = nn.Sequential(
            nn.Linear(NUM_PRIVILEGED_ROWS * NUM_TYPES, 2 * cd),
            nn.GELU(),
            nn.Linear(2 * cd, cd),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * cd, cd), nn.GELU(), nn.Linear(cd, 1)
        )
        self.apply(_init_module)
        for p in (self.cls, self.pos):
            nn.init.normal_(p, std=0.02)
        _scale_residual_outputs(self.blocks)

    def forward(
        self,
        history_enc: Tensor,
        history_len: Tensor,
        privileged: Tensor,
        detach: bool = True,
    ) -> Tensor:
        """-> [B]. ``privileged`` is float[B, 5, 54] of raw counts."""
        if detach:
            history_enc = history_enc.detach()

        keep = _length_mask(history_len, self.cfg.max_history)
        gate = keep.unsqueeze(-1).to(history_enc.dtype)
        h = (self.in_proj(history_enc) + self.pos) * gate

        b = h.shape[0]
        cls = self.cls.unsqueeze(0).expand(b, 1, self.cfg.critic_d)
        x = torch.cat((cls, h), dim=1)
        seq_keep = torch.cat(
            (torch.ones(b, 1, dtype=torch.bool, device=h.device), keep), dim=1
        )
        seq_gate = seq_keep.unsqueeze(-1).to(x.dtype)
        attn = seq_keep[:, None, None, :]

        for block in self.blocks:
            x = block(x, attn) * seq_gate
        pooled = self.norm(x[:, 0])  # [B, critic_d]

        flat = privileged.reshape(b, NUM_PRIVILEGED_ROWS * NUM_TYPES)
        priv = self.privileged(flat)  # [B, critic_d]
        return self.head(torch.cat((pooled, priv), dim=-1)).squeeze(-1)


class ActorCritic(nn.Module):
    """Convenience container. The two towers stay addressable apart."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.actor = TractorNet(self.cfg)
        self.critic = Critic(self.cfg)

    def forward(
        self, obs: ObsBatch, privileged: Tensor, detach_critic: bool = True
    ) -> tuple[ActorOutput, Tensor]:
        out = self.actor(obs)
        value = self.critic(
            out.history_enc, obs.history_len, privileged, detach=detach_critic
        )
        return out, value


# --- building batches -----------------------------------------------------


def _pad_candidates(
    moves: Sequence[np.ndarray], max_candidates: int
) -> tuple[np.ndarray, np.ndarray]:
    padded = np.zeros((max_candidates, NUM_TYPES), dtype=np.float32)
    mask = np.zeros(max_candidates, dtype=bool)
    kept = min(len(moves), max_candidates)
    for i in range(kept):
        padded[i] = moves[i]
        mask[i] = True
    return padded, mask


def batch_observations(
    observations: Sequence[Observation],
    candidates: Sequence[Sequence[np.ndarray]],
    cfg: ModelConfig | None = None,
    device: torch.device | str | None = None,
) -> ObsBatch:
    """Stack encoded observations and their legal moves into an ObsBatch.

    ``candidates[i]`` is the move list for ``observations[i]``: each an
    ``int8[54]`` count vector from ``MoveSet.moves``. Anything past
    ``cfg.max_candidates`` is dropped, which the enumerator's own cap
    already makes rare -- the audit put the p99 at 105.
    """
    cfg = cfg or ModelConfig()
    if len(observations) != len(candidates):
        raise ValueError("one candidate list per observation")
    n = len(observations)
    if n == 0:
        raise ValueError("empty batch")

    cand = np.zeros((n, cfg.max_candidates, NUM_TYPES), dtype=np.float32)
    mask = np.zeros((n, cfg.max_candidates), dtype=bool)
    for i, moves in enumerate(candidates):
        cand[i], mask[i] = _pad_candidates(moves, cfg.max_candidates)

    def stack(name: str, dtype=np.float32) -> Tensor:
        return torch.from_numpy(
            np.stack([getattr(o, name) for o in observations]).astype(dtype)
        )

    batch = ObsBatch(
        history=stack("history"),
        history_len=torch.tensor(
            [int(o.history_len) for o in observations], dtype=torch.int64
        ),
        context=stack("context"),
        unseen=stack("unseen"),
        planes=stack("planes"),
        cards=stack("cards"),
        candidates=torch.from_numpy(cand),
        candidate_mask=torch.from_numpy(mask),
    )
    return batch.to(device) if device is not None else batch


def privileged_view(
    hands: np.ndarray, kitty: np.ndarray, seat: int
) -> np.ndarray:
    """float32[5, 54]: the four hands in relative order, then the kitty.

    Relative order so the critic reads a position the same way from all
    four chairs, matching ``features.py``.
    """
    out = np.zeros((NUM_PRIVILEGED_ROWS, NUM_TYPES), dtype=np.float32)
    for rel in range(NUM_REL_SEATS):
        out[rel] = hands[(seat + rel) % NUM_REL_SEATS]
    out[NUM_REL_SEATS] = kitty
    return out


def batch_privileged(
    views: Sequence[np.ndarray], device: torch.device | str | None = None
) -> Tensor:
    t = torch.from_numpy(np.stack(views).astype(np.float32))
    return t.to(device) if device is not None else t


def dummy_batch(
    cfg: ModelConfig | None = None,
    batch: int = 8,
    history_len: int | None = None,
    n_candidates: int = 12,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[ObsBatch, Tensor]:
    """Synthetic inputs with the right shapes, dtypes and sparsity.

    For benchmarking and for tests that do not need a real position. The
    values are nonsense but the *structure* is honest: history is a
    prefix, candidates are a prefix, the hand is a subset of card types,
    and every count is a small non-negative integer.
    """
    cfg = cfg or ModelConfig()
    rng = np.random.default_rng(seed)
    if history_len is None:
        history_len = cfg.max_history
    history_len = min(history_len, cfg.max_history)
    n_candidates = min(n_candidates, cfg.max_candidates)

    hist = rng.standard_normal(
        (batch, cfg.max_history, HISTORY_DIM), dtype=np.float32
    )
    lens = rng.integers(0, history_len + 1, size=batch)
    for i, ln in enumerate(lens):
        hist[i, ln:] = 0.0

    planes = rng.integers(
        0, 3, size=(batch, NUM_PLANES, NUM_TYPES)
    ).astype(np.float32)
    cards = rng.standard_normal((batch, NUM_TYPES, D_CARD), dtype=np.float32)

    cand = np.zeros((batch, cfg.max_candidates, NUM_TYPES), dtype=np.float32)
    mask = np.zeros((batch, cfg.max_candidates), dtype=bool)
    for i in range(batch):
        k = int(rng.integers(1, n_candidates + 1))
        mask[i, :k] = True
        for j in range(k):
            picks = rng.integers(0, NUM_TYPES, size=int(rng.integers(1, 9)))
            for p in picks:
                cand[i, j, p] = min(cand[i, j, p] + 1.0, 2.0)

    obs = ObsBatch(
        history=torch.from_numpy(hist),
        history_len=torch.from_numpy(lens.astype(np.int64)),
        context=torch.from_numpy(
            rng.standard_normal((batch, CONTEXT_DIM), dtype=np.float32)
        ),
        unseen=torch.from_numpy(
            rng.integers(0, 3, size=(batch, NUM_TYPES)).astype(np.float32)
        ),
        planes=torch.from_numpy(planes),
        cards=torch.from_numpy(cards),
        candidates=torch.from_numpy(cand),
        candidate_mask=torch.from_numpy(mask),
    )
    privileged = torch.from_numpy(
        rng.integers(
            0, 3, size=(batch, NUM_PRIVILEGED_ROWS, NUM_TYPES)
        ).astype(np.float32)
    )
    if device is not None:
        obs, privileged = obs.to(device), privileged.to(device)
    return obs, privileged


# --- 6. analytic FLOPs ----------------------------------------------------

# Parameter counts for the default configuration, derived by hand so a
# structural change to the model shows up as a test failure rather than
# as a quietly different number. FLOPs below are derived the same way.
#
# d = 384, ff = 1024, heads = 6, history layers = 3, query layers = 3.
#
# One pre-LN self-attention + FFN block at width d:
#   2 LayerNorms         2 * 2d                        =     1,536
#   qkv                  d * 3d + 3d                   =   443,520
#   out projection       d * d + d                     =   147,840
#   ff1                  d * ff + ff                   =   394,240
#   ff2                  ff * d + d                    =   393,600
#                                                        ---------
#                                                        1,380,736
#
# History encoder: in_proj 37*384+384 = 14,592; positional 128*384 =
#   49,152; 3 blocks = 4,142,208; final LayerNorm 768.
#                                                        4,206,720
#
# One query layer adds cross-attention to the block above:
#   block pieces as above                              = 1,380,736
#   cross LayerNorm      2d                            =       768
#   cross q              d * d + d                     =   147,840
#   cross kv             d * 2d + 2d                   =   295,680
#   cross out            d * d + d                     =   147,840
#                                                        ---------
#                                                        1,972,864
# Query encoder: 3 layers = 5,918,592; ctx 14,592; unseen 21,120;
#   seat 41,856; seat embed 1,536; hand 11,904; card embed 20,736;
#   sink 384; mem LayerNorm 768; out LayerNorm 768.
#                                                        6,032,256
#
# Scorer: card_proj 30*384 = 11,520; scalar fc1 448; scalar fc2 24,960;
#   LayerNorm 768; score_q 147,840; cand_bias 385.
#                                                          185,921
#
# Heads: value 73,920 + 193; outcome 73,920 + 1,351; belief_h 73,920;
#   belief_c 30*192 = 5,760; belief_out 965.
#                                                          230,029
#
# Actor total  4,206,720 + 6,032,256 + 185,921 + 230,029 = 10,654,926
#
# Critic at width 128, 2 heads, ff 256, 2 layers:
#   in_proj 49,280; cls 128; positional 16,384; 2 blocks 2*132,480 =
#   264,960; LayerNorm 256; privileged 69,376 + 32,896; head 32,896 +
#   129.
#                                                          466,305
ANALYTIC_ACTOR_PARAMS = 10_676_046
ANALYTIC_CRITIC_PARAMS = 466_305


def analytic_flops(
    cfg: ModelConfig,
    batch: int,
    n_candidates: int,
    history_len: int,
    *,
    matmul_only: bool = False,
    causal_discount: bool = True,
) -> dict[str, int]:
    """Forward FLOPs, derived from the configuration, by component.

    Returns ``history``, ``query``, ``scorer``, ``heads``, ``critic``,
    their sum as ``actor`` (the first four) and ``total``.

    Three conventions, stated because the number is meaningless without
    them.

    *Two FLOPs per multiply-accumulate.* A ``[n, i] @ [i, o]`` matmul is
    ``2 * n * i * o``; a bias adds ``n * o``.

    *``n_candidates`` and ``history_len`` are tensor extents, not valid
    counts.* Masking saves no arithmetic on a dense padded batch: if the
    forward runs at 128 history slots, pass 128, even when the longest
    row holds 99. Passing the real length answers the different question
    of what the work would be with perfect packing.

    *Causal attention is counted over the lower triangle* --
    ``L(L+1)/2`` query-key pairs, what a causal-aware kernel actually
    does. Pass ``causal_discount=False`` for the dense ``L * L`` count,
    which is what ``torch.utils.flop_counter`` reports and what a math
    fallback would execute.

    ``matmul_only=True`` drops biases, LayerNorm, GELU and softmax,
    leaving only what a matmul-level counter sees.
    """
    d = cfg.d_model
    ff = cfg.d_ff
    heads = cfg.n_heads
    n = n_candidates
    lh = history_len
    lmem = lh + 1  # + the learned sink token
    tok = NUM_QUERY_TOKENS

    def lin(rows: int, d_in: int, d_out: int, bias: bool = True) -> int:
        f = rows * 2 * d_in * d_out
        if bias and not matmul_only:
            f += rows * d_out
        return f

    def elem(count: int, per: int = 1) -> int:
        """Elementwise work: adds, gates, scalings."""
        return 0 if matmul_only else count * per

    def norm(rows: int, width: int) -> int:
        return elem(rows * width, FLOPS_PER_LAYERNORM_ELEM)

    def gelu(count: int) -> int:
        return elem(count, FLOPS_PER_GELU)

    def attn(lq: int, lk: int, width: int, nh: int, causal: bool = False) -> int:
        pairs = lq * (lq + 1) // 2 if (causal and causal_discount) else lq * lk
        # Q@K^T and A@V are each 2 FLOPs per (pair, head, head-dim).
        f = 2 * pairs * width + 2 * pairs * width
        # The scale and the softmax run on [pairs, heads] scores.
        f += elem(pairs * nh)
        f += elem(pairs * nh, FLOPS_PER_SOFTMAX_ELEM)
        return f

    def sa_ffn_block(rows: int, width: int, nh: int, width_ff: int, causal: bool,
                     keys: int) -> int:
        f = norm(rows, width)
        f += lin(rows, width, 3 * width)
        f += attn(rows, keys, width, nh, causal)
        f += lin(rows, width, width)
        f += elem(rows * width)  # residual add
        f += norm(rows, width)
        f += lin(rows, width, width_ff)
        f += gelu(rows * width_ff)
        f += lin(rows, width_ff, width)
        f += elem(rows * width)  # residual add
        return f

    # -- history encoder ---------------------------------------------------
    hist = lin(lh, HISTORY_DIM, d)
    hist += elem(lh * d)  # + positional
    hist += elem(lh * d)  # * keep gate
    for _ in range(cfg.n_layers_history):
        hist += sa_ffn_block(lh, d, heads, ff, causal=True, keys=lh)
        hist += elem(lh * d)  # * keep gate
    hist += norm(lh, d)
    hist += elem(lh * d)  # * keep gate
    hist *= batch

    # -- query block -------------------------------------------------------
    query = lin(1, CONTEXT_DIM, d)  # context token
    query += lin(1, NUM_TYPES, d)  # unseen token
    query += lin(1, NUM_TYPES, d)  # the seat's own buried kitty
    query += lin(NUM_REL_SEATS, 2 * NUM_TYPES, d)  # four seat tokens
    query += elem(NUM_REL_SEATS * d)  # + seat embedding
    query += lin(NUM_TYPES, D_CARD, d)  # 54 hand tokens
    query += elem(NUM_TYPES * d)  # + card embedding
    query += elem(tok * d)  # * held gate
    query += norm(lmem, d)  # LayerNorm over sink + history, once
    for _ in range(cfg.n_layers_query):
        # self-attention over the query tokens
        query += norm(tok, d)
        query += lin(tok, d, 3 * d)
        query += attn(tok, tok, d, heads)
        query += lin(tok, d, d)
        query += elem(tok * d)
        # cross-attention to the history memory
        query += norm(tok, d)
        query += lin(tok, d, d)  # q
        query += lin(lmem, d, 2 * d)  # k, v over the memory
        query += attn(tok, lmem, d, heads)
        query += lin(tok, d, d)  # out
        query += elem(tok * d)
        # feed-forward
        query += norm(tok, d)
        query += lin(tok, d, ff)
        query += gelu(tok * ff)
        query += lin(tok, ff, d)
        query += elem(tok * d)
        query += elem(tok * d)  # * held gate
    query += norm(1, d)  # out_norm, token 0 only
    query *= batch

    # -- candidate scorer --------------------------------------------------
    scorer = lin(NUM_TYPES, D_CARD, d, bias=False)  # card_proj basis
    scorer += 2 * n * NUM_TYPES * d  # counts @ basis
    scorer += 2 * NUM_TYPES * D_CARD * NUM_CARD_SCALARS  # cards @ scalar basis
    scorer += 2 * n * NUM_TYPES * NUM_CARD_SCALARS  # counts @ per-card scalars
    scorer += elem(n * 12)  # sums, clamps and comparisons for 6 scalars
    scorer += lin(n, NUM_CAND_SCALARS, cfg.scalar_d)
    scorer += gelu(n * cfg.scalar_d)
    scorer += lin(n, cfg.scalar_d, d)
    scorer += elem(n * d)  # embedding + scalar term
    scorer += norm(n, d)
    scorer += lin(1, d, d)  # score_q on the summary
    scorer += 2 * n * d  # bilinear dot product
    scorer += elem(n)  # * scale
    scorer += lin(n, d, 1)  # candidate-only bias term
    scorer += elem(n)  # + that term
    scorer += elem(n)  # masked overwrite
    scorer += elem(n, FLOPS_PER_SOFTMAX_ELEM)  # log_softmax
    scorer *= batch

    # -- auxiliary heads ---------------------------------------------------
    hd = cfg.head_d
    bd = cfg.belief_d
    aux = lin(1, d, hd) + gelu(hd) + lin(1, hd, 1)  # value
    aux += lin(1, d, hd) + gelu(hd) + lin(1, hd, NUM_OUTCOMES)  # outcome
    aux += lin(1, d, bd)  # W_a @ h, once
    aux += lin(NUM_TYPES, D_CARD, bd, bias=False)  # W_b @ cards, once
    aux += elem(NUM_TYPES * bd)  # broadcast add
    aux += gelu(NUM_TYPES * bd)
    aux += lin(NUM_TYPES, bd, NUM_BELIEF_LOCATIONS)
    aux += elem(NUM_TYPES * NUM_BELIEF_LOCATIONS, FLOPS_PER_SOFTMAX_ELEM)
    aux *= batch

    # -- critic ------------------------------------------------------------
    cd = cfg.critic_d
    cff = cfg.critic_d_ff
    ch = cfg.critic_heads
    lc = lh + 1  # + the CLS token
    crit = lin(lh, d, cd)
    crit += elem(lh * cd)  # + positional
    crit += elem(lh * cd)  # * keep gate
    for _ in range(cfg.critic_layers):
        crit += sa_ffn_block(lc, cd, ch, cff, causal=False, keys=lc)
        crit += elem(lc * cd)  # * keep gate
    crit += norm(1, cd)  # pooled CLS
    crit += lin(1, NUM_PRIVILEGED_ROWS * NUM_TYPES, 2 * cd)
    crit += gelu(2 * cd)
    crit += lin(1, 2 * cd, cd)
    crit += gelu(cd)
    crit += lin(1, 2 * cd, cd)
    crit += gelu(cd)
    crit += lin(1, cd, 1)
    crit *= batch

    actor = hist + query + scorer + aux
    return {
        "history": hist,
        "query": query,
        "scorer": scorer,
        "heads": aux,
        "actor": actor,
        "critic": crit,
        "total": actor + crit,
    }


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad or not trainable_only
    )


__all__ = [
    "ActorCritic",
    "ActorOutput",
    "AuxHeads",
    "CandidateScorer",
    "Critic",
    "HistoryEncoder",
    "MASK_LOGIT",
    "ModelConfig",
    "NUM_BELIEF_LOCATIONS",
    "NUM_OUTCOMES",
    "NUM_QUERY_TOKENS",
    "ObsBatch",
    "QueryEncoder",
    "TractorNet",
    "analytic_flops",
    "batch_observations",
    "batch_privileged",
    "count_parameters",
    "dummy_batch",
    "privileged_view",
]
