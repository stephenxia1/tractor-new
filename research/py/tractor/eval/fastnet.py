"""The play network's forward pass for search on CPU.

Search runs the network on a few dozen simulations per ply and reads one
thing from each forward: the policy's logits at the plies, the critic's
value at the leaves. ``ActorCritic.forward`` computes everything the
training loop needs, every time. This computes what search reads, and
the same function, with the work search never reads left out:

* **Query block, held cards only.** The query block has 61 tokens: seven
  fixed ones and one per card type. Types the seat does not hold are
  excluded as attention keys and zeroed after every block, so nothing
  they hold reaches the summary; leaving them out of the computation is
  exact (``rl/zero/net.ts`` does the same). Late in a hand that is most
  of the 54.
* **Last query layer, context token only.** Only the context token's
  output is read, so the last layer computes keys and values for every
  token and the rest for that one token.
* **Actor or critic, not both.** A ply forward runs the history encoder,
  the query block and the scorer; a leaf forward runs the history
  encoder and the critic.
* **Candidates trimmed** to the batch's longest list rather than the
  padded 128. The scorer has no interaction between slots, so this is
  exact; the trimmed slots come back at ``MASK_LOGIT``.
* **History to the batch's longest, and shared prefixes encoded once.**
  The training forward pads the encoded history back to 128 positions,
  and the query block's cross-attention keys and values and the critic
  then run over all of them; here they run over the longest history in
  the batch. And a search batch is a few dozen continuations of one
  position: the rows share the history up to the decision point (per
  seat to act -- the tokens are seat-relative), and the encoder is
  causal, so that prefix's encoding, its keys and values in every
  history layer, and its cross-attention keys and values in every query
  layer are computed once per group of rows that share it, and only each
  row's own suffix is encoded per row. ``rl/zero/net.ts`` caches the same
  things across decisions.

``tests/test_fastnet.py`` holds it to the reference forward on real
positions, to float32 rounding.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..nn.model import MASK_LOGIT, PLANE_TRICK, ObsBatch, _sdpa

__all__ = ["SearchNet"]

# The rows of the query block's token list (``QueryEncoder.build_tokens``):
# context first, then unseen, kitty, four seats, then the 54 card types.
_CONTEXT = 0
_FIXED = 7
_REL_SEATS = 4


class SearchNet:
    """``logits`` and ``value`` for a candidate-list ``ActorCritic``,
    computed as its forward computes them."""

    def __init__(self, net: Any) -> None:
        actor = getattr(net, "actor", None)
        if actor is None or getattr(net, "critic", None) is None:
            raise ValueError("search needs an ActorCritic with its own critic")
        if actor.cfg.cards_mode:
            raise ValueError("SearchNet scores candidate lists; card mode has its own policy")
        self.net = net
        self.actor = actor
        self.critic = net.critic
        self.cfg = actor.cfg

    # --- the two things search reads -----------------------------------

    @torch.no_grad()
    def logits(self, obs: ObsBatch) -> Tensor:
        """float32[B, C]: the policy's logits, ``MASK_LOGIT`` where no
        candidate is (C as in ``obs``)."""
        hist = self._history(obs.history, obs.history_len)
        summary = self._summary(obs, hist)
        return self._score(summary, obs)

    @torch.no_grad()
    def value(self, obs: ObsBatch, privileged: Tensor) -> Tensor:
        """float32[B]: the privileged critic's value for the seat to act."""
        history_enc = self._history(obs.history, obs.history_len).encoded
        return self.critic(
            history_enc,
            obs.history_len,
            privileged,
            detach=True,
            context=obs.context,
            cards=obs.cards,
            trick=obs.planes[:, PLANE_TRICK : PLANE_TRICK + _REL_SEATS],
        ).float()

    # --- the history, shared prefixes once -------------------------------

    def _history(self, history: Tensor, history_len: Tensor) -> "_History":
        enc = self.actor.history
        b = history.shape[0]
        n = min(max(int(history_len.max()), 1) if b else 1, self.cfg.max_history)
        tok = history[:, :n]
        keep = torch.arange(n, device=tok.device).unsqueeze(0) < history_len.unsqueeze(1)
        # The encoder's output dtype: its final LayerNorm's (float32 under
        # bfloat16 autocast, as in the training forward).
        out_dtype = enc.norm(tok.new_zeros(1, self.cfg.d_model)).dtype
        out = torch.zeros(b, n, self.cfg.d_model, dtype=out_dtype, device=tok.device)
        groups: list[_Group] = []
        for rows in _prefix_groups(tok, keep):
            sub, sub_keep = tok[rows], keep[rows]
            p = _common_prefix(sub, sub_keep)
            if len(rows) < 2 or p < _MIN_PREFIX:
                x = (enc.in_proj(sub) + enc.pos[:n]) * sub_keep.unsqueeze(-1).to(sub.dtype)
                gate = sub_keep.unsqueeze(-1).to(x.dtype)
                for block in enc.blocks:
                    x = block(x) * gate
                out[rows] = (enc.norm(x) * gate).to(out_dtype)
                continue
            # The shared prefix, once: every position real, so no gate.
            xp = enc.in_proj(sub[:1, :p]) + enc.pos[:p]
            keys, vals = [], []
            for block in enc.blocks:
                h = block.norm1(xp)
                q, k, v = block.attn.qkv(h).chunk(3, dim=-1)
                keys.append(k)
                vals.append(v)
                xp = xp + block.attn.proj(_sdpa(q, k, v, block.attn.n_heads, is_causal=True))
                xp = xp + block.ff(block.norm2(xp))
            prefix = enc.norm(xp)
            m = len(rows)
            out[rows, :p] = prefix.to(out_dtype).expand(m, p, -1)
            s_len = n - p
            if s_len > 0:
                sk = sub_keep[:, p:]
                gate = sk.unsqueeze(-1).to(sub.dtype)
                x = (enc.in_proj(sub[:, p:]) + enc.pos[p:n]) * gate
                r = torch.arange(s_len, device=x.device).unsqueeze(1)
                c = torch.arange(p + s_len, device=x.device).unsqueeze(0)
                causal = ((c < p) | (c - p <= r)).view(1, 1, s_len, p + s_len)
                for li, block in enumerate(enc.blocks):
                    h = block.norm1(x)
                    q, k, v = block.attn.qkv(h).chunk(3, dim=-1)
                    kk = torch.cat((keys[li].expand(m, p, -1), k), dim=1)
                    vv = torch.cat((vals[li].expand(m, p, -1), v), dim=1)
                    x = x + block.attn.proj(_sdpa(q, kk, vv, block.attn.n_heads, causal))
                    x = x + block.ff(block.norm2(x))
                    x = x * gate
                out[rows, p:] = (enc.norm(x) * gate).to(out_dtype)
            groups.append(_Group(rows=rows, prefix=p))
        return _History(encoded=out, keep=keep, groups=groups)

    # --- the query block, held cards only ---------------------------------

    def _summary(self, obs: ObsBatch, hist: "_History") -> Tensor:
        q = self.actor.query
        tokens, token_mask = q.build_tokens(obs)
        b, _n, d = tokens.shape
        # Kept tokens first, in their original order (a stable sort on
        # "dropped"), cut to the batch's widest row. The context token is
        # always kept, so it stays at position 0.
        order = torch.argsort((~token_mask).to(torch.int8), dim=1, stable=True)
        width = int(token_mask.sum(dim=1).max())
        idx = order[:, :width]
        x = torch.gather(tokens, 1, idx.unsqueeze(-1).expand(b, width, d))
        keep = torch.gather(token_mask, 1, idx)
        gate = keep.unsqueeze(-1).to(x.dtype)
        x = x * gate

        sink = q.mem_sink.unsqueeze(0).expand(b, 1, d)
        mem = q.mem_norm(torch.cat((sink.to(hist.encoded.dtype), hist.encoded), dim=1))
        mem_keep = torch.cat(
            (torch.ones(b, 1, dtype=torch.bool, device=x.device), hist.keep), dim=1
        )
        token_attn = keep[:, None, None, :]
        mem_attn = mem_keep[:, None, None, :]
        # Cross-attention keys and values per query layer; a shared
        # prefix's (the sink and its history) once per group.
        mem_kv = [_mem_kv(block.ca.kv, mem, hist.groups) for block in q.blocks]

        for block, (k, v) in zip(q.blocks[:-1], mem_kv[:-1]):
            x = x + block.sa(block.norm_sa(x), attn_mask=token_attn)
            ca = block.ca
            x = x + ca.proj(_sdpa(ca.q(block.norm_ca(x)), k, v, ca.n_heads, mem_attn))
            x = x + block.ff(block.norm_ff(x))
            x = x * gate
        x0 = self._last_block(q.blocks[-1], x, token_attn, mem_kv[-1], mem_attn)
        return q.out_norm(x0)

    @staticmethod
    def _last_block(block: Any, x: Tensor, token_attn: Tensor,
                    mem_kv: tuple[Tensor, Tensor], mem_attn: Tensor) -> Tensor:
        """``_QueryBlock.forward`` for the context token alone -> [B, d]."""
        d = x.shape[-1]
        h = block.norm_sa(x)
        if isinstance(block.sa.qkv, torch.nn.Linear):
            w, bias = block.sa.qkv.weight, block.sa.qkv.bias
            q_ctx = F.linear(h[:, _CONTEXT : _CONTEXT + 1], w[:d], bias[:d])
            k, v = F.linear(h, w[d:], bias[d:]).chunk(2, dim=-1)
        else:  # a quantized copy: no weight to slice
            q_all, k, v = block.sa.qkv(h).chunk(3, dim=-1)
            q_ctx = q_all[:, _CONTEXT : _CONTEXT + 1]
        x0 = x[:, _CONTEXT : _CONTEXT + 1]
        x0 = x0 + block.sa.proj(_sdpa(q_ctx, k, v, block.sa.n_heads, token_attn))
        ca = block.ca
        k_mem, v_mem = mem_kv
        x0 = x0 + ca.proj(_sdpa(ca.q(block.norm_ca(x0)), k_mem, v_mem, ca.n_heads, mem_attn))
        x0 = x0 + block.ff(block.norm_ff(x0))
        return x0[:, 0]

    # --- the scorer, on the candidates there are --------------------------

    def _score(self, summary: Tensor, obs: ObsBatch) -> Tensor:
        s = self.actor.scorer
        mask = obs.candidate_mask
        full = mask.shape[1]
        width = max(int(mask.sum(dim=1).max()), 1)
        candidates = obs.candidates[:, :width]
        live = mask[:, :width]
        basis = s.card_proj(obs.cards)
        emb = torch.bmm(candidates, basis)
        emb = s.norm(emb + s.scalar_mlp(s.candidate_scalars(candidates, obs.cards)))
        qv = s.score_q(summary).unsqueeze(-1)
        logits = torch.bmm(emb, qv).squeeze(-1) * (1.0 / math.sqrt(self.cfg.d_model))
        logits = (logits + s.cand_bias(emb).squeeze(-1)).float()
        logits = torch.where(live, logits, torch.full_like(logits, MASK_LOGIT))
        if width < full:
            pad = torch.full((logits.shape[0], full - width), MASK_LOGIT,
                             dtype=logits.dtype, device=logits.device)
            logits = torch.cat((logits, pad), dim=1)
        return logits


# A shared prefix shorter than this is not worth splitting the batch for.
_MIN_PREFIX = 8


class _Group:
    __slots__ = ("rows", "prefix")

    def __init__(self, rows: Tensor, prefix: int) -> None:
        self.rows = rows
        self.prefix = prefix


class _History:
    __slots__ = ("encoded", "keep", "groups")

    def __init__(self, encoded: Tensor, keep: Tensor, groups: list) -> None:
        self.encoded = encoded  # [B, n, d], n the batch's longest history
        self.keep = keep  # bool[B, n]
        self.groups = groups  # the rows that share a prefix, and its length


def _prefix_groups(tok: Tensor, keep: Tensor) -> list[Tensor]:
    """Rows grouped by their first token: the seat-relative features make
    rows seen from different seats differ from the first token on."""
    b = tok.shape[0]
    if b == 1:
        return [torch.zeros(1, dtype=torch.int64, device=tok.device)]
    first = torch.where(keep[:, :1], tok[:, 0], torch.full_like(tok[:, 0], float("nan")))
    _uniq, inverse = torch.unique(first.nan_to_num(-1e9), dim=0, return_inverse=True)
    return [torch.nonzero(inverse == g).squeeze(1) for g in range(int(inverse.max()) + 1)]


def _common_prefix(tok: Tensor, keep: Tensor) -> int:
    """Leading positions real and identical in every row."""
    same = ((tok == tok[:1]).all(dim=-1) & keep).all(dim=0)
    return int(same.to(torch.int64).cumprod(0).sum())


def _mem_kv(kv: Any, mem: Tensor, groups: list) -> tuple[Tensor, Tensor]:
    """``kv(mem)`` split in two, with each group's shared prefix (memory
    positions 0..prefix: the sink and the shared history) projected once."""
    if not groups:
        return kv(mem).chunk(2, dim=-1)
    b, length, d = mem.shape
    # The projection's own output dtype (bfloat16 under autocast).
    probe = kv(mem[:1, :1])
    out = torch.empty(b, length, 2 * d, dtype=probe.dtype, device=mem.device)
    covered = torch.zeros(b, dtype=torch.bool, device=mem.device)
    for g in groups:
        p = g.prefix + 1  # the sink leads the memory
        m = len(g.rows)
        out[g.rows, :p] = kv(mem[g.rows[:1], :p]).to(out.dtype).expand(m, p, -1)
        if p < length:
            out[g.rows, p:] = kv(mem[g.rows, p:]).to(out.dtype)
        covered[g.rows] = True
    rest = torch.nonzero(~covered).squeeze(1)
    if len(rest):
        out[rest] = kv(mem[rest]).to(out.dtype)
    k, v = out.chunk(2, dim=-1)
    return k, v
