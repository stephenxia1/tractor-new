"""Tests for the policy-value network.

Four of these are structural rather than statistical, and they are the
reason the model is written the way it is.

``test_candidate_masking_is_exact`` scrambles the contents of the
padding slots and demands every output back bit-identical. A network
that scored candidates with attention across slots would fail it, and
would fail it *quietly*: the leak is a few ulps at initialisation and
grows with training.

``test_history_is_causal`` edits one history token and checks nothing
before it moved. ``test_history_padding_is_inert`` fills the region
past the recorded length with noise and checks nothing moved at all.
Both are exact-equality tests; an approximate one would pass on a
model with a real leak.

``test_critic_gradient_does_not_reach_the_actor`` backpropagates a
critic loss and checks that no actor parameter has a gradient. The
detach that makes this true is one word in ``Critic.forward``, which is
exactly the kind of thing that gets refactored away.

Each is asserted in both directions where it could pass vacuously: a
change to a *valid* slot, or to a token *before* the cut, must move the
output.
"""

from __future__ import annotations

import numpy as np
import torch

from tractor.gen.episode import generate
from tractor.nn import model as M
from tractor.nn.features import (
    CONTEXT_DIM,
    D_CARD,
    HISTORY_DIM,
    NUM_PLANES,
    PLANE_HAND,
    Cache,
    encode,
)
from tractor.nn.model import (
    ActorCritic,
    Critic,
    ModelConfig,
    ObsBatch,
    TractorNet,
    analytic_flops,
    batch_observations,
    batch_privileged,
    count_parameters,
    dummy_batch,
    privileged_view,
)
from tractor.rules.cards import NUM_TYPES

DECKS = 2


def _net(seed: int = 0, cfg: ModelConfig | None = None) -> TractorNet:
    torch.manual_seed(seed)
    return TractorNet(cfg or ModelConfig()).eval()


def _full_batch(cfg: ModelConfig, batch: int, seed: int) -> tuple[ObsBatch, torch.Tensor]:
    """A dense batch: every history slot real, every candidate legal.

    The worst case for the masking tests is the *easy* case, so these
    start from a batch with nothing masked and mask it deliberately.
    """
    obs, priv = dummy_batch(
        cfg,
        batch=batch,
        seed=seed,
        history_len=cfg.max_history,
        n_candidates=cfg.max_candidates,
    )
    obs = obs._replace(
        history_len=torch.full((batch,), cfg.max_history, dtype=torch.int64),
        candidate_mask=torch.ones(batch, cfg.max_candidates, dtype=torch.bool),
    )
    return obs, priv


# --- 1. shapes and dtypes -------------------------------------------------


def test_shapes_and_dtypes() -> None:
    cfg = ModelConfig()
    net = _net(1, cfg)
    critic = Critic(cfg).eval()

    for batch in (1, 8):
        obs, priv = dummy_batch(cfg, batch=batch, seed=batch)
        assert obs.history.shape == (batch, cfg.max_history, HISTORY_DIM)
        assert obs.context.shape == (batch, CONTEXT_DIM)
        assert obs.cards.shape == (batch, NUM_TYPES, D_CARD)
        assert obs.planes.shape == (batch, NUM_PLANES, NUM_TYPES)
        assert obs.candidates.shape == (batch, cfg.max_candidates, NUM_TYPES)
        assert obs.candidate_mask.dtype == torch.bool
        assert obs.history_len.dtype == torch.int64

        with torch.no_grad():
            out = net(obs)
            value = critic(out.history_enc, obs.history_len, priv)

        expected = {
            "logits": (batch, cfg.max_candidates),
            "log_probs": (batch, cfg.max_candidates),
            "value": (batch,),
            "outcome_logits": (batch, M.NUM_OUTCOMES),
            "belief_logits": (batch, NUM_TYPES, M.NUM_BELIEF_LOCATIONS),
            "belief_log_probs": (batch, NUM_TYPES, M.NUM_BELIEF_LOCATIONS),
            "summary": (batch, cfg.d_model),
            "history_enc": (batch, cfg.max_history, cfg.d_model),
        }
        for name, shape in expected.items():
            tensor = getattr(out, name)
            assert tuple(tensor.shape) == shape, f"{name}: {tuple(tensor.shape)}"
            assert tensor.dtype == torch.float32, f"{name}: {tensor.dtype}"
            assert torch.isfinite(tensor).all(), f"{name} is not finite"

        assert tuple(value.shape) == (batch,)
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()

        # Probabilities are a distribution over the legal slots.
        probs = out.log_probs.exp()
        assert torch.allclose(probs.sum(-1), torch.ones(batch), atol=1e-5)
        belief = out.belief_log_probs.exp()
        assert torch.allclose(
            belief.sum(-1), torch.ones(batch, NUM_TYPES), atol=1e-5
        )
    print(f"    outputs checked at batch 1 and 8; {len(expected)} tensors each")


# --- 2. candidate masking -------------------------------------------------


def test_candidate_masking_is_exact() -> None:
    cfg = ModelConfig()
    net = _net(2, cfg)
    batch = 6
    obs, _priv = _full_batch(cfg, batch, seed=21)

    # Row i keeps i + 1 legal candidates, so row 0 is the forced-move
    # case that 14.8% of real decisions land in.
    mask = torch.zeros(batch, cfg.max_candidates, dtype=torch.bool)
    for i in range(batch):
        mask[i, : i + 1] = True
    obs = obs._replace(candidate_mask=mask)

    with torch.no_grad():
        base = net(obs)

    # (a) Invalid slots carry the sentinel and zero probability.
    assert torch.all(base.logits[~mask] == M.MASK_LOGIT)
    invalid = base.log_probs.exp()[~mask]
    assert invalid.max().item() < 1e-12, invalid.max().item()

    # (b) A row with one legal move puts all of its mass on it.
    assert base.log_probs[0, 0].exp().item() == 1.0
    assert base.log_probs[0, 0].item() == 0.0

    # (c) Scrambling what sits in the invalid slots changes nothing.
    rng = torch.Generator().manual_seed(7)
    scrambled = obs.candidates.clone()
    noise = torch.randint(
        0, DECKS + 1, scrambled[~mask].shape, generator=rng
    ).float()
    scrambled[~mask] = noise
    with torch.no_grad():
        moved = net(obs._replace(candidates=scrambled))
    for name in base._fields:
        a, b = getattr(base, name), getattr(moved, name)
        assert torch.equal(a, b), f"{name} moved when a padding slot changed"

    # (d) The same test run against a *valid* slot must fail, or (c) is
    #     measuring nothing.
    touched = obs.candidates.clone()
    touched[-1, 0] = torch.randint(0, DECKS + 1, (NUM_TYPES,), generator=rng).float()
    with torch.no_grad():
        other = net(obs._replace(candidates=touched))
    assert not torch.equal(base.logits, other.logits)
    print(
        "    padding scrambled: all 8 outputs bit-identical; "
        f"forced-move row p = {base.log_probs[0, 0].exp().item():.1f}"
    )


# --- 3. causality ---------------------------------------------------------


def test_history_is_causal() -> None:
    cfg = ModelConfig()
    net = _net(3, cfg)
    batch = 4
    obs, _priv = _full_batch(cfg, batch, seed=31)

    with torch.no_grad():
        base = net(obs)

    for cut in (1, 17, 64, cfg.max_history - 1):
        edited = obs.history.clone()
        torch.manual_seed(100 + cut)
        edited[:, cut] = torch.randn_like(edited[:, cut])
        with torch.no_grad():
            moved = net(obs._replace(history=edited))
        before = torch.equal(base.history_enc[:, :cut], moved.history_enc[:, :cut])
        assert before, f"token {cut} leaked backwards"
        at = base.history_enc[:, cut]
        assert not torch.equal(at, moved.history_enc[:, cut]), (
            f"token {cut} changed nothing at its own position"
        )
    print(f"    causal at cuts 1, 17, 64, {cfg.max_history - 1}")


# --- 4. history length masking -------------------------------------------


def test_history_padding_is_inert() -> None:
    cfg = ModelConfig()
    net = _net(4, cfg)
    critic = Critic(cfg).eval()
    batch = 5
    obs, priv = dummy_batch(cfg, batch=batch, seed=41)
    # Include the empty history: the opening lead of a hand has no
    # tokens at all, and the cross-attention would divide by an
    # all-masked softmax without the learned sink token.
    lengths = torch.tensor([0, 1, 12, 64, 99], dtype=torch.int64)
    obs = obs._replace(history_len=lengths)

    keep = torch.arange(cfg.max_history)[None, :] < lengths[:, None]
    with torch.no_grad():
        base = net(obs)
        base_v = critic(base.history_enc, obs.history_len, priv)

    torch.manual_seed(404)
    noisy = obs.history.clone()
    noisy[~keep] = torch.randn_like(noisy[~keep]) * 10.0
    with torch.no_grad():
        moved = net(obs._replace(history=noisy))
        moved_v = critic(moved.history_enc, obs.history_len, priv)

    for name in base._fields:
        a, b = getattr(base, name), getattr(moved, name)
        assert torch.equal(a, b), f"{name} moved when padding changed"
    assert torch.equal(base_v, moved_v), "critic moved when padding changed"

    # The encoder zeroes the padded region outright, which is what makes
    # a cached history safe to append to.
    assert torch.all(base.history_enc[~keep] == 0.0)

    # Non-vacuous: a real token still matters, including in the row
    # whose history is a single token.
    real = obs.history.clone()
    real[1, 0] = torch.randn(HISTORY_DIM)
    with torch.no_grad():
        other = net(obs._replace(history=real))
    assert not torch.equal(base.summary[1], other.summary[1])
    print(f"    lengths {lengths.tolist()}: summary, logits, heads and critic all fixed")


def test_history_truncation_is_exact() -> None:
    """Encoding only the longest real prefix changes no row's output.

    The encoder runs to the batch's longest history. A row must come
    out the same whether its batch forces the full width (a row of
    ``max_history`` tokens beside it) or lets it stop early.
    """
    cfg = ModelConfig()
    net = _net(5, cfg)
    critic = Critic(cfg).eval()
    obs, priv = dummy_batch(cfg, batch=4, seed=43)
    short = torch.tensor([0, 3, 17, 40], dtype=torch.int64)
    obs = obs._replace(history_len=short)
    full = torch.tensor([0, 3, 17, cfg.max_history], dtype=torch.int64)
    with torch.no_grad():
        a = net(obs)
        a_v = critic(a.history_enc, obs.history_len, priv)
        b = net(obs._replace(history_len=full))
        b_v = critic(b.history_enc, full, priv)
    for name in ("logits", "value", "summary", "belief_logits", "history_enc"):
        x, y = getattr(a, name)[:3], getattr(b, name)[:3]
        assert torch.allclose(x, y, atol=1e-5, rtol=1e-5), name
    assert torch.allclose(a_v[:3], b_v[:3], atol=1e-5, rtol=1e-5)
    assert a.history_enc.shape == b.history_enc.shape
    # An all-empty batch -- a burial's -- still runs and is zero.
    with torch.no_grad():
        e = net(obs._replace(history_len=torch.zeros(4, dtype=torch.int64)))
    assert torch.all(e.history_enc == 0.0)
    assert torch.isfinite(e.value).all()


# --- 5. gradients ---------------------------------------------------------


def test_gradients_reach_every_actor_parameter() -> None:
    cfg = ModelConfig()
    torch.manual_seed(5)
    net = TractorNet(cfg)
    batch = 4
    obs, _priv = _full_batch(cfg, batch, seed=51)
    # Hold every card in every row so the hand tokens, and with them the
    # card embedding, are never masked out of the graph.
    planes = obs.planes.clone()
    planes[:, PLANE_HAND] = 1.0
    obs = obs._replace(planes=planes)

    out = net(obs)
    loss = (
        out.log_probs.mean()
        + out.value.mean()
        + out.outcome_logits.mean()
        + out.belief_log_probs.mean()
    )
    loss.backward()

    missing = [
        name
        for name, p in net.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0
    ]
    assert not missing, f"no gradient at: {missing[:8]} ({len(missing)} total)"
    print(f"    {len(list(net.parameters()))} actor tensors, all with finite nonzero grad")


def test_critic_gradient_does_not_reach_the_actor() -> None:
    cfg = ModelConfig()
    torch.manual_seed(6)
    ac = ActorCritic(cfg)
    obs, priv = _full_batch(cfg, 4, seed=61)

    _out, value = ac(obs, priv, detach_critic=True)
    value.mean().backward()

    leaked = [n for n, p in ac.actor.named_parameters() if p.grad is not None]
    assert not leaked, f"critic loss reached the actor: {leaked[:8]}"
    starved = [n for n, p in ac.critic.named_parameters() if p.grad is None]
    assert not starved, f"critic parameters without gradient: {starved[:8]}"

    # Without the detach the path exists, so the detach is doing work.
    ac.zero_grad(set_to_none=True)
    _out, value = ac(obs, priv, detach_critic=False)
    value.mean().backward()
    attached = [n for n, p in ac.actor.named_parameters() if p.grad is not None]
    assert attached, "detach=False severed the path too, so detach=True proves nothing"
    # Only the history tower is upstream of the critic; the query block,
    # scorer and heads are not.
    assert all(n.startswith("history.") for n in attached), sorted(attached)[:8]
    print(
        f"    detached: 0 actor tensors touched; attached: {len(attached)}, "
        "all in the history tower"
    )


# --- 6. parameter and FLOP accounting ------------------------------------


def test_parameter_count_matches_the_estimate() -> None:
    cfg = ModelConfig()
    actor = TractorNet(cfg)
    critic = Critic(cfg)
    got_a = count_parameters(actor)
    got_c = count_parameters(critic)

    for got, want, what in (
        (got_a, M.ANALYTIC_ACTOR_PARAMS, "actor"),
        (got_c, M.ANALYTIC_CRITIC_PARAMS, "critic"),
    ):
        # Exact. The constant exists so that a structural change to the
        # model shows up here and has to be acknowledged; a tolerance
        # wide enough to absorb a whole projection defeats that, and
        # did: the actor estimate sat one kitty_proj (21,120 parameters,
        # 0.2%) below the model and every run passed.
        assert got == want, (
            f"{what}: {got:,} parameters, estimate says {want:,} "
            f"(difference {got - want:+,}). If the model changed on "
            f"purpose, update the constant and its derivation comment."
        )

    parts = {
        "history": count_parameters(actor.history),
        "query": count_parameters(actor.query),
        "scorer": count_parameters(actor.scorer),
        "heads": count_parameters(actor.heads),
    }
    assert sum(parts.values()) == got_a
    print(
        f"    actor {got_a:,} (estimate {M.ANALYTIC_ACTOR_PARAMS:,}), "
        f"critic {got_c:,} (estimate {M.ANALYTIC_CRITIC_PARAMS:,})"
    )
    print("    " + "  ".join(f"{k} {v:,}" for k, v in parts.items()))


def test_analytic_flops_matches_hand_arithmetic() -> None:
    """A configuration small enough to count by hand, counted by hand.

    Everything below is written out from the architecture rather than
    from the implementation's helpers, so an error in one will not
    reproduce itself in the other. Constants: 2 FLOPs per
    multiply-accumulate, plus one add per bias element; 7 per LayerNorm
    element, 8 per GELU, 5 per softmax element.
    """
    cfg = ModelConfig(
        d_model=8,
        n_layers_history=1,
        n_layers_query=1,
        n_heads=2,
        d_ff=16,
        critic_d=4,
        critic_layers=1,
        max_history=4,
        max_candidates=2,
        critic_heads=1,
        critic_d_ff=8,
        belief_d=4,
        scalar_d=4,
        head_d=4,
    )
    got = analytic_flops(cfg, batch=1, n_candidates=2, history_len=4)

    # -- history encoder: L = 4, d = 8, ff = 16, 2 heads, 1 block ------
    #    causal attention over the lower triangle: 4 * 5 / 2 = 10 pairs
    history = (
        (4 * 2 * 37 * 8 + 4 * 8)  # in_proj, with bias
        + 4 * 8  # + positional
        + 4 * 8  # * keep gate
        # block:
        + 4 * 8 * 7  # LayerNorm
        + (4 * 2 * 8 * 24 + 4 * 24)  # qkv
        + (2 * 10 * 8 + 2 * 10 * 8 + 10 * 2 + 5 * 10 * 2)  # QK, AV, scale, softmax
        + (4 * 2 * 8 * 8 + 4 * 8)  # attention out
        + 4 * 8  # residual
        + 4 * 8 * 7  # LayerNorm
        + (4 * 2 * 8 * 16 + 4 * 16)  # ff1
        + 8 * 4 * 16  # GELU
        + (4 * 2 * 16 * 8 + 4 * 8)  # ff2
        + 4 * 8  # residual
        + 4 * 8  # * keep gate
        + 4 * 8 * 7  # final LayerNorm
        + 4 * 8  # * keep gate
    )
    assert got["history"] == history, (got["history"], history)

    # -- query block: 61 tokens, memory 4 + 1 sink = 5 -----------------
    tokens = (
        (2 * 37 * 8 + 8)  # context token
        + (2 * 54 * 8 + 8)  # unseen token
        + (2 * 54 * 8 + 8)  # the seat's own buried kitty
        + (4 * 2 * 108 * 8 + 4 * 8)  # four seat tokens from two planes each
        + 4 * 8  # + seat embedding
        + (54 * 2 * 30 * 8 + 54 * 8)  # 54 hand tokens
        + 54 * 8  # + card embedding
        + 61 * 8  # * held gate
        + 5 * 8 * 7  # LayerNorm over the memory
    )
    layer = (
        # self-attention, 61 x 61 = 3721 pairs
        61 * 8 * 7
        + (61 * 2 * 8 * 24 + 61 * 24)
        + (2 * 3721 * 8 + 2 * 3721 * 8 + 3721 * 2 + 5 * 3721 * 2)
        + (61 * 2 * 8 * 8 + 61 * 8)
        + 61 * 8
        # cross-attention, 61 x 5 = 305 pairs
        + 61 * 8 * 7
        + (61 * 2 * 8 * 8 + 61 * 8)  # q
        + (5 * 2 * 8 * 16 + 5 * 16)  # k and v over the memory
        + (2 * 305 * 8 + 2 * 305 * 8 + 305 * 2 + 5 * 305 * 2)
        + (61 * 2 * 8 * 8 + 61 * 8)  # out
        + 61 * 8
        # feed-forward
        + 61 * 8 * 7
        + (61 * 2 * 8 * 16 + 61 * 16)
        + 8 * 61 * 16
        + (61 * 2 * 16 * 8 + 61 * 8)
        + 61 * 8
        + 61 * 8  # * held gate
    )
    query = tokens + layer + 1 * 8 * 7  # out_norm on token 0
    assert got["query"] == query, (got["query"], query)

    # -- scorer: 2 candidate slots -------------------------------------
    scorer = (
        54 * 2 * 30 * 8  # card_proj, no bias
        + 2 * 2 * 54 * 8  # counts @ basis
        + 2 * 54 * 30 * 3  # cards @ scalar basis
        + 2 * 2 * 54 * 3  # counts @ per-card scalars
        + 2 * 12  # deriving the six scalars
        + (2 * 2 * 6 * 4 + 2 * 4)  # scalar fc1
        + 8 * 2 * 4  # GELU
        + (2 * 2 * 4 * 8 + 2 * 8)  # scalar fc2
        + 2 * 8  # + into the embedding
        + 2 * 8 * 7  # LayerNorm
        + (2 * 8 * 8 + 8)  # score_q on the summary
        + 2 * 2 * 8  # bilinear dot
        + 2  # * scale
        + (2 * 2 * 8 * 1 + 2)  # candidate-only bias term
        + 2  # + it
        + 2  # masked overwrite
        + 5 * 2  # log_softmax
    )
    assert got["scorer"] == scorer, (got["scorer"], scorer)

    # -- heads ----------------------------------------------------------
    heads = (
        (2 * 8 * 4 + 4) + 8 * 4 + (2 * 4 * 1 + 1)  # value
        + (2 * 8 * 4 + 4) + 8 * 4 + (2 * 4 * 7 + 7)  # outcome
        + (2 * 8 * 4 + 4)  # W_a @ h, once
        + 54 * 2 * 30 * 4  # W_b @ cards, once, no bias
        + 54 * 4  # broadcast add
        + 8 * 54 * 4  # GELU
        + (54 * 2 * 4 * 5 + 54 * 5)  # belief_out
        + 5 * 54 * 5  # log_softmax over locations
    )
    assert got["heads"] == heads, (got["heads"], heads)

    # -- critic: width 4, 1 head, ff 8, sequence 4 + 1 CLS = 5 ----------
    critic = (
        (4 * 2 * 8 * 4 + 4 * 4)  # in_proj from d_model
        + 4 * 4  # + positional
        + 4 * 4  # * keep gate
        # block over 5 tokens, non-causal: 25 pairs
        + 5 * 4 * 7
        + (5 * 2 * 4 * 12 + 5 * 12)
        + (2 * 25 * 4 + 2 * 25 * 4 + 25 * 1 + 5 * 25 * 1)
        + (5 * 2 * 4 * 4 + 5 * 4)
        + 5 * 4
        + 5 * 4 * 7
        + (5 * 2 * 4 * 8 + 5 * 8)
        + 8 * 5 * 8
        + (5 * 2 * 8 * 4 + 5 * 4)
        + 5 * 4
        + 5 * 4  # * keep gate
        + 1 * 4 * 7  # LayerNorm on the pooled CLS
        + (2 * 270 * 8 + 8) + 8 * 8 + (2 * 8 * 4 + 4) + 8 * 4  # privileged MLP
        + (2 * CONTEXT_DIM * 8 + 8) + 8  # context_proj, added into layer one
        + (2 * 8 * 4 + 4) + 8 * 4 + (2 * 4 * 1 + 1)  # head
    )
    assert got["critic"] == critic, (got["critic"], critic)

    assert got["actor"] == history + query + scorer + heads
    assert got["total"] == got["actor"] + critic

    # Scaling: linear in the batch, and the query block is linear in the
    # candidate count only through the scorer.
    doubled = analytic_flops(cfg, batch=2, n_candidates=2, history_len=4)
    assert doubled["total"] == 2 * got["total"]
    print(f"    tiny config total {got['total']:,} FLOPs, matched term by term")


def test_analytic_flops_matches_a_matmul_counter() -> None:
    """Cross-check the matmul terms against torch's own FLOP counter.

    ``FlopCounterMode`` sees ``mm``/``bmm``/``addmm`` and nothing else,
    so the comparison is against ``matmul_only=True``. Attention is
    forced onto the math backend for two reasons: the fused CPU kernel
    is a single op the counter has no formula for, and the math path
    materialises the full L x L scores, which is the
    ``causal_discount=False`` convention.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as exc:  # pragma: no cover - older torch
        print(f"    skipped: {exc}")
        return

    cfg = ModelConfig()
    torch.manual_seed(8)
    net = ActorCritic(cfg).eval()
    batch = 2
    obs, priv = _full_batch(cfg, batch, seed=81)

    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        counter = FlopCounterMode(display=False)
        with counter:
            net(obs, priv)
        measured = counter.get_total_flops()

    predicted = analytic_flops(
        cfg,
        batch,
        cfg.max_candidates,
        cfg.max_history,
        matmul_only=True,
        causal_discount=False,
    )["total"]
    assert measured == predicted, f"counted {measured}, predicted {predicted}"

    # And the causal discount is a real saving, not a rounding.
    discounted = analytic_flops(
        cfg, batch, cfg.max_candidates, cfg.max_history, matmul_only=True
    )["total"]
    assert discounted < predicted
    print(f"    torch counted {measured:,} matmul FLOPs, analytic predicted the same")


# --- 7. a real position ---------------------------------------------------


def test_forward_on_a_real_position() -> None:
    cfg = ModelConfig()
    net = _net(9, cfg)
    critic = Critic(cfg).eval()

    rng = np.random.default_rng(20260921)
    observations: list = []
    candidates: list = []
    privileged: list = []
    forced = 0

    for _ in range(3):
        state, _info = generate(rng, DECKS)
        cache = Cache()
        while not state.done and len(observations) < 24:
            seat = state.to_act
            moves = state.legal_moves(cap=64)
            observations.append(encode(state, seat, cache))
            candidates.append(moves.moves)
            privileged.append(privileged_view(state.hands, state.kitty, seat))
            forced += int(len(moves) == 1)
            state.step(moves.moves[int(rng.integers(len(moves.moves)))])
        if len(observations) >= 24:
            break

    obs = batch_observations(observations, candidates, cfg)
    priv = batch_privileged(privileged)
    with torch.no_grad():
        out = net(obs)
        value = critic(out.history_enc, obs.history_len, priv)

    for name in out._fields:
        tensor = getattr(out, name)
        assert torch.isfinite(tensor).all(), f"{name} is not finite"
    assert torch.isfinite(value).all()

    n_legal = obs.candidate_mask.sum(-1)
    assert int(n_legal.min()) >= 1
    probs = out.log_probs.exp()
    assert torch.allclose(probs.sum(-1), torch.ones(obs.batch), atol=1e-5)
    assert torch.all(probs[~obs.candidate_mask] == 0.0)
    # Every row's mass sits on slots the enumerator actually filled.
    assert torch.allclose(
        (probs * obs.candidate_mask).sum(-1), torch.ones(obs.batch), atol=1e-5
    )
    # The first decision of a hand has an empty history.
    assert int(obs.history_len.min()) == 0

    print(
        f"    {obs.batch} real decisions, history {int(obs.history_len.min())}"
        f"..{int(obs.history_len.max())}, candidates {int(n_legal.min())}"
        f"..{int(n_legal.max())}, {forced} forced"
    )


# --- reference numbers ----------------------------------------------------


def test_report_default_flops() -> None:
    """Not an assertion so much as the number the benchmark divides by."""
    cfg = ModelConfig()
    flops = analytic_flops(cfg, batch=256, n_candidates=128, history_len=99)
    for key in ("history", "query", "scorer", "heads", "actor", "critic", "total"):
        assert flops[key] > 0
    assert flops["total"] == flops["actor"] + flops["critic"]
    order = sorted(
        ("history", "query", "scorer", "heads", "critic"),
        key=lambda k: -flops[k],
    )
    print("    batch 256, 128 candidates, history 99:")
    for key in order:
        print(f"      {key:<8} {flops[key] / 1e9:8.2f} GFLOP")
    print(f"      {'total':<8} {flops['total'] / 1e9:8.2f} GFLOP")


def main() -> int:
    import time

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        started = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - reporting harness
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}"[:1500])
        else:
            print(f"ok    {fn.__name__}  ({time.time() - started:.1f}s)")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_critic_context_starts_inert_then_matters() -> None:
    cfg = ModelConfig()
    torch.manual_seed(6)
    critic = Critic(cfg).eval()
    obs, priv = dummy_batch(cfg, batch=3, seed=47)
    enc = torch.zeros(3, cfg.max_history, cfg.d_model)
    lengths = torch.zeros(3, dtype=torch.int64)
    with torch.no_grad():
        without = critic(enc, lengths, priv)
        with_ctx = critic(enc, lengths, priv, context=obs.context)
    assert torch.equal(without, with_ctx), "a zero-initialised context moved the critic"
    with torch.no_grad():
        critic.context_proj.weight.normal_()
        moved = critic(enc, lengths, priv, context=obs.context)
        other = critic(enc, lengths, priv, context=obs.context.flip(0))
    assert not torch.allclose(moved, other), "the critic ignores its context"


def test_critic_empty_history_fast_path_is_exact() -> None:
    """With no history the pooled vector is computed once and broadcast;
    it must equal what the full sequence computes."""
    cfg = ModelConfig()
    torch.manual_seed(8)
    critic = Critic(cfg).eval()
    with torch.no_grad():
        critic.context_proj.weight.normal_(0, 0.1)
    obs, priv = dummy_batch(cfg, batch=5, seed=51)
    enc = torch.randn(5, cfg.max_history, cfg.d_model)
    zero = torch.zeros(5, dtype=torch.int64)
    one_long = torch.tensor([0, 0, 0, 0, 1], dtype=torch.int64)
    with torch.no_grad():
        fast = critic(enc, zero, priv, context=obs.context)
        # A batch with one real token takes the full path; its first four
        # rows have empty histories and must agree with the fast path.
        full = critic(enc, one_long, priv, context=obs.context)
    assert torch.allclose(fast[:4], full[:4], atol=1e-6), (fast[:4], full[:4])
