"""The landlord's burial, chosen by the value network.

After picking up the kitty the landlord holds 33 cards and puts 8 back
face-down (rules.md §6). This picks the 8 by asking the actor's own
value head -- the imperfect-information one, which sees exactly what
the landlord sees: its hand, its burial, the trump context -- how good
the resulting play-start position is, and climbing.

Why swaps between complete burials rather than removing one card at a
time: the value head has only ever been shown 25-card hands. A greedy
removal passes through 32-, 31-, ... card hands, positions it has no
opinion about. A swap of one kept card for one buried card moves from
one legal play start to another, so every position scored is one the
network was trained on.

The search starts from a plain heuristic burial (low, pointless,
off-trump, short-suited cards first) and takes the best improving swap
each round until none improves. Every hand's candidates for a round go
through one batched forward, and many hands are buried at once.

Two things make a candidate cheap. At play start nothing has been played,
so the context vector, the unseen counts and every card feature but the
held count are the same for every burial of one hand: a candidate is the
base observation with its hand plane, kitty plane and one card-feature
column rewritten. And the history is empty, so the history encoder is
skipped and the query block reads a single masked memory slot -- the
same computation the full forward does for an empty history
(``tests/test_bury.py`` checks it), without the 128-position tensors.

Nothing here reads another seat's cards. The other three hands in the
base state are filled from the landlord's unseen cards in a fixed order,
and the landlord's observation depends only on their sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from ..nn.features import (
    D_STATIC,
    DYN_COUNT,
    NUM_PLANES,
    PLANE_HAND,
    PLANE_KITTY,
    Cache,
    encode,
)
from ..nn.model import ObsBatch
from ..rules.cards import NUM_TYPES, POINTS, Ctx, full_deck, rank_of
from .state import NUM_SEATS, PlayState

__all__ = [
    "BuryConfig",
    "heuristic_burial",
    "random_burial",
    "play_start",
    "value_burials",
]

KITTY_SIZE = 8

# The held-count column of a card-feature row: the only card feature a
# burial changes at play start.
_COUNT_COL = D_STATIC + DYN_COUNT


@dataclass(frozen=True)
class BuryConfig:
    # Swap rounds at most. Each is one batched forward over every hand
    # still improving.
    rounds: int = 8
    # Kept card types considered for a swap each round, lowest heuristic
    # keep-score first. 0 considers every kept type.
    consider: int = 0
    # An improvement smaller than this, in value units, ends the climb.
    tol: float = 1e-4
    # Rows per forward. A candidate row is ~60 query tokens.
    chunk: int = 4096
    # Hands climbed together. A round's candidates for a group are one
    # array of kitties -- 54 bytes each, up to ~200 a hand -- and the
    # forward over them is chunked separately.
    group: int = 256
    # What scores a candidate. "actor": the actor's own value head from
    # the landlord's seat. "critic": the privileged critic averaged over
    # ``worlds`` deals of the landlord's unseen cards, sampled once per
    # hand and shared by every candidate of it. At play start nothing has
    # been played, so a uniform deal of the other 75 cards is exactly
    # what the landlord knows (declarations aside), and the critic reads
    # only the hands, the burial and the context -- no actor forward.
    #
    # Measured at v4-full step 2400 against smart-v18/v28/v34: the actor
    # climb won 0.52, its own heuristic starting point 0.544, and 0.45
    # against 0.50 in the hands the network was landlord. Taking the best
    # of ~100 noisy estimates picks the burials the estimate overrates.
    evaluator: str = "actor"
    worlds: int = 8


def _keep_scores(pool: np.ndarray, ctx: Ctx) -> np.ndarray:
    """float[54]: how much the landlord wants to keep each card type.

    Only a starting point for the climb, so only roughly right: trumps
    and high cards are kept, points are kept rather than exposed to the
    kitty bonus, pairs are kept, and short plain suits are offered up
    so they can be voided.
    """
    score = np.zeros(NUM_TYPES, dtype=np.float64)
    suit_len = np.zeros(5, dtype=np.int64)
    for t in np.nonzero(pool)[0]:
        suit_len[ctx.eff[t]] += int(pool[t])
    for t in range(NUM_TYPES):
        if ctx.is_trump[t]:
            score[t] = 100.0 + float(ctx.strength[t])
            continue
        score[t] = float(rank_of(t)) + 2.0 * float(POINTS[t])
        if pool[t] >= 2:
            score[t] += 6.0
        short = max(0, 6 - int(suit_len[ctx.eff[t]]))
        score[t] -= 2.0 * short
    return score


def heuristic_burial(
    pool: np.ndarray, ctx: Ctx, kitty_size: int = KITTY_SIZE
) -> np.ndarray:
    """int8[54]: the ``kitty_size`` lowest-scoring cards of ``pool``."""
    score = _keep_scores(pool, ctx)
    cards = np.repeat(np.arange(NUM_TYPES), pool.astype(np.int64))
    order = np.argsort(score[cards], kind="stable")
    return np.bincount(cards[order[:kitty_size]], minlength=NUM_TYPES).astype(np.int8)


def random_burial(
    pool: np.ndarray, rng: np.random.Generator, kitty_size: int = KITTY_SIZE
) -> np.ndarray:
    """int8[54]: a uniform ``kitty_size``-card subset of ``pool``."""
    cards = np.repeat(np.arange(NUM_TYPES), pool.astype(np.int64))
    picked = rng.choice(cards, size=kitty_size, replace=False)
    return np.bincount(picked, minlength=NUM_TYPES).astype(np.int8)


def play_start(
    ctx: Ctx,
    hands: np.ndarray,
    pool: np.ndarray,
    burial: np.ndarray,
    landlord: int,
    num_decks: int = 2,
) -> PlayState:
    """The play-start position once ``landlord`` has buried ``burial``.

    ``hands`` are the four dealt hands; the landlord's row is replaced by
    ``pool - burial`` (``pool`` being its hand plus the kitty it took).
    """
    out = hands.astype(np.int8).copy()
    out[landlord] = (pool - burial).astype(np.int8)
    return PlayState.new(ctx, out, burial.astype(np.int8), landlord, num_decks)


def _landlord_view_state(
    ctx: Ctx, pool: np.ndarray, burial: np.ndarray, landlord: int, num_decks: int
) -> PlayState:
    """A play start that only the landlord's view of is ever read.

    The other hands hold the landlord's unseen cards in a fixed order,
    so nothing about the real deal can reach the observation.
    """
    unseen = (full_deck(num_decks) - pool.astype(np.int64)).astype(np.int64)
    cards = np.repeat(np.arange(NUM_TYPES), unseen)
    size = int(pool.sum()) - int(burial.sum())
    hands = np.zeros((NUM_SEATS, NUM_TYPES), dtype=np.int8)
    at = 0
    for s in range(NUM_SEATS):
        if s == landlord:
            continue
        hands[s] = np.bincount(cards[at : at + size], minlength=NUM_TYPES)
        at += size
    return play_start(ctx, hands, pool, burial, landlord, num_decks)


@dataclass
class _Base:
    """One hand's observation parts that no burial changes."""

    context: np.ndarray  # float32[CONTEXT_DIM]
    unseen: np.ndarray  # float32[54]
    cards: np.ndarray  # float32[54, D_CARD], count column to be rewritten
    num_decks: int


def _base(
    ctx: Ctx, pool: np.ndarray, landlord: int, num_decks: int, cache: Cache
) -> _Base:
    state = _landlord_view_state(ctx, pool, heuristic_burial(pool, ctx), landlord, num_decks)
    obs = encode(state, landlord, cache)
    # Through float16 and back, as every observation the network was
    # trained on went (``rollout._pack``), so a candidate is scored on
    # exactly the inputs a real play start would present.
    def f16(a: np.ndarray) -> np.ndarray:
        return a.astype(np.float16).astype(np.float32)

    return _Base(
        context=f16(obs.context),
        unseen=f16(obs.unseen),
        cards=f16(obs.cards),
        num_decks=num_decks,
    )


def _rows(
    bases: Sequence[_Base], owner: np.ndarray, hands: np.ndarray, kitties: np.ndarray
) -> dict[str, np.ndarray]:
    """Observation arrays for candidate rows. ``owner[i]`` indexes ``bases``."""
    n = len(owner)
    context = np.stack([bases[o].context for o in owner])
    unseen = np.stack([bases[o].unseen for o in owner])
    cards = np.stack([bases[o].cards for o in owner])
    decks = np.array([bases[o].num_decks for o in owner], dtype=np.float32)
    cards[:, :, _COUNT_COL] = hands.astype(np.float32) / decks[:, None]
    planes = np.zeros((n, NUM_PLANES, NUM_TYPES), dtype=np.float32)
    planes[:, PLANE_HAND] = hands
    planes[:, PLANE_KITTY] = kitties
    return {"context": context, "unseen": unseen, "cards": cards, "planes": planes}


def _actor(net: Any) -> Any:
    return getattr(net, "actor", net)


def play_start_values(
    net: Any, rows: dict[str, np.ndarray], device: Any, chunk: int = 4096
) -> np.ndarray:
    """float32[N]: the actor's value head on empty-history positions."""
    actor = _actor(net)
    cfg = actor.cfg
    out = np.empty(len(rows["context"]), dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, len(out), chunk):
            hi = min(lo + chunk, len(out))
            b = hi - lo

            def t(name: str) -> torch.Tensor:
                return torch.from_numpy(np.ascontiguousarray(rows[name][lo:hi])).to(device)

            obs = ObsBatch(
                history=torch.zeros(b, 1, 1, device=device),
                history_len=torch.zeros(b, dtype=torch.int64, device=device),
                context=t("context"),
                unseen=t("unseen"),
                planes=t("planes"),
                cards=t("cards"),
                candidates=torch.zeros(b, 1, NUM_TYPES, device=device),
                candidate_mask=torch.ones(b, 1, dtype=torch.bool, device=device),
            )
            # An empty history: one memory slot, masked, behind the sink.
            enc = torch.zeros(b, 1, cfg.d_model, device=device)
            keep = torch.zeros(b, 1, dtype=torch.bool, device=device)
            summary = actor.query(obs, enc, keep)
            out[lo:hi] = actor.heads.value(summary).squeeze(-1).float().cpu().numpy()
    return out


class _DeviceBases:
    """Every hand's unchanging observation parts, once, on the device.

    A candidate row is then an index into these plus its hand and kitty
    counts -- 108 bytes crossing to the device instead of the ~8 KB a
    full row is. Building the rows on the host copied the card-feature
    block once per candidate, about 10 GB a step at training scale, and
    was most of what a burial cost.
    """

    def __init__(self, bases: Sequence[_Base], pools: np.ndarray, device: Any,
                 landlords: Sequence[int] | None = None, worlds: int = 0,
                 rng: np.random.Generator | None = None) -> None:
        dev = torch.device(device)
        self.device = dev
        # For the critic: ``worlds`` deals of each hand's unseen cards to
        # the three other seats, in the landlord's relative order.
        self.others = None
        if worlds > 0:
            rng = rng or np.random.default_rng(0)
            n = len(bases)
            others = np.zeros((n, worlds, 3, NUM_TYPES), dtype=np.float32)
            for i in range(n):
                unseen = full_deck(bases[i].num_decks).astype(np.int64) - pools[i]
                cards = np.repeat(np.arange(NUM_TYPES), unseen)
                size = len(cards) // 3
                for w in range(worlds):
                    rng.shuffle(cards)
                    for s in range(3):
                        others[i, w, s] = np.bincount(
                            cards[s * size : (s + 1) * size], minlength=NUM_TYPES
                        )
            self.others = torch.from_numpy(others).to(dev)
        self.context = torch.from_numpy(np.stack([b.context for b in bases])).to(dev)
        self.unseen = torch.from_numpy(np.stack([b.unseen for b in bases])).to(dev)
        self.cards = torch.from_numpy(np.stack([b.cards for b in bases])).to(dev)
        self.decks = torch.tensor([float(b.num_decks) for b in bases], device=dev)
        self.pools = torch.from_numpy(pools.astype(np.float32)).to(dev)

    def critic_values(self, net: Any, owner: np.ndarray, kitties: np.ndarray,
                      chunk: int) -> np.ndarray:
        """float32[N]: the critic's value, averaged over the sampled deals."""
        critic = net.critic
        cfg = net.cfg
        dev = self.device
        assert self.others is not None
        worlds = self.others.shape[1]
        own_all = torch.from_numpy(owner).to(dev)
        kit_all = torch.from_numpy(kitties.astype(np.float32)).to(dev)
        out = np.empty(len(owner), dtype=np.float32)
        per = max(1, chunk // worlds)
        with torch.no_grad():
            for lo in range(0, len(owner), per):
                hi = min(lo + per, len(owner))
                b = hi - lo
                own = own_all[lo:hi]
                kit = kit_all[lo:hi]
                hand = self.pools[own] - kit
                priv = torch.empty(b, worlds, 5, NUM_TYPES, device=dev)
                priv[:, :, 0] = hand[:, None]
                priv[:, :, 1:4] = self.others[own]
                priv[:, :, 4] = kit[:, None]
                rows = b * worlds
                # Empty histories: the critic's fast path reads only the
                # batch size from this, so one position is enough.
                v = critic(
                    torch.zeros(rows, 1, cfg.d_model, device=dev),
                    torch.zeros(rows, dtype=torch.int64, device=dev),
                    priv.reshape(rows, 5, NUM_TYPES),
                    context=self.context[own].repeat_interleave(worlds, dim=0),
                )
                out[lo:hi] = v.float().reshape(b, worlds).mean(1).cpu().numpy()
        return out

    def values(self, net: Any, owner: np.ndarray, kitties: np.ndarray, chunk: int) -> np.ndarray:
        actor = _actor(net)
        cfg = actor.cfg
        dev = self.device
        own_all = torch.from_numpy(owner).to(dev)
        kit_all = torch.from_numpy(kitties.astype(np.float32)).to(dev)
        out = np.empty(len(owner), dtype=np.float32)
        # TF32 matmuls for this forward only. Measured on an L40S: plain
        # fp32 took 9.7 s to bury 810 hands in a training step, against
        # about 7 s with TF32 in the pipeline benchmark. bf16 would be
        # faster again, but its rounding is the size of one swap's gain.
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            self._values_into(actor, cfg, dev, own_all, kit_all, out, chunk)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        return out

    def _values_into(self, actor: Any, cfg: Any, dev: Any, own_all: Any,
                     kit_all: Any, out: np.ndarray, chunk: int) -> None:
        with torch.no_grad():
            for lo in range(0, len(own_all), chunk):
                hi = min(lo + chunk, len(own_all))
                b = hi - lo
                own = own_all[lo:hi]
                kit = kit_all[lo:hi]
                hand = self.pools[own] - kit
                cards = self.cards[own].clone()
                cards[:, :, _COUNT_COL] = hand / self.decks[own, None]
                planes = torch.zeros(b, NUM_PLANES, NUM_TYPES, device=dev)
                planes[:, PLANE_HAND] = hand
                planes[:, PLANE_KITTY] = kit
                obs = ObsBatch(
                    history=torch.zeros(b, 1, 1, device=dev),
                    history_len=torch.zeros(b, dtype=torch.int64, device=dev),
                    context=self.context[own],
                    unseen=self.unseen[own],
                    planes=planes,
                    cards=cards,
                    candidates=torch.zeros(b, 1, NUM_TYPES, device=dev),
                    candidate_mask=torch.ones(b, 1, dtype=torch.bool, device=dev),
                )
                enc = torch.zeros(b, 1, cfg.d_model, device=dev)
                keep = torch.zeros(b, 1, dtype=torch.bool, device=dev)
                summary = actor.query(obs, enc, keep)
                out[lo:hi] = actor.heads.value(summary).squeeze(-1).float().cpu().numpy()


def _swaps(burial: np.ndarray, pool: np.ndarray, kept_types: np.ndarray) -> np.ndarray:
    """int8[M, 54]: every burial one swap away (a buried card for a kept one)."""
    buried = np.nonzero(burial)[0]
    bi, ki = np.meshgrid(buried, kept_types, indexing="ij")
    bi, ki = bi.ravel(), ki.ravel()
    keep = bi != ki
    bi, ki = bi[keep], ki[keep]
    out = np.repeat(burial[None].astype(np.int8), len(bi), axis=0)
    rows = np.arange(len(bi))
    out[rows, bi] -= 1
    out[rows, ki] += 1
    return out


def value_burials(
    net: Any,
    pools: Sequence[np.ndarray],
    ctxs: Sequence[Ctx],
    landlords: Sequence[int],
    device: Any = "cpu",
    bury: BuryConfig | None = None,
    cache: Cache | None = None,
    num_decks: int = 2,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Bury for many landlords at once.

    ``pools[i]`` is landlord ``i``'s 33 cards (hand plus kitty). Returns
    each burial and summary statistics: the mean value the climb started
    and ended at, and the rounds it took.
    """
    bury = bury or BuryConfig()
    cache = cache or Cache()
    n = len(pools)
    if n > bury.group:
        burials: list[np.ndarray] = []
        parts: list[dict[str, float]] = []
        for lo in range(0, n, bury.group):
            hi = min(lo + bury.group, n)
            b, st = value_burials(
                net, pools[lo:hi], ctxs[lo:hi], landlords[lo:hi],
                device, bury, cache, num_decks,
            )
            burials.extend(b)
            parts.append({**st, "n": hi - lo})
        total = sum(p["n"] for p in parts)
        return burials, {
            "value_start": sum(p["value_start"] * p["n"] for p in parts) / total,
            "value_end": sum(p["value_end"] * p["n"] for p in parts) / total,
            "rounds": max(p["rounds"] for p in parts),
        }
    if n == 0:
        return [], {"value_start": 0.0, "value_end": 0.0, "rounds": 0.0}
    pool_arr = np.stack([np.asarray(p, dtype=np.int8) for p in pools])
    use_critic = bury.evaluator == "critic"
    bases = _DeviceBases(
        [_base(ctxs[i], pool_arr[i], landlords[i], num_decks, cache) for i in range(n)],
        pool_arr, device, landlords,
        worlds=bury.worlds if use_critic else 0,
        rng=np.random.default_rng(int(pool_arr.sum(axis=1)[0]) * 7919 + n),
    )
    score = bases.critic_values if use_critic else bases.values
    keep_scores = [_keep_scores(pool_arr[i], ctxs[i]) for i in range(n)]
    current = [heuristic_burial(pool_arr[i], ctxs[i]) for i in range(n)]
    value = np.full(n, -np.inf, dtype=np.float64)
    start_value = np.zeros(n, dtype=np.float64)
    live = list(range(n))
    rounds = 0

    for r in range(bury.rounds + 1):
        if not live:
            break
        owners: list[np.ndarray] = []
        kitties: list[np.ndarray] = []
        spans: list[tuple[int, int]] = []
        at = 0
        for i in live:
            b = current[i]
            rows = b[None].astype(np.int8)
            if r < bury.rounds:
                kept = np.nonzero(pool_arr[i] - b)[0]
                if bury.consider > 0 and len(kept) > bury.consider:
                    kept = kept[np.argsort(keep_scores[i][kept], kind="stable")[: bury.consider]]
                rows = np.concatenate((rows, _swaps(b, pool_arr[i], kept)))
            owners.append(np.full(len(rows), i, dtype=np.int64))
            kitties.append(rows)
            spans.append((at, at + len(rows)))
            at += len(rows)
        all_k = np.concatenate(kitties)
        v = score(net, np.concatenate(owners), all_k, bury.chunk)
        rounds = r
        still: list[int] = []
        for i, (lo, hi) in zip(live, spans):
            here = v[lo]
            if r == 0:
                start_value[i] = here
            value[i] = here
            if hi - lo > 1:
                best = lo + 1 + int(np.argmax(v[lo + 1 : hi]))
                if v[best] > here + bury.tol:
                    current[i] = all_k[best].copy()
                    value[i] = v[best]
                    still.append(i)
        live = still

    stats = {
        "value_start": float(start_value.mean()),
        "value_end": float(value.mean()),
        "rounds": float(rounds),
    }
    return current, stats


# --- a learned policy: eight picks, one card each ----------------------------


@dataclass
class BuryRows:
    """The eight picks of every landlord in a batch, for training."""

    cards: np.ndarray  # float32[R, 54, BURY_CARD_DIM]
    ctx: np.ndarray  # float32[R, BURY_CTX_DIM]
    mask: np.ndarray  # bool[R, 54]
    privileged: np.ndarray  # float32[R, BURY_PRIV_DIM]
    action: np.ndarray  # int64[R], a card type
    logp: np.ndarray  # float32[R]
    value: np.ndarray  # float32[R]
    hand: np.ndarray  # int64[R], index into the batch

    def __len__(self) -> int:
        return len(self.action)


def bury_context(
    level_rank: int, trump_suit: int, step: int, standing_tier: int,
    declarer_rel: int,
) -> np.ndarray:
    """float32[BURY_CTX_DIM]. ``declarer_rel`` is the standing
    declaration's seat relative to the landlord, or -1 for none."""
    from ..nn.bury_net import BURY_CTX_DIM, BURY_STEPS

    out = np.zeros(BURY_CTX_DIM, dtype=np.float32)
    out[level_rank] = 1.0
    out[13 + trump_suit] = 1.0
    out[18 + step] = 1.0
    out[18 + BURY_STEPS + standing_tier] = 1.0
    if declarer_rel >= 0:
        out[18 + BURY_STEPS + 5 + declarer_rel] = 1.0
    return out


def policy_burials(
    net: Any,
    pools: Sequence[np.ndarray],
    ctxs: Sequence[Ctx],
    standings: Sequence[tuple[int, int]],
    rng: np.random.Generator,
    device: Any = "cpu",
    greedy: bool = False,
    others: np.ndarray | None = None,
    record: bool = False,
    cache: Cache | None = None,
    num_decks: int = 2,
) -> tuple[list[np.ndarray], BuryRows | None]:
    """Bury for many landlords at once by the policy's eight picks.

    ``standings[i]`` is ``(tier, declarer seat relative to the landlord
    or -1)``. ``others`` is float32[n, 3, 54], the other hands after the
    landlord, and only feeds the value head; it may be omitted where
    nothing is trained. Picks are sampled unless ``greedy``.
    """
    from ..nn.bury_net import BURY_STEPS

    cache = cache or Cache()
    n = len(pools)
    pool = np.stack([np.asarray(p, dtype=np.float32) for p in pools])
    buried = np.zeros_like(pool)
    static = np.stack([cache.static_card_features(c, num_decks) for c in ctxs])
    priv = (
        np.zeros((n, 3 * NUM_TYPES), dtype=np.float32)
        if others is None else np.asarray(others, dtype=np.float32).reshape(n, -1)
    )
    parts: dict[str, list[np.ndarray]] = {k: [] for k in
        ("cards", "ctx", "mask", "privileged", "action", "logp", "value", "hand")}
    for step in range(BURY_STEPS):
        held = pool - buried
        cards = np.concatenate(
            (static, (held / num_decks)[..., None], (buried / num_decks)[..., None],
             (pool / num_decks)[..., None]),
            axis=2,
        ).astype(np.float32)
        ctx = np.stack([
            bury_context(ctxs[i].level_rank, ctxs[i].trump_suit, step,
                         standings[i][0], standings[i][1])
            for i in range(n)
        ])
        mask = held > 0
        with torch.no_grad():
            _logits, lp, v = net(
                torch.from_numpy(cards).to(device), torch.from_numpy(ctx).to(device),
                torch.from_numpy(mask).to(device), torch.from_numpy(priv).to(device),
            )
        logp_all = lp.float().cpu().numpy()
        if greedy:
            action = logp_all.argmax(axis=1)
        else:
            action = (logp_all + rng.gumbel(size=logp_all.shape)).argmax(axis=1)
        if not np.all(mask[np.arange(n), action]):
            raise RuntimeError("a burial pick fell outside the held cards")
        buried[np.arange(n), action] += 1
        if record:
            parts["cards"].append(cards)
            parts["ctx"].append(ctx)
            parts["mask"].append(mask)
            parts["privileged"].append(priv)
            parts["action"].append(action.astype(np.int64))
            parts["logp"].append(logp_all[np.arange(n), action].astype(np.float32))
            parts["value"].append(v.float().cpu().numpy().astype(np.float32))
            parts["hand"].append(np.arange(n, dtype=np.int64))
    burials = [buried[i].astype(np.int8) for i in range(n)]
    rows = None
    if record:
        rows = BuryRows(**{k: np.concatenate(v) for k, v in parts.items()})
    return burials, rows
