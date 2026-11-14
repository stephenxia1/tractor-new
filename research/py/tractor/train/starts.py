"""Play-start positions produced by the pre-play agents.

``gen.episode.generate`` draws a play start with no judgement in it:
a declaration uniform over the legal ones, a burial uniform over the
landlord's cards. That is the warm-up distribution. Once there is a
declaration policy and a value network to bury with, play should start
from the positions those produce, which is what this builds:

1. deal ``n`` hands (``sim.deal.DealEnv``), every one in lockstep, with
   one batched forward of the declaration network per round of pending
   decisions;
2. bury, by the value network's climb (``sim.bury.value_burials``) or,
   for a share of hands, uniformly at random -- the share keeps the value
   head honest on burials the climb would never propose;
3. hand back each result as a ``Start``: the four play-start hands, the
   burial and the trump context, about 300 bytes, which is what crosses
   to a rollout worker instead of a ``PlayState``.

With ``record`` set it also returns every real declaration decision --
the rows the declaration phase trains on -- and ``critic_values``
scores each seat's play start with the privileged critic, which is that
phase's reward.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..nn.features import D_STATIC, HISTORY_DIM, Cache, context_vector
from ..nn.model import privileged_view, relative_seat_agents
from ..rules.cards import Ctx
from ..sim.bury import (
    BuryConfig,
    BuryRows,
    play_start,
    policy_burials,
    random_burial,
    value_burials,
)
from ..sim.deal import (
    DECLARE_OBS_DIM,
    DECLARE_PRIV_DIM,
    NUM_ACTIONS,
    PASS,
    DealEnv,
    declare_obs,
    declare_privileged,
)
from ..sim.state import NUM_SEATS, PlayState
from ..rules.cards import POINTS as _POINTS

__all__ = [
    "Start",
    "PrePlayConfig",
    "DeclareRows",
    "make_starts",
    "critic_values",
    "pack_start",
    "start_state",
]


@dataclass(frozen=True)
class Start:
    """Everything a rollout worker needs to begin a hand."""

    hands: np.ndarray  # int8[4, 54], after the burial
    kitty: np.ndarray  # int8[54], the burial
    landlord: int
    level_rank: int
    trump_suit: int

    def state(self, num_decks: int = 2) -> PlayState:
        return PlayState.new(
            Ctx(self.level_rank, self.trump_suit),
            self.hands,
            self.kitty,
            self.landlord,
            num_decks,
        )


def pack_start(start: Start) -> tuple:
    """A ``Start`` as plain bytes and ints, for a worker's command queue."""
    return (
        start.hands.astype(np.int8).tobytes(),
        start.kitty.astype(np.int8).tobytes(),
        int(start.landlord),
        int(start.level_rank),
        int(start.trump_suit),
    )


def start_state(packed: tuple, num_decks: int = 2) -> PlayState:
    hands, kitty, landlord, level_rank, trump_suit = packed
    return PlayState.new(
        Ctx(level_rank, trump_suit),
        np.frombuffer(hands, dtype=np.int8).reshape(NUM_SEATS, -1).copy(),
        np.frombuffer(kitty, dtype=np.int8).copy(),
        landlord,
        num_decks,
    )


@dataclass(frozen=True)
class PrePlayConfig:
    # Sample from the declaration policy, or take its argmax.
    greedy_declare: bool = False
    # Burials by the value network's climb, except this share at random.
    random_burial_frac: float = 0.2
    bury: BuryConfig = field(default_factory=BuryConfig)
    # "climb": ``sim.bury.value_burials``. "policy": the learned burial
    # network's eight picks (``nn/bury_net.py``), sampled unless
    # ``greedy_bury``.
    burial: str = "climb"
    greedy_bury: bool = False


@dataclass
class DeclareRows:
    """The declaration decisions of one batch of hands."""

    obs: np.ndarray  # float32[R, DECLARE_OBS_DIM]
    privileged: np.ndarray  # float32[R, DECLARE_PRIV_DIM]
    mask: np.ndarray  # bool[R, NUM_ACTIONS]
    action: np.ndarray  # int64[R]
    logp: np.ndarray  # float32[R]
    value: np.ndarray  # float32[R]
    hand: np.ndarray  # int64[R], index into the batch's starts
    seat: np.ndarray  # int64[R]

    def __len__(self) -> int:
        return len(self.action)

    def subset(self, idx: np.ndarray) -> "DeclareRows":
        return DeclareRows(**{k: v[idx] for k, v in vars(self).items()})


def _groups(owner: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """``(agent, rows)`` for each agent in ``owner``, in agent order."""
    return [(int(a), np.nonzero(owner == a)[0]) for a in np.unique(owner)]


def _pick(nets: Any, agent: int) -> Any:
    """A population's ``agent``-th network, or the one network."""
    return nets[agent] if isinstance(nets, (list, tuple)) else nets


@dataclass
class PrePlayRows:
    """Everything a pre-play phase trains on, for one batch of hands."""

    declare: DeclareRows
    bury: BuryRows | None


def _declare_round(
    declare_net: Any,
    envs: list[DealEnv],
    pending: list[tuple[int, int]],
    rng: np.random.Generator,
    device: Any,
    greedy: bool,
    agents: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = np.stack([declare_obs(envs[h], s) for h, s in pending])
    priv = np.stack([declare_privileged(envs[h], s) for h, s in pending])
    mask = np.stack([envs[h].mask(s) for h, s in pending])
    if declare_net is None:
        # Uniform over the legal actions.
        logp_all = np.where(mask, -np.log(mask.sum(1, keepdims=True)), -np.inf)
        value = np.zeros(len(pending), dtype=np.float32)
    else:
        # Each decision through its seat's network. One agent is one
        # forward over every row, as before routing existed.
        owner = (
            np.zeros(len(pending), dtype=np.int64) if agents is None
            else agents[[h for h, _s in pending], [s for _h, s in pending]]
        )
        logp_all = np.empty(mask.shape, dtype=np.float32)
        value = np.empty(len(pending), dtype=np.float32)
        for agent, idx in _groups(owner):
            with torch.no_grad():
                _logits, lp, v = _pick(declare_net, agent)(
                    torch.from_numpy(obs[idx]).to(device),
                    torch.from_numpy(mask[idx]).to(device),
                    torch.from_numpy(priv[idx]).to(device),
                )
            logp_all[idx] = lp.float().cpu().numpy()
            value[idx] = v.float().cpu().numpy()
    if greedy:
        action = logp_all.argmax(axis=1)
    else:
        action = (logp_all + rng.gumbel(size=logp_all.shape)).argmax(axis=1)
    logp = logp_all[np.arange(len(pending)), action].astype(np.float32)
    return obs, priv, mask, action.astype(np.int64), logp, value.astype(np.float32)


def make_starts(
    rng: np.random.Generator,
    n: int,
    play_net: Any,
    declare_net: Any,
    device: Any = "cpu",
    pre: PrePlayConfig | None = None,
    record: bool = False,
    cache: Cache | None = None,
    num_decks: int = 2,
    bury_net: Any = None,
    agents: Any = None,
) -> tuple[list[Start], PrePlayRows | None, dict[str, float]]:
    """Deal, declare and bury ``n`` hands.

    ``declare_net`` of ``None`` declares uniformly over the legal
    actions; ``play_net`` of ``None`` buries every hand at random, as
    does ``pre.burial == "policy"`` with no ``bury_net``.

    A population passes lists for the networks and ``agents``,
    int64[n, 4], the agent in each seat of each hand. Each seat then
    declares by its own agent's network, and the landlord buries by its
    agent's burial policy (or climbs on its agent's play network).
    """
    pre = pre or PrePlayConfig()
    if agents is not None:
        agents = np.asarray(agents, dtype=np.int64)
        if agents.shape != (n, NUM_SEATS):
            raise ValueError(f"seating is {agents.shape}, expected {(n, NUM_SEATS)}")
    t_start = time.perf_counter()
    envs = [
        DealEnv(rng, landlord=int(rng.integers(NUM_SEATS)),
                level_rank=int(rng.integers(13)), num_decks=num_decks)
        for _ in range(n)
    ]
    parts: dict[str, list[np.ndarray]] = {k: [] for k in
        ("obs", "privileged", "mask", "action", "logp", "value", "hand", "seat")}
    decisions = 0
    while True:
        pending = []
        for h, env in enumerate(envs):
            seat = env.next_decision()
            if seat is not None:
                pending.append((h, seat))
        if not pending:
            break
        obs, priv, mask, action, logp, value = _declare_round(
            declare_net, envs, pending, rng, device, pre.greedy_declare, agents
        )
        for (h, s), a in zip(pending, action):
            envs[h].act(s, int(a))
        decisions += len(pending)
        if record:
            parts["obs"].append(obs)
            parts["privileged"].append(priv)
            parts["mask"].append(mask)
            parts["action"].append(action)
            parts["logp"].append(logp)
            parts["value"].append(value)
            parts["hand"].append(np.array([h for h, _s in pending], dtype=np.int64))
            parts["seat"].append(np.array([s for _h, s in pending], dtype=np.int64))

    t_declared = time.perf_counter()
    pools = [env.pool for env in envs]
    ctxs = [env.ctx for env in envs]
    landlords = [env.landlord for env in envs]
    burials: list[np.ndarray | None] = [None] * n
    by_value = []
    use_policy = pre.burial == "policy"
    able = bury_net is not None if use_policy else play_net is not None
    for i in range(n):
        if not able or rng.random() < pre.random_burial_frac:
            burials[i] = random_burial(pools[i], rng)
        else:
            by_value.append(i)
    bury_stats = {"value_start": 0.0, "value_end": 0.0, "rounds": 0.0}
    bury_rows = None
    valued = np.asarray(by_value, dtype=np.int64)
    # Each landlord buries by its own agent; one agent is one group.
    owner = (
        np.zeros(len(valued), dtype=np.int64) if agents is None
        else agents[valued, np.asarray(landlords, dtype=np.int64)[valued]]
    )
    if by_value and use_policy:
        standings = []
        others = np.zeros((len(by_value), 3, pools[0].shape[0]), dtype=np.float32)
        for j, i in enumerate(by_value):
            env = envs[i]
            st = env.standing
            rel = -1 if st is None else (st.seat - env.landlord) % NUM_SEATS
            standings.append((env.standing_tier, rel))
            for k in range(3):
                others[j, k] = env.final[(env.landlord + 1 + k) % NUM_SEATS]
        parts_rows: list[BuryRows] = []
        for agent, members in _groups(owner):
            hands = valued[members]
            got, rows_a = policy_burials(
                _pick(bury_net, agent), [pools[i] for i in hands],
                [ctxs[i] for i in hands], [standings[j] for j in members], rng,
                device, greedy=pre.greedy_bury, others=others[members],
                record=record, cache=cache, num_decks=num_decks,
            )
            for i, b in zip(hands, got):
                burials[i] = b
            if rows_a is not None:
                # Rows index the policy's own batch; point them at the hands.
                rows_a.hand = hands[rows_a.hand]
                parts_rows.append(rows_a)
        if parts_rows:
            bury_rows = parts_rows[0] if len(parts_rows) == 1 else BuryRows(**{
                k: np.concatenate([vars(p)[k] for p in parts_rows])
                for k in vars(parts_rows[0])
            })
    elif by_value:
        weighted = []
        for agent, members in _groups(owner):
            hands = valued[members]
            got, st = value_burials(
                _pick(play_net, agent), [pools[i] for i in hands],
                [ctxs[i] for i in hands], [landlords[i] for i in hands], device,
                pre.bury, cache, num_decks,
            )
            for i, b in zip(hands, got):
                burials[i] = b
            weighted.append((len(hands), st))
        if len(weighted) == 1:
            bury_stats = weighted[0][1]
        else:
            total = sum(m for m, _st in weighted)
            bury_stats = {
                "value_start": sum(m * st["value_start"] for m, st in weighted) / total,
                "value_end": sum(m * st["value_end"] for m, st in weighted) / total,
                "rounds": max(st["rounds"] for _m, st in weighted),
            }

    t_buried = time.perf_counter()
    starts = []
    for i, env in enumerate(envs):
        state = play_start(ctxs[i], env.final, pools[i], burials[i], landlords[i], num_decks)
        starts.append(Start(
            hands=state.hands.copy(), kitty=state.kitty.copy(),
            landlord=landlords[i], level_rank=env.level_rank,
            trump_suit=env.trump_suit,
        ))

    rows = None
    if record:
        def cat(k: str, empty: tuple[int, ...], dtype: Any) -> np.ndarray:
            return np.concatenate(parts[k]) if parts[k] else np.zeros(empty, dtype=dtype)
        rows = PrePlayRows(declare=DeclareRows(
            obs=cat("obs", (0, DECLARE_OBS_DIM), np.float32),
            privileged=cat("privileged", (0, DECLARE_PRIV_DIM), np.float32),
            mask=cat("mask", (0, NUM_ACTIONS), bool),
            action=cat("action", (0,), np.int64),
            logp=cat("logp", (0,), np.float32),
            value=cat("value", (0,), np.float32),
            hand=cat("hand", (0,), np.int64),
            seat=cat("seat", (0,), np.int64),
        ), bury=bury_rows)
    declared = [env.standing is not None for env in envs]
    stats = {
        "declare/decisions_per_hand": decisions / max(n, 1),
        "declare/declared_frac": float(np.mean(declared)) if n else 0.0,
        "declare/declarations_per_hand": float(np.mean([len(e.log) for e in envs])) if n else 0.0,
        "declare/no_trump_frac": float(np.mean([e.trump_suit == 4 for e in envs])) if n else 0.0,
        "declare/by_landlord_team_frac": float(np.mean([
            (e.standing is not None and (e.standing.seat - e.landlord) % 2 == 0) for e in envs
        ])) if n else 0.0,
        "bury/value_frac": len(by_value) / max(n, 1),
        "bury/kitty_points": float(np.mean([
            float((b.astype(np.int64) * _POINTS).sum()) for b in burials
        ])) if n else 0.0,
        "bury/value_start": bury_stats["value_start"],
        "bury/value_end": bury_stats["value_end"],
        "bury/rounds": bury_stats["rounds"],
        "time/declare_s": t_declared - t_start,
        "time/bury_s": t_buried - t_declared,
        "time/starts_s": time.perf_counter() - t_start,
    }
    return starts, rows, stats


def critic_values(
    play_net: Any, starts: list[Start], device: Any = "cpu", chunk: int = 512,
    num_decks: int = 2, agents: Any = None, shared: Any = None,
) -> np.ndarray:
    """float32[n, 4]: the privileged critic at each seat's play start.

    In the critic's units: that seat's team's expected future point
    margin over ``2 * 100 * num_decks``. At play start the history is
    empty, so the critic reads only the four hands, the burial and the
    seat's context vector.

    A population passes its list of play networks and ``agents``,
    int64[n, 4]; each seat is then valued by its own agent's critic --
    or, with ``shared``, by the population's one critic, told which agent
    sits in each seat.
    """
    ctx_rows: list[np.ndarray] = []
    priv_rows: list[np.ndarray] = []
    # Per start, not per row: every seat of a hand shares its trump context.
    # Read only by ``CardCritic``; the flat critic never asks for them.
    cache = Cache()
    static = []
    for st in starts:
        state = st.state(num_decks)
        static.append(cache.static_card_features(state.ctx, num_decks))
        for seat in range(NUM_SEATS):
            ctx_rows.append(context_vector(state, seat).astype(np.float16).astype(np.float32))
            priv_rows.append(privileged_view(state.hands, state.kitty, seat))
    context = np.stack(ctx_rows) if ctx_rows else np.zeros((0, 1), np.float32)
    priv = np.stack(priv_rows) if priv_rows else np.zeros((0, 5, 54), np.float32)
    static_all = np.stack(static) if static else np.zeros((0, 54, D_STATIC), np.float32)
    out = np.empty(len(ctx_rows), dtype=np.float32)
    if shared is not None:
        if agents is None:
            raise ValueError("a shared critic needs the seating to know who is who")
        table = np.repeat(np.asarray(agents, dtype=np.int64), NUM_SEATS, axis=0)
        seats = np.tile(np.arange(NUM_SEATS), len(starts))
        seat_agents = relative_seat_agents(table, seats)
        with torch.no_grad():
            for lo in range(0, len(out), chunk):
                r = np.arange(lo, min(lo + chunk, len(out)))
                v = shared(
                    torch.zeros(len(r), 1, HISTORY_DIM, device=device),
                    torch.zeros(len(r), dtype=torch.int64, device=device),
                    torch.from_numpy(priv[r]).to(device),
                    context=torch.from_numpy(context[r]).to(device),
                    cards=torch.from_numpy(static_all[r // NUM_SEATS]).to(device),
                    seat_agents=torch.from_numpy(seat_agents[r]).to(device),
                )
                out[r] = v.float().cpu().numpy()
        return out.reshape(len(starts), NUM_SEATS)
    # Rows are hand-major, seat-minor, as ``agents`` flattens.
    owner = (
        np.zeros(len(out), dtype=np.int64) if agents is None
        else np.asarray(agents, dtype=np.int64).reshape(-1)
    )
    with torch.no_grad():
        for agent, rows in _groups(owner):
            net = _pick(play_net, agent)
            for lo in range(0, len(rows), chunk):
                r = rows[lo : lo + chunk]
                # Empty histories: the critic's fast path reads only the
                # batch size from this, so one position is enough.
                enc = torch.zeros(len(r), 1, net.cfg.d_model, device=device)
                length = torch.zeros(len(r), dtype=torch.int64, device=device)
                v = net.critic(
                    enc, length,
                    torch.from_numpy(priv[r]).to(device),
                    context=torch.from_numpy(context[r]).to(device),
                    cards=torch.from_numpy(static_all[r // NUM_SEATS]).to(device),
                )
                out[r] = v.float().cpu().numpy()
    return out.reshape(len(starts), NUM_SEATS)


def burial_samples(
    bury_net: Any,
    starts: list[Start],
    rows: BuryRows,
    k: int,
    rng: np.random.Generator,
    device: Any = "cpu",
    cache: Cache | None = None,
    num_decks: int = 2,
    agents: Any = None,
) -> tuple[BuryRows, np.ndarray, np.ndarray, list[Start]]:
    """``k - 1`` more burials for every landlord ``rows`` recorded, from the
    same policy, beside the one it made.

    For a leave-one-out baseline: a landlord's burials share its pool and
    the other hands, so the difference between their critic values is the
    burial's doing alone, where a learned baseline also has to predict
    how good the hand was. -> (every sample's rows, each row's sample
    index, each start's hand index, one start per landlord and sample in
    sample-major order -- sample 0 is the burial already made).
    """
    from ..sim.bury import pools_from_rows

    first = rows.ctx[:, 18] > 0.5
    hand_of = rows.hand[first]
    pools, ctxs, standings, others = pools_from_rows(rows, num_decks)
    n = len(hand_of)
    owner = (np.zeros(n, dtype=np.int64) if agents is None else
             np.asarray(agents)[hand_of, [starts[h].landlord for h in hand_of]])

    def start_with(s: Start, burial: np.ndarray) -> Start:
        hands = s.hands.astype(np.int8).copy()
        pool = s.hands[s.landlord].astype(np.int64) + s.kitty.astype(np.int64)
        hands[s.landlord] = (pool - burial).astype(np.int8)
        return Start(hands=hands, kitty=burial.astype(np.int8), landlord=s.landlord,
                     level_rank=s.level_rank, trump_suit=s.trump_suit)

    all_rows = [rows]
    sample_of = [np.zeros(len(rows), dtype=np.int64)]
    out_starts = [starts[h] for h in hand_of]
    out_hands = [hand_of]
    for sample in range(1, k):
        burials: list = [None] * n
        parts: list[BuryRows] = []
        for agent, members in _groups(owner):
            got, r = policy_burials(
                _pick(bury_net, agent), [pools[j] for j in members],
                [ctxs[j] for j in members], [standings[j] for j in members], rng, device,
                others=others[members], record=True, cache=cache, num_decks=num_decks,
            )
            for j, b in zip(members, got):
                burials[j] = b
            r.hand = hand_of[members][r.hand]
            parts.append(r)
        r_all = parts[0] if len(parts) == 1 else BuryRows(**{
            key: np.concatenate([vars(p)[key] for p in parts]) for key in vars(parts[0])})
        all_rows.append(r_all)
        sample_of.append(np.full(len(r_all), sample, dtype=np.int64))
        out_starts += [start_with(starts[hand_of[j]], burials[j]) for j in range(n)]
        out_hands.append(hand_of)
    merged = BuryRows(**{key: np.concatenate([vars(p)[key] for p in all_rows])
                         for key in vars(rows)})
    return merged, np.concatenate(sample_of), np.concatenate(out_hands), out_starts


def leave_one_out(values: np.ndarray) -> np.ndarray:
    """float[k, n] -> each entry minus the mean of the other ``k - 1`` in
    its column."""
    k = values.shape[0]
    if k < 2:
        raise ValueError("a leave-one-out baseline needs at least two samples")
    return values - (values.sum(axis=0, keepdims=True) - values) / (k - 1)
