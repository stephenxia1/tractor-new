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

from ..nn.features import Cache, context_vector
from ..nn.model import privileged_view
from ..rules.cards import Ctx
from ..sim.bury import BuryConfig, play_start, random_burial, value_burials
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


def _declare_round(
    declare_net: Any,
    envs: list[DealEnv],
    pending: list[tuple[int, int]],
    rng: np.random.Generator,
    device: Any,
    greedy: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = np.stack([declare_obs(envs[h], s) for h, s in pending])
    priv = np.stack([declare_privileged(envs[h], s) for h, s in pending])
    mask = np.stack([envs[h].mask(s) for h, s in pending])
    if declare_net is None:
        # Uniform over the legal actions.
        logp_all = np.where(mask, -np.log(mask.sum(1, keepdims=True)), -np.inf)
        value = np.zeros(len(pending), dtype=np.float32)
    else:
        with torch.no_grad():
            _logits, lp, v = declare_net(
                torch.from_numpy(obs).to(device),
                torch.from_numpy(mask).to(device),
                torch.from_numpy(priv).to(device),
            )
        logp_all = lp.float().cpu().numpy()
        value = v.float().cpu().numpy()
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
) -> tuple[list[Start], DeclareRows | None, dict[str, float]]:
    """Deal, declare and bury ``n`` hands.

    ``declare_net`` of ``None`` declares uniformly over the legal
    actions; ``play_net`` of ``None`` buries every hand at random.
    """
    pre = pre or PrePlayConfig()
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
            declare_net, envs, pending, rng, device, pre.greedy_declare
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
    for i in range(n):
        if play_net is None or rng.random() < pre.random_burial_frac:
            burials[i] = random_burial(pools[i], rng)
        else:
            by_value.append(i)
    bury_stats = {"value_start": 0.0, "value_end": 0.0, "rounds": 0.0}
    if by_value:
        got, bury_stats = value_burials(
            play_net, [pools[i] for i in by_value], [ctxs[i] for i in by_value],
            [landlords[i] for i in by_value], device, pre.bury, cache, num_decks,
        )
        for i, b in zip(by_value, got):
            burials[i] = b

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
        rows = DeclareRows(
            obs=cat("obs", (0, DECLARE_OBS_DIM), np.float32),
            privileged=cat("privileged", (0, DECLARE_PRIV_DIM), np.float32),
            mask=cat("mask", (0, NUM_ACTIONS), bool),
            action=cat("action", (0,), np.int64),
            logp=cat("logp", (0,), np.float32),
            value=cat("value", (0,), np.float32),
            hand=cat("hand", (0,), np.int64),
            seat=cat("seat", (0,), np.int64),
        )
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
    num_decks: int = 2,
) -> np.ndarray:
    """float32[n, 4]: the privileged critic at each seat's play start.

    In the critic's units: that seat's team's expected future point
    margin over ``2 * 100 * num_decks``. At play start the history is
    empty, so the critic reads only the four hands, the burial and the
    seat's context vector.
    """
    critic = play_net.critic
    cfg = play_net.cfg
    ctx_rows: list[np.ndarray] = []
    priv_rows: list[np.ndarray] = []
    for st in starts:
        state = st.state(num_decks)
        for seat in range(NUM_SEATS):
            ctx_rows.append(context_vector(state, seat).astype(np.float16).astype(np.float32))
            priv_rows.append(privileged_view(state.hands, state.kitty, seat))
    context = np.stack(ctx_rows) if ctx_rows else np.zeros((0, 1), np.float32)
    priv = np.stack(priv_rows) if priv_rows else np.zeros((0, 5, 54), np.float32)
    out = np.empty(len(ctx_rows), dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, len(out), chunk):
            hi = min(lo + chunk, len(out))
            b = hi - lo
            enc = torch.zeros(b, cfg.max_history, cfg.d_model, device=device)
            length = torch.zeros(b, dtype=torch.int64, device=device)
            v = critic(
                enc, length,
                torch.from_numpy(priv[lo:hi]).to(device),
                context=torch.from_numpy(context[lo:hi]).to(device),
            )
            out[lo:hi] = v.float().cpu().numpy()
    return out.reshape(len(starts), NUM_SEATS)
