"""Which critic generalises: v5's flat critic, or v6's card critic?

The critic is two things at once. It is PPO's baseline, and at play
start -- where it reads no history -- it is the reward for every
declaration and every burial. What matters in both roles is how well it
predicts hands it was not fitted to, which is what ``critic_diag.py``
measured for the flat critic (held-out explained variance ~0.47 at 1024
hands against ~0.62 fitted). This asks the same question of a choice of
architectures, on identical data.

1. Play ``rounds`` rounds of ``hands`` hands with a trained checkpoint
   (v5-bury final by default), from its own declarations and burials --
   a fifth buried at random, as it trained -- and keep what a critic
   reads: history, context, card features, the trick in progress, the
   four hands and the kitty. The target is each row's Monte-Carlo
   return (gamma 1, lambda 1): the points still to come, which no
   critic's own predictions have shaped.
2. Hold the last round's hands out entirely. The critic reads all four
   hands, so only a whole unseen hand is unseen.
3. Train every arm from fresh initialisation on the same rows in the
   same order with the same optimiser, reading the checkpoint's frozen
   history encodings.
4. After every pass, score each arm on the held-out hands: every row's
   return, and each seat's whole-hand return from its play-start value
   (the pre-play reward). Intervals resample hands; differences between
   arms are paired on the same hands.

Arms: ``v5`` is the checkpoint's own trained critic, scored for
reference and never trained here. ``flat`` is a fresh critic of that
shape. ``flat-wide`` is the flat design at the card critic's width and
depth, so a difference between ``cards`` and ``flat`` can be told apart
into structure and size. ``cards`` is ``CardCritic`` at v6's shape.

This is a small-data measurement -- seven rounds is about 28 PPO steps
of critic data, where a run fits thousands -- so it answers how
efficiently each design learns and how well it generalises from a
round's worth of hands, which is the regime PPO's critic lives in.
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..nn.arch import bury_net_from, declare_net_from
from ..nn.model import (
    ActorCritic,
    CardCritic,
    Critic,
    ModelConfig,
    load_model,
    shared_critic,
)
from ..sim.state import NUM_SEATS
from .buffer import RolloutBuffer
from .critic_diag import _clustered_ev_interval, _slope
from .gae import compute_gae
from .parallel import RolloutPool
from .ppo import explained_variance
from .starts import PrePlayConfig, critic_values, make_starts

__all__ = ["ARMS", "collect_dataset", "run"]

ARMS = ("v5", "flat", "flat-wide", "cards")
_TRICK = slice(5, 9)  # the trick-in-progress planes, one per relative seat


def _arm_config(arm: str, actor_cfg: ModelConfig) -> ModelConfig:
    """The critic config an arm builds, sized against the actor it reads.

    ``cards-<d>`` is the card critic at width ``d``: v6's depth, a head
    per 64 of width, feed-forward ``4 * d``. ``cards-raw-<d>`` is the same
    as a population's shared critic builds it (``model.shared_critic``):
    reading the raw history rather than the actor's encoding.
    """
    if arm.startswith("cards-"):
        d = int(arm.rsplit("-", 1)[1])
        return replace(actor_cfg, critic_kind="cards", ffn="swiglu", critic_d=d,
                       critic_layers=4, critic_heads=max(1, d // 64), critic_d_ff=4 * d)
    # The first card critic's size, 256 x 4, which ``cards`` and
    # ``flat-wide`` were measured at; v6 has since come down to 128.
    wide = dict(critic_d=256, critic_layers=4, critic_heads=4, critic_d_ff=1024)
    if arm == "flat":
        return replace(actor_cfg, critic_kind="flat", ffn="gelu")
    if arm == "flat-wide":
        return replace(actor_cfg, critic_kind="flat", ffn="gelu", **wide)
    if arm == "cards":
        return replace(actor_cfg, critic_kind="cards", ffn="swiglu", **wide)
    raise ValueError(f"unknown arm {arm!r}")


def collect_dataset(
    blob: dict, net: ActorCritic, rounds: int, hands: int, device: torch.device,
    seed: int, workers: int = 7, pool: int = 512,
) -> dict[str, Any]:
    """``rounds`` rounds of play from the checkpoint, reduced to what a
    critic reads, plus each round's starts for the play-start score."""
    rng = np.random.default_rng(seed)
    declare_net = declare_net_from(blob["declare"]).to(device).eval()
    bury_net = bury_net_from(blob["bury"]).to(device).eval() if blob.get("bury") else None
    pre = PrePlayConfig(random_burial_frac=0.2,
                        burial="policy" if bury_net is not None else "climb")
    parts: dict[str, list] = {k: [] for k in (
        "history", "history_len", "context", "cards", "trick", "privileged",
        "ret", "episode", "seat")}
    starts_all = []
    buffer = RolloutBuffer(hands * 100, net.cfg)
    with RolloutPool(cfg=net.cfg, workers=workers, hands_per_round=hands) as rollout:
        for r in range(rounds):
            t0 = time.perf_counter()
            net.eval()
            starts, _rows, _st = make_starts(rng, hands, net, declare_net, device,
                                             pre=pre, bury_net=bury_net)
            buffer.reset()
            buffer, _stats = rollout.collect(net, rng, buffer, n_hands=hands,
                                             device=device, pool=pool, amp=True,
                                             starts=starts, level_weight=0.0)
            # lambda 1: every row's return is the rewards still to come.
            compute_gae(buffer, gamma=1.0, lam=1.0)
            n = len(buffer)
            parts["history"].append(buffer.history[:n].copy())
            parts["history_len"].append(buffer.history_len[:n].copy())
            parts["context"].append(buffer.context[:n].copy())
            parts["cards"].append(buffer.cards[:n].copy())
            parts["trick"].append(buffer.planes[:n, _TRICK].copy())
            parts["privileged"].append(buffer.privileged[:n].copy())
            parts["ret"].append(buffer.ret[:n].astype(np.float32).copy())
            parts["episode"].append(buffer.episode[:n].astype(np.int64) + r * hands)
            parts["seat"].append(buffer.seat[:n].astype(np.int64).copy())
            starts_all.extend(starts)
            print(f"  round {r + 1}/{rounds}: {n:,} rows, "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
    data = {k: np.concatenate(v) for k, v in parts.items()}
    data["starts"] = starts_all
    return data


def _to_device(data: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """The rows on the device once: minibatches then index there."""
    half = ("history", "context", "cards", "trick", "privileged")
    out = {k: torch.from_numpy(data[k]).to(device) for k in half}
    out["history_len"] = torch.from_numpy(data["history_len"]).to(device)
    out["ret"] = torch.from_numpy(data["ret"]).to(device)
    return out


def _values(critic: Any, actor: Any, gpu: dict, idx: np.ndarray, device: torch.device,
            batch: int = 1024) -> np.ndarray:
    out = np.empty(len(idx), dtype=np.float64)
    critic.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        for s in range(0, len(idx), batch):
            i = torch.from_numpy(idx[s : s + batch]).to(device)
            out[s : s + len(i)] = _forward(critic, actor, gpu, i).float().cpu().numpy()
    return out


def _forward(critic: Any, actor: Any, gpu: dict, i: torch.Tensor) -> torch.Tensor:
    if getattr(critic, "agents", 0):
        # A shared critic reads the raw history, not the actor's encoding;
        # here there is one agent, so every seat is agent 0.
        return critic(gpu["history"][i].float(), gpu["history_len"][i],
                      gpu["privileged"][i].float(), context=gpu["context"][i].float(),
                      cards=gpu["cards"][i].float(), trick=gpu["trick"][i].float(),
                      seat_agents=torch.zeros(len(i), NUM_SEATS, dtype=torch.int64,
                                              device=i.device))
    with torch.no_grad():
        enc, _keep = actor.history(gpu["history"][i].float(), gpu["history_len"][i])
    return critic(enc, gpu["history_len"][i], gpu["privileged"][i].float(), detach=True,
                  context=gpu["context"][i].float(), cards=gpu["cards"][i].float(),
                  trick=gpu["trick"][i].float())


def _seat_totals(data: dict, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each (hand, seat)'s whole-hand return: its first row's, at lambda 1.

    -> ``(hand_seat_keys, totals)`` in ``hand * 4 + seat`` order.
    """
    key = data["episode"][rows] * NUM_SEATS + data["seat"][rows]
    order = np.lexsort((rows, key))  # by key, then by row: first row first
    k_sorted = key[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = k_sorted[1:] != k_sorted[:-1]
    picked = rows[order][first]
    return k_sorted[first], data["ret"][picked].astype(np.float64)


def _paired_diff(a: np.ndarray, b: np.ndarray, actual: np.ndarray, hand: np.ndarray,
                 rng: np.random.Generator, draws: int = 1000) -> tuple[float, float, float]:
    """EV(a) - EV(b) on the same rows, with a 95% interval resampling hands."""
    hands = np.unique(hand)
    by_hand = [np.nonzero(hand == h)[0] for h in hands]
    stats = np.empty(draws)
    for d in range(draws):
        idx = np.concatenate([by_hand[j] for j in rng.integers(0, len(by_hand), len(by_hand))])
        stats[d] = explained_variance(a[idx], actual[idx]) - explained_variance(b[idx], actual[idx])
    mid = explained_variance(a, actual) - explained_variance(b, actual)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(mid), float(lo), float(hi)


def run(checkpoint_blob: dict, device: torch.device, rounds: int = 8, hands: int = 1024,
        epochs: int = 4, batch: int = 256, lr: float = 3e-4, seed: int = 20260928,
        arms: tuple[str, ...] = ARMS, data: dict | None = None, workers: int = 7,
        pool: int = 512) -> dict[str, Any]:
    """The whole measurement. -> per-arm, per-pass scores and paired differences."""
    actor_cfg = ModelConfig(**checkpoint_blob["model_config"])
    net = ActorCritic(actor_cfg)
    load_model(net, checkpoint_blob["model"])
    net = net.to(device).eval()
    if data is None:
        data = collect_dataset(checkpoint_blob, net, rounds, hands, device, seed,
                               workers=workers, pool=pool)
    held_hand_lo = (rounds - 1) * hands
    held = np.nonzero(data["episode"] >= held_hand_lo)[0]
    fit = np.nonzero(data["episode"] < held_hand_lo)[0]
    rng = np.random.default_rng(seed + 1)
    fit_probe = np.sort(rng.choice(fit, size=min(len(fit), 20_000), replace=False))
    gpu = _to_device(data, device)
    actual_held = data["ret"][held].astype(np.float64)
    held_hand = data["episode"][held]
    start_keys, start_actual = _seat_totals(data, held)
    held_starts = data["starts"][held_hand_lo:]
    start_hand = start_keys // NUM_SEATS

    results: dict[str, Any] = {
        "rows_fit": int(len(fit)), "rows_held": int(len(held)),
        "hands_fit": int(held_hand_lo), "hands_held": int(hands),
        "epochs": epochs, "batch": batch, "lr": lr, "arms": {}, "paired": {},
    }
    preds: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}

    def score(arm: str, critic: Any, epoch: int, seconds: float) -> dict[str, float]:
        p_held = _values(critic, net.actor, gpu, held, device)
        p_fit = _values(critic, net.actor, gpu, fit_probe, device)
        if getattr(critic, "agents", 0):
            p_start = critic_values(
                None, held_starts, device,
                agents=np.zeros((len(held_starts), NUM_SEATS), dtype=np.int64), shared=critic,
            )
        else:
            p_start = critic_values(SimpleNamespace(critic=critic, cfg=actor_cfg),
                                    held_starts, device)
        p_start = p_start.reshape(-1)[start_keys - held_hand_lo * NUM_SEATS]
        preds.setdefault(arm, {})[epoch] = (p_held, p_start)
        lo, hi = _clustered_ev_interval(p_held, actual_held, held_hand, rng, draws=500)
        s_lo, s_hi = _clustered_ev_interval(p_start, start_actual, start_hand, rng, draws=500)
        row = {
            "epoch": epoch,
            "ev_fit": explained_variance(p_fit, data["ret"][fit_probe]),
            "ev_held": explained_variance(p_held, actual_held),
            "ev_held_lo": lo, "ev_held_hi": hi,
            "slope_held": _slope(p_held, actual_held),
            "ev_start": explained_variance(p_start, start_actual),
            "ev_start_lo": s_lo, "ev_start_hi": s_hi,
            "slope_start": _slope(p_start, start_actual),
            "seconds": seconds,
        }
        print(f"  {arm:9} pass {epoch}: fit {row['ev_fit']:+.3f}  held {row['ev_held']:+.3f} "
              f"[{lo:+.3f},{hi:+.3f}] slope {row['slope_held']:.2f}  "
              f"start {row['ev_start']:+.3f} [{s_lo:+.3f},{s_hi:+.3f}] "
              f"slope {row['slope_start']:.2f}  {seconds:6.1f}s", flush=True)
        return row

    for arm in arms:
        torch.manual_seed(seed)
        if arm == "v5":
            critic = net.critic
            params = sum(p.numel() for p in critic.parameters())
            results["arms"][arm] = {"params": params, "passes": [score(arm, critic, 0, 0.0)]}
            continue
        cfg = _arm_config(arm, actor_cfg)
        if arm.startswith("cards-raw-"):
            critic = shared_critic(cfg, agents=1).to(device)
        else:
            critic = (CardCritic(cfg) if cfg.critic_kind == "cards" else Critic(cfg)).to(device)
        params = sum(p.numel() for p in critic.parameters())
        opt = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=0.0)
        order_rng = np.random.default_rng(seed + 2)  # same order for every arm
        passes = []
        for epoch in range(1, epochs + 1):
            critic.train()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            perm = order_rng.permutation(fit)
            for s in range(0, len(perm), batch):
                i = torch.from_numpy(perm[s : s + batch]).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    v = _forward(critic, net.actor, gpu, i)
                loss = F.mse_loss(v.float(), gpu["ret"][i])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            passes.append(score(arm, critic, epoch, time.perf_counter() - t0))
        results["arms"][arm] = {"params": params, "config": asdict(cfg), "passes": passes}

    # Paired on the same held-out hands: every arm against the flat critic
    # it would replace, pass for pass.
    if "flat" in preds:
        for arm in preds:
            if arm == "flat":
                continue
            out = []
            for epoch in sorted(preds["flat"]):
                mine = preds[arm].get(epoch, preds[arm].get(0))
                base_held, base_start = preds["flat"][epoch]
                d_held = _paired_diff(mine[0], base_held, actual_held, held_hand, rng)
                d_start = _paired_diff(mine[1], base_start, start_actual, start_hand, rng)
                out.append({"epoch": epoch, "held": d_held, "start": d_start})
            results["paired"][arm] = out
    return results
