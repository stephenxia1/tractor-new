"""Training loop v2: points-primary DMC, privileged critic, opponent mixture.

    PYTHONPATH=py python py/train2.py --workers 10 --games 64 --threads 22 --minutes 900

Differences from train_dmc.py, which stays as the v1 control:

* targets are point returns with a level auxiliary, distilled through a
  privileged critic (see dmc2.py)
* one network with per-role heads, causal move history, value and belief
  heads (model2.py)
* seats are not all the current policy. A share of games run against
  frozen snapshots, and a share put a snapshot in the *partner* seat --
  pure self-play lets the network assume its partner is a copy of itself,
  which is exactly the assumption that breaks against a human.

Only decisions taken by the current network are trained on; snapshot
seats shape the environment, and their returns follow a different
continuation policy.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from tractor_rl import layout as L
from tractor_rl.dmc2 import (
    AsyncLearner, BatchPrefetcher, DMC2Config, Learner2, ReplayBuffer2, hand_returns)
from tractor_rl.features2 import (
    action_features,
    field_u8,
    field_u16,
    field_u32,
    history_features,
    merge_batches,
    parse_episodes,
    split_per_worker,
    state_features,
)
from tractor_rl.model2 import NetConfig, PrivilegedCritic, TractorNet
from tractor_rl.protocol import WorkerConfig, WorkerPool

RESEARCH_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = RESEARCH_DIR / "checkpoints"
RUN_DIR = RESEARCH_DIR / "runs"

# name, share of workers, learner-seat mask, seats the *current* net plays,
# bot filling any non-learner seats
# Every seat is the agent or one of its own past selves. The heuristic
# `vs_bot` share was dropped: the agent beats smart-v2 by ~40 points, so
# those games were blowouts carrying almost no gradient signal, and the
# capacity is better spent on opponents that can still punish a mistake.
#
# What that gives up is an *external* reference. Self-play plus snapshots is
# a closed system, and a closed system can drift into an equilibrium that is
# self-consistent and still exploitable by anything outside it. The ladder is
# now the only thing that would catch that, so it has to be run on a
# schedule rather than when convenient.
#
# `preplay` stays smart-v34 everywhere: the network plays cards only, and
# declaration and kitty burial are still heuristic by design.
MATCHUPS = [
    ("self", 0.65, 0b1111, (0, 1, 2, 3), "smart-v34"),
    ("vs_snapshot", 0.25, 0b1111, (0, 2), "smart-v34"),
    ("mixed_partner", 0.10, 0b1111, (0,), "smart-v34"),
]


class RoundStore:
    __slots__ = ("states", "actions")

    def __init__(self, states: np.ndarray, actions: np.ndarray) -> None:
        self.states = states
        self.actions = actions


def assign_matchups(num_workers: int) -> list[tuple[str, int, tuple[int, ...], str]]:
    """Spread workers over matchups by largest remainder.

    Plain rounding silently drops the small shares -- at 10 workers,
    round(0.05 * 10) is 0 under banker's rounding, so the bot matchup
    would never run. Every matchup gets at least one worker once there are
    enough to go round.
    """
    counts = []
    for name, share, mask, current, bot in MATCHUPS:
        exact = share * num_workers
        counts.append([name, mask, current, bot, int(exact), exact - int(exact)])
    if num_workers >= len(MATCHUPS):
        for c in counts:
            c[4] = max(c[4], 1)
    total = sum(c[4] for c in counts)
    order = sorted(range(len(counts)), key=lambda i: -counts[i][5])
    i = 0
    while total < num_workers:
        counts[order[i % len(order)]][4] += 1
        total += 1
        i += 1
    while total > num_workers:
        # Trim from the largest non-self matchup first.
        j = max(range(len(counts)), key=lambda k: (counts[k][4], counts[k][0] != "self"))
        if counts[j][4] <= 1 and counts[j][0] == "self":
            break
        counts[j][4] -= 1
        total -= 1
    out: list[tuple[str, int, tuple[int, ...], str]] = []
    for name, mask, current, bot, count, _ in counts:
        out.extend([(name, mask, current, bot)] * count)
    return out[:num_workers]


def subset_indices(rows: np.ndarray, merged, total_decisions: int):
    """Candidate rows and remapped owners for a subset of decisions."""
    keep = np.zeros(total_decisions, dtype=bool)
    keep[rows] = True
    cand_mask = keep[merged.owner]
    pos = np.full(total_decisions, -1, dtype=np.int64)
    pos[rows] = np.arange(rows.shape[0])
    return cand_mask, pos[merged.owner[cand_mask]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--threads", type=int, default=22)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--history", type=str, default="attn", choices=["attn", "none"])
    ap.add_argument("--lambda-mc", type=float, default=DMC2Config().lambda_mc)
    # The target blend is a regulariser on the teacher (see dmc2.py): at the
    # default 0.3 it outweighs the data 7:3. Right early, when the return is
    # mostly deal noise; suspect late, when the teacher's own error is what
    # binds. These anneal it out. Left unset the blend is constant, which is
    # what every run before this one did.
    ap.add_argument("--lambda-mc-final", type=float, default=None,
                    help="anneal the blend to this value (default: constant)")
    ap.add_argument("--lambda-anneal-start", type=int, default=0,
                    help="learner steps at which the anneal begins")
    ap.add_argument("--lambda-anneal-end", type=int, default=0,
                    help="learner steps at which it reaches --lambda-mc-final")
    ap.add_argument("--batch-size", type=int, default=DMC2Config().batch_size)
    ap.add_argument("--train-every", type=int, default=DMC2Config().train_every)
    ap.add_argument("--steps-per-train", type=int, default=DMC2Config().steps_per_train)
    # The critic costs about as much per step as the policy does, so its
    # cadence is a direct lever on learner time -- the dominant cost once
    # featurisation was fixed.
    ap.add_argument("--critic-every", type=int, default=DMC2Config().critic_every)
    # Depth of the featurisation prefetch queue; 0 disables the thread.
    ap.add_argument("--prefetch", type=int, default=0)
    ap.add_argument("--async-learner", action="store_true")
    # Cycle steps_per_train through these values, `--sweep-minutes` on each.
    # Cycling rather than ramping: the metrics drift as training proceeds, so
    # a monotonic sweep would confound the setting with the drift. Visiting
    # each value repeatedly shares the drift across all of them.
    ap.add_argument("--replay-sweep", type=str, default="",
                    help='e.g. "1,2,4,8"')
    ap.add_argument("--sweep-minutes", type=float, default=8.0)
    # fp16 is deliberately absent: autocast in fp16 needs a GradScaler to
    # keep gradients from underflowing, there is none here, and bf16 (which
    # needs no scaling) measured *slower* on both A10G and T4 anyway.
    ap.add_argument("--amp", choices=["off", "bf16"], default="off")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--snapshot-every", type=int, default=10000, help="learner steps")
    # Rounds a pairing is held before resampling the opponent. Long blocks
    # amortise the state_dict copy and keep per-agent batches large, which is
    # what makes batched multi-agent GEMM worth anything on a GPU.
    #
    # The cost is buffer correlation: the replay buffer holds ~55s of play on
    # a GPU box, so a 2500-round block (~113s) means that partway through,
    # every sample involving the snapshot comes from one opponent. That is
    # ~25% of the buffer -- vs_snapshot and mixed_partner -- since self-play
    # and bot games do not use the snapshot net at all.
    ap.add_argument("--snapshot-refresh", type=int, default=2500, help="rounds between resamples")
    ap.add_argument("--snapshot-pool", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--checkpoint-every", type=int, default=100000, help="learner steps")
    # Wall-clock checkpointing matters more than step-count checkpointing on
    # interruptible hardware: a spot reclaim costs whatever has happened
    # since the last save, and a step threshold can easily be longer than
    # the machine's life. One run lost 22k steps to exactly that.
    ap.add_argument("--checkpoint-minutes", type=float, default=10.0)
    ap.add_argument("--run-name", type=str, default="v2")
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RUN_DIR.mkdir(exist_ok=True)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)

    cfg = DMC2Config(
        lambda_mc=args.lambda_mc, batch_size=args.batch_size,
        train_every=args.train_every, steps_per_train=args.steps_per_train,
        critic_every=args.critic_every,
        lambda_mc_final=args.lambda_mc_final,
        lambda_anneal_start=args.lambda_anneal_start,
        lambda_anneal_end=args.lambda_anneal_end)
    if args.lambda_mc_final is not None:
        if args.lambda_anneal_end <= args.lambda_anneal_start:
            raise SystemExit("--lambda-anneal-end must exceed --lambda-anneal-start")
        print(f"lambda_mc anneals {args.lambda_mc} -> {args.lambda_mc_final} "
              f"over steps {args.lambda_anneal_start}..{args.lambda_anneal_end}")
    net_cfg = NetConfig(history=args.history)
    net = TractorNet(net_cfg).to(device)
    critic = PrivilegedCritic().to(device)
    resumed_snapshots: list[dict] = []
    resumed_steps = 0
    hands_base = 0
    resumed_state: dict | None = None
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        resumed_state = state
        net.load_state_dict(state["model"])
        if "critic" in state:
            critic.load_state_dict(state["critic"])
        resumed_snapshots = state.get("snapshots", [])
        resumed_steps = int(state.get("steps", 0))
        hands_base = int(state.get("hands", 0))
        print(f"resumed from {args.resume} (step {resumed_steps}, "
              f"{len(resumed_snapshots)} snapshots)")
    learner = Learner2(net, critic, cfg, device)
    if resumed_state is not None:
        # After the optimisers exist. Checkpoints written before this was
        # added carry no optimiser state, and a shape mismatch (a resume
        # under a different --history) must not be fatal at boot time on a
        # spot box: losing the moments is a transient, losing the run is not.
        for key, opt in (("opt", learner.opt), ("critic_opt", learner.critic_opt)):
            blob = resumed_state.get(key)
            if blob is None:
                print(f"  no {key} in checkpoint; optimiser starts cold")
                continue
            try:
                opt.load_state_dict(blob)
            except (ValueError, KeyError, RuntimeError) as err:
                print(f"  dropping {key}: {' '.join(str(err).split())[:140]}")
            else:
                # load_state_dict also restores param_groups, so a checkpoint
                # written under a different --lr would silently override the
                # one just asked for: a config edit that appears to take
                # effect and does not. Only the moments are wanted here.
                lr = cfg.critic_lr if key == "critic_opt" else cfg.lr
                for group in opt.param_groups:
                    group["lr"] = lr
                print(f"  restored {key} (lr held at {lr})")
    if args.amp != "off":
        learner.amp_dtype = torch.bfloat16
    if device.type == "cuda":
        # TF32 accelerates fp32 matmuls on Ampere and later at reduced
        # mantissa. Free on A10G (g5); a no-op on the T4 we benchmarked,
        # which is Turing and has no TF32 path.
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
    learner.steps = resumed_steps
    buffer = ReplayBuffer2(cfg.buffer_capacity, L.STATE_BYTES_PRIVILEGED)
    # Featurising the next batch while the current one trains. Worth it when
    # the device half is fast relative to the numpy half -- which is the
    # normal case on a GPU and marginal on CPU, so it is opt-in.
    prefetch = (
        BatchPrefetcher(buffer, cfg.batch_size, args.seed + 977, depth=args.prefetch,
                        min_buffer=cfg.min_buffer,
                        featurise=device.type != "cuda")
        if args.prefetch > 0 and not args.async_learner else None)
    # Gradient steps off the game loop entirely. Supersedes the prefetch
    # thread, which only hid featurisation; this hides the whole learner.
    async_learner = (
        AsyncLearner(learner, buffer, cfg, args.seed + 613)
        if args.async_learner else None)
    rng = np.random.default_rng(args.seed)

    snapshot_net = TractorNet(net_cfg).to(device)
    snapshot_net.load_state_dict(net.state_dict())
    snapshot_net.eval()
    snapshots: deque[dict] = deque(maxlen=args.snapshot_pool)
    # Restored snapshots are only opponents, so a shape mismatch (a resume
    # under a different --history, say) must not be fatal: crashing here
    # costs the whole run, since this is boot time on a spot box.
    for snap in resumed_snapshots[-args.snapshot_pool:]:
        state = _to_float(snap["state"])
        try:
            snapshot_net.load_state_dict(state)
        except RuntimeError as err:
            detail = " ".join(str(err).split())[:200]
            print(f"dropping snapshot at step {snap['steps']}: {detail}")
            continue
        snapshots.append({"steps": snap["steps"], "state": state})
    if snapshots:
        snapshot_net.load_state_dict(snapshots[-1]["state"])
        print(f"opponent pool warm: {[s['steps'] for s in snapshots]}")
    else:
        snapshot_net.load_state_dict(net.state_dict())

    matchups = assign_matchups(args.workers)
    configs = [
        WorkerConfig(
            games=args.games, seed=args.seed + i * 101, learner_seats=mask,
            baseline=bot, preplay="smart-v34", privileged=True)
        for i, (_, mask, _, bot) in enumerate(matchups)
    ]
    pool = WorkerPool(args.workers, configs[0], configs=configs, num_groups=args.groups)

    # [worker, seat] -> is this seat played by the *current* network
    current_table = np.zeros((args.workers, 4), dtype=bool)
    for i, (_, _, current_seats, _) in enumerate(matchups):
        for s in current_seats:
            current_table[i, s] = True
    counts = {name: sum(1 for m in matchups if m[0] == name) for name, *_ in MATCHUPS}
    print(f"net {net.num_parameters()/1e6:.2f}M params, critic {critic.num_parameters()/1e6:.2f}M | "
          f"device {device} | history {args.history}")
    print(f"matchups: {counts}")

    log_file = (RUN_DIR / f"{args.run_name}.jsonl").open("a", encoding="utf-8")

    # Spot gives a termination notice and then SIGTERM; save on the way out
    # so the last minutes are not lost on top of whatever the periodic
    # checkpoint missed.
    interrupted = {"flag": False}

    def on_term(signum, _frame):
        interrupted["flag"] = True
        try:
            save(net, critic, learner, cfg, net_cfg, args, hands_base + hands, snapshots, final=True)
            print(f"caught signal {signum}; checkpointed at step {learner.steps}", flush=True)
        finally:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    pending: dict[tuple[int, int, int], list] = {}
    started = time.perf_counter()
    deadline = started + args.minutes * 60
    sweep = [int(x) for x in args.replay_sweep.split(",") if x.strip()]
    sweep_at = time.perf_counter()
    sweep_i = 0
    if sweep:
        cfg = replace(cfg, steps_per_train=sweep[0])
        print(f"replay sweep: {sweep}, {args.sweep_minutes} min per phase")
    rounds = decisions = trained_decisions = hands = 0
    # Per-phase counters, so each setting is scored on its own window rather
    # than on run-to-date totals.
    phase_dec = phase_steps = phase_trained = 0
    phase_start_step = 0
    last_checkpoint = time.perf_counter()
    # Watermarks rather than modulo tests; see the snapshot block below.
    next_snapshot_at = ((resumed_steps // args.snapshot_every) + 1) * args.snapshot_every
    next_archive_at = ((resumed_steps // args.checkpoint_every) + 1) * args.checkpoint_every
    recent_points: list[int] = []
    t_env = t_feat = t_fwd = t_learn = t_book = 0.0

    def score(model, rows, merged, tokens_all, mask_all, sf_all, af_all):
        """Greedy candidate index for a subset of decisions, under `model`."""
        if rows.shape[0] == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        cand_mask, new_owner = subset_indices(rows, merged, merged.num_decisions)
        # inference_mode rather than no_grad: it also skips version-counter
        # and view tracking, measured 12% faster on a realistic actor round.
        with torch.inference_mode():
            out = model(
                sf_all[rows], af_all[cand_mask],
                torch.from_numpy(new_owner).to(device),
                torch.from_numpy(
                    merged.states[rows, L.H_ROLE].astype(np.int64)).to(device),
                None if tokens_all is None else tokens_all[rows],
                None if mask_all is None else mask_all[rows],
                want="q")
        q = out["q"].cpu().numpy()
        counts_sub = merged.cand_counts[rows]
        offsets_sub = np.cumsum(counts_sub) - counts_sub
        max_k = int(counts_sub.max())
        padded = np.full((rows.shape[0], max_k), -np.inf, dtype=np.float32)
        within = np.arange(q.shape[0]) - offsets_sub[new_owner]
        padded[new_owner, within] = q
        return padded.argmax(axis=1), counts_sub

    try:
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            group = pool.next_group()
            group_workers = pool.workers_of(group)
            batches = pool.read_group(group)
            merged = merge_batches(batches)
            t1 = time.perf_counter()
            if merged.num_decisions == 0:
                pool.write_group(group, batches, [np.zeros(0, dtype=np.int64)] * len(batches))
                continue

            sf_all = torch.from_numpy(state_features(merged.states)).to(device)
            af_all = torch.from_numpy(
                action_features(merged.cands, merged.states, merged.owner)).to(device)
            if args.history == "attn":
                tok_np, msk_np = history_features(merged.states)
                tokens_all = torch.from_numpy(tok_np).to(device)
                mask_all = torch.from_numpy(msk_np).to(device)
            else:
                tokens_all = mask_all = None
            t2 = time.perf_counter()

            seats = field_u8(merged.states, L.H_SEAT)
            local_of = np.searchsorted(
                merged.bounds[1:], np.arange(merged.num_decisions), side="right")
            worker_ids = np.array([group_workers[int(j)].index for j in local_of])
            is_current = current_table[worker_ids, seats]
            rows_cur = np.nonzero(is_current)[0]
            rows_snap = np.nonzero(~is_current)[0]

            picks = np.zeros(merged.num_decisions, dtype=np.int64)
            best_cur, counts_cur = score(net, rows_cur, merged, tokens_all, mask_all, sf_all, af_all)
            explore = rng.random(rows_cur.shape[0]) < cfg.epsilon
            rand_pick = (rng.random(rows_cur.shape[0]) * counts_cur).astype(np.int64)
            picks[rows_cur] = np.where(explore, rand_pick, best_cur)
            if rows_snap.shape[0] > 0:
                best_snap, _ = score(
                    snapshot_net, rows_snap, merged, tokens_all, mask_all, sf_all, af_all)
                picks[rows_snap] = best_snap
            pool.write_group(group, batches, split_per_worker(picks, merged.bounds))
            t3 = time.perf_counter()

            # Stash only what the current network decided.
            store = RoundStore(
                merged.states[rows_cur].copy(),
                merged.cands[merged.cand_offsets[rows_cur] + picks[rows_cur]].copy())
            game_ids = field_u16(merged.states, L.H_GAME_ID)
            hand_ids = field_u32(merged.states, L.H_HAND_ID)
            trick_idx = field_u8(merged.states, L.H_TRICK_INDEX)
            attacker = field_u8(merged.states, L.H_I_AM_ATTACKER)
            for local_i, i in enumerate(rows_cur):
                key = (int(worker_ids[i]), int(game_ids[i]), int(hand_ids[i]))
                pending.setdefault(key, []).append(
                    (store, local_i, int(trick_idx[i]), 1.0 if attacker[i] else -1.0))
            decisions += merged.num_decisions
            phase_dec += merged.num_decisions
            trained_decisions += rows_cur.shape[0]
            phase_trained += rows_cur.shape[0]

            for local_w, batch in enumerate(batches):
                w = group_workers[local_w].index
                for ep in parse_episodes(batch.episodes):
                    hands += 1
                    recent_points.append(ep.attacker_points)
                    entries = pending.pop((w, ep.game_id, ep.hand_id), None)
                    if not entries:
                        continue
                    returns, level = hand_returns(ep, cfg)
                    if returns.shape[0] == 0:
                        continue
                    st = np.stack([e[0].states[e[1]] for e in entries])
                    ac = np.stack([e[0].actions[e[1]] for e in entries])
                    tr = np.clip(np.fromiter((e[2] for e in entries), np.int64, len(entries)),
                                 0, returns.shape[0] - 1)
                    sg = np.fromiter((e[3] for e in entries), np.float32, len(entries))
                    buffer.add(st, ac, (returns[tr] * sg).astype(np.float32),
                               (level * sg).astype(np.float32))

            t4 = time.perf_counter()
            if rounds % cfg.train_every == 0:
                if async_learner is not None:
                    # A learner thread that dies leaves `request` succeeding
                    # until the queue fills and then silently dropping, so the
                    # run burns its whole --minutes collecting data and taking
                    # no gradient steps. The class records the exception; this
                    # is the half that was missing.
                    if async_learner.error is not None:
                        raise RuntimeError(
                            "async learner thread died: "
                            f"{async_learner.error!r}") from async_learner.error
                    # Hand the work off and keep the game loop moving.
                    async_learner.request(cfg.steps_per_train)
                else:
                    for _ in range(cfg.steps_per_train):
                        if prefetch is None:
                            learner.step(buffer, rng)
                        elif buffer.size >= cfg.min_buffer:
                            nxt = prefetch.next()
                            learner.step_batch(
                                learner.featurise_on_device(nxt)
                                if learner.device_featurise else nxt)
            t5 = time.perf_counter()

            # A watermark, not `steps % snapshot_every == 0`. The main loop
            # samples `learner.steps` once a round, and the step counter can
            # stride past an exact multiple -- always with --async-learner,
            # where a background thread advances it between samples, and
            # whenever --steps-per-train is not 1 and the phase lands on an
            # odd offset. Missing the multiple leaves the pool empty, and an
            # empty pool makes `snapshot_net` track the live net, quietly
            # turning 35% of the matchups into plain self-play.
            if learner.steps >= next_snapshot_at:
                next_snapshot_at = ((learner.steps // args.snapshot_every) + 1
                                    ) * args.snapshot_every
                if not snapshots or snapshots[-1]["steps"] != learner.steps:
                    # Cloned under the lock for the same reason `save` takes
                    # it: a snapshot spanning two optimiser steps would put a
                    # network that never existed into the opponent pool.
                    with learner.weights_lock:
                        snap_state = copy.deepcopy(
                            {k: v.detach().cpu().clone()
                             for k, v in net.state_dict().items()})
                    snapshots.append({"steps": learner.steps, "state": snap_state})
            # Reloading a snapshot copies parameters, so do it rarely.
            # Until the pool has anything in it, track the current network:
            # otherwise those seats would play with the random initial
            # weights for the whole warm-up, which is neither self-play nor
            # a useful opponent.
            if rounds % args.snapshot_refresh == 0:
                if snapshots:
                    pick = snapshots[int(rng.integers(0, len(snapshots)))]
                    snapshot_net.load_state_dict(pick["state"])
                else:
                    with learner.weights_lock:
                        snapshot_net.load_state_dict(net.state_dict())

            t_env += t1 - t0; t_feat += t2 - t1; t_fwd += t3 - t2
            t_book += t4 - t3; t_learn += t5 - t4
            rounds += 1

            if rounds % args.log_every == 0:
                el = time.perf_counter() - started
                row = {
                    "elapsed_s": round(el, 1), "rounds": rounds, "hands": hands_base + hands,
                    "decisions": decisions, "trained_decisions": trained_decisions,
                    "decisions_per_s": round(decisions / el, 1),
                    "hands_per_s": round(hands / el, 2),
                    "buffer": buffer.size, "learner_steps": learner.steps,
                    "snapshots": len(snapshots),
                    "attacker_points_mean": (
                        round(float(np.mean(recent_points[-500:])), 1) if recent_points else None),
                    "pct_env": round(100 * t_env / el, 1),
                    "pct_features": round(100 * t_feat / el, 1),
                    "pct_forward": round(100 * t_fwd / el, 1),
                    "pct_learn": round(100 * t_learn / el, 1),
                    # Buffer inserts, episode assembly and the pending map.
                    # Untimed until now, which is why these used to sum to 96.
                    "pct_bookkeeping": round(100 * t_book / el, 1),
                    "async_dropped": (async_learner.dropped
                                      if async_learner is not None else 0),
                    **{k: round(v, 5) for k, v in learner.stats.items()},
                }
                log_file.write(json.dumps(row) + "\n"); log_file.flush()
                # lambda only earns a column while it is actually moving.
                lam = (f" lam {row.get('lambda_mc')}"
                       if cfg.lambda_mc_final is not None else "")
                print(
                    f"[{row['elapsed_s']:7.1f}s] hands {row['hands']:7d} dec/s {row['decisions_per_s']:7.0f} "
                    f"steps {learner.steps:7d} loss {row.get('loss')} "
                    f"q {row.get('loss_q')} belief {row.get('belief_acc')} "
                    f"tgt_std {row.get('target_std')} resid_std {row.get('critic_residual_std')}{lam}",
                    flush=True)

            now = time.perf_counter()
            if sweep and buffer.size < cfg.min_buffer:
                # Warm-up is not a data point: hold the clock until the
                # buffer is full so every phase measures its own setting.
                sweep_at = now
                phase_dec = phase_trained = 0
                phase_start_step = learner.steps
            if sweep and now - sweep_at >= args.sweep_minutes * 60:
                took = now - sweep_at
                gained = learner.steps - phase_start_step
                print(f"[sweep] steps_per_train={cfg.steps_per_train} "
                      f"dec/s {phase_dec/took:7.0f} steps/s {gained/took:6.2f} "
                      f"replay {gained*cfg.batch_size/max(1,phase_trained):5.2f} "
                      f"| q {learner.stats.get('loss_q')} "
                      f"resid {learner.stats.get('critic_residual_std')} "
                      f"belief {learner.stats.get('belief_acc')}", flush=True)
                sweep_i = (sweep_i + 1) % len(sweep)
                cfg = replace(cfg, steps_per_train=sweep[sweep_i])
                if async_learner is not None:
                    async_learner.cfg = cfg
                sweep_at = now
                phase_dec = phase_trained = 0
                phase_start_step = learner.steps
            archive_due = learner.steps >= next_archive_at
            if learner.steps > 0 and (
                archive_due
                or now - last_checkpoint >= args.checkpoint_minutes * 60
            ):
                # Only the step-based trigger keeps a numbered archive.
                # The 10-minute one fires ~40x in a run, and with the
                # opponent pool inside, every one of those is a large
                # file nobody reads.
                save(net, critic, learner, cfg, net_cfg, args, hands_base + hands, snapshots,
                     archive=archive_due)
                if archive_due:
                    next_archive_at = ((learner.steps // args.checkpoint_every) + 1
                                       ) * args.checkpoint_every
                last_checkpoint = now
    finally:
        if async_learner is not None:
            async_learner.close()
        if prefetch is not None:
            prefetch.close()
        pool.close()
        save(net, critic, learner, cfg, net_cfg, args, hands_base + hands, snapshots, final=True)
        log_file.close()

    el = time.perf_counter() - started
    print(f"done: {hands} hands, {decisions} decisions ({trained_decisions} trained) "
          f"in {el/60:.1f} min, {learner.steps} steps")


# Snapshots are frozen *opponents*, never the weights being trained, so
# they are stored at half precision. It halves a file that mirrors to S3
# on a timer, and a few thousandths of drift in an opponent net is not
# something the learner can perceive.
def _to_half(state: dict) -> dict:
    return {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}


def _to_float(state: dict) -> dict:
    return {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}


def _atomic_save(payload: dict, path: Path) -> None:
    # The S3 mirror copies checkpoints on its own timer and cannot tell a
    # finished file from a half-written one; a truncated latest.pt is a
    # failed resume at the next boot. Rename is atomic, so the mirror
    # either sees the old file or the new one.
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save(net, critic, learner, cfg, net_cfg, args, hands, snapshots=(),
         final=False, archive=True) -> None:
    """Write a checkpoint, holding the weights lock for the whole write.

    `state_dict()` hands back references to the live tensors, not copies, so
    taking the lock only while building the payload would prove nothing --
    the optimiser would go on mutating them while `torch.save` serialised.
    Holding it across the write instead costs the learner thread well under
    a second every `--checkpoint-minutes`, and buys a checkpoint that is one
    coherent network rather than parameters from two different optimiser
    steps, which is a network that never existed and which a resume would
    then train from.
    """
    with learner.weights_lock:
        _save_locked(net, critic, learner, cfg, net_cfg, args, hands,
                     snapshots, final, archive)


def _save_locked(net, critic, learner, cfg, net_cfg, args, hands, snapshots=(),
                 final=False, archive=True) -> None:
    payload = {
        "arch": "v2",
        "model": net.state_dict(),
        "critic": critic.state_dict(),
        "net_config": asdict(net_cfg),
        "config": asdict(cfg),
        "steps": learner.steps,
        "hands": hands,
        # The opponent pool is training state, not a by-product. Left out,
        # a resume restarts self-play against copies of the current net
        # and needs snapshot_every * snapshot_pool steps -- 100k at the
        # current settings -- to rebuild its diversity, which is longer
        # than a spot instance has been living.
        "snapshots": [
            {"steps": s["steps"], "state": _to_half(s["state"])} for s in snapshots
        ],
        # RMSprop's running second moments. Without these a resume restarts
        # the optimiser from zero, so the first steps after every reclaim
        # take effectively uncalibrated step sizes. One 12-hour history-A/B
        # arm was reclaimed seven times, and the checkpoint taken 35k steps
        # after one of those resets measured worse than the checkpoint
        # 100k steps *before* it, on two independent opponents. Roughly
        # +17 MB on the file, which is cheap against that.
        "opt": learner.opt.state_dict(),
        "critic_opt": learner.critic_opt.state_dict(),
    }
    if archive:
        name = f"{args.run_name}_final.pt" if final else f"{args.run_name}_{learner.steps}.pt"
        _atomic_save(payload, CHECKPOINT_DIR / name)
    _atomic_save(payload, CHECKPOINT_DIR / f"{args.run_name}_latest.pt")


if __name__ == "__main__":
    main()
