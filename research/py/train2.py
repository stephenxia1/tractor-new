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
import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from tractor_rl import layout as L
from tractor_rl.dmc2 import DMC2Config, Learner2, ReplayBuffer2, hand_returns
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
MATCHUPS = [
    ("self", 0.65, 0b1111, (0, 1, 2, 3), "smart-v34"),
    ("vs_snapshot", 0.20, 0b1111, (0, 2), "smart-v34"),
    ("mixed_partner", 0.10, 0b1111, (0,), "smart-v34"),
    ("vs_bot", 0.05, 0b0101, (0, 2), "smart-v2"),
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
    ap.add_argument("--batch-size", type=int, default=DMC2Config().batch_size)
    ap.add_argument("--train-every", type=int, default=DMC2Config().train_every)
    ap.add_argument("--steps-per-train", type=int, default=DMC2Config().steps_per_train)
    ap.add_argument("--snapshot-every", type=int, default=10000, help="learner steps")
    ap.add_argument("--snapshot-refresh", type=int, default=250, help="rounds between resamples")
    ap.add_argument("--snapshot-pool", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--checkpoint-every", type=int, default=100000)
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
        train_every=args.train_every, steps_per_train=args.steps_per_train)
    net_cfg = NetConfig(history=args.history)
    net = TractorNet(net_cfg).to(device)
    critic = PrivilegedCritic().to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        net.load_state_dict(state["model"])
        if "critic" in state:
            critic.load_state_dict(state["critic"])
        print(f"resumed from {args.resume} (step {state.get('steps', 0)})")
    learner = Learner2(net, critic, cfg, device)
    buffer = ReplayBuffer2(cfg.buffer_capacity, L.STATE_BYTES_PRIVILEGED)
    rng = np.random.default_rng(args.seed)

    snapshot_net = TractorNet(net_cfg).to(device)
    snapshot_net.load_state_dict(net.state_dict())
    snapshot_net.eval()
    snapshots: deque[dict] = deque(maxlen=args.snapshot_pool)

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
    pending: dict[tuple[int, int, int], list] = {}
    started = time.perf_counter()
    deadline = started + args.minutes * 60
    rounds = decisions = trained_decisions = hands = 0
    recent_points: list[int] = []
    t_env = t_feat = t_fwd = t_learn = 0.0

    def score(model, rows, merged, tokens_all, mask_all, sf_all, af_all):
        """Greedy candidate index for a subset of decisions, under `model`."""
        if rows.shape[0] == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        cand_mask, new_owner = subset_indices(rows, merged, merged.num_decisions)
        with torch.no_grad():
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
            trained_decisions += rows_cur.shape[0]

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
                for _ in range(cfg.steps_per_train):
                    learner.step(buffer, rng)
            t5 = time.perf_counter()

            if learner.steps > 0 and learner.steps % args.snapshot_every == 0:
                if not snapshots or snapshots[-1]["steps"] != learner.steps:
                    snapshots.append({
                        "steps": learner.steps,
                        "state": copy.deepcopy(
                            {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}),
                    })
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
                    snapshot_net.load_state_dict(net.state_dict())

            t_env += t1 - t0; t_feat += t2 - t1; t_fwd += t3 - t2; t_learn += t5 - t4
            rounds += 1

            if rounds % args.log_every == 0:
                el = time.perf_counter() - started
                row = {
                    "elapsed_s": round(el, 1), "rounds": rounds, "hands": hands,
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
                    **{k: round(v, 5) for k, v in learner.stats.items()},
                }
                log_file.write(json.dumps(row) + "\n"); log_file.flush()
                print(
                    f"[{row['elapsed_s']:7.1f}s] hands {hands:7d} dec/s {row['decisions_per_s']:7.0f} "
                    f"steps {learner.steps:7d} loss {row.get('loss')} "
                    f"q {row.get('loss_q')} belief {row.get('belief_acc')} "
                    f"tgt_std {row.get('target_std')} resid_std {row.get('critic_residual_std')}",
                    flush=True)

            if learner.steps > 0 and learner.steps % args.checkpoint_every == 0:
                save(net, critic, learner, cfg, net_cfg, args, hands)
    finally:
        pool.close()
        save(net, critic, learner, cfg, net_cfg, args, hands, final=True)
        log_file.close()

    el = time.perf_counter() - started
    print(f"done: {hands} hands, {decisions} decisions ({trained_decisions} trained) "
          f"in {el/60:.1f} min, {learner.steps} steps")


def save(net, critic, learner, cfg, net_cfg, args, hands, final=False) -> None:
    payload = {
        "arch": "v2",
        "model": net.state_dict(),
        "critic": critic.state_dict(),
        "net_config": asdict(net_cfg),
        "config": asdict(cfg),
        "steps": learner.steps,
        "hands": hands,
    }
    name = f"{args.run_name}_final.pt" if final else f"{args.run_name}_{learner.steps}.pt"
    torch.save(payload, CHECKPOINT_DIR / name)
    torch.save(payload, CHECKPOINT_DIR / f"{args.run_name}_latest.pt")


if __name__ == "__main__":
    main()
