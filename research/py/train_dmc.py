"""DMC self-play training loop.

    PYTHONPATH=py python py/train_dmc.py --workers 4 --games 96 --minutes 30

Each round: every worker's pending decisions are merged, featurised and
scored in one forward pass; the chosen moves go back over the pipe; hands
that finished this round are converted into Monte-Carlo targets and pushed
into the replay buffer; the learner takes a few gradient steps.

Decisions are held in `pending` until their hand ends, because a DMC target
is only known once the hand is over.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from tractor_rl import layout as L
from tractor_rl.dmc import DMCConfig, Learner, ReplayBuffer, hand_returns
from tractor_rl.features import (
    action_features,
    field_u8,
    field_u16,
    field_u32,
    merge_batches,
    parse_episodes,
    split_per_worker,
    state_features,
)
from tractor_rl.model import ModelConfig, QNet, epsilon_greedy, group_argmax
from tractor_rl.protocol import WorkerConfig, WorkerPool

RESEARCH_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = RESEARCH_DIR / "checkpoints"
RUN_DIR = RESEARCH_DIR / "runs"


class RoundStore:
    """Copies of one round's states and chosen actions, kept alive until
    every hand that touched them has finished."""

    __slots__ = ("states", "actions")

    def __init__(self, states: np.ndarray, actions: np.ndarray) -> None:
        self.states = states
        self.actions = actions


DEFAULTS = DMCConfig()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games", type=int, default=96)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epsilon", type=float, default=DEFAULTS.epsilon)
    ap.add_argument("--lr", type=float, default=DEFAULTS.lr)
    ap.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    ap.add_argument("--train-every", type=int, default=DEFAULTS.train_every)
    ap.add_argument("--steps-per-train", type=int, default=DEFAULTS.steps_per_train)
    ap.add_argument("--preplay", type=str, default="smart-v34")
    ap.add_argument("--log-every", type=int, default=50, help="rounds between log lines")
    ap.add_argument("--checkpoint-every", type=int, default=2000, help="learner steps")
    ap.add_argument("--run-name", type=str, default="dmc")
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RUN_DIR.mkdir(exist_ok=True)

    cfg = DMCConfig(
        epsilon=args.epsilon,
        lr=args.lr,
        batch_size=args.batch_size,
        train_every=args.train_every,
        steps_per_train=args.steps_per_train,
    )
    device = torch.device("cpu")
    model = QNet(ModelConfig()).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        print(f"resumed from {args.resume} (step {state.get('steps', 0)})")
    learner = Learner(model, cfg, device)
    buffer = ReplayBuffer(cfg.buffer_capacity)
    rng = np.random.default_rng(args.seed)

    pool = WorkerPool(
        args.workers,
        WorkerConfig(games=args.games, seed=args.seed, preplay=args.preplay),
    )
    print(
        f"model {model.num_parameters() / 1e6:.2f}M params | "
        f"{args.workers} workers x {args.games} games | threads {args.threads}"
    )

    log_path = RUN_DIR / f"{args.run_name}.jsonl"
    log_file = log_path.open("a", encoding="utf-8")

    pending: dict[tuple[int, int, int], list[tuple[RoundStore, int, int, float]]] = {}
    started = time.perf_counter()
    deadline = started + args.minutes * 60.0
    rounds = 0
    decisions = 0
    hands = 0
    recent_points: list[int] = []
    recent_attacker_wins: list[int] = []
    t_env = t_feat = t_forward = t_learn = 0.0

    try:
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            batches = pool.read_round()
            merged = merge_batches(batches)
            t1 = time.perf_counter()

            if merged.num_decisions == 0:
                pool.write_round(batches, [np.zeros(0, dtype=np.int64)] * len(batches))
                continue

            sf_np = state_features(merged.states)
            af_np = action_features(merged.cands, merged.states, merged.owner)
            t2 = time.perf_counter()

            with torch.no_grad():
                q = model(
                    torch.from_numpy(sf_np),
                    torch.from_numpy(af_np),
                    torch.from_numpy(merged.owner),
                ).numpy()
            best = group_argmax(q, merged.owner, merged.cand_offsets, merged.cand_counts)
            picks = epsilon_greedy(best, merged.cand_counts, cfg.epsilon, rng)
            pool.write_round(batches, split_per_worker(picks, merged.bounds))
            t3 = time.perf_counter()

            # --- stash this round's decisions until their hands finish ---
            store = RoundStore(
                merged.states.copy(),
                merged.cands[merged.cand_offsets + picks].copy(),
            )
            game_ids = field_u16(merged.states, L.H_GAME_ID)
            hand_ids = field_u32(merged.states, L.H_HAND_ID)
            trick_idx = field_u8(merged.states, L.H_TRICK_INDEX)
            is_attacker = field_u8(merged.states, L.H_I_AM_ATTACKER)
            worker_of = np.searchsorted(merged.bounds[1:], np.arange(merged.num_decisions), side="right")
            for i in range(merged.num_decisions):
                key = (int(worker_of[i]), int(game_ids[i]), int(hand_ids[i]))
                sign = 1.0 if is_attacker[i] else -1.0
                pending.setdefault(key, []).append((store, i, int(trick_idx[i]), sign))
            decisions += merged.num_decisions

            # --- finished hands become Monte-Carlo targets ---
            for w, batch in enumerate(batches):
                for ep in parse_episodes(batch.episodes):
                    hands += 1
                    recent_points.append(ep.attacker_points)
                    recent_attacker_wins.append(0 if ep.defenders_won else 1)
                    entries = pending.pop((w, ep.game_id, ep.hand_id), None)
                    if not entries:
                        continue
                    returns = hand_returns(ep, cfg)
                    if returns.shape[0] == 0:
                        continue
                    states = np.stack([e[0].states[e[1]] for e in entries])
                    actions = np.stack([e[0].actions[e[1]] for e in entries])
                    tricks = np.clip(
                        np.fromiter((e[2] for e in entries), dtype=np.int64, count=len(entries)),
                        0,
                        returns.shape[0] - 1,
                    )
                    signs = np.fromiter(
                        (e[3] for e in entries), dtype=np.float32, count=len(entries)
                    )
                    buffer.add(states, actions, (returns[tricks] * signs).astype(np.float32))

            # --- learner ---
            t4 = time.perf_counter()
            if rounds % cfg.train_every == 0:
                for _ in range(cfg.steps_per_train):
                    learner.step(buffer, rng)
            t5 = time.perf_counter()

            t_env += t1 - t0
            t_feat += t2 - t1
            t_forward += t3 - t2
            t_learn += t5 - t4
            rounds += 1

            if rounds % args.log_every == 0:
                elapsed = time.perf_counter() - started
                row = {
                    "elapsed_s": round(elapsed, 1),
                    "rounds": rounds,
                    "hands": hands,
                    "decisions": decisions,
                    "decisions_per_s": round(decisions / elapsed, 1),
                    "hands_per_s": round(hands / elapsed, 2),
                    "buffer": buffer.size,
                    "learner_steps": learner.steps,
                    "loss": None if learner.steps == 0 else round(learner.last_loss, 5),
                    "attacker_points_mean": (
                        round(float(np.mean(recent_points[-500:])), 1) if recent_points else None
                    ),
                    "attacker_win_rate": (
                        round(float(np.mean(recent_attacker_wins[-500:])), 3)
                        if recent_attacker_wins
                        else None
                    ),
                    "pending_hands": len(pending),
                    "pct_env": round(100 * t_env / elapsed, 1),
                    "pct_features": round(100 * t_feat / elapsed, 1),
                    "pct_forward": round(100 * t_forward / elapsed, 1),
                    "pct_learn": round(100 * t_learn / elapsed, 1),
                }
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()
                print(
                    f"[{row['elapsed_s']:7.1f}s] hands {hands:6d} "
                    f"dec/s {row['decisions_per_s']:7.0f} "
                    f"buf {buffer.size:7d} steps {learner.steps:6d} "
                    f"loss {row['loss']} "
                    f"atk_pts {row['attacker_points_mean']} "
                    f"env/feat/fwd/learn "
                    f"{row['pct_env']:.0f}/{row['pct_features']:.0f}/"
                    f"{row['pct_forward']:.0f}/{row['pct_learn']:.0f}%"
                )

            if learner.steps > 0 and learner.steps % args.checkpoint_every == 0:
                save_checkpoint(model, learner, cfg, args, hands)
    finally:
        pool.close()
        save_checkpoint(model, learner, cfg, args, hands, final=True)
        log_file.close()

    elapsed = time.perf_counter() - started
    print(
        f"done: {hands} hands, {decisions} decisions in {elapsed / 60:.1f} min "
        f"({decisions / elapsed:,.0f} decisions/s), {learner.steps} learner steps"
    )


def save_checkpoint(model, learner, cfg, args, hands: int, final: bool = False) -> None:
    name = f"{args.run_name}_final.pt" if final else f"{args.run_name}_{learner.steps}.pt"
    path = CHECKPOINT_DIR / name
    torch.save(
        {
            "model": model.state_dict(),
            "steps": learner.steps,
            "hands": hands,
            "config": asdict(cfg),
        },
        path,
    )
    latest = CHECKPOINT_DIR / f"{args.run_name}_latest.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "steps": learner.steps,
            "hands": hands,
            "config": asdict(cfg),
        },
        latest,
    )


if __name__ == "__main__":
    main()
