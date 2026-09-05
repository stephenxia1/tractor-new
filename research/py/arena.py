"""Duplicate-deal evaluation.

Tractor hands are dominated by the deal. A team handed four jokers and a
long trump suit wins with almost any policy, which is why this project's
heuristic lineage kept being fooled: one variant measured 56% over 32
games and 25% on a re-run with fresh seeds, and several "ties" were
indistinguishable from ±35% swings.

Duplicate scoring removes that. Every deal is played twice from identical
cards -- once with the agent on seats {0,2}, once on {1,3} -- and the two
runs are compared to each other rather than to an average. What is left is
the difference the *policy* made on those cards.

Two statistics per deal, both from the agent's point of view:

* **point margin** -- the attackers' final score when the agent attacked,
  minus their score when the opponent attacked, on the same cards. Finer
  grained, lower variance; the primary number.
* **level margin** -- the §8.3 level swing the agent earned across the two
  runs minus the opponent's. This is what the game is actually played for,
  but it is coarse (an 80-119 attacker win scores zero for both sides).

Usage:

    # trained checkpoint against a built-in bot
    PYTHONPATH=py python py/arena.py --agent checkpoints/dmc_latest.pt \
        --opponent smart-v34 --pairs 200

    # two heuristics against each other -- no network involved
    PYTHONPATH=py python py/arena.py --agent bot:smart-v34 \
        --opponent smart-v18 --pairs 200
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from tractor_rl.features import (
    Episode,
    action_features,
    merge_batches,
    parse_episodes,
    split_per_worker,
    state_features,
)
from tractor_rl.model import ModelConfig, QNet, group_argmax
from tractor_rl.protocol import WorkerConfig, WorkerPool

RESEARCH_DIR = Path(__file__).resolve().parents[1]

# Seat masks. Side A is the agent on seats 0 and 2, side B on 1 and 3.
SIDE_A_SEATS = 0b0101
SIDE_B_SEATS = 0b1010
SIDE_A_LIST = (0, 2)
SIDE_B_LIST = (1, 3)


@dataclass(slots=True)
class PairResult:
    point_margin: float
    level_margin: float
    agent_attack_points: int
    opponent_attack_points: int


def agent_defends(landlord_seat: int, agent_seats: tuple[int, ...]) -> bool:
    """The defending team is the landlord's. Teams are {0,2} and {1,3}, so
    the landlord's seat alone says which side is defending."""
    return landlord_seat in agent_seats


def signed_levels(ep: Episode, agent_is_defender: bool) -> tuple[int, int]:
    """(levels the agent gained, levels the opponent gained) for one hand."""
    if ep.defenders_won:
        return (ep.level_delta, 0) if agent_is_defender else (0, ep.level_delta)
    return (0, ep.level_delta) if agent_is_defender else (ep.level_delta, 0)


def score_pair(ep_a: Episode, ep_b: Episode) -> PairResult:
    """Compare the same deal played with the agent on each team.

    Exactly one of the two runs has the agent attacking, because the
    landlord seat is a property of the deal and does not move between runs.
    """
    a_defends = agent_defends(ep_a.landlord_seat, SIDE_A_LIST)
    b_defends = agent_defends(ep_b.landlord_seat, SIDE_B_LIST)
    if a_defends == b_defends:
        raise AssertionError(
            "the agent attacked in both runs or defended in both; the seat "
            "swap did not take effect"
        )
    if a_defends:
        agent_attack, opponent_attack = ep_b.attacker_points, ep_a.attacker_points
    else:
        agent_attack, opponent_attack = ep_a.attacker_points, ep_b.attacker_points

    agent_lv_a, opp_lv_a = signed_levels(ep_a, a_defends)
    agent_lv_b, opp_lv_b = signed_levels(ep_b, b_defends)
    return PairResult(
        point_margin=float(agent_attack - opponent_attack),
        level_margin=float((agent_lv_a + agent_lv_b) - (opp_lv_a + opp_lv_b)),
        agent_attack_points=agent_attack,
        opponent_attack_points=opponent_attack,
    )


def summarize(values: np.ndarray, label: str, unit: str) -> str:
    n = values.shape[0]
    if n == 0:
        return f"{label}: no pairs"
    mean = float(values.mean())
    if n > 1:
        se = float(values.std(ddof=1)) / math.sqrt(n)
        half = 1.96 * se
    else:
        se = half = float("nan")
    wins = int((values > 0).sum())
    losses = int((values < 0).sum())
    verdict = "better" if mean - half > 0 else "worse" if mean + half < 0 else "not distinguishable"
    return (
        f"{label}: {mean:+.2f} {unit} [95% CI {mean - half:+.2f}, {mean + half:+.2f}]  "
        f"({wins} pairs better / {losses} worse / {n - wins - losses} level) -> {verdict}"
    )


def load_agent(path: str, device: torch.device) -> QNet | None:
    """Returns None when `--agent bot:<name>` asked for a heuristic."""
    if path.startswith("bot:"):
        return None
    model = QNet(ModelConfig()).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    steps = state.get("steps", "?")
    hands = state.get("hands", "?")
    print(f"agent: {path} (learner steps {steps}, hands trained on {hands})")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--agent",
        required=True,
        help="checkpoint path, or bot:<BotType> to evaluate a heuristic",
    )
    ap.add_argument("--opponent", default="smart-v34", help="BotType for the other team")
    ap.add_argument("--pairs", type=int, default=200, help="deals played twice each")
    ap.add_argument("--worker-pairs", type=int, default=2, help="worker pairs (2 processes each)")
    ap.add_argument("--games", type=int, default=48, help="game slots per worker")
    ap.add_argument("--base", type=int, default=90210, help="deal seed base")
    ap.add_argument("--preplay", type=str, default="smart-v34")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-seconds", type=float, default=1800.0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = load_agent(args.agent, device)
    agent_bot = args.agent[4:] if model is None else None
    if agent_bot is not None:
        print(f"agent: bot {agent_bot}")
    print(f"opponent: {args.opponent}")

    # Worker i of side A and worker i of side B share a deal-seed base, so
    # they deal the same cards; different pairs get different bases so the
    # pairs are not all replaying one sequence of deals.
    base_cfg = WorkerConfig(
        games=args.games,
        preplay=args.preplay,
        baseline=args.opponent,
        randomize=False,
    )
    configs: list[WorkerConfig] = []
    for i in range(args.worker_pairs):
        pair_base = args.base + i * 1_000_003
        for seats, seat_list in ((SIDE_A_SEATS, SIDE_A_LIST), (SIDE_B_SEATS, SIDE_B_LIST)):
            if model is None:
                # No learner seats: the agent is itself a bot, so fill its
                # seats with it via the per-seat override.
                cfg = replace(
                    base_cfg,
                    seed=args.base + i,
                    learner_seats=0,
                    baseline=args.opponent,
                    baseline_alt=agent_bot,
                    alt_seats=seats,
                    deal_seed_base=pair_base,
                )
            else:
                cfg = replace(
                    base_cfg,
                    seed=args.base + i,
                    learner_seats=seats,
                    deal_seed_base=pair_base,
                )
            configs.append(cfg)

    pool = WorkerPool(len(configs), base_cfg, configs=configs)
    # Worker order is [pair0-sideA, pair0-sideB, pair1-sideA, ...].
    side_of = ["A" if w % 2 == 0 else "B" for w in range(len(configs))]
    pair_of = [w // 2 for w in range(len(configs))]

    waiting: dict[tuple[int, int, int], tuple[str, Episode]] = {}
    results: list[PairResult] = []
    digest_mismatches = 0
    hands_seen = 0
    started = time.perf_counter()

    try:
        while len(results) < args.pairs and time.perf_counter() - started < args.max_seconds:
            batches = pool.read_round()
            merged = merge_batches(batches)

            if merged.num_decisions > 0 and model is not None:
                sf = torch.from_numpy(state_features(merged.states)).to(device)
                af = torch.from_numpy(
                    action_features(merged.cands, merged.states, merged.owner)
                ).to(device)
                owner = torch.from_numpy(merged.owner).to(device)
                with torch.no_grad():
                    q = model(sf, af, owner).cpu().numpy()
                picks = group_argmax(q, merged.owner, merged.cand_offsets, merged.cand_counts)
            else:
                picks = np.zeros(merged.num_decisions, dtype=np.int64)
            pool.write_round(batches, split_per_worker(picks, merged.bounds))

            for w, batch in enumerate(batches):
                for ep in parse_episodes(batch.episodes):
                    hands_seen += 1
                    key = (pair_of[w], ep.game_id, ep.hand_id)
                    other = waiting.pop(key, None)
                    if other is None:
                        waiting[key] = (side_of[w], ep)
                        continue
                    other_side, other_ep = other
                    if other_side == side_of[w]:
                        raise AssertionError(f"two {other_side}-side hands for {key}")
                    # The deal digest is taken right after the shuffle, so
                    # matching digests mean the two runs really were the
                    # same cards -- the whole premise of the comparison.
                    if ep.deal_digest != other_ep.deal_digest:
                        digest_mismatches += 1
                        continue
                    ep_a, ep_b = (other_ep, ep) if other_side == "A" else (ep, other_ep)
                    results.append(score_pair(ep_a, ep_b))
    finally:
        pool.close()

    elapsed = time.perf_counter() - started
    if digest_mismatches:
        raise SystemExit(
            f"{digest_mismatches} pairs had mismatched deal digests -- the two "
            f"halves were not the same cards; fix the seeding before trusting any number"
        )
    if not results:
        raise SystemExit("no complete pairs; try more time or fewer pairs")

    points = np.array([r.point_margin for r in results], dtype=np.float64)
    levels = np.array([r.level_margin for r in results], dtype=np.float64)
    agent_attack = np.array([r.agent_attack_points for r in results], dtype=np.float64)
    opp_attack = np.array([r.opponent_attack_points for r in results], dtype=np.float64)

    agent_name = args.agent if model is not None else f"bot:{agent_bot}"
    print()
    print(f"{len(results)} duplicate pairs ({hands_seen} hands) in {elapsed / 60:.1f} min")
    print(f"  {agent_name}  vs  {args.opponent}")
    print("  " + summarize(points, "point margin", "pts"))
    print("  " + summarize(levels, "level margin", "levels"))
    print(
        f"  attacker points: agent {agent_attack.mean():.1f} / "
        f"opponent {opp_attack.mean():.1f} on the same deals"
    )
    print(
        f"  agent reached 80+ as attacker on {(agent_attack >= 80).mean():.1%} of deals, "
        f"opponent {(opp_attack >= 80).mean():.1%}"
    )


if __name__ == "__main__":
    main()
