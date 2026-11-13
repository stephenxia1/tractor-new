"""Finds the fastest training configuration on the current machine.

Runs the real training loop for a short slice at each setting and reports
sustained decisions/s, so the answer accounts for CPU contention between
the TypeScript workers and torch rather than assuming they compose.

    PYTHONPATH=py python py/sweep_throughput.py --seconds 75
    PYTHONPATH=py python py/sweep_throughput.py --trainer train2 --seconds 120

For train2 the row also reports learner steps/s and the replay ratio
(samples drawn per decision trained on). Once the actors are fast, the
replay ratio is what keeps a run comparable to earlier ones: the v2
checkpoints were trained at about 0.79 (RESULTS.md, "Tuning flipped
objective").

Settings that matter here: worker count (actors), games per worker (batch
size per forward pass), torch threads (contends with the workers), and
groups (>1 overlaps worker compute with Python compute instead of
alternating).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = RESEARCH_DIR / "runs"


def run_one(workers: int, games: int, threads: int, groups: int, seconds: float,
            trainer: str = "train_dmc", extra: list[str] | None = None,
            batch_size: int = 512) -> dict:
    name = f"sweep_{trainer}_w{workers}_g{games}_t{threads}_gr{groups}"
    log = RUN_DIR / f"{name}.jsonl"
    if log.exists():
        log.unlink()
    cmd = [
        sys.executable,
        str(RESEARCH_DIR / "py" / f"{trainer}.py"),
        "--workers", str(workers),
        "--games", str(games),
        "--threads", str(threads),
        "--groups", str(groups),
        "--minutes", str(seconds / 60.0),
        "--log-every", "25",
        "--run-name", name,
        *(extra or []),
    ]
    env = {"PYTHONPATH": str(RESEARCH_DIR / "py")}
    import os

    subprocess.run(
        cmd,
        cwd=str(RESEARCH_DIR),
        env={**os.environ, **env},
        capture_output=True,
        check=False,
    )
    if not log.exists():
        return {"workers": workers, "games": games, "threads": threads, "groups": groups, "error": "no log"}
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"workers": workers, "games": games, "threads": threads, "groups": groups, "error": "no rows"}
    # Skip the first row: it carries worker startup and buffer fill.
    row = rows[-1]
    first = rows[0] if len(rows) > 1 else None
    if first is not None and row["elapsed_s"] > first["elapsed_s"]:
        span = row["elapsed_s"] - first["elapsed_s"]
        decisions = row["decisions"] - first["decisions"]
        hands = row["hands"] - first["hands"]
        steady = decisions / span
        steady_hands = hands / span
        steps_per_s = (row["learner_steps"] - first["learner_steps"]) / span
        trained = row.get("trained_decisions", 0) - first.get("trained_decisions", 0)
        replay = ((row["learner_steps"] - first["learner_steps"]) * batch_size / trained
                  if trained > 0 else None)
    else:
        steady = row["decisions_per_s"]
        steady_hands = row["hands_per_s"]
        steps_per_s = None
        replay = None
    return {
        "workers": workers,
        "games": games,
        "threads": threads,
        "groups": groups,
        "decisions_per_s": round(steady, 1),
        "hands_per_s": round(steady_hands, 2),
        "learner_steps": row["learner_steps"],
        "steps_per_s": None if steps_per_s is None else round(steps_per_s, 2),
        "replay": None if replay is None else round(replay, 2),
        "pct_env": row["pct_env"],
        "pct_features": row["pct_features"],
        "pct_forward": row["pct_forward"],
        "pct_learn": row["pct_learn"],
    }


def build_configs(cores: int) -> list[tuple[int, int, int, int]]:
    """Configurations to try, scaled to the machine.

    The split that matters is how many cores go to the Node actors versus
    torch: on the dev laptop the wrong split cost 3x, and the crossover is
    machine-specific, so the sweep brackets it rather than assuming it.
    Games per worker is chosen to keep the per-round batch near 400-800
    decisions, which is what the forward pass wants.
    """
    cores = max(4, cores)
    out: list[tuple[int, int, int, int]] = []
    for worker_share in (0.5, 0.65, 0.8):
        workers = max(2, int(round(cores * worker_share)))
        threads = max(2, cores - workers)
        games = max(16, int(round(512 / max(workers, 1))) * 2)
        out.append((workers, games, threads, 2))
        out.append((workers, games, threads, 1))
    # A torch-heavy corner, in case the learner rather than the actors is
    # the binding constraint (likely on a GPU box).
    workers = max(2, cores // 3)
    out.append((workers, 96, max(2, cores - workers), 2))
    # Deduplicate while preserving order.
    seen: set[tuple[int, int, int, int]] = set()
    unique = []
    for cfg in out:
        if cfg in seen:
            continue
        seen.add(cfg)
        unique.append(cfg)
    return unique


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=75.0, help="per configuration")
    ap.add_argument(
        "--cores",
        type=int,
        default=0,
        help="override the detected core count when sizing the sweep",
    )
    ap.add_argument("--trainer", choices=["train_dmc", "train2"], default="train_dmc")
    ap.add_argument("--extra", type=str, default="",
                    help="further trainer arguments, e.g. '--action-space sequential'")
    args = ap.parse_args()
    extra = args.extra.split()

    configs = build_configs(args.cores or os.cpu_count() or 8)
    print(f"sweeping {len(configs)} configurations on {os.cpu_count()} logical cores\n")
    results = []
    for workers, games, threads, groups in configs:
        started = time.perf_counter()
        row = run_one(workers, games, threads, groups, args.seconds, args.trainer, extra)
        row["wall_s"] = round(time.perf_counter() - started, 1)
        results.append(row)
        print(
            f"w{row['workers']} g{row['games']} t{row['threads']} groups{row['groups']}: "
            + (
                row["error"]
                if "error" in row
                else (
                    f"{row['decisions_per_s']:>7.0f} dec/s  {row['hands_per_s']:>5.1f} hands/s  "
                    f"env/feat/fwd/learn {row['pct_env']:.0f}/{row['pct_features']:.0f}/"
                    f"{row['pct_forward']:.0f}/{row['pct_learn']:.0f}%"
                    + (f"  {row['steps_per_s']} steps/s, replay {row['replay']}"
                       if row.get("steps_per_s") is not None else "")
                )
            ),
            flush=True,
        )

    best = max(
        (r for r in results if "error" not in r),
        key=lambda r: r["decisions_per_s"],
        default=None,
    )
    print()
    if best is not None:
        print(
            f"best: {best['workers']} workers x {best['games']} games, "
            f"{best['threads']} threads, {best['groups']} groups "
            f"-> {best['decisions_per_s']:.0f} decisions/s ({best['hands_per_s']:.1f} hands/s)"
        )
    RUN_DIR.mkdir(exist_ok=True)
    (RUN_DIR / f"sweep_results_{args.trainer}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
