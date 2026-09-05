"""Finds the fastest training configuration on the current machine.

Runs the real training loop for a short slice at each setting and reports
sustained decisions/s, so the answer accounts for CPU contention between
the TypeScript workers and torch rather than assuming they compose.

    PYTHONPATH=py python py/sweep_throughput.py --seconds 75

Settings that matter here: worker count (actors), games per worker (batch
size per forward pass), torch threads (contends with the workers), and
groups (>1 overlaps worker compute with Python compute instead of
alternating).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = RESEARCH_DIR / "runs"


def run_one(workers: int, games: int, threads: int, groups: int, seconds: float) -> dict:
    name = f"sweep_w{workers}_g{games}_t{threads}_gr{groups}"
    log = RUN_DIR / f"{name}.jsonl"
    if log.exists():
        log.unlink()
    cmd = [
        sys.executable,
        str(RESEARCH_DIR / "py" / "train_dmc.py"),
        "--workers", str(workers),
        "--games", str(games),
        "--threads", str(threads),
        "--groups", str(groups),
        "--minutes", str(seconds / 60.0),
        "--log-every", "25",
        "--run-name", name,
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
    else:
        steady = row["decisions_per_s"]
        steady_hands = row["hands_per_s"]
    return {
        "workers": workers,
        "games": games,
        "threads": threads,
        "groups": groups,
        "decisions_per_s": round(steady, 1),
        "hands_per_s": round(steady_hands, 2),
        "learner_steps": row["learner_steps"],
        "pct_env": row["pct_env"],
        "pct_features": row["pct_features"],
        "pct_forward": row["pct_forward"],
        "pct_learn": row["pct_learn"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=75.0, help="per configuration")
    args = ap.parse_args()

    configs = [
        # (workers, games, threads, groups)
        (4, 96, 8, 1),  # the current default, for reference
        (4, 96, 8, 2),  # same, with worker/Python overlap
        (6, 64, 8, 2),
        (6, 64, 6, 2),
        (8, 48, 6, 2),
        (4, 192, 8, 2),
        (3, 128, 10, 1),
    ]
    results = []
    for workers, games, threads, groups in configs:
        started = time.perf_counter()
        row = run_one(workers, games, threads, groups, args.seconds)
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
    (RUN_DIR / "sweep_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
