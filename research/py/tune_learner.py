"""Tunes the learner side, for machines where data is not the constraint.

`sweep_throughput.py` maximises decisions/s, which is the right objective
when the environment is the bottleneck. On a big box it stops being: the
actors produce far more experience than the learner can consume, and the
useful axis becomes gradient steps per second at a sane replay ratio.

Replay ratio here is (samples trained per second) / (decisions generated
per second). Around 1.0 each sample is used about once. Well below 1 the
box is throwing away experience it paid to generate; well above 1 it is
grinding the same samples and the actors are idle.

    PYTHONPATH=py python3 py/tune_learner.py --seconds 90
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


def run_one(workers, games, threads, train_every, steps_per_train, batch, seconds):
    name = f"tune_w{workers}_t{threads}_te{train_every}_sp{steps_per_train}_b{batch}"
    log = RUN_DIR / f"{name}.jsonl"
    if log.exists():
        log.unlink()
    cmd = [
        sys.executable, str(RESEARCH_DIR / "py" / "train_dmc.py"),
        "--workers", str(workers), "--games", str(games), "--threads", str(threads),
        "--groups", "2", "--train-every", str(train_every),
        "--steps-per-train", str(steps_per_train), "--batch-size", str(batch),
        "--minutes", str(seconds / 60.0), "--log-every", "50", "--run-name", name,
    ]
    subprocess.run(
        cmd, cwd=str(RESEARCH_DIR),
        env={**os.environ, "PYTHONPATH": str(RESEARCH_DIR / "py")},
        capture_output=True, check=False,
    )
    if not log.exists():
        return None
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) < 2:
        return None
    a, b = rows[0], rows[-1]
    dt = b["elapsed_s"] - a["elapsed_s"]
    if dt <= 0:
        return None
    dps = (b["decisions"] - a["decisions"]) / dt
    sps = (b["learner_steps"] - a["learner_steps"]) / dt
    return {
        "workers": workers, "games": games, "threads": threads,
        "train_every": train_every, "steps_per_train": steps_per_train,
        "batch": batch,
        "decisions_per_s": round(dps, 1),
        "steps_per_s": round(sps, 2),
        "samples_per_s": round(sps * batch, 1),
        "replay_ratio": round(sps * batch / max(dps, 1e-9), 3),
        "loss": b["loss"],
        "split": f"{b['pct_env']:.0f}/{b['pct_features']:.0f}/{b['pct_forward']:.0f}/{b['pct_learn']:.0f}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--cores", type=int, default=0)
    args = ap.parse_args()
    cores = args.cores or os.cpu_count() or 16

    # Shift cores from the actors to torch, and ask the learner for more
    # work per round, since experience is the abundant side here.
    configs = [
        # workers, games, threads, train_every, steps_per_train, batch
        (max(4, cores // 3), 64, cores - max(4, cores // 3), 2, 1, 512),
        (max(4, cores // 4), 64, cores - max(4, cores // 4), 1, 2, 512),
        (max(4, cores // 4), 64, cores - max(4, cores // 4), 1, 4, 512),
        (max(4, cores // 4), 64, cores - max(4, cores // 4), 1, 2, 1024),
        (max(4, cores // 6), 96, cores - max(4, cores // 6), 1, 4, 1024),
    ]
    print(f"tuning the learner on {cores} cores, {args.seconds:.0f}s per configuration\n")
    results = []
    for cfg in configs:
        row = run_one(*cfg, args.seconds)
        if row is None:
            print(f"w{cfg[0]} t{cfg[2]} every{cfg[3]} x{cfg[4]} b{cfg[5]}: no usable rows")
            continue
        results.append(row)
        print(
            f"w{row['workers']:>2} t{row['threads']:>2} every{row['train_every']} "
            f"x{row['steps_per_train']} b{row['batch']:>4}: "
            f"{row['decisions_per_s']:>8.0f} dec/s  {row['steps_per_s']:>6.2f} steps/s  "
            f"replay {row['replay_ratio']:>5.2f}  loss {row['loss']}  {row['split']}",
            flush=True,
        )

    if results:
        # Prefer the most gradient steps among configurations that are not
        # starving the learner of fresh data.
        healthy = [r for r in results if r["replay_ratio"] <= 2.0] or results
        best = max(healthy, key=lambda r: r["steps_per_s"])
        print(
            f"\nbest: {best['workers']} workers, {best['threads']} threads, "
            f"--train-every {best['train_every']} --steps-per-train {best['steps_per_train']} "
            f"--batch-size {best['batch']}  ->  {best['steps_per_s']:.2f} steps/s, "
            f"replay {best['replay_ratio']:.2f}"
        )
        (RUN_DIR / "tune_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
