"""Tunes the learner side, for machines where data is not the constraint.

`sweep_throughput.py` maximises decisions/s, which is the right objective
when the environment is the bottleneck. On a big box it stops being: the
actors produce far more experience than the learner can consume, and the
useful axis becomes training throughput at a sane replay ratio. That is
*samples* per second, not gradient steps: `batch` varies down the grid
below, so a step is not a constant unit of work and ranking on step count
alone would hand the win to whichever arm used the smallest batch.

Replay ratio here is (samples trained per second) / (decisions that reach
the buffer per second). Around 1.0 each sample is used about once. Well
below 1 the box is throwing away experience it paid to generate; well
above 1 it is grinding the same samples and the actors are idle. The
denominator is the *trained* decision count, not every decision played:
train2 discards the seats a snapshot opponent chose, and counting those
as fresh data hides between a fifth and a third of the grind, depending
on how the matchup shares round out at that worker count.

    PYTHONPATH=py python3 py/tune_learner.py --seconds 90

It tunes `py/train2.py` because that is what the answer gets used on: a
winner measured on the v1 loop -- no privileged critic, no belief head, no
causal-history attention -- describes a different cost split from the one
being configured. `--script` points it at another trainer that takes the
same flags.
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


def run_one(workers, games, threads, train_every, steps_per_train, batch, seconds, script):
    name = f"tune_w{workers}_t{threads}_te{train_every}_sp{steps_per_train}_b{batch}"
    log = RUN_DIR / f"{name}.jsonl"
    if log.exists():
        log.unlink()
    cmd = [
        sys.executable, str(RESEARCH_DIR / script),
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
    # The replay ratio has to be measured against the decisions that
    # actually reach the buffer, not against every decision the pool
    # produced. train2 splits each round into the rows the current net
    # chose and the rows a snapshot opponent chose, and stashes only the
    # former, so snapshot-played seats are generated, scored and thrown
    # away -- they can never be trained on. Dividing by `decisions`
    # therefore counts them as fresh data and understates the real grind:
    # the stored fraction is 0.6875 at 4 workers, 0.781 at 8 and 0.813 at
    # 12, so the old number was 1.23x-1.45x too low. train2 logs the
    # trained count for exactly this reason and its own sweep line divides
    # by it.
    #
    # train_dmc.py trains on every decision and its rows carry no such
    # field, and `--script` still accepts that trainer, so fall back to
    # `decisions` there -- and label the fallback wherever it is printed
    # or saved, so a number computed the old way is never read as one
    # computed the new way.
    trained_known = "trained_decisions" in a and "trained_decisions" in b
    tps = ((b["trained_decisions"] - a["trained_decisions"]) / dt
           if trained_known else dps)
    return {
        "workers": workers, "games": games, "threads": threads,
        "train_every": train_every, "steps_per_train": steps_per_train,
        "batch": batch,
        "decisions_per_s": round(dps, 1),
        "trained_decisions_per_s": round(tps, 1),
        "steps_per_s": round(sps, 2),
        "samples_per_s": round(sps * batch, 1),
        "replay_ratio": round(sps * batch / max(tps, 1e-9), 3),
        "replay_basis": "trained_decisions" if trained_known else "decisions",
        # train2 only puts the learner's stats in the row on its own cadence
        # (every `stats_every` learner steps, to keep a device sync off the
        # hot path), so a short slice can end on a row that has no loss at
        # all. It is a diagnostic here, not the objective, so its absence
        # must not throw the configuration away.
        "loss": b.get("loss"),
        "split": f"{b['pct_env']:.0f}/{b['pct_features']:.0f}/{b['pct_forward']:.0f}/{b['pct_learn']:.0f}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--cores", type=int, default=0)
    ap.add_argument("--script", type=str, default="py/train2.py",
                    help="trainer to tune, relative to research/")
    args = ap.parse_args()
    # os.cpu_count() is the host's core count inside a Modal container, not
    # the reservation -- measured 24 in a cpu=8 container, with no cgroup
    # quota and sched_getaffinity agreeing. Tuning a 24-core machine and
    # then running the winner on 8 reserved cores is the actor-starvation
    # pathology in RESULTS.md. OMP_NUM_THREADS is the one thing Modal does
    # set from the reservation, so believe it first.
    cores = args.cores or int(os.environ.get("OMP_NUM_THREADS") or 0) or os.cpu_count() or 16
    # OMP_NUM_THREADS is 1 on plenty of machines, and the configurations
    # below hand the actors `max(4, cores // n)` workers and torch the
    # rest: at four cores torch is left `--threads 0`, which it rejects
    # outright. Five is the smallest count that leaves it one. An explicit
    # --cores below that is a mistake worth saying out loud rather than
    # quietly tuning a machine the operator did not ask for.
    if cores < 5:
        if args.cores:
            raise SystemExit(
                f"--cores {args.cores} leaves torch no threads: every "
                f"configuration gives the actors at least 4 workers and torch "
                f"the remaining cores - 4"
            )
        print(f"detected {cores} core(s); tuning as 5, the smallest count that "
              f"leaves torch a thread", flush=True)
        cores = 5

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
    print(f"tuning {args.script} on {cores} cores, "
          f"{args.seconds:.0f}s per configuration\n")
    results = []
    for cfg in configs:
        row = run_one(*cfg, args.seconds, args.script)
        if row is None:
            print(f"w{cfg[0]} t{cfg[2]} every{cfg[3]} x{cfg[4]} b{cfg[5]}: no usable rows")
            continue
        results.append(row)
        basis = "" if row["replay_basis"] == "trained_decisions" else "*"
        print(
            f"w{row['workers']:>2} t{row['threads']:>2} every{row['train_every']} "
            f"x{row['steps_per_train']} b{row['batch']:>4}: "
            f"{row['decisions_per_s']:>8.0f} dec/s  "
            f"{row['trained_decisions_per_s']:>8.0f} trained/s  "
            f"{row['steps_per_s']:>6.2f} steps/s  "
            f"replay {row['replay_ratio']:>5.2f}{basis:1s} loss {row['loss']}  {row['split']}",
            flush=True,
        )
    if any(r["replay_basis"] != "trained_decisions" for r in results):
        print("\n* replay ratio taken against all decisions: this trainer's "
              "log rows carry no trained_decisions, so the ratio shown is a "
              "lower bound on the real one")

    if results:
        # Prefer the most *samples* per second among configurations that are
        # not starving the learner of fresh data. Steps/s is the wrong key
        # because `batch` is not constant down the grid: a batch-1024 arm
        # does twice the work in one gradient step, so it would have to win
        # on step count as well to be picked, which no amount of extra
        # throughput lets it do. steps/s x batch is the same unit in every
        # row and is what the learner actually consumes.
        healthy = [r for r in results if r["replay_ratio"] <= 2.0]
        if not healthy:
            # `or results` here would quietly switch the health filter off
            # and print the winner as though it had passed one. Say instead
            # that the whole grid is regrinding and the pick is the least
            # bad of it.
            print("\nno configuration held replay <= 2.0; every arm is "
                  "regrinding the buffer, so the pick below is the least "
                  "bad of an unhealthy grid, not a healthy winner")
            healthy = results
        best = max(healthy, key=lambda r: r["samples_per_s"])
        fallback = "" if best["replay_basis"] == "trained_decisions" else " (vs all decisions)"
        print(
            f"\nbest: {best['workers']} workers, {best['threads']} threads, "
            f"--train-every {best['train_every']} --steps-per-train {best['steps_per_train']} "
            f"--batch-size {best['batch']}  ->  {best['samples_per_s']:,.0f} samples/s "
            f"({best['steps_per_s']:.2f} steps/s), replay {best['replay_ratio']:.2f}{fallback}"
        )
        (RUN_DIR / "tune_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
