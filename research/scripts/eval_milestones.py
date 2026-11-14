"""Evaluate a run's milestones as they appear, from a bridge job.

    python scripts/eval_milestones.py zero1 --first 1500 --last 4000

Every ``--every`` steps (250) from ``--first`` to ``--last``, once
``/vol/<run>/milestones/step_XXXXXXX.pt`` exists, this runs
``modal_eval.py::smart`` against ``--bots`` at ``--deals`` (770) with the
run's own declarations and burials (``--pre-play agent-policy``), and
every ``--random-every`` steps (500) with random pre-play as well. Each
record lands in ``runs/<run>-<step>-<policy|random>.jsonl`` -- on the
bridge's volume when run there -- and its ladder line is printed.

A bridge job can run for a day, where an agent session's background
shell is cut off after two hours. It needs the Modal CLI and a token,
which bridge jobs have.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

FAIR = "smart-v18,smart-v28,smart-v34"


def milestones(run: str) -> set[int]:
    out = subprocess.run(["modal", "volume", "ls", "tractor-rl", f"{run}/milestones"],
                         capture_output=True, text=True)
    return {int(m) for m in re.findall(r"step_(\d{7})\.pt", out.stdout)}


def evaluate(run: str, step: int, kind: str, bots: str, deals: int) -> int:
    job = f"{run}-{step}-{kind}"
    if os.path.exists(f"runs/{job}.jsonl") and os.path.getsize(f"runs/{job}.jsonl") > 0:
        print(f"{job}: already measured", flush=True)
        return 0
    cmd = ["modal", "run", "--detach", "modal_eval.py::smart",
           "--checkpoints", f"/vol/{run}/milestones/step_{step:07d}.pt",
           "--bots", bots, "--deals", str(deals),
           "--pre-play", "agent-policy" if kind == "policy" else "random",
           "--out", f"runs/{job}.jsonl"]
    env = {**os.environ, "TRACTOR_EVAL_APP": f"tractor-eval-{job}"}
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    text = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout + proc.stderr)
    ladder = [line.strip() for line in text.splitlines() if " ladder " in line]
    print(f"{job}: exit {proc.returncode}  {ladder[0] if ladder else 'no ladder line'}",
          flush=True)
    if proc.returncode != 0:
        print(text[-2000:], flush=True)
    return proc.returncode


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--first", type=int, default=250)
    ap.add_argument("--last", type=int, default=4000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--random-every", type=int, default=500)
    ap.add_argument("--bots", default=FAIR)
    ap.add_argument("--deals", type=int, default=770)
    ap.add_argument("--poll-s", type=int, default=180)
    args = ap.parse_args(argv)
    for step in range(args.first, args.last + 1, args.every):
        while step not in milestones(args.run):
            time.sleep(args.poll_s)
        print(f"[{time.strftime('%H:%M', time.gmtime())}] {args.run} milestone {step}",
              flush=True)
        evaluate(args.run, step, "policy", args.bots, args.deals)
        if args.random_every and step % args.random_every == 0:
            evaluate(args.run, step, "random", args.bots, args.deals)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
