"""Verdict on a population smoke run: did every agent actually train?

The failure this guards against is quiet. A population trainer that misroutes
seats, or files decisions under the wrong agent, or lets one agent starve
while the others learn, produces a loss curve that looks entirely normal --
because *some* agent is always learning. So the checks here are per-agent and
structural rather than "did the loss go down".

    PYTHONPATH=py python py/check_pop_smoke.py runs/popsmoke.jsonl
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

FAILS: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def main(path: str) -> int:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        print("no metrics at all -- the run produced nothing")
        return 1
    last = rows[-1]
    n = len(last["steps"])
    print(f"{len(rows)} metric rows, {n} agents, "
          f"{last['iteration']} iterations, {last['rounds']} rounds\n")

    check(n >= 2, f"more than one agent in the population ({n})")
    check(all(s > 0 for s in last["steps"]),
          f"every agent took gradient steps: {last['steps']}")
    check(all(b > 0 for b in last["buffers"]),
          f"every agent's buffer filled: {last['buffers']}")

    # Round-robin training should leave the agents within a step or two of
    # each other; a large spread means one agent is being starved of seats
    # or its decisions are being filed elsewhere.
    spread = max(last["steps"]) - min(last["steps"])
    check(spread <= max(3, 0.15 * max(last["steps"])),
          f"gradient steps are balanced across agents (spread {spread})")

    live = [d for d in last["loss_q"] if not math.isnan(d)]
    check(len(live) == n, f"every agent reported a loss ({len(live)}/{n})")
    check(all(0.0 < d < 10.0 for d in live),
          f"losses are finite and in range: {[round(x, 4) for x in live]}")

    beliefs = [b for b in last["belief_acc"] if not math.isnan(b)]
    check(all(0.3 < b < 1.0 for b in beliefs),
          f"belief accuracy is sane: {[round(b, 3) for b in beliefs]}")

    # Replay ratio is the number that says whether the cadence is right.
    replay = last["replay"]
    check(all(r > 0 for r in replay), f"every agent trained on its own data: {replay}")
    check(max(replay) < 4.5,
          f"replay is held near the target (max {max(replay)})")
    check(abs(max(replay) - min(replay)) < 1.0,
          f"replay is even across agents: {replay}")

    check(last["iteration"] >= 3,
          f"the run crossed several iteration boundaries ({last['iteration']})")
    check(last["archive"] > 0,
          f"the archive filled after burn-in ({last['archive']} entries)")

    # Every agent should have been seated over the run, not just at the start.
    first = rows[0]
    grew = [b - a for a, b in zip(first["buffers"], last["buffers"])]
    check(all(g > 0 for g in grew), f"every agent kept collecting: {grew}")

    dec = last["decisions_per_s"]
    check(dec > 0, f"decisions were produced ({dec:.0f}/s)")
    print(f"\ntime: env {last['pct_env']}% features {last['pct_features']}% "
          f"forward {last['pct_forward']}% book {last['pct_bookkeeping']}% "
          f"learn {last['pct_learn']}%")

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed")
        return 1
    print("population smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/popsmoke.jsonl"))
