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


def newest_run(rows: list[dict]) -> list[dict]:
    """Only the rows the most recent invocation appended.

    train_pop opens runs/<name>.jsonl in append mode so that --resume
    continues one logical run, which means a rerun under the same name
    inherits every earlier invocation's rows. Read whole, the file makes a
    run that collapsed early look like one that trained for hours: the row
    count, the first row the "kept collecting" check below diffs against,
    and the archive and iteration numbers all come from someone else's run.

    `rounds` only ever grows within one invocation and a fresh rerun starts
    it over, so the last place it fails to grow is where the newest one
    began. Not "the last place it drops": train_pop logs on
    `rounds % --log-every == 0`, so every fresh invocation's *first* row
    carries exactly --log-every rounds. A previous invocation that wrote a
    single row and then died therefore left a row holding the same number
    the next invocation's first row holds, and a strict `<` reads those two
    equal values as one continuous run -- the one case where the stale rows
    are also the ones most likely to be mistaken for a healthy start. Within
    an invocation the logged value is strictly increasing (one row per
    distinct multiple of --log-every), so `<=` cannot split a run in half.

    Two reruns stay invisible to that, and both would need a run-instance id
    in the rows, which lives on the trainer side: one that appended *nothing
    at all* -- the --agents mismatch SystemExit fires before the log file is
    even opened -- and a --resume, which carries `rounds` on from the
    checkpoint instead of restarting it, so its rows stay joined to the
    invocation they continue.
    """
    start = 0
    for i in range(1, len(rows)):
        if rows[i]["rounds"] <= rows[i - 1]["rounds"]:
            start = i
    return rows[start:]


def main(path: str) -> int:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        print("no metrics at all -- the run produced nothing")
        return 1
    dropped = len(rows)
    rows = newest_run(rows)
    if not rows:
        print("no metrics from the newest run -- the file holds only older ones")
        return 1
    dropped -= len(rows)
    if dropped:
        print(f"ignoring {dropped} row(s) from earlier invocations of this run name")
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
    # Diffed from the newest invocation's own first row: buffers start empty
    # on every invocation, so measuring from an earlier one's first row
    # compares a fresh buffer against a full one and reports growth nobody
    # observed. A --resume is the boundary newest_run cannot see, so there
    # this check still spans it.
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
