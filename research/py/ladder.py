"""Runs one checkpoint against a list of opponents and prints the table.

    python py/ladder.py --checkpoint checkpoints/v2_latest.pt --label 966k

One rung cannot tell saturation from a plateau. A checkpoint that has
stopped improving and a checkpoint whose opponent is simply too weak to
separate look identical against that one opponent, and different across a
ladder -- the plateau is flat everywhere, the saturation only flat at the
easy end.

Every rung uses the same deal base, so two checkpoints run through this are
compared on identical cards.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parents[1]

# Cheap heuristics first: the determinization bots cost three orders of
# magnitude more per hand, so a run that has to be cut short still returns
# the fast rungs.
DEFAULT_RUNGS = [
    "random", "lowball", "point-hoarder", "aggressive",
    "smart", "smart-v2", "smart-v4", "smart-v9",
]

MARGIN = re.compile(r"point margin: ([-+\d.]+) pts \[95% CI ([-+\d.]+), ([-+\d.]+)\]")
LEVELS = re.compile(r"level margin: ([-+\d.]+) levels")
PAIRS = re.compile(r"(\d+) duplicate pairs")
ATTACK = re.compile(r"attacker points: agent ([\d.]+) / opponent ([\d.]+)")


def run_one(checkpoint: str, opponent: str, pairs: int, base: int,
            seconds: float, threads: int) -> dict | None:
    cmd = [
        sys.executable, "-u", "py/arena.py",
        "--agent", checkpoint, "--opponent", opponent,
        "--pairs", str(pairs), "--base", str(base),
        "--worker-pairs", "2", "--games", "48",
        "--threads", str(threads), "--max-seconds", str(seconds),
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=RESEARCH_DIR, capture_output=True, text=True)
    out = proc.stdout
    m = MARGIN.search(out)
    if m is None:
        print(f"  {opponent:<16} no result ({proc.returncode}) "
              f"{(proc.stderr or out).strip().splitlines()[-1][:80] if (proc.stderr or out).strip() else ''}")
        return None
    lv = LEVELS.search(out)
    pr = PAIRS.search(out)
    at = ATTACK.search(out)
    row = {
        "opponent": opponent,
        "margin": float(m.group(1)),
        "lo": float(m.group(2)),
        "hi": float(m.group(3)),
        "levels": float(lv.group(1)) if lv else None,
        "pairs": int(pr.group(1)) if pr else None,
        "agent_attack": float(at.group(1)) if at else None,
        "opp_attack": float(at.group(2)) if at else None,
        "seconds": round(time.perf_counter() - started, 1),
    }
    sig = "  " if row["lo"] <= 0 <= row["hi"] else ("+" if row["margin"] > 0 else "-")
    print(f"  {opponent:<16} {row['margin']:+7.2f}  [{row['lo']:+.2f}, {row['hi']:+.2f}] "
          f"{sig}  {row['pairs']} pairs, {row['seconds']:.0f}s")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--rungs", nargs="*", default=DEFAULT_RUNGS)
    ap.add_argument("--pairs", type=int, default=120)
    ap.add_argument("--base", type=int, default=90210)
    ap.add_argument("--seconds", type=float, default=900)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    print(f"{args.label}  ({args.checkpoint})  base {args.base}, {args.pairs} pairs")
    rows = []
    for opponent in args.rungs:
        row = run_one(args.checkpoint, opponent, args.pairs, args.base,
                      args.seconds, args.threads)
        if row is not None:
            rows.append(row)

    out = RESEARCH_DIR / "runs" / f"ladder_{args.label}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
