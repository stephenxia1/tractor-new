"""Read the history-encoder A/B at matched steps and at matched wall clock.

The first attempt at this experiment could only compare the two arms where
they happened to finish -- 195k steps against 353k -- which cannot tell
"the encoder adds nothing" apart from "the encoder helps per step but not
enough to pay for being 1.8x slower". This reads a grid of checkpoints
mirrored at fixed step counts, so both questions get separate answers:

* **matched steps** -> sample efficiency. Does a gradient step with the
  encoder buy more than one without it?
* **matched wall clock** -> what a machine-hour is worth. The wall clock
  comes from S3's LastModified on each numbered upload, which survives the
  resume that resets the in-process elapsed counter.

Every evaluation is 1000 duplicate pairs on one deal base, which gives a
+/-2.5 point interval; 120 pairs gives +/-7 and cannot resolve anything
under about ten points, which is how a 6.7 point lead on three rungs once
turned out to be 0.13.

    PYTHONPATH=py python py/hist2_eval.py --fetch      # pull the grid
    PYTHONPATH=py python py/hist2_eval.py --evaluate
    PYTHONPATH=py python py/hist2_eval.py --report
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = RESEARCH_DIR / "checkpoints"
RUN_DIR = RESEARCH_DIR / "runs"
RESULTS = RUN_DIR / "hist2_eval.json"

BUCKET = "tractor-rl-staging-387293"
ARMS = {"h2attn": "history=attn", "h2none": "history=none"}
GRID = [50_000, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000]
RUNGS = ["smart-v2", "smart-v4"]
BASE = 424200
PAIRS = 1000

MARGIN = re.compile(r"point margin: ([-+\d.]+) pts \[95% CI ([-+\d.]+), ([-+\d.]+)\]")


def fetch() -> dict:
    """Download every mirrored checkpoint and record when it was written."""
    import boto3

    s3 = boto3.Session(region_name="us-east-1").client("s3")
    meta: dict = {}
    for arm in ARMS:
        meta[arm] = {}
        first = None
        for step in GRID:
            key = f"results/{arm}_{step}.pt"
            try:
                head = s3.head_object(Bucket=BUCKET, Key=key)
            except Exception:
                continue
            out = CHECKPOINT_DIR / f"{arm}_{step}.pt"
            if not out.exists():
                s3.download_file(BUCKET, key, str(out))
            ts = head["LastModified"]
            first = ts if first is None else min(first, ts)
            meta[arm][step] = {"path": str(out), "written": ts.isoformat()}
        # Wall clock is measured from this arm's own first mirrored
        # checkpoint, not from launch: boot and bootstrap are not training.
        for step, rec in meta[arm].items():
            import datetime
            t = datetime.datetime.fromisoformat(rec["written"])
            rec["hours"] = (t - first).total_seconds() / 3600 if first else 0.0
        print(f"{arm}: {len(meta[arm])} checkpoints "
              f"{sorted(meta[arm])}")
    (RUN_DIR / "hist2_grid.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def arena(path: str, opponent: str) -> tuple[float, float, float] | None:
    cmd = [sys.executable, "-u", "py/arena.py", "--agent", path,
           "--opponent", opponent, "--pairs", str(PAIRS), "--worker-pairs", "3",
           "--games", "48", "--threads", "6", "--base", str(BASE),
           "--max-seconds", "600"]
    proc = subprocess.run(cmd, cwd=RESEARCH_DIR, capture_output=True, text=True)
    m = MARGIN.search(proc.stdout)
    if not m:
        print(f"    no margin parsed for {path} vs {opponent}")
        print("   ", proc.stdout.strip().splitlines()[-2:] if proc.stdout else proc.stderr[-300:])
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def evaluate() -> None:
    meta = json.loads((RUN_DIR / "hist2_grid.json").read_text(encoding="utf-8"))
    out = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    for arm in ARMS:
        out.setdefault(arm, {})
        for step in sorted(int(s) for s in meta.get(arm, {})):
            rec = meta[arm][str(step)] if str(step) in meta[arm] else meta[arm][step]
            out[arm].setdefault(str(step), {"hours": rec["hours"]})
            for rung in RUNGS:
                if rung in out[arm][str(step)]:
                    continue
                r = arena(rec["path"], rung)
                if r is None:
                    continue
                out[arm][str(step)][rung] = r
                print(f"  {arm} @{step:>7,} vs {rung:9s} "
                      f"{r[0]:+7.2f} [{r[1]:+.2f}, {r[2]:+.2f}]")
                RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


def report() -> None:
    out = json.loads(RESULTS.read_text(encoding="utf-8"))
    for rung in RUNGS:
        print(f"\n=== vs {rung}, 1000 duplicate pairs, base {BASE} ===")
        print("\nmatched steps (sample efficiency)")
        print(f"  {'steps':>9s}{'attn':>22s}{'none':>22s}{'diff':>9s}")
        for step in GRID:
            a = out.get("h2attn", {}).get(str(step), {}).get(rung)
            b = out.get("h2none", {}).get(str(step), {}).get(rung)
            if not a and not b:
                continue
            fa = f"{a[0]:+7.2f} [{a[1]:+.1f},{a[2]:+.1f}]" if a else " " * 21
            fb = f"{b[0]:+7.2f} [{b[1]:+.1f},{b[2]:+.1f}]" if b else " " * 21
            d = f"{b[0] - a[0]:+9.2f}" if (a and b) else " " * 9
            print(f"  {step:9,}{fa:>22s}{fb:>22s}{d}")

        print("\nmatched wall clock (what a machine-hour buys)")
        print(f"  {'arm':8s}{'hours':>7s}{'steps':>9s}{'margin':>22s}")
        for arm in ARMS:
            for step in GRID:
                rec = out.get(arm, {}).get(str(step))
                if not rec or rung not in rec:
                    continue
                m = rec[rung]
                print(f"  {arm:8s}{rec['hours']:7.2f}{step:9,}"
                      f"{m[0]:+10.2f} [{m[1]:+.1f},{m[2]:+.1f}]")

    print("\nPre-registered reading:")
    print("  none == attn at matched steps          -> encoder adds nothing")
    print("  attn > none at steps, not at hours     -> more efficient, not worth 61%")
    print("  attn > none at both                    -> it earns its cost")
    print("  none > attn at matched steps           -> investigate before believing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    if a.evaluate:
        evaluate()
    if a.report or not (a.fetch or a.evaluate):
        report()
