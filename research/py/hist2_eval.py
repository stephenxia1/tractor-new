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
  resume that resets the in-process elapsed counter. Both arms are placed
  on a common origin of gradient step 0, extrapolated back from their own
  mirrored checkpoints, so equal hours really are equal machine time.

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


def step_zero(arm: str, written: dict[int, float]) -> float:
    """Epoch seconds at which this arm was at gradient step 0.

    The matched-wall-clock table only says anything if both arms are
    measured from the same instant, and measuring each from its own first
    mirrored checkpoint does not do that: GRID[0] is 50,000 gradient
    steps, not boot, and the arms reach it at different times. At the
    measured 8.4 and 15.9 steps/s that is 1.653 h for h2attn against
    0.874 h for h2none, so every h2attn row used to be credited about 47
    free minutes -- and h2attn is the slower arm, i.e. exactly the one the
    comparison exists to charge for its cost.

    So extrapolate each arm back to step 0 at its own throughput, fitted
    by least squares over its own (step, timestamp) points. That assumes
    throughput is constant across the run; a resume or a busy host breaks
    the assumption, and the boot itself is not measured at all. It is an
    estimate -- but the same estimate for both arms, which is what a
    comparison needs and what the per-arm origin never was.
    """
    pts = sorted(written.items())
    if len(pts) >= 2:
        n = len(pts)
        mean_step = sum(s for s, _ in pts) / n
        mean_ts = sum(t for _, t in pts) / n
        sxx = sum((s - mean_step) ** 2 for s, _ in pts)
        sxy = sum((s - mean_step) * (t - mean_ts) for s, t in pts)
        if sxx > 0 and sxy > 0:
            # sxy/sxx is seconds per gradient step; step 0 is that many
            # seconds per step before the mean point.
            return mean_ts - (sxy / sxx) * mean_step
    if not pts:
        return 0.0
    # Either one checkpoint, which fixes no rate, or a non-positive slope,
    # which means the timestamps do not increase with the step number --
    # a re-upload, a clock skew, a mirror that copied out of order. Both
    # leave nothing honest to extrapolate with, so fall back to this arm's
    # own earliest upload rather than invent a throughput. Earliest by
    # *timestamp*, not by step: in the non-monotone case the lowest-
    # numbered checkpoint is not the first one written, and using it would
    # put negative hours in the column.
    origin = min(t for _, t in pts)
    print(f"WARNING: no usable steps/hour for {arm} from its {len(pts)} "
          f"mirrored checkpoint(s). Its hours column is measured from that "
          f"arm's earliest mirrored checkpoint, not step 0, and is NOT "
          f"comparable with the other arm.")
    return origin


def fetch() -> dict:
    """Download every mirrored checkpoint and record when it was written."""
    import boto3

    s3 = boto3.Session(region_name="us-east-1").client("s3")
    meta: dict = {}
    for arm in ARMS:
        meta[arm] = {}
        written: dict[int, float] = {}
        for step in GRID:
            key = f"results/{arm}_{step}.pt"
            try:
                head = s3.head_object(Bucket=BUCKET, Key=key)
            except Exception:
                continue
            out = CHECKPOINT_DIR / f"{arm}_{step}.pt"
            if not out.exists():
                # checkpoints/ is gitignored, so it is absent on a tree where
                # no trainer has run -- which is exactly the tree someone runs
                # --fetch on.
                out.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(BUCKET, key, str(out))
            ts = head["LastModified"]
            written[step] = ts.timestamp()
            meta[arm][step] = {"path": str(out), "written": ts.isoformat()}
        origin = step_zero(arm, written)
        for step, rec in meta[arm].items():
            rec["hours"] = (written[step] - origin) / 3600
        print(f"{arm}: {len(meta[arm])} checkpoints "
              f"{sorted(meta[arm])}")
    grid = RUN_DIR / "hist2_grid.json"
    grid.parent.mkdir(parents=True, exist_ok=True)
    grid.write_text(json.dumps(meta, indent=2), encoding="utf-8")
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
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    for arm in ARMS:
        out.setdefault(arm, {})
        for step in sorted(int(s) for s in meta.get(arm, {})):
            rec = meta[arm][str(step)] if str(step) in meta[arm] else meta[arm][step]
            # `hours` comes from the grid, not from an evaluation, so it is
            # refreshed rather than set once: a results file written before
            # the step-0 origin above would otherwise keep serving the old
            # per-arm hours to --report forever, and the expensive arena
            # runs beside it would make it look freshly computed.
            out[arm].setdefault(str(step), {})
            out[arm][str(step)]["hours"] = rec["hours"]
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

        print("\nmatched wall clock (what a machine-hour buys; hours run "
              "from each arm's own extrapolated step 0)")
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
