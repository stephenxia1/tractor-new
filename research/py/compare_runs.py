"""Compare two arena runs *paired* on the deals they share.

Two mistakes this replaces, both of which I made while drawing conclusions
from this project.

**Reading overlapping intervals as "no difference".** They are not the same
thing. For two independent means the difference has
`SE = sqrt(se_a^2 + se_b^2)`, and intervals can overlap while the difference
is comfortably significant -- the history A/B at matched 100k steps came out
"marginal" by the overlap eyeball and `-3.93 [-7.36, -0.50]` when computed
properly.

**Throwing away the pairing.** `arena.py` already plays each deal twice with
the teams swapped so that deal luck cancels *within* a comparison. Two runs
against the same opponent from the same `--base` see the same deals, so the
same trick works one level up: match the runs on `deal_digest` and take the
per-deal difference. Deal variance cancels there too, and the interval on
the difference collapses -- typically several times narrower than treating
the runs as independent samples.

    PYTHONPATH=py python py/arena.py --agent A.pt ... --dump-pairs a.jsonl
    PYTHONPATH=py python py/arena.py --agent B.pt ... --dump-pairs b.jsonl
    PYTHONPATH=py python py/compare_runs.py a.jsonl b.jsonl --labels A B

Both runs must use the same `--opponent` and `--base`; the digests are the
proof, and the tool says how many deals actually matched.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        # A digest can repeat if the same deal was played more than once;
        # keep the first, so both sides drop the same extras.
        out.setdefault(rec["digest"], rec)
    return out


def interval(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan"), float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    half = 1.96 * math.sqrt(var / n)
    return mean, mean - half, mean + half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--field", default="points", choices=["points", "levels"])
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit(
            "no deals in common -- the two runs used a different --base or "
            "a different --opponent, so they cannot be compared paired")
    la, lb = args.labels

    va = [a[d][args.field] for d in shared]
    vb = [b[d][args.field] for d in shared]
    diff = [b[d][args.field] - a[d][args.field] for d in shared]

    ma, loa, hia = interval(va)
    mb, lob, hib = interval(vb)
    md, lod, hid = interval(diff)

    print(f"{len(shared)} deals in common "
          f"({len(a)} in {la}, {len(b)} in {lb})\n")
    print(f"  {la:<10s} {ma:+7.2f} [{loa:+.2f}, {hia:+.2f}]")
    print(f"  {lb:<10s} {mb:+7.2f} [{lob:+.2f}, {hib:+.2f}]")

    # The unpaired interval, for contrast: this is what treating the two runs
    # as independent samples would give.
    sea, seb = (hia - loa) / 2 / 1.96, (hib - lob) / 2 / 1.96
    sed = math.sqrt(sea ** 2 + seb ** 2)
    print(f"\n  {lb} - {la}, unpaired  {mb - ma:+7.2f} "
          f"[{mb - ma - 1.96 * sed:+.2f}, {mb - ma + 1.96 * sed:+.2f}]")
    print(f"  {lb} - {la}, paired    {md:+7.2f} [{lod:+.2f}, {hid:+.2f}]"
          f"   <- use this one")
    width_u, width_p = 2 * 1.96 * sed, hid - lod
    if width_p > 0:
        print(f"\n  pairing narrowed the interval {width_u / width_p:.1f}x")
    better = sum(1 for d in diff if d > 0)
    worse = sum(1 for d in diff if d < 0)
    print(f"  {lb} better on {better} deals, worse on {worse}, "
          f"level on {len(diff) - better - worse}")
    verdict = ("significant" if not (lod <= 0 <= hid) else "not significant")
    print(f"\n  difference is {verdict} at 95%")


if __name__ == "__main__":
    main()
