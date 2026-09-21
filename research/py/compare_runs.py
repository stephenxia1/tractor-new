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

Both runs must use the same `--opponent`, `--preplay` and `--base`. The
digests prove only the cards: `computeDealDigest` is FNV over the dealt
hands, the kitty, the landlord and the trump rank, which makes it a pure
function of the deal seed and blind to which bot sat in the other two
seats. Two runs against different opponents therefore join just as
cleanly as two against the same one, and every point of the difference
would be filed against the agent. So arena.py stamps the opponent, the
pre-play bot and the base on each dumped row and this tool refuses to
compare rows that disagree; it also says how many deals actually matched.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


# The arena settings that have to match for a difference to be about the
# agent. The deal digest covers none of them.
IDENTITY = ("opponent", "preplay", "base")


def load(path: str) -> tuple[dict[int, dict], dict | None]:
    """Rows keyed by deal digest, plus the run identity they all agree on.

    The identity is None for a dump written before arena.py recorded it.
    Callers must treat that as "unchecked", not as "checked and fine".
    """
    out: dict[int, dict] = {}
    idents: list[dict] = []
    untagged = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        # A digest can repeat if the same deal was played more than once;
        # keep the first, so both sides drop the same extras.
        out.setdefault(rec["digest"], rec)
        if all(k in rec for k in IDENTITY):
            ident = {k: rec[k] for k in IDENTITY}
            if ident not in idents:
                idents.append(ident)
        else:
            untagged += 1
    # Every row carries the same stamp within one arena run, so more than
    # one distinct stamp -- or a mixture of stamped and unstamped rows --
    # means the file is two runs concatenated. Half a file cannot be the
    # A side of a paired comparison, so refuse rather than average them.
    if len(idents) > 1:
        raise SystemExit(
            f"{path} is not a single run: it carries {len(idents)} different "
            f"opponent/preplay/base settings ({idents})")
    if untagged and idents:
        raise SystemExit(
            f"{path} mixes {untagged} row(s) with no run identity into rows "
            f"tagged {idents[0]}; a concatenated dump cannot be compared")
    return out, (idents[0] if idents else None)


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

    a, ident_a = load(args.a)
    b, ident_b = load(args.b)
    la, lb = args.labels
    if ident_a is None or ident_b is None:
        # An older dump has nothing to check against, and the digests do
        # not stand in for it: they match on the cards alone. Say so
        # loudly rather than let the reader assume the guarantee held.
        untagged = [p for p, i in ((args.a, ident_a), (args.b, ident_b)) if i is None]
        print(f"WARNING: no opponent/preplay/base fields in "
              f"{' and '.join(untagged)}, so the two runs could NOT be "
              f"checked for a matching opponent. Re-run arena.py to get a "
              f"dump that can be. Until then a difference below may be the "
              f"opponent changing rather than the agent.\n")
    elif ident_a != ident_b:
        differ = ", ".join(f"{k} {ident_a[k]!r} vs {ident_b[k]!r}"
                           for k in IDENTITY if ident_a[k] != ident_b[k])
        raise SystemExit(
            f"{la} and {lb} are not comparable: {differ}. The deal digests "
            f"join on the cards alone, so this difference would be reported "
            f"as the agent's")
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        # One shared deal is worse than none: interval() has no variance to
        # work with and returns NaN bounds, every NaN comparison is False, and
        # the verdict below would read "significant at 95%" off an interval
        # that was never computed. A one-pair smoke run reaches here.
        raise SystemExit(
            f"{len(shared)} deal(s) in common -- the two runs must share a "
            "--base and an --opponent, and a paired interval needs at least "
            "two deals")

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
    # Belt and braces with the shared-deal guard above: a non-finite bound
    # makes every comparison False, which would otherwise read as "the
    # interval excludes zero" and print the strongest claim this tool makes.
    significant = (math.isfinite(lod) and math.isfinite(hid)
                   and not (lod <= 0 <= hid))
    verdict = "significant" if significant else "not significant"
    print(f"\n  difference is {verdict} at 95%")


if __name__ == "__main__":
    main()
