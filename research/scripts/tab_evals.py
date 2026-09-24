"""Tabulate `modal_eval.py::smart` result files, and pair two of them.

    python scripts/tab_evals.py evals/2026-09-24/v4-final-*.jsonl
    python scripts/tab_evals.py A.jsonl B.jsonl --paired [--a step_0002000] [--b final]

One line per (checkpoint, pre-play, bot), then the mean over smart-v18 /
v28 / v34 -- the three fair search bots the README tables average.
`--paired` takes exactly two files measured on the same seed and deal
count and prints B - A hand for hand, bot by bot. Same seed means the
same deals, so the difference is valid for any two runs; it is tightest
when both sides also start play from the same positions (two
`--pre-play random` files), and wider when pre-play differs (two burial
modes), because the burial then changes the position being played.
"""

from __future__ import annotations

import argparse
import json
import math

Z = 1.959963984540054
FAIR = ("smart-v18", "smart-v28", "smart-v34")
DEFAULT_SEED = 20260923  # modal_eval.py::smart's default


def load(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def label(r: dict) -> tuple[str, str]:
    parts = r["checkpoint"].rstrip("/").split("/")
    ckpt = f"{parts[-3]}/{parts[-1].removesuffix('.pt')}" if len(parts) >= 3 else r["checkpoint"]
    mode = r.get("pre_play", "bot")
    if r.get("search"):
        mode += "+search"
    return ckpt, mode


def mean_ci(x: list[float]) -> tuple[float, float]:
    n = len(x)
    m = sum(x) / n
    var = sum((a - m) ** 2 for a in x) / max(n - 1, 1)
    return m, Z * math.sqrt(var / n)


def table(rows: list[dict]) -> None:
    print(f"{'checkpoint':24} {'pre-play':20} {'bot':10} {'win':>12} {'levels':>8} "
          f"{'points':>7} {'win LL':>7} {'win att':>7} {'net decl':>8}")
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in sorted(rows, key=lambda r: (label(r), r["bot"])):
        ck, mode = label(r)
        groups.setdefault((ck, mode), []).append(r)
        half = r["winrate"] - r["winrate_ci_low"]
        print(f"{ck:24} {mode:20} {r['bot']:10} {r['winrate']:6.3f}±{half:.3f} "
              f"{r['levels']:+8.3f} {r['points']:+7.1f} "
              f"{r.get('winrate_defending', float('nan')):7.3f} "
              f"{r.get('winrate_attacking', float('nan')):7.3f} "
              f"{r.get('net_declared_frac', float('nan')):8.2f}")
    print()
    print("mean over", ", ".join(FAIR))
    for (ck, mode), rs in groups.items():
        fair = [r for r in rs if r["bot"] in FAIR]
        if len(fair) != len(FAIR):
            continue
        wins = [w for r in fair for w in r["per_hand_wins"]]
        m, h = mean_ci(wins)
        avg = lambda k: sum(r.get(k, float("nan")) for r in fair) / len(fair)  # noqa: E731
        print(f"  {ck:24} {mode:20} win {m:.3f}±{h:.3f}  levels {avg('levels'):+.3f}  "
              f"points {avg('points'):+.1f}  win LL {avg('winrate_defending'):.3f}  "
              f"win att {avg('winrate_attacking'):.3f}")


def paired(a: list[dict], b: list[dict]) -> None:
    by_bot = {r["bot"]: r for r in a}
    print(f"{'bot':10} {'A':>6} {'B':>6} {'B - A win':>16} {'B - A points':>16}")
    pooled_w: list[float] = []
    pooled_p: list[float] = []
    for rb in sorted(b, key=lambda r: r["bot"]):
        ra = by_bot.get(rb["bot"])
        if ra is None:
            continue
        # Rows from before 2026-09-24 carry no seed; they used the default.
        seeds = ra.get("seed", DEFAULT_SEED), rb.get("seed", DEFAULT_SEED)
        if seeds[0] != seeds[1] or ra["deals"] != rb["deals"]:
            raise SystemExit(f"{rb['bot']}: different seed or deal count, not pairable")
        dw = [y - x for x, y in zip(ra["per_hand_wins"], rb["per_hand_wins"])]
        dp = [y - x for x, y in zip(ra["per_deal_points"], rb["per_deal_points"])]
        if rb["bot"] in FAIR:
            pooled_w += dw
            pooled_p += dp
        mw, hw = mean_ci(dw)
        mp, hp = mean_ci(dp)
        print(f"{rb['bot']:10} {ra['winrate']:6.3f} {rb['winrate']:6.3f} "
              f"{mw:+8.3f}±{hw:.3f} {mp:+9.2f}±{hp:.2f}")
    if pooled_w:
        mw, hw = mean_ci(pooled_w)
        mp, hp = mean_ci(pooled_p)
        print(f"{'v18-v34':10} {'':6} {'':6} {mw:+8.3f}±{hw:.3f} {mp:+9.2f}±{hp:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--paired", action="store_true")
    # A file holding several checkpoints: keep the rows whose checkpoint
    # path contains this text (e.g. `step_0002000`).
    ap.add_argument("--a", default="", help="checkpoint filter for the first file")
    ap.add_argument("--b", default="", help="checkpoint filter for the second file")
    args = ap.parse_args()
    if args.paired:
        if len(args.files) != 2:
            raise SystemExit("--paired takes exactly two files")
        a = [r for r in load(args.files[0]) if args.a in r["checkpoint"]]
        b = [r for r in load(args.files[1]) if args.b in r["checkpoint"]]
        paired(a, b)
    else:
        table([r for f in args.files for r in load(f)])


if __name__ == "__main__":
    main()
