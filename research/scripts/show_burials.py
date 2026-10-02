"""Print `modal_perf.py::sample_burials` output readably.

    python scripts/show_burials.py evals/2026-10-02/burials-pop8c-2800.json [N]
"""
import json
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\xiast\OneDrive\Documents\GitHub\tractor-new\research\py")
from tractor.rules.cards import POINTS, RANK_NAMES, Ctx, card_name  # noqa: E402

SYM = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
SUIT_ORDER = ["S", "H", "D", "C"]
REL = {-1: "nobody (kitty flip)", 0: "the landlord", 1: "the opponent after",
       2: "the partner", 3: "the opponent before"}


def name(t):
    n = card_name(t)
    return n if n in ("SJ", "BJ") else n[:-1] + SYM[n[-1]]


def cards_of(counts):
    return [t for t in range(54) for _ in range(int(counts[t]))]


def grouped(counts, ctx, mark=None):
    """Trumps strongest first, then each plain suit high to low."""
    groups = {"T": [], **{s: [] for s in SUIT_ORDER}}
    for t in cards_of(counts):
        key = "T" if ctx.is_trump[t] else card_name(t)[-1]
        groups[key].append(t)
    out = []
    for key in ["T"] + SUIT_ORDER:
        ts = groups[key]
        if not ts:
            out.append(f"{'trump' if key == 'T' else SYM[key]}: —")
            continue
        ts.sort(key=lambda t: -float(ctx.strength[t]))
        out.append(f"{'trump' if key == 'T' else SYM[key]}({len(ts)}): " + " ".join(name(t) for t in ts))
    return "   ".join(out)


def main():
    path = sys.argv[1]
    n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    d = json.load(open(path))
    hands = d["hands"]
    for i, h in enumerate(hands[:n_show]):
        ctx = Ctx(h["level_rank"], h["trump_suit"])
        trump = "no trump" if h["trump_suit"] == 4 else SYM[SUIT_ORDER[h["trump_suit"]]]
        pool = np.array(h["pool"])
        print(f"\n### Hand {i + 1}: agent {h['agent']} is landlord; level {RANK_NAMES[h['level_rank']]}, "
              f"trump {trump}, declared by {REL[h['declarer_rel']]}")
        print("33 cards:  " + grouped(pool, ctx))
        for kind in ("greedy", "sampled", "heuristic"):
            b = np.array(h[kind])
            kept = pool - b
            label = {"greedy": "policy (greedy, as evaluated)", "sampled": "policy (sampled, as trained)",
                     "heuristic": "heuristic"}[kind]
            print(f"  {label:31} buries: {' '.join(name(t) for t in sorted(cards_of(b), key=lambda t: (not ctx.is_trump[t], card_name(t)[-1], -float(ctx.strength[t]))))}"
                  f"   [{h['points'][kind]} pts, critic {h['critic'][kind]:+.3f}]")
        same = int(np.minimum(np.array(h["greedy"]), np.array(h["heuristic"])).sum())
        print(f"  greedy and heuristic share {same} of 8 cards")

    # Over every hand.
    g = np.array([h["critic"]["greedy"] for h in hands])
    s = np.array([h["critic"]["sampled"] for h in hands])
    he = np.array([h["critic"]["heuristic"] for h in hands])
    overlap = np.array([np.minimum(np.array(h["greedy"]), np.array(h["heuristic"])).sum() for h in hands])
    tr = lambda h, k: sum(int(c) for t, c in enumerate(h[k]) if Ctx(h["level_rank"], h["trump_suit"]).is_trump[t])
    voids = lambda h, k: sum(1 for su in range(4)
                             if any(h["pool"][t] and not Ctx(h["level_rank"], h["trump_suit"]).is_trump[t] and t // 13 == su for t in range(52))
                             and not any((h["pool"][t] - h[k][t]) and not Ctx(h["level_rank"], h["trump_suit"]).is_trump[t] and t // 13 == su for t in range(52)))
    print(f"\n### Over all {len(hands)} hands")
    print(f"critic value, landlord seat: greedy {g.mean():+.4f}, sampled {s.mean():+.4f}, heuristic {he.mean():+.4f}; "
          f"greedy > heuristic in {np.mean(g > he):.0%} of hands")
    print(f"greedy and heuristic share on average {overlap.mean():.1f} of 8 cards; identical in {np.mean(overlap == 8):.0%}")
    for k in ("greedy", "heuristic"):
        print(f"{k:9}: points buried {np.mean([h['points'][k] for h in hands]):.2f}, trumps buried "
              f"{np.mean([tr(h, k) for h in hands]):.2f}, plain suits voided {np.mean([voids(h, k) for h in hands]):.2f}")


main()
