"""Every follow the engine judged, judged again by the Python rules.

    npx tsx src/dumpFollowAll.ts 600 > follow_all.json        (from research/)
    python scripts/compare_follow_all.py follow_all.json

`tests/test_differential.py` checks a hundred random follows per
position. Few of those are legal, and the legal ones are where an
obligation rule can differ, so this checks every distinct follow of the
right size instead. Exits non-zero on any disagreement and prints the
first few positions where it happened.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "py"))
from tractor.rules import cards as C  # noqa: E402
from tractor.rules import combos as K  # noqa: E402
from tractor.rules import follow as F  # noqa: E402


def main() -> int:
    data = json.loads(Path(sys.argv[1]).read_text())
    decks = data["decks"]
    total = 0
    mismatches: Counter = Counter()
    shown = 0
    for i, case in enumerate(data["cases"]):
        ctx = C.Ctx(case["levelRank"], case["trumpSuit"])
        lead = C.counts_from_ids(case["lead"])
        hand = C.counts_from_ids(case["hand"])
        ob = F.lead_obligations(K.decompose(lead, ctx, decks), decks)
        k = int(lead.sum())
        for probe in case["probes"]:
            total += 1
            mine = bool(F.is_legal_follow(
                ob, k, case["ledSuit"], C.counts_from_ids(probe["cand"]), hand, ctx, decks))
            if mine == probe["ok"]:
                continue
            kind = "python accepts, engine refuses" if mine else "python refuses, engine accepts"
            mismatches[kind] += 1
            if shown < 5:
                shown += 1
                name = lambda ids: " ".join(C.card_name(s) for s in sorted(ids))  # noqa: E731
                print(f"case {i}: {kind}; level {C.RANK_NAMES[case['levelRank']]}, "
                      f"trump suit {case['trumpSuit']}, led suit {case['ledSuit']}")
                print(f"  lead {name(case['lead'])}\n  hand {name(case['hand'])}\n  cand {name(probe['cand'])}")
    legal = sum(p["ok"] for c in data["cases"] for p in c["probes"])
    print(f"{len(data['cases'])} positions, {total} follows ({legal} legal per the engine), "
          f"disagreements: {dict(mismatches) or 'none'}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
