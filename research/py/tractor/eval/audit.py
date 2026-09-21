"""Action-space audit.

Measures the distribution of legal moves per decision under real
play-phase positions, so the static tensor widths the network needs are
set from the observed tail rather than from a guess. A cap that binds is
not a performance knob: a move dropped by the enumerator is a move the
agent can never learn to make, so the rate at which each candidate cap
would bind is reported alongside the raw counts.

    python -m tractor.eval.audit --hands 2000
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from dataclasses import dataclass, field

import numpy as np

from ..gen.episode import generate
from ..rules.combos import decompose

# Large enough that it effectively never binds; reported if it ever does.
UNCAPPED = 4096

# Caps worth pricing. The tensor is padded to this width, so the cost of
# a wider one is only arithmetic in the scoring head.
CANDIDATE_CAPS = (16, 32, 48, 64, 96, 128, 192, 256)


@dataclass
class Audit:
    lead_counts: list[int] = field(default_factory=list)
    follow_counts: list[int] = field(default_factory=list)
    history_lengths: list[int] = field(default_factory=list)
    play_sizes: collections.Counter = field(default_factory=collections.Counter)
    throw_leads: int = 0
    forced: int = 0
    decisions: int = 0
    hands: int = 0
    tricks: int = 0
    hit_uncapped: int = 0
    unenumerated: int = 0
    seconds: float = 0.0


def run(hands: int, seed: int, num_decks: int = 2) -> Audit:
    rng = np.random.default_rng(seed)
    out = Audit()
    started = time.perf_counter()

    for _ in range(hands):
        state, _info = generate(rng, num_decks)
        plays_so_far = 0
        while not state.done:
            moves = state.legal_moves(cap=UNCAPPED)
            n = len(moves)
            out.decisions += 1
            out.history_lengths.append(plays_so_far)
            if moves.truncated:
                out.hit_uncapped += 1
            if moves.unenumerated:
                out.unenumerated += 1
            if n == 1:
                out.forced += 1
            if state.is_lead:
                out.lead_counts.append(n)
            else:
                out.follow_counts.append(n)

            choice = moves.moves[int(rng.integers(n))]
            played = state.step(choice)
            plays_so_far += 1
            out.play_sizes[int(played.sum())] += 1
            if state.is_lead and not state.done:
                parts = decompose(played, state.ctx, num_decks)
                if parts is not None and len(parts) > 1:
                    out.throw_leads += 1

        out.hands += 1
        out.tricks += len(state.tricks)

    out.seconds = time.perf_counter() - started
    return out


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {}
    a = np.array(values)
    keys = (50, 90, 99, 99.9, 100)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 2),
        **{f"p{k}": int(np.percentile(a, k)) for k in keys},
    }


def report(a: Audit) -> dict:
    allc = a.lead_counts + a.follow_counts
    counts = np.array(allc)
    binds = {
        cap: round(100.0 * float((counts > cap).mean()), 3) for cap in CANDIDATE_CAPS
    }
    hist = np.array(a.history_lengths)
    return {
        "hands": a.hands,
        "decisions": a.decisions,
        "decisions_per_hand": round(a.decisions / max(a.hands, 1), 2),
        "tricks_per_hand": round(a.tricks / max(a.hands, 1), 2),
        "hands_per_second": round(a.hands / max(a.seconds, 1e-9), 2),
        "forced_decision_pct": round(100.0 * a.forced / max(a.decisions, 1), 2),
        "throw_leads_per_hand": round(a.throw_leads / max(a.hands, 1), 2),
        "uncapped_limit_hit": a.hit_uncapped,
        "enumerator_fallbacks": a.unenumerated,
        "leads": _percentiles(a.lead_counts),
        "follows": _percentiles(a.follow_counts),
        "all": _percentiles(allc),
        "history_length": _percentiles(a.history_lengths),
        "max_history_observed": int(hist.max()) if hist.size else 0,
        "play_sizes": dict(sorted(a.play_sizes.items())),
        "pct_of_decisions_truncated_at_cap": binds,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    audit = run(args.hands, args.seed)
    result = report(audit)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
