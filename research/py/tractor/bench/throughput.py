"""Measured cost against predicted cost.

Every figure this prints is paired with the estimate it is testing and
the ratio between them. A benchmark that only reports timings leaves
the reader to remember what was expected, which is how an estimate that
is wrong by a factor of two survives for weeks.

    python -m tractor.bench.throughput --hands 200
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass

import numpy as np

from ..gen.episode import generate
from ..telemetry import Spans, memory_snapshot

# Predictions on record, so the table can score itself. Each is the
# figure quoted in the build plan or by the component's author.
PREDICTIONS = {
    "simulator_hands_per_s": 12.74,  # audit run, full enumeration at cap 4096
    "encode_ms": 0.40,  # features.py author's measurement
    "candidates_per_decision": 12.06,  # 265,032-decision audit
    "decisions_per_hand": 88.34,  # 265,032-decision audit
    "forced_decision_pct": 14.77,  # 265,032-decision audit
}


@dataclass
class Row:
    name: str
    measured: float
    predicted: float | None
    unit: str

    @property
    def ratio(self) -> float | None:
        if self.predicted in (None, 0):
            return None
        return self.measured / self.predicted


def _fmt(rows: list[Row]) -> str:
    width = max(len(r.name) for r in rows)
    out = [f"{'measurement'.ljust(width)}   measured   predicted     ratio  unit"]
    out.append("-" * (width + 42))
    for r in rows:
        pred = "-" if r.predicted is None else f"{r.predicted:9.3f}"
        ratio = "-" if r.ratio is None else f"{r.ratio:8.2f}x"
        out.append(
            f"{r.name.ljust(width)}   {r.measured:8.3f}   {pred}  {ratio}  {r.unit}"
        )
    return "\n".join(out)


def bench_simulator(hands: int, seed: int, cap: int) -> list[Row]:
    """Full self-play with a uniform policy, the environment side only."""
    rng = np.random.default_rng(seed)
    spans = Spans(enabled=True)

    decisions = 0
    forced = 0
    candidates = 0
    started = time.perf_counter()
    for _ in range(hands):
        with spans.span("generate"):
            state, _info = generate(rng, 2)
        while not state.done:
            with spans.span("legal_moves"):
                moves = state.legal_moves(cap=cap)
            decisions += 1
            candidates += len(moves)
            if len(moves) == 1:
                forced += 1
            pick = moves.moves[int(rng.integers(len(moves)))]
            with spans.span("step"):
                state.step(pick)
    elapsed = time.perf_counter() - started

    summary = spans.summary()
    per_decision_us = 1e6 * elapsed / max(decisions, 1)
    rows = [
        Row("simulator: hands/s", hands / elapsed, PREDICTIONS["simulator_hands_per_s"], "hands/s"),
        Row("simulator: decisions/s", decisions / elapsed, None, "dec/s"),
        Row("simulator: us/decision", per_decision_us, None, "us"),
        Row(
            "enumeration: candidates/decision",
            candidates / max(decisions, 1),
            PREDICTIONS["candidates_per_decision"],
            "",
        ),
        Row(
            "enumeration: decisions/hand",
            decisions / hands,
            PREDICTIONS["decisions_per_hand"],
            "",
        ),
        Row(
            "enumeration: forced decisions",
            100.0 * forced / max(decisions, 1),
            PREDICTIONS["forced_decision_pct"],
            "%",
        ),
    ]
    for name in ("generate", "legal_moves", "step"):
        got = summary.get(name)
        if got is None:
            continue
        share = 100.0 * got["total"] / elapsed
        rows.append(Row(f"  span {name}", share, None, "% of loop"))
    return rows


def bench_encoder(hands: int, seed: int, cap: int) -> list[Row]:
    """Observation encoding, which the loop pays once per decision."""
    try:
        from ..nn import features as FE
    except Exception as exc:  # pragma: no cover - module may not exist yet
        return [Row(f"encoder unavailable ({type(exc).__name__})", 0.0, None, "")]

    rng = np.random.default_rng(seed)
    cache = FE.Cache() if hasattr(FE, "Cache") else None
    count = 0
    started = time.perf_counter()
    for _ in range(hands):
        state, _info = generate(rng, 2)
        while not state.done:
            seat = state.to_act
            if cache is None:
                FE.encode(state, seat)
            else:
                FE.encode(state, seat, cache)
            count += 1
            moves = state.legal_moves(cap=cap)
            state.step(moves.moves[int(rng.integers(len(moves)))])
    elapsed = time.perf_counter() - started
    # The loop above also pays the simulator, so subtract its share using
    # the separately measured per-decision cost rather than pretending
    # this figure is the encoder alone.
    return [
        Row(
            "encoder+sim: ms/decision",
            1e3 * elapsed / max(count, 1),
            None,
            "ms",
        ),
        Row("encoder: observations", float(count), None, "count"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--cap", type=int, default=128)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    print(f"host      {platform.processor() or platform.machine()}")
    print(f"python    {platform.python_version()}")
    print(f"cap       {args.cap} candidates")
    print()

    rows = bench_simulator(args.hands, args.seed, args.cap)
    print(_fmt(rows))
    print()
    enc = bench_encoder(max(args.hands // 4, 10), args.seed + 1, args.cap)
    print(_fmt(enc))
    print()
    mem = memory_snapshot()
    print(f"host rss  {mem['host_rss_bytes'] / 1e6:.0f} MB")

    if args.out:
        payload = {
            r.name: {"measured": r.measured, "predicted": r.predicted, "unit": r.unit}
            for r in rows + enc
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
