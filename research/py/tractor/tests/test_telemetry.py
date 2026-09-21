"""Telemetry: timers, memory, FLOP arithmetic and the metrics sink.

There is no GPU on the machine that runs this, so the CUDA branches are
exercised against a stub ``torch.cuda`` rather than skipped. The stub's
``Event.elapsed_time`` returns a fixed 12.5 ms, a value no wall clock
could produce, so a span that reports exactly 0.0125 s proves the event
path ran and the ``perf_counter`` path did not.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import torch

from tractor.telemetry import (
    PEAK_TFLOPS,
    Run,
    Spans,
    memory_snapshot,
    mfu,
    peak_tflops,
    transformer_flops,
)
from tractor import telemetry


# --------------------------------------------------------------------------
# 1. spans
# --------------------------------------------------------------------------


def test_spans_measure_a_known_sleep() -> None:
    spans = Spans(enabled=True, device="cpu")
    for _ in range(2):
        with spans.span("nap"):
            time.sleep(0.05)
    stat = spans.summary()["nap"]
    assert stat["count"] == 2, stat
    # Lower bound is tight (a sleep never returns early by much); the upper
    # bound is loose because Windows timer granularity and a loaded box
    # both overshoot, and a flaky timing test is worse than a loose one.
    assert 0.09 <= stat["total"] <= 0.40, stat
    assert abs(stat["mean"] - stat["total"] / 2) < 1e-12, stat
    # Nothing nested inside, so exclusive == inclusive.
    assert stat["exclusive"] == stat["total"], stat
    print(f"    two 50ms sleeps measured {stat['total']*1000:.1f}ms")

    spans.reset()
    assert spans.summary() == {}


def test_spans_nesting_splits_inclusive_from_exclusive() -> None:
    spans = Spans(enabled=True, device="cpu")
    with spans.span("outer"):
        time.sleep(0.02)
        with spans.span("inner"):
            time.sleep(0.04)
        with spans.span("inner"):
            time.sleep(0.04)

    summary = spans.summary()
    outer, inner = summary["outer"], summary["inner"]
    assert outer["count"] == 1 and inner["count"] == 2

    # The identity is exact arithmetic, not a timing claim: the outer
    # span's exclusive time is its inclusive time minus everything its
    # direct children spent. Asserting it exactly is what catches a
    # subtraction against the wrong parent.
    assert abs(outer["exclusive"] - (outer["total"] - inner["total"])) < 1e-12
    # A leaf has no children to subtract.
    assert inner["exclusive"] == inner["total"]
    # And the timings are in the right places: outer contains both inners,
    # but only ~20ms of the ~100ms is its own.
    assert outer["total"] > inner["total"] > 0.0
    assert 0.015 <= outer["exclusive"] <= 0.20, outer
    assert 0.07 <= inner["total"] <= 0.40, inner
    print(f"    outer incl {outer['total']*1000:.1f}ms  "
          f"excl {outer['exclusive']*1000:.1f}ms  "
          f"inner incl {inner['total']*1000:.1f}ms")


def test_disabled_spans_are_free_and_record_nothing() -> None:
    n = 50_000
    off = Spans(enabled=False)
    on = Spans(enabled=True, device="cpu")

    def loop(spans, reps: int) -> float:
        # Best of five: a single sample on a laptop measures the scheduler.
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(reps):
                with spans.span("x"):
                    pass
            best = min(best, time.perf_counter() - t0)
        return best

    def bare(reps: int) -> float:
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(reps):
                pass
            best = min(best, time.perf_counter() - t0)
        return best

    baseline = bare(n)
    cost_off = loop(off, n)
    cost_on = loop(on, n)
    per_off = (cost_off - baseline) / n
    per_on = (cost_on - baseline) / n

    assert off.summary() == {}, "a disabled span recorded something"
    assert off.dropped == 0
    # Relative, and only relative. Both loops are timed under whatever
    # load the machine is under, so their ratio survives a busy host
    # while an absolute bound does not: the same disabled span measured
    # 356ns, 405ns and 2058ns on three consecutive isolated runs here,
    # so a 1us threshold fails for reasons that have nothing to do with
    # the code. The ranges overlap badly enough that no absolute number
    # separates "the disabled path is doing real work" from "the
    # scheduler looked away", which is why the claim worth asserting is
    # that disabling a span makes it substantially cheaper.
    assert per_off * 3 < per_on, f"off {per_off*1e9:.0f}ns vs on {per_on*1e9:.0f}ns"
    # A pure smoke bound against the disabled path acquiring real work
    # such as I/O or a device call, which would cost orders of
    # magnitude more than any scheduling noise.
    assert per_off < 5e-5, f"{per_off*1e9:.0f}ns per disabled span"
    print(f"    per span: disabled {per_off*1e9:6.0f}ns   "
          f"enabled {per_on*1e9:6.0f}ns   ({per_on/max(per_off, 1e-12):.0f}x)")


# --------------------------------------------------------------------------
# 2. FLOP arithmetic
# --------------------------------------------------------------------------


def test_transformer_flops_matches_hand_arithmetic() -> None:
    # Hand-worked configuration: 2 layers, d_model 64, ff 256, 8 tokens,
    # 4 heads, batch 1. Every matmul is 2*M*N*K.
    #
    #   one projection   2 * 8 * 64 * 64  =    65,536   (x4 = 262,144)
    #   ffn up           2 * 8 * 64 * 256 =   262,144
    #   ffn down         2 * 8 * 256 * 64 =   262,144
    #   scores (Q@K.T)   2 * 8 * 8  * 64  =     8,192
    #   values (A@V)     2 * 8 * 8  * 64  =     8,192
    #                                        ---------
    #   per layer        262,144 + 524,288 + 16,384 =   802,816
    #   two layers                                   = 1,605,632
    total, parts = transformer_flops(2, 64, 256, 8, 4)
    assert total == 1_605_632, total
    assert parts["attn_q"] == 2 * 65_536
    assert parts["attn_q"] == parts["attn_k"] == parts["attn_v"] == parts["attn_out"]
    assert parts["ffn_up"] == parts["ffn_down"] == 2 * 262_144
    assert parts["attn_scores"] == parts["attn_values"] == 2 * 8_192
    assert sum(parts.values()) == total, "the breakdown must add up to the total"

    # heads partition d_model; they do not change the arithmetic.
    assert transformer_flops(2, 64, 256, 8, 1).total == total
    assert transformer_flops(2, 64, 256, 8, 8).total == total

    # Batch multiplies everything exactly once. The trap this guards is
    # passing batch*seq as `tokens`, which would square the batch into the
    # two attention terms instead: at batch 4 that is 1,605,632*4 here,
    # against 2*transformer_flops(2,64,256,32,4).parts["attn_scores"] being
    # 4x too large.
    assert transformer_flops(2, 64, 256, 8, 4, batch=4).total == 4 * total
    folded = transformer_flops(2, 64, 256, 32, 4)
    assert folded.parts["attn_scores"] == 4 * 4 * parts["attn_scores"]

    for bad in ((0, 64, 256, 8, 4), (2, 64, 256, 8, 0), (2, -1, 256, 8, 4)):
        try:
            transformer_flops(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"transformer_flops{bad} should have raised")
    try:
        transformer_flops(2, 64, 256, 8, 5)       # 64 % 5 != 0
    except ValueError:
        pass
    else:
        raise AssertionError("indivisible heads should have raised")

    print(f"    2x(d=64, ff=256, T=8): {total:,} FLOPs; "
          f"ffn share {(parts['ffn_up']+parts['ffn_down'])/total:.0%}")


# --------------------------------------------------------------------------
# 3. MFU
# --------------------------------------------------------------------------


def test_mfu_is_none_for_unknown_devices_and_a_fraction_for_known_ones() -> None:
    assert mfu(1e12, 0.1, "NVIDIA GeForce RTX 9090") is None
    assert mfu(1e12, 0.1, "Some Accelerator 3000") is None
    assert mfu(1e12, 0.1, "") is None
    assert mfu(1e12, 0.1, None) is None          # no CUDA on this machine
    # A device in the table with no hardware path for the dtype is also
    # None, not a guess: Turing has no bf16 units.
    assert peak_tflops("Tesla T4", "bf16") is None
    assert peak_tflops("Tesla T4", "tf32") is None
    assert peak_tflops("Tesla T4", "fp16") == 65.13

    # Hand-computed: 1e12 FLOPs in 0.1 s is 1e13 FLOP/s. An A100's dense
    # bf16 peak is 312 TFLOP/s, so 1e13 / 3.12e14 = 0.032051282...
    got = mfu(1e12, 0.1, "NVIDIA A100-SXM4-80GB", "bf16")
    assert got is not None and 0.0 < got <= 1.0
    assert abs(got - 1e13 / 312e12) < 1e-12, got
    assert abs(got - 0.03205128205128205) < 1e-15, got
    # Same work against the fp32 peak of 19.5 TFLOP/s: 1e13/1.95e13.
    fp32 = mfu(1e12, 0.1, "NVIDIA A100-SXM4-80GB", "fp32")
    assert abs(fp32 - 1e13 / 19.5e12) < 1e-12, fp32
    assert 0.0 < fp32 <= 1.0

    # Order in the table is load bearing: "NVIDIA A10G" contains "a10".
    # The two cards are genuinely different silicon and must not share a
    # row: the A10G has 80 SMs, the A10 has 72, an 12% gap that lands
    # straight in the MFU denominator. A measured dense GEMM on a Modal
    # A10G reaches 67.22 bf16 TFLOP/s, which is 96% of the figure below
    # and would have been 107% of the A10's.
    assert peak_tflops("NVIDIA A10G", "bf16") == 140.08
    assert peak_tflops("NVIDIA A10G", "fp32") == 35.02
    assert peak_tflops("NVIDIA A10", "bf16") == 125.0
    assert peak_tflops("NVIDIA A10", "fp32") == 31.2

    # The GA102 dense ratios these rows are built from: TF32 twice
    # fp32, BF16 four times it. A row that silently reverts to the
    # halved figures breaks this rather than merely looking low.
    # Relative, because the A10's fp32 entry is the datasheet's rounded
    # 31.2 while its tensor entries are the exact 62.5 and 125.
    for card in ("NVIDIA A10", "NVIDIA A10G"):
        base = peak_tflops(card, "fp32")
        for dtype, ratio in (("tf32", 2.0), ("bf16", 4.0)):
            got = peak_tflops(card, dtype)
            assert abs(got / (ratio * base) - 1.0) < 0.01, (card, dtype, got)
    # ... and the H100 variants must not collapse onto the generic key.
    # 756.5 is the datasheet's 1,513 TFLOPS halved, the same rule that
    # gives 989.5 from 1,979 and 835.5 from 1,671.
    assert peak_tflops("NVIDIA H100 PCIe", "bf16") == 756.5
    assert peak_tflops("NVIDIA H100 80GB HBM3", "bf16") == 989.5
    # "NVIDIA L40" must not fall through to the "l4" row, which would
    # read three times too fast.
    assert peak_tflops("NVIDIA L40", "bf16") == 181.05
    assert peak_tflops("NVIDIA L40S", "bf16") == 362.05
    assert peak_tflops("NVIDIA L4", "bf16") == 121.0

    # Nonsense inputs give None rather than inf or a division error.
    assert mfu(1e12, 0.0, "NVIDIA L4") is None
    assert mfu(1e12, -1.0, "NVIDIA L4") is None
    assert mfu(1e12, float("inf"), "NVIDIA L4") is None
    try:
        peak_tflops("NVIDIA L4", "int4")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown dtype should raise, not guess")

    # The five devices the brief names are all present with an fp32 number,
    # and every entry says where it came from.
    for name in ("Tesla T4", "NVIDIA L4", "NVIDIA A10G",
                 "NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3"):
        assert peak_tflops(name, "fp32") is not None, name
    for key, peak in PEAK_TFLOPS.items():
        assert peak.source, key
        assert peak.fp32 and peak.fp32 > 0, key

    print(f"    A100 bf16 MFU for 10 TFLOP/s: {got:.1%}   fp32: {fp32:.1%}")


# --------------------------------------------------------------------------
# 4. the CUDA branches, against a stub
# --------------------------------------------------------------------------

_FAKE_ELAPSED_MS = 12.5


class _FakeEvent:
    """Stands in for ``torch.cuda.Event(enable_timing=True)``.

    ``elapsed_time`` returns a constant that no wall clock would produce,
    which is what makes "the event branch ran" checkable rather than
    merely plausible.
    """

    created: list["_FakeEvent"] = []

    def __init__(self, enable_timing: bool = False, blocking: bool = False,
                 interprocess: bool = False) -> None:
        self.enable_timing = enable_timing
        self.records = 0
        _FakeEvent.created.append(self)

    def record(self, stream=None) -> None:
        self.records += 1

    def elapsed_time(self, other: "_FakeEvent") -> float:
        return _FAKE_ELAPSED_MS


class _FakeCuda:
    """The slice of ``torch.cuda`` that telemetry.py actually touches."""

    Event = _FakeEvent

    def __init__(self) -> None:
        self.syncs = 0
        self.resets = 0

    def is_available(self) -> bool:
        return True

    def synchronize(self, device=None) -> None:
        self.syncs += 1

    def get_device_name(self, device=None) -> str:
        return "NVIDIA A100-SXM4-80GB"

    def memory_allocated(self, device=None) -> int:
        return 111

    def memory_reserved(self, device=None) -> int:
        return 222

    def max_memory_allocated(self, device=None) -> int:
        return 333

    def max_memory_reserved(self, device=None) -> int:
        return 444

    def reset_peak_memory_stats(self, device=None) -> None:
        self.resets += 1


def _with_fake_cuda(fn):
    """Run ``fn(fake)`` with ``torch.cuda`` replaced, then put it back."""
    real = torch.cuda
    fake = _FakeCuda()
    _FakeEvent.created = []
    torch.cuda = fake
    try:
        return fn(fake)
    finally:
        torch.cuda = real


def test_fake_cuda_drives_the_event_based_span() -> None:
    assert not torch.cuda.is_available(), "this test is for the CPU-only box"

    def body(fake: _FakeCuda) -> dict:
        assert telemetry.cuda_available(), "the stub should look available"
        spans = Spans(enabled=True, device="cuda")
        assert spans.cuda, "device='cuda' must select the event timer"
        with spans.span("step"):
            with spans.span("forward"):
                pass
        # Nothing is resolved until the flush: that laziness is the point.
        assert fake.syncs == 0, "resolving a span must not synchronize inline"
        summary = spans.summary()
        assert fake.syncs == 1, "summary() should synchronize exactly once"
        return summary

    summary = _with_fake_cuda(body)

    # Four events for two spans, all with timing enabled, each recorded once.
    assert len(_FakeEvent.created) == 4, len(_FakeEvent.created)
    assert all(e.enable_timing for e in _FakeEvent.created)
    assert all(e.records == 1 for e in _FakeEvent.created)

    # The reported time is the stub's 12.5 ms exactly. A wall clock could
    # not have produced this, so the perf_counter branch demonstrably did
    # not run.
    assert summary["step"]["total"] == _FAKE_ELAPSED_MS / 1000.0, summary
    assert summary["forward"]["total"] == _FAKE_ELAPSED_MS / 1000.0, summary
    # Both spans "took" 12.5 ms and one contains the other, so the outer
    # span's exclusive time is zero -- the nesting arithmetic runs on the
    # event path too, not only on the host path.
    assert summary["step"]["exclusive"] == 0.0, summary
    assert summary["forward"]["exclusive"] == _FAKE_ELAPSED_MS / 1000.0

    # The real torch is back, and an auto-detected Spans is on the host
    # timer again.
    assert torch.cuda.is_available() is False
    assert Spans(enabled=True).cuda is False
    print(f"    event path: 4 events, 1 synchronize, "
          f"{summary['step']['total']*1000:.1f}ms reported")


def test_fake_cuda_drives_the_memory_snapshot() -> None:
    # Without CUDA: zeros for the device fields, and no exception.
    off = memory_snapshot()
    for key in ("cuda_allocated_bytes", "cuda_reserved_bytes",
                "cuda_max_allocated_bytes", "cuda_max_reserved_bytes"):
        assert off[key] == 0, (key, off)
    assert off["host_rss_bytes"] > 0, "host RSS should be readable here"

    def body(fake: _FakeCuda) -> dict:
        telemetry.reset_memory_peaks()
        assert fake.resets == 1, "reset_memory_peaks skipped its GPU branch"
        return memory_snapshot()

    on = _with_fake_cuda(body)
    # Distinct sentinels, so a snapshot that read the wrong counter shows up.
    assert on["cuda_allocated_bytes"] == 111, on
    assert on["cuda_reserved_bytes"] == 222, on
    assert on["cuda_max_allocated_bytes"] == 333, on
    assert on["cuda_max_reserved_bytes"] == 444, on
    assert on["host_rss_bytes"] > 0
    assert set(on) == set(off), "the two paths must produce the same columns"

    # The device-name helpers run their GPU branch too.
    name = _with_fake_cuda(lambda fake: telemetry.current_device_name())
    assert name == "NVIDIA A100-SXM4-80GB", name
    assert _with_fake_cuda(lambda fake: mfu(1e12, 0.1)) is not None
    assert telemetry.current_device_name() is None, "restore failed"
    print(f"    host rss {off['host_rss_bytes']/2**20:.0f} MiB; "
          f"cuda fields zero without cuda, sentinels with it")


# --------------------------------------------------------------------------
# 5. the metrics sink
# --------------------------------------------------------------------------


def _read(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_run_without_wandb_writes_parseable_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # A directory that does not exist yet: the sink creates it.
        path = Path(tmp) / "runs" / "unit.jsonl"
        run = Run("tractor-rl", "unit test/1", {"lr": 1e-3},
                  enabled=False, segment=3, jsonl=path)
        assert not run.wandb_active
        run.log({"loss": 1.5, "acc": 0.25}, step=10)
        run.log({"loss": 1.25, "per_agent": [1.0, 3.0]}, step=20)
        run.summary({"final": 0.5})
        run.finish()

        rows = _read(path)
        assert len(rows) == 3, rows
        assert all(row["segment"] == 3 for row in rows), rows
        assert all("wall_time" in row for row in rows)
        assert rows[0]["loss"] == 1.5 and rows[0]["learner_steps"] == 10
        assert rows[1]["per_agent"] == [1.0, 3.0], "a list should survive the JSONL"
        assert rows[2].get("_summary") is True and rows[2]["final"] == 0.5
        assert "_summary" not in rows[0]

        # Logging after finish is a no-op, not an exception, and does not
        # reopen the file -- a trainer's `finally` block logs one last row
        # more often than anyone admits.
        run.log({"loss": 0.0}, step=30)
        run.summary({"final": 9.9})
        run.finish()
        assert len(_read(path)) == 3

        # Append, not truncate: this is what makes a chained segment
        # continue one file, and `segment` the only thing separating them.
        again = Run("tractor-rl", "unit test/1", enabled=False,
                    segment=4, jsonl=path)
        again.log({"loss": 1.0}, step=40)
        again.finish()
        rows = _read(path)
        assert len(rows) == 4
        assert [r["segment"] for r in rows] == [3, 3, 3, 4]
        print(f"    {len(rows)} rows across two segments in {path.name}")


def test_run_takes_its_segment_from_the_environment() -> None:
    before = os.environ.get(telemetry.SEGMENT_ENV)
    try:
        os.environ[telemetry.SEGMENT_ENV] = "7"
        assert Run("p", "n", enabled=False, jsonl=False).segment == 7
        os.environ[telemetry.SEGMENT_ENV] = "not-a-number"
        assert Run("p", "n", enabled=False, jsonl=False).segment == 0
        del os.environ[telemetry.SEGMENT_ENV]
        assert Run("p", "n", enabled=False, jsonl=False).segment == 0
        # An explicit argument still wins.
        os.environ[telemetry.SEGMENT_ENV] = "7"
        assert Run("p", "n", enabled=False, jsonl=False, segment=2).segment == 2
    finally:
        os.environ.pop(telemetry.SEGMENT_ENV, None)
        if before is not None:
            os.environ[telemetry.SEGMENT_ENV] = before


def test_run_survives_a_wandb_that_fails() -> None:
    class _DeadWandb:
        def init(self, **kwargs):
            raise RuntimeError("no network")

    class _FlakyRun:
        def __init__(self) -> None:
            self.summary = {}

        def log(self, row):
            raise RuntimeError("socket closed")

        def finish(self):
            raise RuntimeError("closed twice")

    class _FlakyWandb:
        def init(self, **kwargs):
            return _FlakyRun()

        def define_metric(self, *args, **kwargs):
            pass

    real = telemetry._import_wandb
    with tempfile.TemporaryDirectory() as tmp:
        for label, stub in (("init raises", _DeadWandb()),
                            ("log raises", _FlakyWandb())):
            path = Path(tmp) / f"{label.split()[0]}.jsonl"
            telemetry._import_wandb = lambda _stub=stub: _stub
            try:
                run = Run("tractor-rl", "flaky", enabled=True, segment=1,
                          jsonl=path)
                run.log({"loss": 1.0}, step=1)
                run.summary({"final": 2.0})
                run.finish()
            finally:
                telemetry._import_wandb = real
            rows = _read(path)
            assert len(rows) == 2, (label, rows)
            assert rows[0]["loss"] == 1.0 and rows[0]["segment"] == 1
            assert not run.wandb_active, label

        # No wandb at all: still no exception, still a file.
        path = Path(tmp) / "absent.jsonl"
        telemetry._import_wandb = lambda: None
        try:
            with Run("tractor-rl", "absent", enabled=True, jsonl=path) as run:
                run.log({"loss": 3.0}, step=1)
        finally:
            telemetry._import_wandb = real
        assert _read(path)[0]["loss"] == 3.0
    print("    init failure, log failure and a missing wandb all fell back")


def test_run_flattens_what_wandb_can_chart() -> None:
    seen: list[dict] = []

    class _Recorder:
        def __init__(self) -> None:
            self.summary = {}

        def log(self, row):
            seen.append(row)

        def finish(self):
            pass

    class _Wandb:
        def init(self, **kwargs):
            _Wandb.kwargs = kwargs
            return _Recorder()

    real = telemetry._import_wandb
    telemetry._import_wandb = lambda: _Wandb()
    try:
        run = Run("tractor-rl", "chart me", {"lr": 1e-3}, enabled=True,
                  segment=2, jsonl=False)
        run.log({"loss": 0.5, "per_agent": [1.0, 2.0, 3.0], "note": "text",
                 "nested": {"a": 1.0}, "nan": float("nan")}, step=99)
        run.finish()
    finally:
        telemetry._import_wandb = real

    assert len(seen) == 1
    row = seen[0]
    assert row["loss"] == 0.5 and row["segment"] == 2
    # The step is a field, never wandb's own `step=`: a resumed segment
    # whose counter restarts must still chart.
    assert row["learner_steps"] == 99
    assert row["agent0/per_agent"] == 1.0 and row["per_agent/mean"] == 2.0
    assert row["nested/a"] == 1.0
    assert "note" not in row and "nan" not in row, "unchartable values leak"
    # Resume is what stitches a Modal segment chain into one wandb run.
    assert _Wandb.kwargs["resume"] == "allow"
    assert _Wandb.kwargs["id"] == "chart-me", _Wandb.kwargs["id"]
    print(f"    wandb row: {sorted(row)}")


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        started = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - reporting harness
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}"[:1500])
        else:
            print(f"ok    {fn.__name__}  ({time.time() - started:.1f}s)")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
