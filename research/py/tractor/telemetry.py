"""What a training step cost: time, memory, arithmetic, and where it went.

Four independent pieces, deliberately not coupled to any network or
trainer, because none of those exist yet and this has to be usable and
testable before they do.

``Spans``
    A nestable timer. On CUDA it records ``torch.cuda.Event`` pairs and
    resolves them in a batch, because a wall clock measures when the
    *launch* returned, not when the GPU finished -- an asynchronous
    backward pass looks like 200us of wall time and 40ms of GPU time, and
    the wall-clock version of that number has ended more than one
    optimisation session in the wrong place. Events are resolved lazily
    (one ``synchronize`` per ``summary()``) rather than at each span's
    close, since a synchronize inside the loop is itself the stall it is
    supposed to be measuring.

``memory_snapshot``
    CUDA allocated/reserved/peak plus host RSS, as plain bytes. Zeros,
    never an exception, when there is no CUDA -- the same call has to work
    in a CPU rollout worker and on an H100.

``mfu`` / ``transformer_flops``
    Achieved FLOP/s over the device's peak. The peak table is sourced,
    per-dtype, and returns ``None`` for a device it does not know, because
    a fabricated denominator turns a benchmark into a decoration. See
    ``PEAK_TFLOPS`` for each number's provenance.

``Run``
    A wandb wrapper that degrades to a local JSONL file and never raises.
    A logging failure at hour eleven of a twelve-hour segment must cost
    the charts, not the run. Every row carries a ``segment`` field: the
    Modal chain in ``research/modal_app.py`` restarts this process once
    per segment against the same run name and the same JSONL file, so
    without that field the rows of a six-segment run are indistinguishable
    and a step counter that restarts looks like a crash.

Nothing here imports wandb unless a ``Run`` is actually enabled, and
nothing here requires torch to import at all.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Sequence

try:  # torch is optional: a rollout worker has no use for it.
    import torch
except Exception:  # pragma: no cover - a broken install raises non-ImportError
    torch = None  # type: ignore[assignment]

__all__ = [
    "Spans",
    "memory_snapshot",
    "reset_memory_peaks",
    "cuda_available",
    "current_device_name",
    "Peak",
    "PEAK_TFLOPS",
    "peak_tflops",
    "peak_source",
    "mfu",
    "transformer_flops",
    "FlopCount",
    "Run",
    "flatten",
    "SEGMENT_ENV",
]


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

_warned: set[str] = set()


def _warn(msg: str) -> None:
    """Print a degradation notice once. Telemetry never spams a log."""
    if msg in _warned:
        return
    _warned.add(msg)
    print(msg, flush=True)


def cuda_available() -> bool:
    """True when torch is present and reports a usable CUDA device."""
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def current_device_name(index: int = 0) -> str | None:
    """``torch.cuda.get_device_name`` or None; never raises."""
    if not cuda_available():
        return None
    try:
        return str(torch.cuda.get_device_name(index))
    except Exception:
        return None


# --------------------------------------------------------------------------
# 1. spans
# --------------------------------------------------------------------------


class _NullSpan:
    """The disabled span: no allocation, no timer, two trivial calls.

    A fresh object or a ``@contextmanager`` generator per span would put
    an allocation on the hot path of code that is instrumented precisely
    because it is hot. One shared immutable instance costs an attribute
    load and two C-level method calls.
    """

    __slots__ = ()

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


_NULL_SPAN = _NullSpan()


class _Span:
    """One timed region, and its own context manager.

    Holds a link to its parent rather than a depth, so exclusive time is a
    subtraction against the actual parent even when the same name appears
    at two different depths.
    """

    __slots__ = ("_owner", "name", "parent", "child_time", "_a", "_b")

    def __init__(self, owner: "Spans", name: str) -> None:
        self._owner = owner
        self.name = name
        self.parent: _Span | None = None
        self.child_time = 0.0
        self._a: Any = None
        self._b: Any = None

    def __enter__(self) -> "_Span":
        owner = self._owner
        stack = owner._stack()
        self.parent = stack[-1] if stack else None
        stack.append(self)
        if owner._cuda:
            event = owner._event()
            self._a = event
            event.record()
        else:
            self._a = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> bool:
        owner = self._owner
        # Closed before the stack is popped, so the timer bounds the
        # region and not the bookkeeping.
        if owner._cuda:
            event = owner._event()
            event.record()
            self._b = event
        else:
            self._b = time.perf_counter()
        stack = owner._stack()
        if stack and stack[-1] is self:
            stack.pop()
        else:  # `with` cannot produce this; never corrupt the stack if it does
            try:
                stack.remove(self)
            except ValueError:
                pass
        owner._finish(self)
        return False

    def _elapsed(self) -> float:
        """Seconds. Valid only once the device has reached both events."""
        if self._owner._cuda:
            return float(self._a.elapsed_time(self._b)) / 1000.0
        return float(self._b) - float(self._a)


class Spans:
    """Nestable per-name timers, CUDA-event based when the device is CUDA.

    ::

        spans = Spans(enabled=args.profile)
        with spans.span("train_step"):
            with spans.span("forward"):
                out = net(batch)
            with spans.span("backward"):
                loss.backward()
        run.log(spans.metrics())
        spans.reset()

    ``summary()[name]`` gives ``count``, ``total``, ``mean``, ``exclusive``
    and ``exclusive_mean``, all in seconds. Inclusive time is everything
    between enter and exit; exclusive subtracts the direct children, which
    is the number that says whether a parent is doing work of its own or
    only waiting on what it contains.

    The timing source is decided once, at construction:

    * ``device=None`` -- CUDA events if ``torch.cuda.is_available()``,
      else ``time.perf_counter``.
    * ``device="cuda"`` -- CUDA events unconditionally. A real escape
      hatch, and the seam the unit tests use to exercise the GPU branch on
      a CPU-only box.
    * anything else -- ``time.perf_counter``.

    A CUDA span has no duration until the device reaches its events, so
    spans queue and are resolved in one batch by ``flush()``, which
    ``summary()`` and ``reset()`` call for you. That is one ``synchronize``
    per reporting interval instead of one per span. The queue is bounded
    by ``max_pending``; reaching that bound forces an early flush, which
    does perturb the very timings it resolves, so the default is large
    enough that a normal step never gets there.

    The open-span stack is thread-local and the accumulators are locked,
    so a background thread with its own spans is safe. CUDA events are
    recorded on each thread's current stream, which is the caller's
    problem rather than this class's.
    """

    def __init__(self, enabled: bool = True, device: Any = None,
                 max_pending: int = 8192) -> None:
        self.enabled = bool(enabled)
        self._cuda = _use_cuda_timer(device)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._finished: list[_Span] = []
        # name -> [count, inclusive_seconds, exclusive_seconds]
        self._totals: dict[str, list[float]] = {}
        self._pool: list[Any] = []
        self._max_pending = max(1, int(max_pending))
        self.dropped = 0

    @property
    def cuda(self) -> bool:
        """True when this instance times with ``torch.cuda.Event`` pairs."""
        return self._cuda

    # -- the hot path ------------------------------------------------------

    def span(self, name: str) -> Any:
        """Time the ``with`` block under ``name``."""
        if not self.enabled:
            return _NULL_SPAN
        return _Span(self, name)

    # -- internals ---------------------------------------------------------

    def _stack(self) -> list[_Span]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def _event(self) -> Any:
        # Pooled: a span is two events and a training loop closes thousands
        # of spans a minute. `pop` on an empty list is caught rather than
        # pre-checked, so the pool needs no lock of its own.
        try:
            return self._pool.pop()
        except IndexError:
            pass
        try:
            return torch.cuda.Event(enable_timing=True)
        except Exception as exc:  # noqa: BLE001 - no CUDA in this build
            # `device="cuda"` was asked for and cannot be honoured.
            # Fall back rather than raise: this runs inside
            # __enter__, so raising would stop the instrumented
            # block from running at all.
            _warn(f"telemetry: CUDA events unavailable "
                  f"({type(exc).__name__}); spans use the host clock")
            self._cuda = False
            return None

    def _finish(self, span: _Span) -> None:
        self._finished.append(span)
        if not self._cuda:
            # A host timer already knows its answer; resolving now keeps
            # the queue empty and the memory flat.
            self.flush()
        elif len(self._finished) >= self._max_pending:
            self.flush()

    def flush(self) -> None:
        """Resolve queued spans into the accumulators.

        Synchronizes once, outside the lock, and only for spans that were
        already closed when the queue was taken -- so a span closing on
        another thread during the synchronize is left for the next flush
        rather than read before the device reached it.
        """
        if not self._finished:
            return
        with self._lock:
            pending, self._finished = self._finished, []
        if not pending:
            return
        if self._cuda:
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                self.dropped += len(pending)
                _warn(f"telemetry: cuda synchronize failed "
                      f"({type(exc).__name__}: {exc}); "
                      f"{len(pending)} spans dropped")
                return
        with self._lock:
            totals = self._totals
            for span in pending:
                try:
                    inclusive = span._elapsed()
                except Exception as exc:
                    self.dropped += 1
                    _warn(f"telemetry: span {span.name!r} unresolvable "
                          f"({type(exc).__name__}: {exc}); dropped")
                    continue
                # `pending` is in completion order and a child always
                # completes before its parent, so a parent's `child_time`
                # is final by the time this loop reaches it.
                if span.parent is not None:
                    span.parent.child_time += inclusive
                exclusive = inclusive - span.child_time
                slot = totals.get(span.name)
                if slot is None:
                    totals[span.name] = [1.0, inclusive, exclusive]
                else:
                    slot[0] += 1.0
                    slot[1] += inclusive
                    slot[2] += exclusive
                if self._cuda:
                    self._pool.append(span._a)
                    self._pool.append(span._b)
                span._a = span._b = None

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, dict[str, float]]:
        """``{name: {count, total, mean, exclusive, exclusive_mean}}``, seconds."""
        self.flush()
        with self._lock:
            return {
                name: {
                    "count": int(count),
                    "total": total,
                    "mean": total / count if count else 0.0,
                    "exclusive": exclusive,
                    "exclusive_mean": exclusive / count if count else 0.0,
                }
                for name, (count, total, exclusive) in self._totals.items()
            }

    def metrics(self, prefix: str = "time/") -> dict[str, float]:
        """``summary()`` flattened into a row ``Run.log`` can take."""
        out: dict[str, float] = {}
        for name, stat in self.summary().items():
            out[f"{prefix}{name}"] = stat["total"]
            out[f"{prefix}{name}/mean"] = stat["mean"]
            out[f"{prefix}{name}/excl"] = stat["exclusive"]
            out[f"{prefix}{name}/count"] = float(stat["count"])
        return out

    def reset(self) -> None:
        """Zero the accumulators. Call between reporting intervals."""
        self.flush()
        with self._lock:
            self._totals.clear()


def _use_cuda_timer(device: Any) -> bool:
    if device is None:
        return cuda_available()
    name = str(getattr(device, "type", device)).split(":")[0].lower()
    if name == "cuda":
        if torch is None:
            raise RuntimeError("Spans(device='cuda') needs torch installed")
        return True
    return False


# --------------------------------------------------------------------------
# 2. memory
# --------------------------------------------------------------------------

_CUDA_MEMORY_KEYS = (
    "cuda_allocated_bytes",
    "cuda_reserved_bytes",
    "cuda_max_allocated_bytes",
    "cuda_max_reserved_bytes",
)


def memory_snapshot(device: Any = None) -> dict[str, int]:
    """Bytes: CUDA allocated/reserved/peaks, plus this process's host RSS.

    The CUDA fields are zero -- not absent, and not an exception -- when
    there is no CUDA. A metrics row has to have the same columns on a CPU
    worker and on a GPU learner, or the two runs do not plot together.

    ``allocated`` is what torch's caching allocator has handed to tensors;
    ``reserved`` is what it has taken from the driver and will not give
    back. The gap between them is fragmentation, and it is the pair, not
    either number alone, that explains an out-of-memory at a batch size
    that fits on paper. The ``max_`` fields are peaks since the process
    started or since the last ``reset_memory_peaks()``.
    """
    snap: dict[str, int] = {key: 0 for key in _CUDA_MEMORY_KEYS}
    snap["host_rss_bytes"] = _host_rss_bytes()
    if not cuda_available():
        return snap
    try:
        snap["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
        snap["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
        snap["cuda_max_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device))
        snap["cuda_max_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device))
    except Exception as exc:
        _warn(f"telemetry: cuda memory query failed "
              f"({type(exc).__name__}: {exc}); reporting zeros")
    return snap


def reset_memory_peaks(device: Any = None) -> None:
    """Restart the ``max_`` counters. A peak needs a window to be a peak."""
    if not cuda_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception as exc:
        _warn(f"telemetry: reset_peak_memory_stats failed "
              f"({type(exc).__name__}: {exc})")


def _proc_rss_bytes() -> int:
    """Linux, and the cheapest of the three: one read, no dependency."""
    with open("/proc/self/statm", "rb") as fh:
        pages = int(fh.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


def _psutil_rss_bytes() -> int:
    """Portable, but psutil is not a dependency of this tree."""
    import psutil  # type: ignore[import-not-found]

    return int(psutil.Process().memory_info().rss)


_win_rss: Any = None


def _windows_rss_bytes() -> int:
    """Windows. Only the dev box needs it, but a zero there hides leaks."""
    global _win_rss
    if _win_rss is None:
        _win_rss = _build_windows_rss()
    return _win_rss()


def _build_windows_rss() -> Any:
    """Resolve GetProcessMemoryInfo once, with real prototypes.

    The prototypes are the whole point. Without an explicit ``restype``,
    ctypes types ``GetCurrentProcess`` as ``c_int`` and truncates the
    64-bit pseudo-handle (``(HANDLE)-1``) to 32 bits, and the call comes
    back with "the handle is invalid" -- which looks exactly like "this
    platform cannot report RSS" and silently zeroes the column.
    """
    import ctypes
    from ctypes import wintypes

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    # Modern Windows exports this from kernel32 as K32GetProcessMemoryInfo;
    # psapi.dll still forwards to it, and older builds have only psapi.
    get_info = getattr(kernel32, "K32GetProcessMemoryInfo", None)
    if get_info is None:
        get_info = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters),
                         wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()

    def read() -> int:
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        if not get_info(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)

    return read


_rss_reader: Any = None


def _host_rss_bytes() -> int:
    """Resident set size of this process, or 0 where it cannot be read.

    The readers are probed in order of cost and of how much the source can
    be trusted, and the first one that answers is cached: a failed
    ``import psutil`` walks ``sys.path`` every time, and this is called
    once per metrics row.

    ``resource.ru_maxrss`` is deliberately not among them. It is a peak
    rather than a current value, and its unit differs between Linux and
    macOS -- a wrong number in right-looking units is worse than a zero.
    """
    global _rss_reader
    if _rss_reader is None:
        _rss_reader = _pick_rss_reader()
    try:
        return _rss_reader()
    except Exception:
        return 0


def _pick_rss_reader() -> Any:
    candidates = [_proc_rss_bytes, _psutil_rss_bytes]
    if sys.platform == "win32":
        candidates.append(_windows_rss_bytes)
    for candidate in candidates:
        try:
            if candidate() > 0:
                return candidate
        except Exception:
            continue
    _warn("telemetry: no way to read host RSS on this platform; reporting 0")
    return lambda: 0


# --------------------------------------------------------------------------
# 3. model FLOPs and MFU
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Peak:
    """Dense peak throughput in TFLOP/s. ``None`` where the part cannot do it."""

    fp32: float | None
    tf32: float | None
    bf16: float | None
    fp16: float | None
    source: str


# Peak *dense* tensor throughput in TFLOP/s: no structured sparsity, FP32
# accumulate -- the shape a training matmul actually runs in.
#
# Two conventions matter when checking these against a datasheet.
#
#   * NVIDIA prints the sparse number with an asterisk, and frequently
#     prints *only* that number. Every figure below that came from an
#     asterisked cell has been halved, and its comment says so. Quoting
#     the sparse peak is the single most common way an MFU number ends up
#     a clean factor of two too flattering.
#   * "fp32" is the non-tensor SM rate. TF32 is a separate column because
#     `torch.backends.cuda.matmul.allow_tf32 = True` silently moves an
#     fp32 matmul onto the tensor cores, and on an A100 that is an 8x
#     change of denominator for a one-line change of configuration.
#
# Keys are matched as case-insensitive substrings of
# `torch.cuda.get_device_name()`, FIRST MATCH WINS, so the order here is
# load bearing: "a10g" must precede "a10", and the H100 variants must
# precede the generic "h100".
PEAK_TFLOPS: dict[str, Peak] = {
    # "NVIDIA H100 PCIe". Datasheet: FP32 51; TF32 756*, BF16/FP16 1,513*,
    # where * is with sparsity, so dense is half.
    "h100 pcie": Peak(51.0, 378.0, 756.5, 756.5,
                      "NVIDIA H100 datasheet, PCIe column; "
                      "sparse figures halved"),
    # "NVIDIA H100 NVL" (94GB single card). Datasheet: FP32 60; TF32 835*,
    # BF16 1,671*. Halved. Not verified against hardware.
    "h100 nvl": Peak(60.0, 417.5, 835.5, 835.5,
                     "NVIDIA H100 NVL datasheet; sparse figures halved; "
                     "unverified on hardware"),
    # "NVIDIA H200". Same SM count and clocks as H100 SXM; datasheet:
    # FP32 67; TF32 989*, BF16 1,979*. Halved.
    "h200": Peak(67.0, 494.5, 989.5, 989.5,
                 "NVIDIA H200 datasheet; sparse figures halved"),
    # Generic H100 -- in practice "NVIDIA H100 80GB HBM3", the SXM part.
    # Last of the H100 keys so the PCIe and NVL spellings win first. If a
    # future PCIe spelling lands here instead, the peak is overstated and
    # the MFU therefore reads low, which is the safe direction for a
    # number people quote in a write-up.
    "h100": Peak(67.0, 494.5, 989.5, 989.5,
                 "NVIDIA H100 datasheet, SXM column; sparse figures halved"),
    # "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", and the 40GB
    # variants. The datasheet prints both columns: TF32 156 | 312*,
    # BF16 312 | 624*; the dense one is taken. SXM and PCIe share peak
    # rates -- they differ in sustained power, not in the ceiling.
    "a100": Peak(19.5, 156.0, 312.0, 312.0,
                 "NVIDIA A100 datasheet (dense column)"),
    # "NVIDIA L40S". Datasheet prints both: TF32 183 | 366*,
    # BF16 362.05 | 733*.
    "l40s": Peak(91.6, 183.0, 362.05, 362.05,
                 "NVIDIA L40S datasheet (dense column)"),
    # "NVIDIA L4". Datasheet: FP32 30.3; TF32 120*, BF16/FP16 242*. Halved.
    # "l40" would otherwise fall through to the "l4" row below and
    # read three times too fast. Dense figures from the L40 datasheet.
    "l40": Peak(90.5, 90.5, 181.05, 181.05,
            "NVIDIA L40 datasheet, dense column; sparse figures halved"),
    "l4": Peak(30.3, 60.0, 121.0, 121.0,
               "NVIDIA L4 datasheet; sparse figures halved"),
    # The two GA102 datacenter parts, derived from the silicon rather
    # than from reading a datasheet's sparsity footnote. Per SM per
    # clock, dense: 256 FLOP fp32 on the CUDA cores, 512 TF32 and 1024
    # BF16/FP16 on the third-generation tensor cores. So TF32 is 2x
    # fp32 and BF16 is 4x, and the A10's published 31.2 / 62.5 / 125
    # falls out exactly at 72 SMs and 1695 MHz:
    #
    #     fp32  72 * 256  * 1.695e9 =  31.25
    #     tf32  72 * 512  * 1.695e9 =  62.50
    #     bf16  72 * 1024 * 1.695e9 = 125.00
    #
    # which settles what the datasheet's "125 | 250*" means: the first
    # figure is dense and only the asterisked one carries sparsity.
    # Two successive revisions of these rows got this wrong, first by
    # pairing the A10's core count with the A10G's clock, then by
    # halving figures that were already dense. Both showed up the same
    # way -- a measured dense 8192-cube GEMM above the stated peak,
    # which cuBLAS cannot do. Against the numbers below, measurements
    # land at 58-60% on the A10 and 48% on the A10G, consistent across
    # both dtypes on each card, which is what a 150 W part sustains.
    # The a100, l4 and l40 rows already read their datasheets this way.
    "a10g": Peak(35.02, 70.04, 140.08, 140.08,
                 "GA102 dense rates at A10G core count (80 SMs) and "
                 "1710 MHz boost; 256/512/1024 FLOP per SM per clock"),
    # Must follow "a10g", which contains this key as a substring.
    "a10": Peak(31.2, 62.5, 125.0, 125.0,
                "NVIDIA A10 datasheet, dense column; reproduced by "
                "72 SMs * 256/512/1024 FLOP per clock at 1695 MHz"),
    # "Tesla T4". Datasheet: FP32 8.1; mixed precision FP16/FP32 65
    # (65.13 exactly: 320 tensor cores * 64 FMA * 2 * 1.59 GHz);
    # INT8 130 TOPS.
    #
    # Turing (sm_75) has no TF32 and no hardware BF16. torch will happily
    # run a bf16 matmul on a T4 by falling off the tensor cores, so the
    # honest peak for that dtype is not a number at all -- it is "this
    # device cannot do that", and `peak_tflops(..., "bf16")` returns None
    # here. Inventing a bf16 figure would be exactly the guess this table
    # exists to refuse; a T4 mixed-precision run should be measured
    # against "fp16".
    "t4": Peak(8.1, None, None, 65.13,
               "NVIDIA T4 datasheet; Turing has no TF32 and no hardware "
               "BF16, hence the two None entries"),
}

_DTYPE_ALIASES = {
    "fp32": "fp32", "float32": "fp32", "f32": "fp32", "single": "fp32",
    "tf32": "tf32",
    "bf16": "bf16", "bfloat16": "bf16",
    "fp16": "fp16", "float16": "fp16", "f16": "fp16", "half": "fp16",
}


def _lookup(device_name: str | None) -> Peak | None:
    if not device_name:
        return None
    haystack = str(device_name).lower()
    for needle, peak in PEAK_TFLOPS.items():
        if needle in haystack:
            return peak
    return None


def peak_tflops(device_name: str | None, dtype: str = "bf16") -> float | None:
    """Dense peak TFLOP/s for a device name, or ``None`` when not known.

    ``None`` means one of three things, and all three are the same answer
    for a caller: no name was available, the name matched nothing in
    ``PEAK_TFLOPS``, or the device has no hardware path for that dtype
    (a T4 and bf16). Returning a plausible number instead would produce an
    MFU that reads like a measurement and is not one.
    """
    key = _DTYPE_ALIASES.get(str(dtype).lower())
    if key is None:
        raise ValueError(
            f"unknown dtype {dtype!r}; expected one of "
            f"{sorted(set(_DTYPE_ALIASES.values()))}")
    peak = _lookup(device_name)
    return None if peak is None else getattr(peak, key)


def peak_source(device_name: str | None) -> str | None:
    """Where this device's row came from, so a benchmark can print it."""
    peak = _lookup(device_name)
    return None if peak is None else peak.source


def mfu(flops_per_step: float, seconds: float, device_name: str | None = None,
        dtype: str = "bf16") -> float | None:
    """Model FLOP utilisation: achieved FLOP/s over the device's peak.

    ``flops_per_step`` is the analytic count for the work in one step --
    see ``transformer_flops`` -- not a profiler's kernel total. For a
    training step that is about ``3x`` the forward count (a forward, plus
    a backward costing roughly two forwards); this function does not apply
    that factor, because whether a step includes a backward is the
    caller's business, not the denominator's.

    ``device_name`` defaults to the current CUDA device's name. Returns
    ``None`` when the peak is unknown (see ``peak_tflops``) or the
    interval is not a positive finite number of seconds.

    The result is deliberately not clamped. An MFU above 1.0 is real
    information: it means the FLOP count or the ``dtype`` is wrong, most
    often an fp32 denominator against a run that TF32 quietly moved onto
    the tensor cores.
    """
    if device_name is None:
        device_name = current_device_name()
    peak = peak_tflops(device_name, dtype)
    if peak is None or not peak > 0:
        return None
    if not seconds > 0 or not math.isfinite(seconds):
        return None
    return float(flops_per_step) / float(seconds) / (peak * 1e12)


class FlopCount(NamedTuple):
    """``total`` FLOPs and ``parts`` saying where they went. Exact integers."""

    total: int
    parts: dict[str, int]


def transformer_flops(layers: int, d_model: int, ff: int, tokens: int,
                      heads: int, *, batch: int = 1) -> FlopCount:
    """Analytic forward FLOPs for a stack of pre-LayerNorm transformer blocks.

    ``tokens`` is the sequence length of ONE sequence; ``batch`` is how
    many of them. The distinction is not cosmetic: the projections and the
    feed-forward scale with ``batch * tokens``, but the attention score and
    value matmuls scale with ``batch * tokens**2``. Folding the batch into
    ``tokens`` therefore inflates the two attention terms by a factor of
    ``batch``, and at batch 512 that is not a rounding error. It is the
    usual reason a hand-rolled FLOP count disagrees with a profiler.

    Every matmul is counted as ``2 * M * N * K`` -- one multiply and one
    add per MAC. With ``N = batch * tokens`` and ``T = tokens``, per layer:

    ==================  ===========================  =======================
    term                shape                        FLOPs
    ==================  ===========================  =======================
    ``attn_q``          [N, d] @ [d, d]              ``2 * N * d * d``
    ``attn_k``          [N, d] @ [d, d]              ``2 * N * d * d``
    ``attn_v``          [N, d] @ [d, d]              ``2 * N * d * d``
    ``attn_out``        [N, d] @ [d, d]              ``2 * N * d * d``
    ``attn_scores``     Q @ K.T, within a sequence   ``2 * N * T * d``
    ``attn_values``     A @ V, within a sequence     ``2 * N * T * d``
    ``ffn_up``          [N, d] @ [d, ff]             ``2 * N * d * ff``
    ``ffn_down``        [N, ff] @ [ff, d]            ``2 * N * ff * d``
    ==================  ===========================  =======================

    ``parts`` holds each term summed over all ``layers``, so the parts add
    up to ``total`` exactly. Which of them dominates is the point of
    returning them: while ``ff = 4 * d_model`` the feed-forward pair is
    two thirds of the block, and the attention matmuls only overtake the
    projections once ``tokens`` passes ``d_model``.

    ``heads`` appears in no term, and that is correct rather than an
    omission: splitting ``d_model`` into ``heads`` heads of
    ``d_model / heads`` partitions exactly the same arithmetic. It is
    taken as an argument so that the divisibility the shapes assume is
    checked here, rather than discovered at the first forward pass.

    Excluded: LayerNorm, softmax, residual adds, activations, biases, and
    any embedding or output projection. Those are ``O(N * d)`` or
    ``O(N * T)`` elementwise work -- memory bound, not arithmetic -- and
    the standard analytic model leaves them out. That makes this count a
    floor on the work, and so the MFU it feeds is a floor too.
    """
    for label, value in (("layers", layers), ("d_model", d_model), ("ff", ff),
                         ("tokens", tokens), ("heads", heads),
                         ("batch", batch)):
        if int(value) != value or value <= 0:
            raise ValueError(f"{label} must be a positive integer, got {value!r}")
    layers, d_model, ff = int(layers), int(d_model), int(ff)
    tokens, heads, batch = int(tokens), int(heads), int(batch)
    if d_model % heads:
        raise ValueError(f"d_model {d_model} is not divisible by heads {heads}")

    n = batch * tokens              # total tokens through the projections
    proj = 2 * n * d_model * d_model
    attn = 2 * n * tokens * d_model
    parts = {
        "attn_q": layers * proj,
        "attn_k": layers * proj,
        "attn_v": layers * proj,
        "attn_out": layers * proj,
        "attn_scores": layers * attn,
        "attn_values": layers * attn,
        "ffn_up": layers * 2 * n * d_model * ff,
        "ffn_down": layers * 2 * n * ff * d_model,
    }
    return FlopCount(sum(parts.values()), parts)


# --------------------------------------------------------------------------
# 4. metrics
# --------------------------------------------------------------------------

SEGMENT_ENV = "TRACTOR_SEGMENT"

_ID_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _as_number(x: Any) -> float | None:
    """A chartable scalar, or None.

    A training loop produces `loss.detach()` and numpy scalars far more
    often than it produces Python floats, and those are exactly the
    values a plain isinstance check rejects. Zero-dimensional tensors
    and numpy scalars are unwrapped; anything with more than one element
    is not a scalar and is left to the caller's own reduction.
    """
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(float(x)) else None
    item = getattr(x, "item", None)
    if callable(item):
        size = getattr(x, "numel", None)
        count = size() if callable(size) else getattr(x, "size", 1)
        if callable(count):
            count = count()
        try:
            if int(count) != 1:
                return None
            value = float(item())
        except Exception:  # noqa: BLE001 - value is simply not chartable
            return None
        return value if math.isfinite(value) else None
    return None


def _finite(x: Any) -> bool:
    return _as_number(x) is not None


def flatten(row: dict) -> dict[str, float]:
    """Metrics row -> flat ``{name: number}``, dropping what cannot be charted.

    Same contract as ``tractor_rl/wandb_log.py:flatten``, reimplemented
    rather than imported because this package deliberately does not depend
    on that one. A per-agent list becomes ``agent{i}/{key}`` series plus
    mean/min/max, since a list is not a chartable quantity.
    """
    out: dict[str, float] = {}
    for key, value in row.items():
        number = _as_number(value)
        if isinstance(value, bool):
            out[key] = int(value)
        elif number is not None:
            out[key] = number
        elif isinstance(value, (list, tuple)):
            nums = []
            for i, item in enumerate(value):
                got = _as_number(item)
                if got is not None:
                    out[f"agent{i}/{key}"] = got
                    nums.append(got)
            if nums:
                out[f"{key}/mean"] = sum(nums) / len(nums)
                out[f"{key}/min"] = min(nums)
                out[f"{key}/max"] = max(nums)
        elif isinstance(value, dict):
            for sub, item in value.items():
                got = _as_number(item)
                if got is not None:
                    out[f"{key}/{sub}"] = got
        # None, strings and anything else belong in config, not metrics.
    return out


def _jsonable(obj: Any) -> Any:
    """Last-resort JSON encoder: numpy scalars, tensors, then ``str``."""
    for attr in ("item", "tolist"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return str(obj)


def _default_segment() -> int:
    try:
        return int(os.environ.get(SEGMENT_ENV, "0"))
    except (TypeError, ValueError):
        return 0


def _import_wandb() -> Any:
    """Import wandb, or return ``None``. A seam: tests replace it to fail."""
    try:
        import wandb
    except Exception:
        return None
    return wandb


class Run:
    """One training run's metrics: wandb when it is there, JSONL always.

    ::

        run = Run("tractor-rl", args.run_name, config=vars(args),
                  enabled=args.wandb)
        run.log({"loss": 0.31, "steps_per_s": 240.0}, step=learner.steps)
        run.summary({"final_elo": 118.0})
        run.finish()

    Three properties, in order of importance.

    **A logging failure never kills a training run.** Every call is
    wrapped; the first failure of a sink disables that sink for the rest
    of the process and prints one line. Nothing here raises, including
    after ``finish()``.

    **The local JSONL is the source of truth.** The brief for this class
    was "fall back to JSONL when wandb is unavailable"; it writes the
    JSONL unconditionally instead, and ``jsonl=False`` opts out. It costs
    the same either way, and it means a run whose wandb key expires in
    hour eleven still has a complete record rather than a file that starts
    at hour eleven. The file is opened in append mode so a resumed segment
    continues one file.

    **A resumed run continues one chart.** The wandb id is derived from
    the run name with ``resume="allow"``, so the Modal segment chain, and
    any restart after a preemption, append to a single wandb run instead
    of producing N unrelated ones.

    Every row carries ``segment`` -- from the ``segment`` argument, or the
    ``TRACTOR_SEGMENT`` environment variable, or 0. With one JSONL file
    and one wandb run per chain, it is the only thing that tells the rows
    of segment 4 from the rows of segment 1.

    ``step`` is logged as a *field* named ``step_metric``, never as
    wandb's own ``step=``. Two reasons, both already paid for in lost
    runs: a learner does not step at all until the replay buffer fills, so
    an explicit ``step=0`` for that whole phase collapses it to a single
    overwritten point; and wandb's internal step never goes backwards, so
    a resumed segment whose counter restarts would have its rows silently
    discarded. Declaring the metric as the x-axis instead lets the value
    stand still, or restart, and still chart.
    """

    def __init__(self, project: str, name: str, config: dict | None = None,
                 enabled: bool = True, *, segment: int | None = None,
                 entity: str = "", group: str = "", tags: Sequence[str] = (),
                 mode: str = "online", jsonl: Any = True,
                 step_metric: str = "learner_steps") -> None:
        self.project = project
        self.name = name
        self.config = dict(config or {})
        self.segment = _default_segment() if segment is None else int(segment)
        self.step_metric = step_metric
        self.rows = 0
        # Rows the JSONL sink could not serialise. Counted rather than
        # allowed to retire the sink, which one bad key would otherwise do.
        self.dropped_rows = 0
        self._closed = False
        self._run: Any = None
        self._fh: Any = None
        self._path: Path | None = self._resolve_path(jsonl)
        if enabled:
            self._open_wandb(entity, group, tags, mode)
        if self._run is None and self._path is None:
            _warn(f"telemetry: Run {name!r} has no sink; metrics are discarded")

    @property
    def path(self) -> Path | None:
        """The JSONL file being written, or ``None`` if that sink is off."""
        return self._path

    @property
    def wandb_active(self) -> bool:
        """True while a live wandb run is attached."""
        return self._run is not None

    # -- construction ------------------------------------------------------

    def _resolve_path(self, jsonl: Any) -> Path | None:
        if jsonl is None or jsonl is False:
            return None
        try:
            if jsonl is True:
                # Where the existing trainers write: runs/<name>.jsonl under
                # the working directory, which the Modal container symlinks
                # onto the volume.
                stem = _ID_UNSAFE.sub("-", self.name)[:120] or "run"
                return Path("runs") / f"{stem}.jsonl"
            return Path(jsonl)
        except Exception as exc:
            _warn(f"telemetry: bad jsonl path {jsonl!r} "
                  f"({type(exc).__name__}: {exc}); no local log")
            return None

    def _open_wandb(self, entity: str, group: str, tags: Sequence[str],
                    mode: str) -> None:
        wb = _import_wandb()
        if wb is None:
            _warn("telemetry: wandb not installed (pip install wandb); "
                  "logging to JSONL only")
            return
        # wandb ids allow [A-Za-z0-9_-]; run names are ours to choose, but a
        # stray character would otherwise fail init at startup.
        run_id = _ID_UNSAFE.sub("-", self.name)[:120] or "run"
        try:
            self._run = wb.init(
                project=self.project,
                entity=entity or None,
                group=group or None,
                name=self.name,
                id=run_id,
                resume="allow",
                tags=list(tags) or None,
                mode=mode,
                config=dict(self.config),
            )
        except Exception as exc:
            _warn(f"telemetry: wandb init failed ({type(exc).__name__}: {exc}); "
                  f"logging to JSONL only")
            self._run = None
            return
        # Chart against the axis the experiments are compared on rather than
        # wall-clock log order. Best effort: the spelling of this call has
        # moved between wandb majors and it is not worth a run.
        try:
            define = (getattr(self._run, "define_metric", None)
                      or getattr(wb, "define_metric", None))
            if define is not None:
                define(self.step_metric)
                define("*", step_metric=self.step_metric)
        except Exception as exc:
            _warn(f"telemetry: define_metric failed "
                  f"({type(exc).__name__}: {exc}); charts use log order")

    # -- logging -----------------------------------------------------------

    def log(self, row: dict, step: int | None = None) -> None:
        """Log one metrics row. Silently does nothing after ``finish()``."""
        if self._closed:
            return
        try:
            payload = dict(row)
            payload["segment"] = self.segment
            if step is not None and self.step_metric not in payload:
                payload[self.step_metric] = step
        except Exception as exc:
            _warn(f"telemetry: unloggable row ({type(exc).__name__}: {exc})")
            return
        self.rows += 1
        self._to_jsonl(payload)
        self._to_wandb(payload)

    def summary(self, values: dict) -> None:
        """Set run-level summary values: final scores, totals, verdicts."""
        if self._closed:
            return
        try:
            payload = dict(values)
            payload["segment"] = self.segment
        except Exception as exc:
            _warn(f"telemetry: unloggable summary ({type(exc).__name__}: {exc})")
            return
        self._to_jsonl(payload, summary=True)
        if self._run is None:
            return
        try:
            self._run.summary.update(flatten(payload))
        except Exception as exc:
            self._fail_wandb("summary", exc)

    def finish(self) -> None:
        """Close both sinks. Idempotent, and safe from a ``finally`` block."""
        if self._closed:
            return
        self._closed = True
        if self._run is not None:
            try:
                self._run.finish()
            except Exception as exc:
                _warn(f"telemetry: wandb finish failed "
                      f"({type(exc).__name__}: {exc})")
            self._run = None
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception as exc:
                _warn(f"telemetry: closing {self._path} failed "
                      f"({type(exc).__name__}: {exc})")
            self._fh = None

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.finish()
        return False

    # -- sinks -------------------------------------------------------------

    def _to_jsonl(self, payload: dict, summary: bool = False) -> None:
        if self._path is None:
            return
        try:
            if self._fh is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                # Append: a resumed segment continues the same file, which
                # is the reason every row needs its own `segment` field.
                self._fh = self._path.open("a", encoding="utf-8")
            record = dict(payload)
            record["wall_time"] = time.time()
            if summary:
                record["_summary"] = True
            # Flushed per row rather than per buffer: a preempted container
            # that loses the last 8KB loses the minutes that explain why.
            line = json.dumps(record, default=_jsonable)
        except Exception as exc:
            # A row that will not serialise is one bad row, not a
            # broken sink. `default=` is never consulted for dict
            # *keys*, so a numpy key alone lands here; dropping the
            # row keeps every later one.
            self.dropped_rows += 1
            _warn(f"telemetry: dropped an unserialisable row "
                  f"({type(exc).__name__}: {exc})")
            return
        try:
            self._fh.write(line + "\n")
            self._fh.flush()
        except Exception as exc:
            _warn(f"telemetry: JSONL sink {self._path} disabled "
                  f"({type(exc).__name__}: {exc})")
            self._path = None
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None

    def _to_wandb(self, payload: dict) -> None:
        if self._run is None:
            return
        try:
            self._run.log(flatten(payload))
        except Exception as exc:
            self._fail_wandb("log", exc)

    def _fail_wandb(self, what: str, exc: Exception) -> None:
        """One failure retires the mirror; the JSONL carries the run."""
        _warn(f"telemetry: wandb {what} failed ({type(exc).__name__}: {exc}); "
              f"continuing without it")
        self._run = None
