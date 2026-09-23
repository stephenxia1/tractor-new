"""A ``RolloutBuffer`` several processes can fill at once.

The parallel rollout puts the simulator in worker processes and keeps
one batched forward in the parent, which means an encoded observation
has to get from a worker to the parent on every decision. Measured with
``py/bench_ipc.py``, sending those 28,466 bytes through a
``multiprocessing.Queue`` one decision at a time sustains about 1,150
decisions a second -- slower than the single-process rollout it is
meant to replace. The same payload written into shared memory, with the
queue carrying only a row index, sustains 23,600 at a batch of 32.

So nothing is sent. The buffer *is* the transport: workers write their
rows in place, the parent reads them in place, and the messages that
cross the boundary are a few integers.

Layout. One block, fields laid end to end in the order
``RolloutBuffer.__init__`` allocates them, each padded to a 64-byte
boundary. Both sides build their view by calling that same constructor
with a carving allocator, so the layout is not described anywhere and
cannot drift between parent and child -- it is whatever the constructor
does, executed twice.

Ownership. The creating process owns the block and must ``close()`` it,
which unlinks. A child attaches by name, and closes its own handle
without unlinking. On POSIX a block outlives the process that forgot to
unlink it, so the owner's ``close`` runs from a ``finally`` and the
resource tracker is left to catch what a hard kill leaves behind.

Partitioning. Workers are given disjoint, contiguous row ranges so no
two ever write the same row and no lock is needed on the payload. The
cost is that the filled rows are not a prefix -- worker 0 may fill 900
of its 2,000 rows while worker 3 fills 1,100 -- and everything
downstream, ``compute_gae`` included, reads ``0..len(buffer)``. So the
ranges are compacted into a prefix once, at the end, by
``compact_ranges``.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from ..nn.model import ModelConfig
from .buffer import RolloutBuffer

# Fields start on a 64-byte boundary: one cache line on every machine
# this runs on, so no two fields share a line and a worker writing its
# own column cannot invalidate another's.
ALIGN = 64


def _round_up(n: int, to: int = ALIGN) -> int:
    return ((int(n) + to - 1) // to) * to


class _Recorder:
    """Allocates normally and remembers what it was asked for, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], np.dtype]] = []

    def __call__(self, shape: Any, dtype: Any = np.float64) -> np.ndarray:
        array = np.zeros(shape, dtype=dtype)
        # Per *row*: the leading axis is the capacity, and the whole
        # point is to rescale it.
        self.calls.append((tuple(array.shape[1:]), array.dtype))
        return array


class _Carver:
    """Hands out views into one block, bumping an offset per call."""

    def __init__(self, buf: memoryview) -> None:
        self._buf = buf
        self.offset = 0

    def __call__(self, shape: Any, dtype: Any = np.float64) -> np.ndarray:
        array = np.ndarray(shape, dtype=dtype, buffer=self._buf, offset=self.offset)
        self.offset += _round_up(array.nbytes)
        return array


def plan(capacity: int, cfg: ModelConfig | None = None) -> int:
    """Bytes one shared block needs to hold ``capacity`` rows.

    Measured by building a one-row buffer and rescaling what its
    constructor asked for, rather than by restating the field table
    here. A second copy of that table is a second thing to keep in step
    with the first.
    """
    cfg = cfg or ModelConfig()
    recorder = _Recorder()
    RolloutBuffer(1, cfg, alloc=recorder)  # a few tens of kilobytes
    total = 0
    for per_row, dtype in recorder.calls:
        count = int(np.prod(per_row, dtype=np.int64)) if per_row else 1
        total += _round_up(count * int(capacity) * dtype.itemsize)
    return total


@dataclass(frozen=True)
class BufferSpec:
    """What a child needs to build the parent's buffer. Picklable."""

    shm_name: str
    capacity: int
    nbytes: int
    cfg: ModelConfig


class SharedBuffer:
    """A ``RolloutBuffer`` over shared memory, plus the handle behind it.

    Create in the parent, pass ``.spec`` to a child, ``attach`` there.
    Both get an ordinary ``RolloutBuffer``; neither knows it is special.
    """

    def __init__(
        self,
        buffer: RolloutBuffer,
        shm: shared_memory.SharedMemory,
        spec: BufferSpec,
        owner: bool,
    ) -> None:
        self.buffer = buffer
        self.spec = spec
        self._shm = shm
        self._owner = owner
        self._closed = False

    @classmethod
    def create(
        cls, capacity: int, cfg: ModelConfig | None = None
    ) -> "SharedBuffer":
        cfg = cfg or ModelConfig()
        capacity = int(capacity)
        nbytes = plan(capacity, cfg)
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        try:
            carver = _Carver(shm.buf)
            buffer = RolloutBuffer(capacity, cfg, alloc=carver)
        except Exception:
            shm.close()
            shm.unlink()
            raise
        spec = BufferSpec(
            shm_name=shm.name, capacity=capacity, nbytes=nbytes, cfg=cfg
        )
        return cls(buffer, shm, spec, owner=True)

    @classmethod
    def attach(cls, spec: BufferSpec) -> "SharedBuffer":
        shm = shared_memory.SharedMemory(name=spec.shm_name)
        try:
            carver = _Carver(shm.buf)
            buffer = RolloutBuffer(spec.capacity, spec.cfg, alloc=carver)
        except Exception:
            shm.close()
            raise
        return cls(buffer, shm, spec, owner=False)

    def close(self) -> None:
        """Release this process's handle; the owner also unlinks.

        Idempotent, and it drops the buffer's references first: a
        ``SharedMemory`` refuses to close while a numpy view still
        points into it, and the resulting ``BufferError`` at interpreter
        exit is noise that hides whatever really went wrong.
        """
        if self._closed:
            return
        self._closed = True
        self.buffer = None  # type: ignore[assignment]
        try:
            self._shm.close()
        finally:
            if self._owner:
                try:
                    self._shm.unlink()
                except FileNotFoundError:  # pragma: no cover
                    pass

    def __enter__(self) -> "SharedBuffer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def partition(capacity: int, workers: int) -> list[tuple[int, int]]:
    """``[(start, stop), ...]``: disjoint, contiguous, covering.

    The remainder goes to the earliest workers a row at a time rather
    than all to the last one, so the widest and the narrowest range
    differ by at most one row.
    """
    capacity, workers = int(capacity), int(workers)
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    if capacity < workers:
        raise ValueError(
            f"capacity {capacity} cannot be split across {workers} workers"
        )
    base, extra = divmod(capacity, workers)
    out, start = [], 0
    for w in range(workers):
        stop = start + base + (1 if w < extra else 0)
        out.append((start, stop))
        start = stop
    return out


def compact_ranges(
    buffer: RolloutBuffer, filled: list[tuple[int, int]]
) -> int:
    """Move the filled parts of each range into one prefix. -> row count.

    ``filled`` is ``[(start, stop), ...]`` per worker, ``stop`` being one
    past that worker's last written row. Workers finish unevenly, so the
    written rows are a set of gapped intervals, while ``compute_gae``
    and the learner both read ``0..len(buffer)``. This closes the gaps.

    Order is preserved within every range, which is what matters: GAE
    orders a seat's trajectory by row index and a hand lives entirely
    inside one worker's range, so moving whole ranges cannot reorder a
    trajectory. Ranges are copied front to back and every destination is
    at or before its source, so a plain copy is safe in place.
    """
    ordered = sorted(filled)
    for i in range(1, len(ordered)):
        if ordered[i][0] < ordered[i - 1][1]:
            raise ValueError(f"ranges overlap: {ordered[i - 1]} and {ordered[i]}")

    write = 0
    for start, stop in ordered:
        start, stop = int(start), int(stop)
        if stop < start:
            raise ValueError(f"range ({start}, {stop}) ends before it starts")
        count = stop - start
        if count == 0:
            continue
        if write > start:  # pragma: no cover - guarded by the sort above
            raise AssertionError("compaction would overwrite unread rows")
        if write != start:
            for name in buffer.fields:
                array = getattr(buffer, name)
                array[write : write + count] = array[start:stop]
        write += count

    # Everything past the prefix is stale: it is whatever the previous
    # occupant of those rows wrote, and a reader that trusted the
    # capacity rather than the count would train on it.
    for name in buffer.fields:
        getattr(buffer, name)[write:] = 0
    buffer._n = write
    return write
