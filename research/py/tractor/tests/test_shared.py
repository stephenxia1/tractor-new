"""The shared-memory buffer, and the three ways it could corrupt a round.

A buffer several processes write at once fails quietly. A layout that
differs between parent and child reads one field's bytes as another's;
overlapping partitions let two workers write the same row; a compaction
that reorders rows scrambles the trajectories GAE reads by row index.
None of those raise -- they produce a buffer full of plausible
nonsense -- so each is pinned here against a case with a known answer.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from tractor.nn.model import ModelConfig
from tractor.train.buffer import RolloutBuffer
from tractor.train.shared import (
    ALIGN,
    BufferSpec,
    SharedBuffer,
    compact_ranges,
    partition,
    plan,
)

# The real ``max_history`` -- the buffer refuses anything else -- with
# the rest cut down so a test allocates kilobytes and not megabytes.
SMALL = ModelConfig(
    d_model=32,
    n_layers_history=1,
    n_layers_query=1,
    n_heads=2,
    d_ff=64,
    critic_d=16,
    critic_layers=1,
    critic_heads=2,
    critic_d_ff=32,
    belief_d=16,
    scalar_d=8,
    head_d=16,
    max_candidates=8,
)


# --- 1. the layout --------------------------------------------------------


def test_plan_covers_every_field_with_room_to_spare() -> None:
    """The block must hold the fields; padding makes it bigger, never less."""
    capacity = 37  # deliberately not a round number
    want = plan(capacity, SMALL)
    plain = RolloutBuffer(capacity, SMALL)
    raw = sum(getattr(plain, name).nbytes for name in plain.fields)
    assert want >= raw
    # Padding is bounded: at most one alignment unit per field.
    assert want <= raw + ALIGN * len(plain.fields)


def test_a_shared_buffer_is_an_ordinary_buffer() -> None:
    """Same shapes, same dtypes, and it starts zeroed."""
    plain = RolloutBuffer(16, SMALL)
    with SharedBuffer.create(16, SMALL) as shared:
        buf = shared.buffer
        assert buf.fields == plain.fields
        for name in plain.fields:
            got, want = getattr(buf, name), getattr(plain, name)
            assert got.shape == want.shape, name
            assert got.dtype == want.dtype, name
            assert not got.any(), f"{name} came back dirty"


def test_fields_do_not_overlap_in_the_block() -> None:
    """One field writing into another's bytes is the silent corruption."""
    with SharedBuffer.create(24, SMALL) as shared:
        buf = shared.buffer
        # Fill each field with a distinct value, then check none of them
        # disturbed an earlier one.
        for i, name in enumerate(buf.fields):
            array = getattr(buf, name)
            array[...] = np.ones(array.shape, dtype=array.dtype) * (
                1 if array.dtype == np.bool_ else (i % 7) + 1
            )
        for i, name in enumerate(buf.fields):
            array = getattr(buf, name)
            expect = 1 if array.dtype == np.bool_ else (i % 7) + 1
            assert (array == expect).all(), f"{name} was overwritten"


# --- 2. attaching from another process ------------------------------------


def _child_writes(spec: BufferSpec, start: int, stop: int, token: float) -> None:
    """Attach, stamp a range, detach. Runs in a separate process."""
    view = SharedBuffer.attach(spec)
    try:
        view.buffer.logp[start:stop] = token
        view.buffer.seat[start:stop] = int(token)
        view.buffer.history[start:stop] = token
    finally:
        view.close()


def test_a_child_process_writes_into_the_parents_rows() -> None:
    """The whole premise: no copy crosses the boundary, the rows are shared.

    If the child's layout disagreed with the parent's by even one byte
    this would come back as garbage in the wrong field, which is why it
    checks a value the child wrote through *three* fields of different
    dtypes rather than one.
    """
    ctx = mp.get_context("spawn")
    with SharedBuffer.create(64, SMALL) as shared:
        procs = []
        for w, (start, stop) in enumerate(partition(64, 4)):
            procs.append(
                ctx.Process(target=_child_writes,
                            args=(shared.spec, start, stop, float(w + 1)))
            )
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0, f"child exited {p.exitcode}"

        buf = shared.buffer
        for w, (start, stop) in enumerate(partition(64, 4)):
            token = float(w + 1)
            assert (buf.logp[start:stop] == token).all(), w
            assert (buf.seat[start:stop] == w + 1).all(), w
            assert (buf.history[start:stop] == token).all(), w


# --- 3. the partition -----------------------------------------------------


def test_partition_is_disjoint_and_covers_everything() -> None:
    for capacity in (4, 5, 17, 100, 1024):
        for workers in (1, 2, 3, 7, 4):
            if workers > capacity:
                continue
            ranges = partition(capacity, workers)
            assert len(ranges) == workers
            assert ranges[0][0] == 0 and ranges[-1][1] == capacity
            for (_a, stop), (start, _b) in zip(ranges, ranges[1:]):
                assert stop == start, (capacity, workers, ranges)
            widths = [b - a for a, b in ranges]
            assert max(widths) - min(widths) <= 1, (capacity, workers)
            assert sum(widths) == capacity


def test_partition_refuses_what_it_cannot_divide() -> None:
    for capacity, workers in ((3, 4), (0, 1)):
        try:
            partition(capacity, workers)
        except ValueError:
            pass
        else:
            raise AssertionError(f"({capacity}, {workers}) was accepted")


# --- 4. compaction --------------------------------------------------------


def _stamped(capacity: int) -> RolloutBuffer:
    """A buffer whose every row says which row it was."""
    buf = RolloutBuffer(capacity, SMALL)
    buf.episode[:] = np.arange(capacity)
    buf.logp[:] = np.arange(capacity, dtype=np.float32)
    return buf


def test_compaction_closes_the_gaps_and_keeps_the_order() -> None:
    """Workers finish unevenly, so the written rows are gapped intervals."""
    buf = _stamped(40)
    # Four ranges of ten, each only partly filled.
    filled = [(0, 7), (10, 13), (20, 29), (30, 31)]
    want = np.concatenate([np.arange(a, b) for a, b in filled])

    rows = compact_ranges(buf, filled)
    assert rows == want.size == 20
    assert len(buf) == rows
    assert np.array_equal(buf.episode[:rows], want)
    assert np.array_equal(buf.logp[:rows], want.astype(np.float32))


def test_compaction_zeroes_what_it_left_behind() -> None:
    """A stale tail is a legal-looking row the learner would train on."""
    buf = _stamped(40)
    rows = compact_ranges(buf, [(0, 5), (10, 15)])
    assert rows == 10
    assert not buf.episode[rows:].any()
    assert not buf.logp[rows:].any()


def test_compaction_of_full_ranges_is_the_identity() -> None:
    buf = _stamped(40)
    rows = compact_ranges(buf, partition(40, 4))
    assert rows == 40
    assert np.array_equal(buf.episode[:40], np.arange(40))


def test_compaction_rejects_overlapping_ranges() -> None:
    """Two workers owning one row is the bug compaction must not paper over."""
    buf = _stamped(40)
    try:
        compact_ranges(buf, [(0, 12), (10, 20)])
    except ValueError as exc:
        assert "overlap" in str(exc), exc
    else:
        raise AssertionError("overlapping ranges were compacted")


def test_compaction_handles_an_empty_range() -> None:
    """A worker that never got a hand contributes nothing, not a gap."""
    buf = _stamped(40)
    rows = compact_ranges(buf, [(0, 5), (10, 10), (20, 23)])
    assert rows == 8
    assert np.array_equal(buf.episode[:8], np.array([0, 1, 2, 3, 4, 20, 21, 22]))
