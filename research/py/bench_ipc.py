"""Can a process boundary carry the rollout's observations fast enough?

The rollout is CPU-bound in pure Python -- measured on the training run,
``collect`` spends 2.70 s enumerating, 1.94 s encoding and 1.42 s
stepping the simulator against 1.38 s in the network -- so the obvious
move is to put the simulator in worker processes and keep one batched
forward in the parent. Whether that pays depends entirely on what the
boundary costs, because every decision has to cross it twice: an
encoded observation out, an action index back.

The budget, from the same run. A step is 11,100 decisions in 8.8 s of
collect. Driving collect down to ~2 s means the parent must sustain
about **5,500 round trips a second**, each carrying 28,474 bytes of
observation, or 157 MiB/s -- on top of running the network.

Two transports are measured against that number:

``queue``    a ``multiprocessing.Queue`` carrying the arrays directly,
             which pickles each one and writes it down a pipe.
``shm``      a ring of preallocated ``shared_memory`` slots; the worker
             writes into a slot and the queue carries only its index,
             so the payload is copied once instead of pickled.

Run it with a no-op parent first (``--work 0``) to find the transport's
own ceiling, then with a realistic per-batch cost to see what survives.

    python bench_ipc.py --workers 7 --seconds 5
    python bench_ipc.py --workers 7 --transport shm --inflight 32
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from multiprocessing import shared_memory

import numpy as np

# The real observation, field by field, from ``tractor.train.buffer``
# and ``tractor.nn.features``. Hard-coded rather than imported so this
# script runs anywhere, and checked against the real thing by
# ``--verify``.
HISTORY_DIM = 37
CONTEXT_DIM = 37
NUM_PLANES = 10
D_CARD = 30
NUM_TYPES = 54
NUM_PRIVILEGED_ROWS = 5
MAX_HISTORY = 128
MAX_CANDIDATES = 128

FIELDS: tuple[tuple[str, tuple[int, ...], type], ...] = (
    ("history", (MAX_HISTORY, HISTORY_DIM), np.float16),
    ("context", (CONTEXT_DIM,), np.float16),
    ("unseen", (NUM_TYPES,), np.float16),
    ("planes", (NUM_PLANES, NUM_TYPES), np.float16),
    ("cards", (NUM_TYPES, D_CARD), np.float16),
    ("candidates", (MAX_CANDIDATES, NUM_TYPES), np.float16),
    ("candidate_mask", (MAX_CANDIDATES,), np.bool_),
    ("privileged", (NUM_PRIVILEGED_ROWS, NUM_TYPES), np.float16),
)

PAYLOAD = sum(int(np.prod(shape)) * np.dtype(dt).itemsize for _n, shape, dt in FIELDS)

# What the run measured, so the script can say whether a result clears
# the bar instead of leaving the reader to do the arithmetic.
DECISIONS_PER_STEP = 11_100
COLLECT_SECONDS = 8.8
TARGET_RPS = DECISIONS_PER_STEP / 2.0  # collect at ~2 s


def _make_observation(rng: np.random.Generator) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, shape, dt in FIELDS:
        if dt is np.bool_:
            out[name] = rng.integers(0, 2, size=shape).astype(np.bool_)
        else:
            out[name] = rng.standard_normal(shape).astype(dt)
    return out


# --- transport A: pickle through a Queue ----------------------------------


def _queue_worker(wid: int, out_q, in_q, inflight: int, stop,
                  batch: int) -> None:
    rng = np.random.default_rng(wid)
    obs = _make_observation(rng)
    # One message carries ``batch`` decisions, stacked. The worker holds
    # that many hands in flight to fill it, which is what the pool
    # already does inside a single process.
    stacked = {k: np.repeat(v[None], batch, axis=0) for k, v in obs.items()}
    pending = 0
    while not stop.is_set():
        while pending < inflight:
            out_q.put((wid, batch, {k: v.copy() for k, v in stacked.items()}))
            pending += 1
        in_q.get()
        pending -= 1


# --- transport B: a shared-memory ring ------------------------------------


def _shm_worker(wid: int, out_q, in_q, inflight: int, stop, shm_name: str,
                base: int, slots: int, batch: int) -> None:
    rng = np.random.default_rng(wid)
    obs = _make_observation(rng)
    flat = np.concatenate([v.reshape(-1).view(np.uint8) for v in obs.values()])
    block = np.repeat(flat[None], batch, axis=0)
    shm = shared_memory.SharedMemory(name=shm_name)
    ring = np.ndarray((slots, PAYLOAD), dtype=np.uint8, buffer=shm.buf)
    try:
        cursor, pending = 0, 0
        while not stop.is_set():
            while pending < inflight:
                start = base + cursor * batch
                ring[start : start + batch, :] = block  # one copy, no pickle
                out_q.put((wid, start, batch))
                cursor = (cursor + 1) % inflight
                pending += 1
            in_q.get()
            pending -= 1
    finally:
        shm.close()


def run(transport: str, workers: int, inflight: int, seconds: float,
        work_us: float, batch: int = 1) -> dict[str, float]:
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    out_q = ctx.Queue(maxsize=workers * inflight * 2)
    in_qs = [ctx.Queue(maxsize=inflight * 2) for _ in range(workers)]

    shm = None
    procs = []
    if transport == "shm":
        slots = workers * inflight * batch
        shm = shared_memory.SharedMemory(create=True, size=slots * PAYLOAD)
        ring = np.ndarray((slots, PAYLOAD), dtype=np.uint8, buffer=shm.buf)
        for w in range(workers):
            procs.append(ctx.Process(
                target=_shm_worker,
                args=(w, out_q, in_qs[w], inflight, stop, shm.name,
                      w * inflight * batch, slots, batch),
                daemon=True))
    else:
        ring = None
        for w in range(workers):
            procs.append(ctx.Process(
                target=_queue_worker,
                args=(w, out_q, in_qs[w], inflight, stop, batch),
                daemon=True))

    for p in procs:
        p.start()

    # Warm up: process starts and first allocations are not throughput.
    def take() -> tuple[int, int]:
        """One message -> ``(worker, decisions it carried)``, payload read."""
        if transport == "shm":
            wid, start, count = out_q.get()
            # Touch the payload: a transport that is never read is not
            # a transport, and the read is where shm pays its copy.
            _ = ring[start : start + count].sum()
            return wid, count
        wid, count, arrays = out_q.get()
        _ = arrays["candidates"].sum()
        return wid, count

    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline:
        wid, _count = take()
        in_qs[wid].put(0)

    done = 0
    t0 = time.perf_counter()
    end = t0 + seconds
    while time.perf_counter() < end:
        wid, count = take()
        if work_us > 0:
            spin = time.perf_counter() + work_us * 1e-6 * count
            while time.perf_counter() < spin:
                pass
        in_qs[wid].put(0)
        done += count
    elapsed = time.perf_counter() - t0

    stop.set()
    # Drain so the workers' final puts do not block their exit.
    for _ in range(workers * inflight * 2):
        try:
            msg = out_q.get_nowait()
            in_qs[msg[0]].put_nowait(0)
        except Exception:
            break
    for p in procs:
        p.join(timeout=2.0)
        if p.is_alive():
            p.terminate()
    if shm is not None:
        shm.close()
        shm.unlink()

    rps = done / elapsed
    return {
        "transport": transport,
        "workers": float(workers),
        "inflight": float(inflight),
        "batch": float(batch),
        "round_trips": float(done),
        "seconds": elapsed,
        "rps": rps,
        "mib_s": rps * PAYLOAD / 2**20,
        "collect_s": DECISIONS_PER_STEP / rps if rps > 0 else float("inf"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--inflight", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--work-us", type=float, default=0.0,
                    help="microseconds of parent work per decision")
    ap.add_argument("--transport", default="both",
                    choices=("queue", "shm", "both"))
    ap.add_argument("--batch", default="1",
                    help="decisions per message; a comma list sweeps them")
    args = ap.parse_args()
    batches = [int(b) for b in str(args.batch).split(",") if b.strip()]

    print(f"payload {PAYLOAD:,} B per decision")
    print(f"target  {TARGET_RPS:,.0f} round trips/s "
          f"({TARGET_RPS * PAYLOAD / 2**20:,.0f} MiB/s) to put collect at 2.0 s")
    print(f"today   {DECISIONS_PER_STEP / COLLECT_SECONDS:,.0f} decisions/s "
          f"single-threaded, collect {COLLECT_SECONDS} s")
    print()
    names = ("queue", "shm") if args.transport == "both" else (args.transport,)
    print(f"{'transport':10s} {'batch':>5} {'decisions/s':>12} {'MiB/s':>8} "
          f"{'collect':>8}  verdict")
    for name in names:
        for batch in batches:
            r = run(name, args.workers, args.inflight, args.seconds,
                    args.work_us, batch)
            verdict = "CLEARS" if r["rps"] >= TARGET_RPS else "short"
            print(f"{name:10s} {batch:5d} {r['rps']:12,.0f} {r['mib_s']:8,.0f} "
                  f"{r['collect_s']:7.2f}s  {verdict}")


if __name__ == "__main__":
    main()
