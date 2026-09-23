"""``collect`` with the simulator in worker processes.

The single-process rollout is CPU-bound in Python. Measured on the
training run, an 8.8 s ``collect`` spends 2.70 s enumerating legal
moves, 1.94 s encoding features and 1.42 s stepping the simulator
against 1.38 s in the network -- about 70% host work, 16% accelerator,
on a container reserving eight cores and using one. Enumeration is pure
Python, so threads cannot help; processes can.

Two measurements shaped what this file does.

*The transport.* Every decision has to get an observation to the
process holding the network and an action back. ``py/bench_ipc.py``
puts a ``multiprocessing.Queue`` carrying the 28,466-byte observation
per decision at ~1,150 decisions a second -- slower than the loop it
replaces -- against ~23,600 for a shared-memory ring with the queue
carrying a row index, at a batch of 32. So nothing is sent: the buffer
*is* the transport (see ``shared.py``) and the queues carry tags and
integers.

*The workers are persistent.* Spawning them per round does not pay.
Measured, a spawned worker takes 9.5 s to reach its first instruction
and seven of them 12.3 s, because each one re-imports torch; a round of
128 hands takes about 8 s in total. The first version of this file
spawned per call and measured 1.08x against the serial rollout on an
A10G, with the whole gain inside the startup. So ``RolloutPool`` starts
its workers once and keeps them across training steps, and a round is a
message rather than a process.

Shape of a round. Each worker owns a disjoint row range and its own
pool of hands. It fills the pool, encodes and enumerates every seat to
act, writes those rows, and tells the parent which ones. The parent
waits until every live worker has reported, runs one forward over all
of their rows at once, writes the sampled action, log-probability and
value back, and releases the workers to step their simulators.

That is lockstep, and it costs the overlap: while the parent runs the
forward its workers idle, and vice versa. The arithmetic still favours
it, and the alternative -- two batches in flight per worker -- needs
each worker to hold two independent pools. Worth doing once this is
proven, not before.

What this does *not* change: the rewards, the credit rule, the
sampling, or what lands in a row. Those come from ``rollout.py``'s own
helpers, called here rather than reimplemented, because a second copy
of the trick-credit rule that drifts from the first is the kind of bug
that survives a training run disguised as a hyperparameter.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any

import numpy as np
import torch

from ..gen.episode import deal_sizes, generate
from ..nn.features import POINTS_PER_DECK, Cache
from ..nn.model import ModelConfig
from .buffer import RolloutBuffer, obs_batch_from_arrays
from .rollout import (
    DECKS,
    _choose,
    _credit_terminal,
    _credit_trick,
    _evaluate,
    _InFlight,
    _pack,
    _room_for_another,
    _span,
)
from .shared import SharedBuffer, compact_ranges, partition

__all__ = ["RolloutPool", "collect_parallel", "DEFAULT_WORKERS"]

# One core is the parent's, running the network and the sampler.
DEFAULT_WORKERS = 7

# A worker that cannot reach the parent within this long has been
# orphaned -- the parent died, or is wedged. Exiting beats blocking
# forever inside a container billed until its timeout.
WORKER_TIMEOUT_S = 900.0

_STAT_KEYS = (
    "hands", "decisions", "candidate_total", "forced", "truncated",
    "unenumerated",
)


def _worker(wid: int, spec: Any, start: int, stop: int, req_q: Any,
            rep_q: Any) -> None:
    """Play rounds into rows ``[start, stop)`` until told to stop.

    Everything except the forward pass happens here. The parent never
    touches this worker's ``PlayState`` and this worker never touches
    the network, so the only shared mutable state is the row range --
    which no other worker can reach.
    """
    view = SharedBuffer.attach(spec)
    try:
        window = view.buffer.window(start, stop)
        cfg = spec.cfg
        cache = Cache()  # keyed by trump context; worth keeping across rounds
        point_norm = float(POINTS_PER_DECK * DECKS)
        per_seat, _kitty = deal_sizes(DECKS)
        max_decisions = per_seat * 4

        while True:
            cmd = rep_q.get(timeout=WORKER_TIMEOUT_S)
            if cmd[0] == "stop":
                return
            _tag, n_hands, episode_base, seed, level_weight, pool = cmd

            window.reset()
            rng = np.random.default_rng(seed)
            weight = np.float32(level_weight)
            active: list[_InFlight] = []
            started = 0
            stats = dict.fromkeys(_STAT_KEYS, 0)

            while True:
                while (
                    len(active) < pool
                    and started < n_hands
                    and _room_for_another(window, active, max_decisions)
                ):
                    state, _info = generate(rng, DECKS)
                    active.append(
                        _InFlight(state=state, episode=episode_base + started)
                    )
                    started += 1
                if not active:
                    break

                packed = [_pack(hand, cache, cfg) for hand in active]
                first_local = len(window)
                for hand, decision in zip(active, packed):
                    row = window.add(
                        history=decision.history,
                        history_len=decision.history_len,
                        context=decision.context,
                        unseen=decision.unseen,
                        planes=decision.planes,
                        cards=decision.cards,
                        candidates=decision.candidates,
                        candidate_mask=decision.candidate_mask,
                        privileged=decision.privileged,
                        # Placeholders. The parent overwrites all three
                        # before it releases this worker.
                        action=0,
                        logp=0.0,
                        value=0.0,
                        seat=decision.seat,
                        episode=hand.episode,
                    )
                    hand.trick_rows.append(row)
                    hand.last_row[decision.seat] = row

                # ``add`` fills sequentially, so one batch is one
                # contiguous run and the parent needs two numbers.
                req_q.put(("batch", wid, start + first_local, len(packed)))
                if rep_q.get(timeout=WORKER_TIMEOUT_S)[0] != "go":
                    return  # the parent is shutting the round down

                finished: list[int] = []
                for i, (hand, decision) in enumerate(zip(active, packed)):
                    row = first_local + i
                    slot = int(window.action[row])
                    stats["decisions"] += 1
                    moves = decision.moves
                    stats["candidate_total"] += len(moves.moves)
                    stats["forced"] += int(len(moves.moves) == 1)
                    stats["truncated"] += int(moves.truncated)
                    stats["unenumerated"] += int(moves.unenumerated)

                    cards = decision.candidates[slot].astype(np.int8)
                    before = len(hand.state.tricks)
                    hand.state.step(cards)
                    if len(hand.state.tricks) > before:
                        _credit_trick(window, hand, point_norm, weight)
                    if hand.state.done:
                        _credit_terminal(window, hand, weight)
                        stats["hands"] += 1
                        finished.append(i)
                for i in reversed(finished):
                    active.pop(i)

            req_q.put(("done", wid, len(window), stats))
    finally:
        view.close()


class RolloutPool:
    """Worker processes that outlive a round, and the block they share.

    Start one before the training loop and call ``collect`` per step.
    Starting one per step spends more on ``import torch`` than the round
    it is meant to accelerate -- see the module docstring.
    """

    def __init__(
        self,
        cfg: ModelConfig | None = None,
        workers: int = DEFAULT_WORKERS,
        hands_per_round: int = 128,
        capacity: int | None = None,
    ) -> None:
        self.cfg = cfg or ModelConfig()
        self.workers = max(1, min(int(workers), int(hands_per_round)))
        self.hands_per_round = int(hands_per_round)

        per_seat, _kitty = deal_sizes(DECKS)
        self._max_decisions = per_seat * 4
        # Every hand needs room for its worst case and a worker cannot
        # borrow from its neighbour's range, so the block is sized by
        # the widest share rather than by the total.
        widest = -(-self.hands_per_round // self.workers)
        if capacity is None:
            capacity = widest * self._max_decisions * self.workers
        self.capacity = int(capacity)
        self.ranges = partition(self.capacity, self.workers)

        ctx = mp.get_context("spawn")
        self._shared = SharedBuffer.create(self.capacity, self.cfg)
        self._req_q = ctx.Queue()
        self._rep_qs = [ctx.Queue() for _ in range(self.workers)]
        self._procs = []
        self._closed = False
        try:
            for w in range(self.workers):
                start, stop = self.ranges[w]
                proc = ctx.Process(
                    target=_worker,
                    args=(w, self._shared.spec, start, stop, self._req_q,
                          self._rep_qs[w]),
                    daemon=True,
                )
                proc.start()
                self._procs.append(proc)
        except Exception:
            self.close()
            raise

    # --- one round --------------------------------------------------------

    def collect(
        self,
        net: Any,
        rng: np.random.Generator,
        buffer: RolloutBuffer | None = None,
        *,
        n_hands: int | None = None,
        device: torch.device | str = "cpu",
        temperature: float = 1.0,
        greedy: bool = False,
        level_weight: float = 1.0,
        pool: int | None = None,
        amp: bool = False,
        spans: Any = None,
        episode_base: int = 0,
    ) -> tuple[RolloutBuffer, dict[str, float]]:
        """One round. -> ``(buffer, stats)``, same keys as ``collect``.

        ``spans`` records the parent's own work -- "forward" and
        "sample" -- and nothing else. Encode, enumerate and step happen
        in the workers, where this process cannot time them; the serial
        rollout is where that breakdown comes from.
        """
        if self._closed:
            raise RuntimeError("collect() on a closed RolloutPool")
        n_hands = int(n_hands if n_hands is not None else self.hands_per_round)
        if n_hands > self.hands_per_round:
            raise ValueError(
                f"this pool was sized for {self.hands_per_round} hands a "
                f"round, not {n_hands}; make a wider one"
            )
        workers = self.workers
        pool_total = int(pool) if pool else 128
        per_worker_pool = max(1, pool_total // workers)
        hands_each = [
            n_hands // workers + (1 if w < n_hands % workers else 0)
            for w in range(workers)
        ]

        was_training = getattr(net, "training", False)
        if hasattr(net, "eval"):
            net.eval()

        started_at = time.perf_counter()
        totals = dict.fromkeys(_STAT_KEYS, 0)
        forwards = 0
        batch_total = 0
        buf = self._shared.buffer

        try:
            live = set()
            base = episode_base
            for w in range(workers):
                if hands_each[w] == 0:
                    continue
                self._rep_qs[w].put((
                    "round", hands_each[w], base,
                    int(rng.integers(0, 2**31 - 1)), float(level_weight),
                    per_worker_pool,
                ))
                base += hands_each[w]
                live.add(w)

            ends: dict[int, int] = {}
            with torch.no_grad(), torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=amp,
            ):
                while live:
                    # Every live worker blocks after reporting, so
                    # waiting for all of them is safe and makes the
                    # batch as wide as the pool. It is also lockstep.
                    pending: dict[int, tuple[int, int]] = {}
                    while len(pending) < len(live):
                        msg = self._req_q.get(timeout=WORKER_TIMEOUT_S)
                        if msg[0] == "done":
                            _tag, wid, rows, stats = msg
                            ends[wid] = rows
                            live.discard(wid)
                            for key in totals:
                                totals[key] += int(stats[key])
                        else:
                            _tag, wid, first, count = msg
                            pending[wid] = (first, count)
                    if not pending:
                        break

                    # Fancy indexing, so each of these materialises a
                    # copy: at 1024 hands a round moves about 2.4 GiB
                    # through here. Timed apart from the forward because
                    # the two have different remedies -- this one shrinks
                    # by slicing each worker's contiguous run instead of
                    # gathering, the forward only by a wider batch.
                    with _span(spans, "gather"):
                        rows = np.concatenate([
                            np.arange(first, first + count, dtype=np.int64)
                            for first, count in
                            (pending[w] for w in sorted(pending))
                        ])
                        obs = obs_batch_from_arrays(
                            buf.history[rows], buf.history_len[rows],
                            buf.context[rows], buf.unseen[rows],
                            buf.planes[rows], buf.cards[rows],
                            buf.candidates[rows], buf.candidate_mask[rows],
                            device=device,
                        )
                        privileged = torch.from_numpy(
                            buf.privileged[rows].astype(np.float32)
                        ).to(device)
                    with _span(spans, "forward"):
                        out, values = _evaluate(net, obs, privileged)
                    with _span(spans, "sample"):
                        slots, chosen_logp = _choose(
                            out.logits, obs.candidate_mask, temperature,
                            greedy, rng,
                        )
                    with _span(spans, "writeback"):
                        buf.action[rows] = slots.astype(np.int64)
                        buf.logp[rows] = chosen_logp.astype(np.float32)
                        buf.value[rows] = (
                            values.detach().float().cpu().numpy()
                            .astype(np.float32)
                        )
                    forwards += 1
                    batch_total += int(rows.size)
                    with _span(spans, "release"):
                        for w in pending:
                            self._rep_qs[w].put(("go",))

            with _span(spans, "compact"):
                filled = [
                    (self.ranges[w][0], self.ranges[w][0] + ends.get(w, 0))
                    for w in range(workers)
                ]
                rows_written = compact_ranges(buf, filled)

            out_buffer = buffer if buffer is not None else RolloutBuffer(
                max(rows_written, 1), self.cfg
            )
            if out_buffer.capacity < rows_written:
                raise ValueError(
                    f"the caller's buffer holds {out_buffer.capacity} rows "
                    f"and the round wrote {rows_written}"
                )
            # A second full pass over the round's bytes. Avoidable in
            # principle -- the learner could read the shared block --
            # but only once nothing downstream outlives the pool.
            with _span(spans, "copy_out"):
                for name in buf.fields:
                    getattr(out_buffer, name)[:rows_written] = getattr(
                        buf, name
                    )[:rows_written]
                out_buffer._n = rows_written
        finally:
            if was_training and hasattr(net, "train"):
                net.train()

        elapsed = time.perf_counter() - started_at
        decisions = totals["decisions"]
        return out_buffer, {
            "hands": totals["hands"],
            "decisions": decisions,
            "candidates_per_decision": (
                totals["candidate_total"] / max(decisions, 1)
            ),
            "forced_fraction": totals["forced"] / max(decisions, 1),
            "truncated": totals["truncated"],
            "unenumerated": totals["unenumerated"],
            "forwards": forwards,
            "mean_batch": batch_total / max(forwards, 1),
            "seconds": elapsed,
            "workers": float(workers),
            "rows": float(rows_written),
        }

    # --- teardown ---------------------------------------------------------

    def close(self) -> None:
        """Stop the workers and release the block. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for q in getattr(self, "_rep_qs", []):
            try:
                q.put(("stop",))
            except Exception:  # pragma: no cover - queue already dead
                pass
        for proc in getattr(self, "_procs", []):
            proc.join(timeout=15.0)
            if proc.is_alive():  # pragma: no cover
                proc.terminate()
        shared = getattr(self, "_shared", None)
        if shared is not None:
            shared.close()

    def __enter__(self) -> "RolloutPool":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


def collect_parallel(
    net: Any,
    rng: np.random.Generator,
    buffer: RolloutBuffer | None,
    n_hands: int,
    device: torch.device | str,
    cfg: ModelConfig | None = None,
    temperature: float = 1.0,
    greedy: bool = False,
    spans: Any = None,
    level_weight: float = 1.0,
    pool: int | None = None,
    amp: bool = False,
    workers: int = DEFAULT_WORKERS,
    capacity: int | None = None,
) -> tuple[RolloutBuffer, dict[str, float]]:
    """One round through a throwaway pool.

    For tests and one-off measurements. A training loop should hold a
    ``RolloutPool`` instead: this pays the worker start-up -- about 9.5
    seconds for one worker and 12.3 for seven, most of it ``import
    torch`` -- on every call, which is more than the round costs.
    """
    cfg = cfg or getattr(net, "cfg", None) or ModelConfig()
    with RolloutPool(
        cfg=cfg, workers=workers, hands_per_round=int(n_hands),
        capacity=capacity,
    ) as worker_pool:
        return worker_pool.collect(
            net, rng, buffer, n_hands=int(n_hands), device=device,
            temperature=temperature, greedy=greedy,
            level_weight=level_weight, pool=pool, amp=amp, spans=spans,
        )
