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
import queue
import time
from typing import Any

import numpy as np
import torch

from ..gen.episode import deal_sizes, generate
from ..nn.features import POINTS_PER_DECK, Cache
from ..nn.model import MAX_PICKS, PICK_SUBMIT, ModelConfig
from .buffer import RolloutBuffer, obs_batch_from_arrays
from ..nn.model import relative_seat_agents
from .rollout import (
    DECKS,
    _credit_terminal,
    _credit_trick,
    _InFlight,
    _pack,
    _room_for_another,
    _span,
    as_nets,
    check_agents,
    routed_step,
)
from .pick_rollout import _picker, actor_encode, score_step
from .shared import SharedBuffer, copy_ranges, partition
from .starts import pack_start, start_state

__all__ = ["RolloutPool", "collect_parallel", "DEFAULT_WORKERS"]

# One core is the parent's, running the network and the sampler.
DEFAULT_WORKERS = 7

# A worker mid-round that has not been heard from in this long is
# wedged; the parent gives up on the round rather than blocking forever
# inside a container billed until its timeout.
WORKER_TIMEOUT_S = 900.0
# How often a waiting side looks at whether the other is still alive.
# An idle worker waits for its next command however long the parent is
# busy elsewhere -- a population's pre-play phase uses no workers for
# half an hour -- and exits only if the parent has died. (It used to
# give up after WORKER_TIMEOUT_S of silence: pop8's workers exited a
# quarter of an hour into its first pre-play phase, and the next round
# waited on the dead.)
POLL_S = 30.0


def _command(rep_q: Any, poll_s: float) -> tuple:
    """The worker's next command from the parent, however long in coming;
    ``("stop",)`` once the parent has died."""
    while True:
        try:
            return rep_q.get(timeout=poll_s)
        except queue.Empty:
            parent = mp.parent_process()
            if parent is not None and not parent.is_alive():
                return ("stop",)

_STAT_KEYS = (
    "hands", "decisions", "candidate_total", "forced", "truncated",
    "unenumerated", "forced_steps",
)


def _worker(wid: int, spec: Any, start: int, stop: int, req_q: Any,
            rep_q: Any, poll_s: float = POLL_S) -> None:
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
            cmd = _command(rep_q, poll_s)
            if cmd[0] == "stop":
                return
            (_tag, n_hands, episode_base, seed, level_weight, pool, starts,
             halves) = cmd

            window.reset()
            rng = np.random.default_rng(seed)
            weight = np.float32(level_weight)
            # One pool of hands per half. With one half this is the
            # lockstep round. With two, each half's batch waits at the
            # parent while this process steps and packs the other, so the
            # forward and the simulator overlap.
            pools: list[list[_InFlight]] = [[] for _ in range(halves)]
            per_half = max(1, pool // halves)
            started = 0
            stats = dict.fromkeys(_STAT_KEYS, 0)

            def refill(h: int) -> None:
                nonlocal started
                in_flight = [hand for p in pools for hand in p]
                while (
                    len(pools[h]) < per_half
                    and started < n_hands
                    and _room_for_another(window, in_flight, max_decisions)
                ):
                    if starts is not None:
                        state = start_state(starts[started])
                    else:
                        state, _info = generate(rng, DECKS)
                    hand = _InFlight(state=state, episode=episode_base + started)
                    pools[h].append(hand)
                    in_flight.append(hand)
                    started += 1

            def advance(cur: list, r: int, rules: Any) -> None:
                # Card mode: record each step's mask and prefix, taking
                # forced steps here, until a step offers a choice (whose
                # index the parent reads from ``action``) or the play
                # submits (``action`` -1).
                picked, k = cur[0], cur[2]
                while True:
                    if k >= MAX_PICKS:
                        raise RuntimeError("a play did not submit within MAX_PICKS steps")
                    m = rules.mask(picked, cur[1])
                    window.candidate_mask[r, k] = m
                    window.candidates[r, k] = picked
                    legal = int(m.sum())
                    if legal == 0:
                        raise RuntimeError(f"no legal pick at step {k}")
                    if legal != 1:
                        break
                    a = int(np.argmax(m))
                    k += 1
                    stats["forced_steps"] += 1
                    if a == PICK_SUBMIT:
                        cur[3] = True
                        break
                    picked[a] += 1
                    cur[1] = a
                cur[2] = k
                window.action[r] = -1 if cur[3] else k

            def send(h: int) -> tuple[list, int]:
                packed = [_pack(hand, cache, cfg) for hand in pools[h]]
                first_local = len(window)
                for hand, decision in zip(pools[h], packed):
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
                        # before it releases this half.
                        action=0,
                        logp=0.0,
                        value=0.0,
                        seat=decision.seat,
                        episode=hand.episode,
                        pending_play=cfg.cards_mode,
                    )
                    hand.trick_rows.append(row)
                    hand.last_row[decision.seat] = row
                # ``add`` fills sequentially, so one batch is one
                # contiguous run and the parent needs two numbers. An
                # empty half still reports, so the parent's lockstep per
                # half stays whole.
                if cfg.cards_mode:
                    # [picked, last, step, submitted] per decision.
                    cursors[h] = [
                        [np.zeros(54, dtype=np.int8), -1, 0, False] for _ in packed
                    ]
                    for i, decision in enumerate(packed):
                        advance(cursors[h][i], first_local + i, decision.rules)
                req_q.put(("batch", wid, h, start + first_local, len(packed)))
                return packed, first_local

            def pick(h: int, actions: np.ndarray) -> None:
                # The parent's picks for this half's live plays, one per
                # row of the batch (-1 for rows already submitted).
                packed, first_local = pending[h]
                for i, decision in enumerate(packed):
                    cur = cursors[h][i]
                    if cur[3]:
                        continue
                    a = int(actions[i])
                    r = first_local + i
                    if a < 0 or not window.candidate_mask[r, cur[2], a]:
                        raise RuntimeError(f"illegal pick {a} at step {cur[2]}")
                    cur[2] += 1
                    if a == PICK_SUBMIT:
                        cur[3] = True
                        window.action[r] = -1
                        continue
                    cur[0][a] += 1
                    cur[1] = a
                    advance(cur, r, decision.rules)
                req_q.put(("picked", wid, h))

            def step(h: int, packed: list, first_local: int) -> None:
                finished: list[int] = []
                for i, (hand, decision) in enumerate(zip(pools[h], packed)):
                    row = first_local + i
                    slot = int(window.action[row])
                    stats["decisions"] += 1
                    moves = decision.moves
                    stats["candidate_total"] += len(moves.moves)
                    stats["forced"] += int(len(moves.moves) == 1)
                    stats["truncated"] += int(moves.truncated)
                    stats["unenumerated"] += int(moves.unenumerated)

                    if cfg.cards_mode:
                        cards = cursors[h][i][0].copy()
                    else:
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
                    pools[h].pop(i)

            if n_hands == 0:
                req_q.put(("done", wid, len(window), stats))
                continue
            pending = []
            cursors: list = [None] * halves
            for h in range(halves):
                refill(h)
                pending.append(None)
                pending[h] = send(h)
            h = 0
            while True:
                cmd = _command(rep_q, poll_s)
                while cmd[0] == "pick":
                    pick(cmd[1], cmd[2])
                    cmd = _command(rep_q, poll_s)
                if cmd[0] != "go":
                    return  # the parent is shutting the round down
                step(h, *pending[h])
                refill(h)
                if started >= n_hands and not any(pools):
                    # Each other half has one empty batch at the parent;
                    # take its go first, so none is left in this queue
                    # to be read as the next round's command.
                    for _ in range(halves - 1):
                        if _command(rep_q, poll_s)[0] != "go":
                            return
                    break
                pending[h] = send(h)
                h = (h + 1) % halves

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
        poll_s: float = POLL_S,
    ) -> None:
        self.cfg = cfg or ModelConfig()
        self.poll_s = float(poll_s)
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
                          self._rep_qs[w], self.poll_s),
                    daemon=True,
                )
                proc.start()
                self._procs.append(proc)
        except Exception:
            self.close()
            raise

    def _receive(self) -> tuple:
        """The next message from any worker. A worker that has died is an
        error at once, naming it, rather than a quarter-hour's silence;
        one alive but silent for ``WORKER_TIMEOUT_S`` is wedged."""
        waited = 0.0
        while True:
            try:
                return self._req_q.get(timeout=self.poll_s)
            except queue.Empty:
                dead = [w for w, proc in enumerate(self._procs) if not proc.is_alive()]
                if dead:
                    raise RuntimeError(f"rollout worker(s) {dead} have died")
                waited += self.poll_s
                if waited >= WORKER_TIMEOUT_S:
                    raise TimeoutError(
                        f"no word from the rollout workers in {WORKER_TIMEOUT_S:.0f} s")

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
        starts: Any = None,
        agents: Any = None,
        chunks: int = 1,
        critic: Any = None,
        overlap: bool = False,
        forwards: str = "eager",
    ) -> tuple[RolloutBuffer, dict[str, float]]:
        """One round. -> ``(buffer, stats)``, same keys as ``collect``.

        ``overlap`` splits every worker's hands into two halves that take
        turns: while the parent runs the forward on one half's rows, the
        workers step and pack the other's. Lockstep costs the sum of the
        parent's time and the workers'; this costs about the larger. Each
        forward is then half as wide, so pair it with a doubled ``pool``.

        ``starts``, when given, are the round's play starts (``train.starts.
        Start``), one per hand, in place of the workers' own uniform draw.

        ``net`` may be a population's list of networks, with ``agents``
        int64[n_hands, 4] naming the network in each seat of each hand, as
        in ``rollout.collect``. The workers are unchanged: a row's hand is
        its episode less ``episode_base``, so the parent routes it.

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
        nets = as_nets(net)
        seating = check_agents(agents, n_hands, nets)
        if critic is not None and seating is None:
            raise ValueError("a shared critic needs the seating to know who is who")
        cards_mode = self.cfg.cards_mode
        if cards_mode and critic is not None:
            raise ValueError("card-mode rollout with a shared critic is not supported yet")
        workers = self.workers
        pool_total = int(pool) if pool else 128
        per_worker_pool = max(1, pool_total // workers)
        hands_each = [
            n_hands // workers + (1 if w < n_hands % workers else 0)
            for w in range(workers)
        ]

        was_training = [getattr(n, "training", False) for n in nets]
        for n in nets:
            if hasattr(n, "eval"):
                n.eval()

        started_at = time.perf_counter()
        totals = dict.fromkeys(_STAT_KEYS, 0)
        n_forwards = 0
        batch_total = 0
        pick_rounds = 0
        scored_steps = 0
        buf = self._shared.buffer

        try:
            if starts is not None and len(starts) < n_hands:
                raise ValueError(f"{len(starts)} starts for {n_hands} hands")
            live = set()
            base = episode_base
            for w in range(workers):
                if hands_each[w] == 0:
                    continue
                lo = base - episode_base
                mine = (
                    None if starts is None
                    else [pack_start(s) for s in starts[lo : lo + hands_each[w]]]
                )
                self._rep_qs[w].put((
                    "round", hands_each[w], base,
                    int(rng.integers(0, 2**31 - 1)), float(level_weight),
                    per_worker_pool, mine, 2 if overlap else 1,
                ))
                base += hands_each[w]
                live.add(w)

            ends: dict[int, int] = {}
            halves = 2 if overlap else 1
            # Batches by half. With two halves a worker's next batch for
            # one half can arrive while the other is being collected.
            waiting: list[dict[int, tuple[int, int]]] = [{} for _ in range(halves)]
            half = 0
            with torch.no_grad(), torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=amp,
            ):
                def absorb(msg: tuple) -> None:
                    # A worker's report: a finished round or a batch.
                    if msg[0] == "done":
                        _tag, wid, n_rows, stats = msg
                        ends[wid] = n_rows
                        live.discard(wid)
                        for key in totals:
                            totals[key] += int(stats[key])
                        for w_half in waiting:
                            w_half.pop(wid, None)
                    elif msg[0] == "batch":
                        _tag, wid, h, first, count = msg
                        waiting[h][wid] = (first, count)
                    else:
                        raise RuntimeError(f"unexpected worker message {msg[0]!r}")

                while live:
                    # Every live worker reports each half before it moves
                    # on, so waiting for all of them is safe and makes
                    # the batch as wide as that half. It is lockstep per
                    # half.
                    while len(waiting[half]) < len(live):
                        absorb(self._receive())
                    if not live:
                        break
                    pending = waiting[half]
                    waiting[half] = {}

                    rows = np.concatenate([
                        np.arange(first, first + count, dtype=np.int64)
                        for first, count in
                        (pending[w] for w in sorted(pending))
                    ])
                    if rows.size == 0:
                        # Every worker's half is empty: nothing to act.
                        for w in pending:
                            self._rep_qs[w].put(("go", half))
                        half = (half + 1) % halves
                        continue

                    # Fancy indexing, so each of these materialises a
                    # copy: at 1024 hands a round moves about 2.4 GiB
                    # through here. Timed apart from the forward (as
                    # "gather") because the two have different remedies
                    # -- this one shrinks by slicing each worker's
                    # contiguous run instead of gathering, the forward
                    # only by a wider batch.
                    def gather(idx: np.ndarray) -> tuple[Any, torch.Tensor]:
                        r = rows[idx]
                        obs = obs_batch_from_arrays(
                            buf.history[r], buf.history_len[r],
                            buf.context[r], buf.unseen[r],
                            buf.planes[r], buf.cards[r],
                            buf.candidates[r], buf.candidate_mask[r],
                            device=device,
                        )
                        privileged = torch.from_numpy(
                            buf.privileged[r].astype(np.float32)
                        ).to(device)
                        return obs, privileged

                    hand_idx = buf.episode[rows] - episode_base
                    seats = buf.seat[rows]
                    if seating is None:
                        owner = np.zeros(len(rows), dtype=np.int64)
                    else:
                        owner = seating[hand_idx, seats]
                    if cards_mode:
                        order = sorted(pending)
                        spans_of = {}
                        offset = 0
                        for w in order:
                            spans_of[w] = (offset, pending[w][1])
                            offset += pending[w][1]
                        slots, chosen_logp, values, n_rounds, n_scored = self._pick_round(
                            nets, owner, gather, rows, spans_of, half, absorb,
                            temperature, greedy, rng, spans,
                        )
                        pick_rounds += n_rounds
                        scored_steps += n_scored
                    else:
                        slots, chosen_logp, values = routed_step(
                            nets, owner, gather, temperature, greedy, rng, spans,
                            lengths=buf.history_len[rows], chunks=chunks,
                            critic=critic, forwards=forwards,
                            seat_agents=(
                                None if critic is None
                                else relative_seat_agents(seating[hand_idx], seats)
                            ),
                        )
                    with _span(spans, "writeback"):
                        buf.action[rows] = slots
                        buf.logp[rows] = chosen_logp
                        buf.value[rows] = values
                    n_forwards += 1
                    batch_total += int(rows.size)
                    with _span(spans, "release"):
                        for w in pending:
                            self._rep_qs[w].put(("go", half))
                    half = (half + 1) % halves

            # Each worker's rows go straight to their place in the
            # caller's buffer: one pass over the round's bytes, where
            # compacting the shared block and then copying it out was
            # two, single-threaded -- 7.1 s of a 50 s round at 4096
            # hands. The rows land in the order compaction gave them.
            filled = sorted(
                (self.ranges[w][0], ends.get(w, 0)) for w in range(workers)
            )
            rows_written = sum(count for _start, count in filled)
            out_buffer = buffer if buffer is not None else RolloutBuffer(
                max(rows_written, 1), self.cfg
            )
            if out_buffer.capacity < rows_written:
                raise ValueError(
                    f"the caller's buffer holds {out_buffer.capacity} rows "
                    f"and the round wrote {rows_written}"
                )
            with _span(spans, "copy_out"):
                copy_ranges(buf, out_buffer, filled)
                out_buffer._n = rows_written
        finally:
            for n, was in zip(nets, was_training):
                if was and hasattr(n, "train"):
                    n.train()

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
            "forwards": n_forwards,
            "mean_batch": batch_total / max(n_forwards, 1),
            **({
                "forced_steps": float(totals["forced_steps"]),
                "scored_steps": float(scored_steps),
                "pick_rounds": float(pick_rounds),
            } if cards_mode else {}),
            "seconds": elapsed,
            "workers": float(workers),
            "rows": float(rows_written),
        }

    # --- card-by-card plays (E7) -------------------------------------------

    def _pick_round(self, nets, owner, gather, rows, spans_of, half, absorb,
                    temperature, greedy, rng, spans):
        """Every row's play, one pick at a time, with the workers holding
        the rules. -> ``(steps, logp, value, sub-rounds, scored steps)``.

        Each worker has written, for each of its rows, the mask and
        prefix of the first step that offers a choice and that step's
        index in ``action`` (-1 once the play has submitted). The encoders
        run once; each sub-round scores the live rows' current steps,
        sends each worker its picks, and waits for every worker it sent
        to, which applies them, takes the forced steps that follow and
        reports the next step to choose. Other reports that arrive
        meanwhile (the other half's batches) are kept for later.
        """
        buf = self._shared.buffer
        n = len(rows)
        logp = np.zeros(n, dtype=np.float32)
        value = np.zeros(n, dtype=np.float32)
        groups = []
        with _span(spans, "forward"):
            for agent in np.unique(owner):
                idx = np.nonzero(owner == agent)[0]
                net = nets[int(agent)]
                obs, privileged = gather(idx)
                summary, card_states, v = actor_encode(net, obs, privileged)
                value[idx] = v.detach().float().cpu().numpy()
                groups.append((idx, summary, card_states, _picker(net)))
        rounds = 0
        scored = 0
        with _span(spans, "sample"):
            while True:
                step = buf.action[rows]
                live = step >= 0
                if not live.any():
                    break
                actions = np.full(n, -1, dtype=np.int64)
                for idx, summary, card_states, picker in groups:
                    pos = np.nonzero(live[idx])[0]
                    if len(pos) == 0:
                        continue
                    j = idx[pos]
                    r = rows[j]
                    k = step[j]
                    sel = torch.from_numpy(pos).to(summary.device)
                    a, lp = score_step(
                        picker, summary[sel], card_states[sel],
                        buf.candidates[r, k], buf.candidate_mask[r, k], k,
                        temperature, greedy, rng,
                    )
                    actions[j] = a
                    logp[j] += lp
                    scored += len(j)
                asked = []
                for w, (off, count) in spans_of.items():
                    if live[off : off + count].any():
                        self._rep_qs[w].put(("pick", half, actions[off : off + count].copy()))
                        asked.append(w)
                answered = set()
                while len(answered) < len(asked):
                    msg = self._receive()
                    if msg[0] == "picked":
                        if msg[2] != half:
                            raise RuntimeError("a worker answered picks for the wrong half")
                        answered.add(msg[1])
                    else:
                        absorb(msg)
                rounds += 1
        steps = buf.candidate_mask[rows].any(axis=-1).sum(axis=-1).astype(np.int64)
        return steps, logp, value, rounds, scored

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
    agents: Any = None,
    chunks: int = 1,
    critic: Any = None,
    overlap: bool = False,
    forwards: str = "eager",
) -> tuple[RolloutBuffer, dict[str, float]]:
    """One round through a throwaway pool.

    For tests and one-off measurements. A training loop should hold a
    ``RolloutPool`` instead: this pays the worker start-up -- about 9.5
    seconds for one worker and 12.3 for seven, most of it ``import
    torch`` -- on every call, which is more than the round costs.
    """
    cfg = cfg or getattr(as_nets(net)[0], "cfg", None) or ModelConfig()
    with RolloutPool(
        cfg=cfg, workers=workers, hands_per_round=int(n_hands),
        capacity=capacity,
    ) as worker_pool:
        return worker_pool.collect(
            net, rng, buffer, n_hands=int(n_hands), device=device,
            temperature=temperature, greedy=greedy,
            level_weight=level_weight, pool=pool, amp=amp, spans=spans,
            agents=agents, chunks=chunks, critic=critic, overlap=overlap,
            forwards=forwards,
        )
