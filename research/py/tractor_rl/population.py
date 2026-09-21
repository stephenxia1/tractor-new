"""Seating and opponent history for population training.

The shape of a run: N agents train at once. An *iteration* freezes every
agent's acting weights, seats four of them at each table, lets them play
for a fixed number of rounds, and then publishes each agent's trained
weights as its new acting weights and reshuffles the seating. Within an
iteration every opponent is stationary, which is the property RL
best-response learning assumes and which ordinary simultaneous self-play
does not provide.

Freezing alone is not enough. Four agents best-responding to a frozen
profile is iterated best response, and iterated best response cycles --
rock-paper-scissors is the standard counterexample. What makes fictitious
play converge is that the best response is computed against the *average*
of the opponent's past strategies, and the averaging is what the archive
here supplies: two of the four seats at each table are drawn from
publications past rather than from the current population.

The sampling window sets the step size. Seating only the newest
publication is alpha = 1, the cycling case. Sampling uniformly over every
publication so far makes any one of them appear with probability ~1/t,
which is exactly FP's alpha_t -> 0 schedule. Algorithm-R reservoir
sampling gives that in bounded memory, so the archive can behave as though
it kept every iteration while holding a couple of dozen.

Nothing here imports torch: payloads are opaque, so the seating rules and
the sampling distribution can be tested without a GPU or a working local
torch.
"""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple, Sequence

import numpy as np

# Seats 0 and 2 are one team, 1 and 3 the other (engine `seatToTeam`), so
# whether two agents are partners or opponents is decided by which seat
# indices they take.
PARTNER_PAIRS = frozenset({(0, 2), (1, 3)})
SEAT_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


class ModelRef(NamedTuple):
    """Which set of weights sits in a seat.

    `iteration < 0` means the agent's current acting weights; anything else
    names a specific past publication held in the archive.
    """

    agent: int
    iteration: int = -1

    @property
    def is_live(self) -> bool:
        return self.iteration < 0

    def __str__(self) -> str:
        return f"a{self.agent}" if self.is_live else f"a{self.agent}@{self.iteration}"


class Archive:
    """A uniform sample of each agent's publication history, bounded.

    `capacity` slots per agent. Publications before `burn_in` are refused:
    a near-random agent is a waste of a seat, and FP's guarantee says
    nothing about how much compute the early average deserves.
    """

    def __init__(self, n_agents: int, capacity: int = 24, burn_in: int = 3) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.n_agents = n_agents
        self.capacity = capacity
        self.burn_in = burn_in
        # Per agent: the retained sample, and how many publications have been
        # offered to it. The counter -- not the sample size -- drives the
        # reservoir's acceptance probability, which is what keeps the sample
        # uniform over the whole history rather than over the last `capacity`.
        self._slots: list[list[tuple[int, Any]]] = [[] for _ in range(n_agents)]
        self._offered: list[int] = [0] * n_agents

    def publish(self, agent: int, iteration: int, payload: Any,
                rng: np.random.Generator) -> bool:
        """Offer a publication. Returns whether it was retained."""
        if iteration < self.burn_in:
            return False
        self._offered[agent] += 1
        n = self._offered[agent]
        slots = self._slots[agent]
        if len(slots) < self.capacity:
            slots.append((iteration, payload))
            return True
        # Algorithm R: the n-th item survives with probability capacity/n.
        j = int(rng.integers(0, n))
        if j < self.capacity:
            slots[j] = (iteration, payload)
            return True
        return False

    def restore(self, entries: list[tuple[int, int, Any]],
                offered: list[int]) -> int:
        """Rebuild from a checkpoint index. Returns how many entries landed.

        The payloads a run stores are paths, and a resume onto a fresh box
        will find some of them missing; the caller filters those out, so this
        takes whatever survived rather than assuming the index is complete.
        """
        self._slots = [[] for _ in range(self.n_agents)]
        for agent, iteration, payload in entries:
            if 0 <= agent < self.n_agents and len(self._slots[agent]) < self.capacity:
                self._slots[agent].append((iteration, payload))
        saved = [int(n) for n in offered[:self.n_agents]]
        while len(saved) < self.n_agents:
            saved.append(0)
        # The counter has to describe the same history the sample does. Kept
        # verbatim against a sample that lost entries, the empty slots refill
        # with one contiguous block of post-resume iterations and are then
        # evicted at only capacity/n -- so the archive stops averaging over
        # the history and the population is back to the iterated best
        # response it exists to prevent, with nothing in the logs to show it.
        # Scaling to what survived is exact when nothing was dropped, since
        # a full reservoir means len(slots) == capacity and a partial one was
        # never offered more than it holds.
        self._offered = [
            max(len(self._slots[a]),
                int(saved[a] * len(self._slots[a]) / self.capacity))
            for a in range(self.n_agents)
        ]
        return self.size()

    def discard(self, ref: ModelRef) -> bool:
        """Forget one entry whose payload turned out to be unusable.

        The offered counter is deliberately left alone: the publication was
        still offered, and lowering it would raise every later acceptance
        probability and bias the sample toward what comes after the loss.
        """
        slots = self._slots[ref.agent]
        for i, (iteration, _) in enumerate(slots):
            if iteration == ref.iteration:
                del slots[i]
                return True
        return False

    def refs(self, agent: int) -> list[ModelRef]:
        return [ModelRef(agent, it) for it, _ in self._slots[agent]]

    def all_refs(self) -> list[ModelRef]:
        return [r for a in range(self.n_agents) for r in self.refs(a)]

    def payload(self, ref: ModelRef) -> Any:
        for it, payload in self._slots[ref.agent]:
            if it == ref.iteration:
                return payload
        raise KeyError(f"{ref} is not in the archive")

    def size(self) -> int:
        return sum(len(s) for s in self._slots)

    def offered(self, agent: int) -> int:
        return self._offered[agent]


def choose_archive_refs(rng: np.random.Generator, archive: Archive,
                        budget: int) -> list[ModelRef]:
    """Pick up to `budget` distinct archived opponents for one iteration.

    Distinct weights cost a forward pass each -- the acting loop groups
    decisions by model -- so the budget is a compute knob, not a statistical
    one. Agents are drawn first and then an iteration within the agent, so
    the mixture stays spread across the population instead of loading up on
    whichever agent happens to hold the most slots.
    """
    have = [a for a in range(archive.n_agents) if archive.refs(a)]
    if not have or budget <= 0:
        return []
    picked: list[ModelRef] = []
    seen: set[ModelRef] = set()
    # Round-robin over a shuffled agent order, so a budget below the agent
    # count still covers distinct agents rather than one agent repeatedly.
    order = list(rng.permutation(have))
    i = 0
    while len(picked) < budget and i < budget * 8:
        agent = int(order[i % len(order)])
        options = [r for r in archive.refs(agent) if r not in seen]
        if options:
            ref = options[int(rng.integers(0, len(options)))]
            picked.append(ref)
            seen.add(ref)
        i += 1
        if i % len(order) == 0 and all(
                all(r in seen for r in archive.refs(a)) for a in have):
            break  # archive exhausted
    return picked


def _live_assignment(rng: np.random.Generator, n_agents: int, n_tables: int,
                     live_per_table: int) -> list[list[int]]:
    """Which live agents sit at each table, spread as evenly as possible.

    Every agent gets floor or ceil of `n_tables * live_per_table / n_agents`
    seats; an agent is never seated twice at the same table, since two copies
    of one agent at a table is self-play wearing a population's clothes.
    """
    total = n_tables * live_per_table
    if live_per_table > n_agents:
        raise ValueError("cannot seat more distinct live agents than exist")
    pool: list[int] = []
    while len(pool) < total:
        pool.extend(int(x) for x in rng.permutation(n_agents))
    pool = pool[:total]
    rng.shuffle(pool)

    tables: list[list[int]] = [[] for _ in range(n_tables)]
    leftovers: list[int] = []
    for agent in pool:
        # First table with room that does not already hold this agent.
        for t in range(n_tables):
            if len(tables[t]) < live_per_table and agent not in tables[t]:
                tables[t].append(agent)
                break
        else:
            leftovers.append(agent)

    # A leftover only happens when every table with room already holds this
    # agent. Swap it against a table that has room, moving that table's
    # occupant to a table that can take it.
    for agent in leftovers:
        placed = False
        for t in range(n_tables):
            if len(tables[t]) >= live_per_table:
                continue
            for u in range(n_tables):
                if u == t:
                    continue
                for k, other in enumerate(tables[u]):
                    if other not in tables[t] and agent not in tables[u]:
                        tables[u][k] = agent
                        tables[t].append(other)
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:  # pragma: no cover - needs n_agents < live_per_table
            raise RuntimeError("could not seat live agents without a duplicate")
    return tables


def plan_seating(rng: np.random.Generator, n_agents: int, n_tables: int,
                 live_per_table: int,
                 archive_refs: Sequence[ModelRef] = ()) -> list[list[ModelRef]]:
    """Four ModelRefs per table.

    `live_per_table` seats hold currently-training agents and the rest hold
    archived opponents. Which *seat indices* the live agents take is drawn
    uniformly over the six pairs, so two live agents are partners a third of
    the time and opponents the rest: a live agent that only ever partnered
    other live agents would be free to build conventions with the current
    population, which is the failure the archive exists to prevent.

    With an empty archive -- the first iteration, before anything has been
    published -- all four seats hold live agents. That is ordinary
    population self-play, which is the right behaviour for a population
    with no history to average over yet.
    """
    if not 1 <= live_per_table <= 4:
        raise ValueError("live_per_table must be between 1 and 4")
    if not archive_refs:
        # Cold start: no history to average over, so every seat is a live
        # agent. With fewer than four agents a table cannot hold four
        # distinct ones, so seats repeat -- two agents give a0/a1/a0/a1,
        # which is still both of them playing each other. `_live_assignment`
        # forbids repeats by design, so this path does its own dealing.
        if n_agents >= 4:
            # Enough agents for four distinct per table, so use the balanced
            # assignment: it keeps every agent's seat count within one, and
            # an agent with three times another's seats this iteration would
            # get three times the training data from it.
            return [[ModelRef(a) for a in row]
                    for row in _live_assignment(rng, n_agents, n_tables, 4)]
        # Fewer agents than seats: deal from one pool of cycled permutations
        # so the counts stay balanced across the whole iteration, then cut it
        # into tables. Dealing each table its own pool balances within a
        # table and not across them -- three agents over sixteen seats came
        # out 3/6/7 that way, against 5/5/6 here.
        total = n_tables * 4
        pool: list[int] = []
        while len(pool) < total:
            pool.extend(int(x) for x in rng.permutation(n_agents))
        return [[ModelRef(a) for a in pool[t * 4:(t + 1) * 4]]
                for t in range(n_tables)]
    live = _live_assignment(rng, n_agents, n_tables, live_per_table)

    seating: list[list[ModelRef]] = []
    for t in range(n_tables):
        row: list[ModelRef | None] = [None, None, None, None]
        if live_per_table == 4:
            order = rng.permutation(4)
            for k, seat in enumerate(order):
                row[int(seat)] = ModelRef(live[t][k])
        else:
            if live_per_table == 2:
                seats = SEAT_PAIRS[int(rng.integers(0, len(SEAT_PAIRS)))]
            else:
                seats = tuple(sorted(
                    int(x) for x in rng.permutation(4)[:live_per_table]))
            for k, seat in enumerate(seats):
                row[seat] = ModelRef(live[t][k])
            # Without replacement while the budget allows it: two identical
            # frozen policies at one table is a needlessly correlated
            # opponent, and avoiding it costs nothing.
            empty = [seat for seat in range(4) if row[seat] is None]
            order = list(rng.permutation(len(archive_refs)))
            for k, seat in enumerate(empty):
                row[seat] = archive_refs[int(order[k % len(order)])]
        seating.append([r for r in row if r is not None])
        assert len(seating[-1]) == 4
    return seating


def distinct_models(seating: Iterable[Sequence[ModelRef]]) -> list[ModelRef]:
    """Every distinct set of weights on the board, in a stable order.

    One forward pass per entry, so this is the acting cost of an iteration.
    """
    seen = {ref for table in seating for ref in table}
    return sorted(seen)


def slot_index(seating: Sequence[Sequence[ModelRef]]
               ) -> tuple[list[ModelRef], np.ndarray]:
    """Distinct refs, and a [table, seat] array indexing into them.

    This is the routing table the acting loop uses: decisions arrive as a
    flat batch keyed by (table, seat), and one forward pass is run per slot.
    Building it here rather than in the trainer keeps it testable without
    torch -- misrouting a seat would train an agent on another agent's
    decisions, which no loss curve would reveal.
    """
    refs = distinct_models(seating)
    index = {ref: i for i, ref in enumerate(refs)}
    table = np.array([[index[ref] for ref in row] for row in seating],
                     dtype=np.int64)
    return refs, table


def seating_table(seating: Sequence[Sequence[ModelRef]],
                  n_agents: int) -> tuple[np.ndarray, np.ndarray]:
    """[n_tables, 4] of agent ids, and a matching live mask.

    The trainer needs this as arrays: decisions arrive as a flat batch keyed
    by (worker, seat), and routing them means indexing, not iterating.
    """
    agents = np.array([[r.agent for r in row] for row in seating], dtype=np.int16)
    live = np.array([[r.is_live for r in row] for row in seating], dtype=bool)
    if agents.size and (agents.max() >= n_agents or agents.min() < 0):
        raise ValueError("seating references an agent outside the population")
    return agents, live


def live_seat_counts(seating: Sequence[Sequence[ModelRef]], n_agents: int) -> np.ndarray:
    """How many live seats each agent holds -- its share of the training data."""
    counts = np.zeros(n_agents, dtype=np.int64)
    for row in seating:
        for ref in row:
            if ref.is_live:
                counts[ref.agent] += 1
    return counts
