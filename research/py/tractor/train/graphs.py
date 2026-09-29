"""A training step captured as a CUDA graph, one per history width.

With the round on the device and nothing read back (``DeviceRows``,
``PPOLearner._update_on_device``), what is left of a minibatch on a fast
GPU is the host issuing its ~1000 kernels one by one. Measured on an H100
(v6, one agent's rows; ``modal_perf.py::graph_bench``): the actor's step
16.4 -> 9.4 ms, the shared critic's 8.5 -> 3.3 ms, the actor's weights
bit-identical to the uncaptured step's.

What is captured is the whole step -- the minibatch's gather from the
device rows, the forward and loss, the backward, the clip, the fused
AdamW's step (made ``capturable``) with its on-device refusal of a
non-finite step, the metric sums -- on one static index tensor, which is
all a replay needs refreshed. A graph replays the kernels it recorded at
the addresses it recorded, so it is dropped whenever something it read by
address may have moved: the device rows, the optimiser's state, the
learning rate (``key``).

Every graph of a learner shares one memory pool. That is safe in any
replay order because nothing allocated in the pool outlives a replay:
gradients are recomputed each step, and the parameters, the optimiser
state and the metric sums were all allocated outside it.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

__all__ = ["StepGraphs"]


class StepGraphs:
    """``body(pos, width)``, one optimiser step on the rows ``pos``,
    captured after ``warm`` uncaptured steps of each width.

    ``body`` must read its rows through ``pos`` -- this object's static
    index tensor -- and must not read anything back to the host."""

    def __init__(self, opt: torch.optim.Optimizer, size: int, device: Any,
                 warm: int = 2) -> None:
        self.opt = opt
        self.size = int(size)
        self.warm = int(warm)
        self.pos = torch.zeros(self.size, dtype=torch.int64, device=device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.seen: dict[int, int] = {}
        self.key: Any = None
        self.pool: Any = None
        self.side: torch.cuda.Stream | None = None
        self.captures = 0
        self.replays = 0

    def reset(self) -> None:
        self.graphs.clear()
        self.seen.clear()
        self.pool = None

    def key_of(self, rows: Any) -> tuple:
        """What the graphs read by address: the rows' tensors, the
        optimiser's state (``load_state_dict`` replaces it, and a first
        step creates it), and the learning rates they baked in."""
        state = self.opt.state
        params = [p for g in self.opt.param_groups for p in g["params"]]
        ends = (params[0], params[-1]) if params else ()
        tensors = tuple(
            id(v) for p in ends for v in state.get(p, {}).values()
            if isinstance(v, torch.Tensor)
        )
        lrs = tuple(float(g["lr"]) for g in self.opt.param_groups)
        return (id(rows), getattr(rows, "generation", 0), id(state), tensors, lrs)

    def step(self, body: Callable[[torch.Tensor, int], None], pos: torch.Tensor,
             width: int, rows: Any) -> None:
        key = self.key_of(rows)
        if key != self.key:
            self.reset()
            self.key = key
        self.pos.copy_(pos)
        graph = self.graphs.get(width)
        if graph is not None:
            graph.replay()
            self.replays += 1
            return
        seen = self.seen.get(width, 0)
        self.seen[width] = seen + 1
        if self.side is None:
            self.side = torch.cuda.Stream()
        if seen < self.warm:
            # Uncaptured, on a side stream as capture wants its warm-up.
            self.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side):
                self.opt.zero_grad(set_to_none=True)
                body(self.pos, width)
            torch.cuda.current_stream().wait_stream(self.side)
            return
        for group in self.opt.param_groups:
            group["capturable"] = True
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        # The gradients are allocated in the capture, from the pool.
        self.opt.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            body(self.pos, width)
        self.graphs[width] = graph
        self.captures += 1
        # Capturing records without running: this step is the first replay.
        graph.replay()
        self.replays += 1
