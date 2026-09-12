"""What does the real stacked net cost against N separate ones?

`bench_multiagent.py` answered this for a hand-written stand-in covering the
MLP path only, and measured 5.69x for eight agents on a T4. This runs the
same question against the network that will actually train: full trunk,
causal history attention, action encoder, per-role heads, value and belief
heads -- forward and backward, exactly as the learner steps it.

Expect a different answer on CPU and GPU, and expect CPU to look bad. The
stacked net computes every role head on every row and gathers, because bmm
needs equal batches per group; `TractorNet` on CPU masks instead and runs
each head only on its own rows, measured 3.2x faster there. The batched
GEMM exists for the GPU, where launch overhead dominates and the masked
form costs three host synchronisations per forward.

    PYTHONPATH=py python py/bench_stacked.py --device cuda --agents 1 2 4 8
"""

from __future__ import annotations

import argparse
import time

import torch

from tractor_rl.features2 import ACTION_DIM, HISTORY_STEPS, HISTORY_TOKEN_DIM, STATE_DIM
from tractor_rl.model2 import NetConfig, TractorNet
from tractor_rl.stacked import StackedTractorNet
from tractor_rl import layout as L


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, dev: torch.device, reps: int) -> float:
    for _ in range(3):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) / reps * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=512, help="per agent")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--agents", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--history", default="attn", choices=["attn", "none"])
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    cfg = NetConfig(history=args.history)
    B = args.batch
    if dev.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch {B} per agent | history {args.history} | "
          f"threads {args.threads} | reps {args.reps}\n")
    print(f"{'N':>3}{'separate ms':>13}{'stacked ms':>12}{'speedup':>9}"
          f"{'cost vs N=1':>13}{'per agent':>11}")

    base = None
    for n in args.agents:
        nets = [TractorNet(cfg).to(dev) for _ in range(n)]
        stacked = StackedTractorNet(n, cfg).to(dev)
        stacked.load_from(nets)

        states = torch.randn(n, B, STATE_DIM, device=dev)
        actions = torch.randn(n, B, ACTION_DIM, device=dev)
        roles = torch.randint(0, L.NUM_ROLES, (n, B), device=dev)
        tokens = mask = None
        if args.history == "attn":
            tokens = torch.randn(n, B, HISTORY_STEPS, HISTORY_TOKEN_DIM, device=dev)
            mask = (torch.rand(n, B, HISTORY_STEPS, device=dev) > 0.3).float()
        owner = torch.arange(B, device=dev)

        def separate():
            for i, net in enumerate(nets):
                out = net(states[i], actions[i], owner, roles[i],
                          None if tokens is None else tokens[i],
                          None if mask is None else mask[i], want="all")
                loss = (out["q"].square().mean() + out["v"].square().mean()
                        + out["belief"].square().mean()
                        + out["q_level"].square().mean())
                net.zero_grad(set_to_none=True)
                loss.backward()

        def batched():
            out = stacked(states, actions, roles, tokens, mask, want="all")
            loss = (out["q"].square().mean() + out["v"].square().mean()
                    + out["belief"].square().mean() + out["q_level"].square().mean())
            stacked.zero_grad(set_to_none=True)
            loss.backward()

        t_sep = timed(separate, dev, args.reps)
        t_stk = timed(batched, dev, args.reps)
        if base is None:
            base = t_stk
        print(f"{n:3d}{t_sep:13.2f}{t_stk:12.2f}{t_sep / t_stk:8.2f}x"
              f"{t_stk / base:12.2f}x{t_stk / n:10.2f} ms")

        del nets, stacked, states, actions, tokens, mask
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    print("\n'cost vs N=1' is the number that decides the design: N agents")
    print("should cost far less than N times one. 'per agent' is the wall")
    print("clock each agent actually pays for a gradient step.")


if __name__ == "__main__":
    main()
