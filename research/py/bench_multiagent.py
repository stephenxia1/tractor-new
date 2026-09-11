"""Does training N agents together cost N steps, or fewer?

The population design lives or dies on this. At N=8 each agent needs ~3.5
gradient steps/s and one g4dn.2xlarge supplies ~27 in total, so if eight
agents cost 8x one agent we are exactly at capacity with no margin. If a
batched GEMM makes them cost ~4x, there is room.

A single [512,512] layer was measured 1.85x better batched. That is not the
same claim as a whole forward *and backward* over the real stack, which is
what this measures: N separate models stepped in a loop, against one
weight-stacked model stepped with bmm.

The stacked model covers the MLP path -- stem, residual blocks, fuse, action
encoder, head. The history attention is left out and timed separately, so
the result is not quietly overstated by omitting the awkward part.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from tractor_rl.features2 import ACTION_DIM, STATE_DIM

W, A, HM = 512, 128, 256


class Single(nn.Module):
    """The MLP path of one agent, shapes matching TractorNet."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(STATE_DIM, W)
        self.b1 = nn.ModuleList([nn.Linear(W, W), nn.Linear(W, W)])
        self.b2 = nn.ModuleList([nn.Linear(W, W), nn.Linear(W, W)])
        self.fuse = nn.Linear(W + 96, W)
        self.act = nn.Sequential(nn.Linear(ACTION_DIM, A), nn.ReLU(), nn.Linear(A, A), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(W + A, HM), nn.ReLU(),
                                  nn.Linear(HM, HM // 2), nn.ReLU(), nn.Linear(HM // 2, 1))

    def forward(self, s, a, h):
        x = self.stem(s)
        x = x + self.b1[1](torch.relu(self.b1[0](x)))
        x = x + self.b2[1](torch.relu(self.b2[0](x)))
        x = torch.relu(self.fuse(torch.cat([x, h], dim=1)))
        return self.head(torch.cat([x, self.act(a)], dim=1)).squeeze(-1)


class Stacked(nn.Module):
    """N agents as one module: every Linear becomes a bmm over the agent axis."""

    def __init__(self, n: int) -> None:
        super().__init__()
        def p(i, o):
            return nn.Parameter(torch.randn(n, i, o) * (1.0 / i ** 0.5))
        def b(o):
            return nn.Parameter(torch.zeros(n, 1, o))
        self.n = n
        self.ws = nn.ParameterDict({
            "stem": p(STATE_DIM, W), "stem_b": b(W),
            "b1a": p(W, W), "b1a_b": b(W), "b1b": p(W, W), "b1b_b": b(W),
            "b2a": p(W, W), "b2a_b": b(W), "b2b": p(W, W), "b2b_b": b(W),
            "fuse": p(W + 96, W), "fuse_b": b(W),
            "act1": p(ACTION_DIM, A), "act1_b": b(A), "act2": p(A, A), "act2_b": b(A),
            "h1": p(W + A, HM), "h1_b": b(HM), "h2": p(HM, HM // 2), "h2_b": b(HM // 2),
            "h3": p(HM // 2, 1), "h3_b": b(1),
        })

    def forward(self, s, a, h):
        w = self.ws
        x = torch.baddbmm(w["stem_b"], s, w["stem"])
        x = x + torch.baddbmm(w["b1b_b"], torch.relu(torch.baddbmm(w["b1a_b"], x, w["b1a"])), w["b1b"])
        x = x + torch.baddbmm(w["b2b_b"], torch.relu(torch.baddbmm(w["b2a_b"], x, w["b2a"])), w["b2b"])
        x = torch.relu(torch.baddbmm(w["fuse_b"], torch.cat([x, h], dim=2), w["fuse"]))
        y = torch.relu(torch.baddbmm(w["act1_b"], a, w["act1"]))
        y = torch.relu(torch.baddbmm(w["act2_b"], y, w["act2"]))
        z = torch.cat([x, y], dim=2)
        z = torch.relu(torch.baddbmm(w["h1_b"], z, w["h1"]))
        z = torch.relu(torch.baddbmm(w["h2_b"], z, w["h2"]))
        return torch.baddbmm(w["h3_b"], z, w["h3"]).squeeze(-1)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--agents", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    B = args.batch
    if dev.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch {B} per agent\n")
    print(f"{'N':>4}{'separate ms':>13}{'stacked ms':>12}{'speedup':>9}{'cost vs N=1':>13}")

    base = None
    for n in args.agents:
        singles = [Single().to(dev) for _ in range(n)]
        s = [torch.randn(B, STATE_DIM, device=dev) for _ in range(n)]
        a = [torch.randn(B, ACTION_DIM, device=dev) for _ in range(n)]
        h = [torch.randn(B, 96, device=dev) for _ in range(n)]

        def sep():
            for i, m in enumerate(singles):
                out = m(s[i], a[i], h[i])
                loss = out.square().mean()
                m.zero_grad(set_to_none=True)
                loss.backward()

        stacked = Stacked(n).to(dev)
        S = torch.randn(n, B, STATE_DIM, device=dev)
        Aa = torch.randn(n, B, ACTION_DIM, device=dev)
        H = torch.randn(n, B, 96, device=dev)

        def stk():
            out = stacked(S, Aa, H)
            loss = out.square().mean()
            stacked.zero_grad(set_to_none=True)
            loss.backward()

        times = []
        for fn in (sep, stk):
            fn(); sync(dev)
            t0 = time.perf_counter()
            for _ in range(args.reps):
                fn()
            sync(dev)
            times.append((time.perf_counter() - t0) / args.reps)
        if base is None:
            base = times[0]
        print(f"{n:>4}{times[0]*1000:>13.2f}{times[1]*1000:>12.2f}"
              f"{times[0]/times[1]:>9.2f}x{times[1]/base:>12.2f}x")
    print("\n  'cost vs N=1' is what matters: N agents should cost far less than Nx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
