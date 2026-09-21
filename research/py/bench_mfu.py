"""Achieved FLOP/s against this machine's own GEMM peak, and whether
stacking several agents into one batched GEMM recovers utilisation.

Different agents have different weights, so they cannot share a matmul the
way samples in one batch do. They *can* share a batched GEMM: stack N
weight matrices into [N, out, in] and N batches into [N, B, in] and let one
bmm do all of it. That is the question this measures.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from tractor_rl.features2 import ACTION_DIM, HISTORY_STEPS, HISTORY_TOKEN_DIM, STATE_DIM
from tractor_rl.model2 import NetConfig, TractorNet

W, A, HD, HM, HL, HF = 512, 128, 96, 256, 2, 192


def policy_macs(n_cands_per_state: float) -> float:
    """MACs for one state + its candidates, from the layer shapes."""
    stem = STATE_DIM * W
    blocks = 2 * (W * W * 2)
    hist = HISTORY_STEPS * HISTORY_TOKEN_DIM * HD
    per_layer = (HISTORY_STEPS * HD * 3 * HD + HISTORY_STEPS * HISTORY_STEPS * HD
                 + HISTORY_STEPS * HD * HD + HISTORY_STEPS * HD * HF * 2)
    hist += HL * per_layer
    fuse = (W + HD) * W
    state = stem + blocks + hist + fuse
    per_cand = (ACTION_DIM * A + A * A) + (W + A) * HM + HM * (HM // 2) + (HM // 2)
    return state + n_cands_per_state * per_cand


def gemm_peak(dev=None, reps: int = 12) -> float:
    """This machine's achievable fp32 GEMM rate -- the honest denominator."""
    n = 2048
    a, b = torch.randn(n, n, device=dev), torch.randn(n, n, device=dev)
    a @ b
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    return 2 * n ** 3 / dt / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--cands", type=float, default=8.0,
                    help="mean legal moves per decision, the actor's candidate "
                         "rows; the measured mean is about 8, and the engine "
                         "caps enumeration at 48, so pass --cands 48 for the "
                         "worst case rather than a typical round")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)

    peak = gemm_peak(dev)
    print(f"machine GEMM peak (fp32, {args.threads} threads): {peak:,.0f} GFLOP/s\n")

    net = TractorNet(NetConfig()).to(dev)
    B = args.batch
    sf = torch.randn(B, STATE_DIM, device=dev)
    af = torch.randn(B, ACTION_DIM, device=dev)
    owner = torch.arange(B, device=dev)
    roles = torch.randint(0, 3, (B,), device=dev)
    tok = torch.randn(B, HISTORY_STEPS, HISTORY_TOKEN_DIM, device=dev)
    msk = (torch.rand(B, HISTORY_STEPS, device=dev) > 0.4).float()

    def train_step():
        out = net(sf, af, owner, roles, tok, msk, want="all")
        loss = out["q"].square().mean() + out["v"].square().mean()
        net.zero_grad(set_to_none=True)
        loss.backward()

    train_step()
    if dev.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(6):
        train_step()
    if dev.type == "cuda": torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 6
    macs = policy_macs(1.0) * B
    gflops = 3 * 2 * macs / dt / 1e9        # fwd + bwd ~ 3x forward
    print(f"learner step  batch {B}: {dt*1000:7.1f} ms   {gflops:6.1f} GFLOP/s"
          f"   MFU {100*gflops/peak:4.1f}%")

    with torch.inference_mode():
        # The actor scores every legal move of every state, so candidate
        # rows outnumber states by the mean legal-move count -- which is the
        # whole reason for the owner gather below. This used to read
        # `26 * B / 26`, i.e. exactly B, so the row re-measured the learner's
        # own shape and `policy_macs(C / B)` collapsed to `policy_macs(1.0)`.
        # The trunk is 4.36M of the 4.59M MACs, so that understated the work
        # by 1.35x at the measured mean of about 8, not 26x. The stray 26
        # outlived the bug as the --cands default, where it was neither the
        # measured mean nor the engine's 48-candidate cap -- the default is
        # the mean now, so this row describes a typical round unless asked
        # for something else.
        # On --device cuda policy_macs undercounts the per-candidate term by
        # a further ~3x on top of this, because model2 runs all three role
        # q-heads over every candidate row there rather than masking, to keep
        # the output shape static and the queue unstalled.
        C = int(args.cands * B)
        af_act = torch.randn(C, ACTION_DIM, device=dev)
        own_act = torch.from_numpy(np.sort(np.random.randint(0, B, C))).to(dev)
        net(sf, af_act, own_act, roles, tok, msk, want="q")
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(6):
            net(sf, af_act, own_act, roles, tok, msk, want="q")
        if dev.type == "cuda": torch.cuda.synchronize()
        dta = (time.perf_counter() - t0) / 6
    macs_a = policy_macs(C / B) * B
    ga = 2 * macs_a / dta / 1e9
    print(f"actor forward batch {B}: {dta*1000:7.1f} ms   {ga:6.1f} GFLOP/s"
          f"   MFU {100*ga/peak:4.1f}%\n")

    # N agents: N separate GEMMs vs one batched GEMM over the agent axis.
    N, hid = args.agents, W
    xs = torch.randn(N, B, hid, device=dev)
    ws = torch.randn(N, hid, hid, device=dev)
    def looped():
        return torch.stack([xs[i] @ ws[i] for i in range(N)])
    def stacked():
        return torch.bmm(xs, ws)
    for label, fn in [(f"{N} agents, separate matmuls", looped),
                      (f"{N} agents, one bmm", stacked)]:
        fn()
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            fn()
        if dev.type == "cuda": torch.cuda.synchronize()
        d = (time.perf_counter() - t0) / 20
        g = 2 * N * B * hid * hid / d / 1e9
        print(f"  {label:<30} {d*1000:6.2f} ms   {g:6.1f} GFLOP/s   MFU {100*g/peak:4.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
