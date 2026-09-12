"""What fraction of a learner step is the causal history encoder?

This decides how much of the stacked multi-agent net has to exist. The
multi-agent benchmark covered the MLP path only and said so; if attention
is a few percent of the step it can stay unstacked, run once per agent in a
loop, and the batched GEMM only has to handle the trunk and heads. If it is
a third of the step, a loop over N agents would eat the win and the
attention has to be batched over the agent axis as well -- which is a far
larger piece of work, since `nn.MultiheadAttention` has no per-agent weight
form and would have to be hand-rolled.

    PYTHONPATH=py python py/bench_history_share.py --batch 512 --reps 20
"""

from __future__ import annotations

import argparse
import time

import torch

from tractor_rl.features2 import ACTION_DIM, HISTORY_STEPS, HISTORY_TOKEN_DIM, STATE_DIM
from tractor_rl.model2 import CausalHistoryEncoder, NetConfig, TractorNet
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
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    B = args.batch

    states = torch.randn(B, STATE_DIM, device=dev)
    # Candidates outnumber decisions; the learner's batch is one candidate
    # per decision, so owner is the identity.
    actions = torch.randn(B, ACTION_DIM, device=dev)
    owner = torch.arange(B, device=dev)
    roles = torch.randint(0, L.NUM_ROLES, (B,), device=dev)
    tokens = torch.randn(B, HISTORY_STEPS, HISTORY_TOKEN_DIM, device=dev)
    mask = (torch.rand(B, HISTORY_STEPS, device=dev) > 0.3).float()

    def step(net, tok, msk):
        def run():
            out = net(states, actions, owner, roles, tok, msk, want="all")
            loss = (out["q"].square().mean() + out["v"].square().mean()
                    + out["belief"].square().mean() + out["q_level"].square().mean())
            net.zero_grad(set_to_none=True)
            loss.backward()
        return run

    with_hist = TractorNet(NetConfig(history="attn")).to(dev)
    without = TractorNet(NetConfig(history="none")).to(dev)

    t_with = timed(step(with_hist, tokens, mask), dev, args.reps)
    t_without = timed(step(without, None, None), dev, args.reps)

    # The encoder on its own, forward and backward, so the difference above
    # can be checked against a direct measurement rather than trusted.
    enc = CausalHistoryEncoder(NetConfig()).to(dev)

    def enc_step():
        out = enc(tokens, mask)
        enc.zero_grad(set_to_none=True)
        out.square().mean().backward()

    t_enc = timed(enc_step, dev, args.reps)

    print(f"device {dev}  batch {B}  threads {args.threads}  reps {args.reps}\n")
    print(f"  full net (history=attn)   {t_with:8.2f} ms")
    print(f"  trunk only (history=none) {t_without:8.2f} ms")
    print(f"  difference                {t_with - t_without:8.2f} ms"
          f"  ({100 * (t_with - t_without) / t_with:.1f}% of the step)")
    print(f"  encoder measured alone    {t_enc:8.2f} ms")
    print()
    share = (t_with - t_without) / t_with
    p_with = with_hist.num_parameters()
    p_without = without.num_parameters()
    print(f"  params: full {p_with/1e6:.2f}M, trunk-only {p_without/1e6:.2f}M, "
          f"history {(p_with - p_without)/1e6:.2f}M "
          f"({100*(p_with - p_without)/p_with:.1f}%)")
    print()
    if share < 0.15:
        print(f"  -> attention is {share:.0%} of the step. Leave it unstacked: even a")
        print(f"     loop over N agents costs at most N x {share:.0%}, and the stacked")
        print(f"     net only needs the trunk, action encoder and heads.")
    else:
        print(f"  -> attention is {share:.0%} of the step. A per-agent loop would")
        print(f"     dominate at N=8, so the stacked net has to batch attention too.")


if __name__ == "__main__":
    main()
