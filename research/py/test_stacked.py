"""Does the stacked net compute exactly what N separate nets compute?

This is the only thing standing between a batched GEMM and a training
result nobody can interpret. A misrouted agent axis, a transposed weight,
a head split in the wrong order -- each produces a network that trains
perfectly happily and learns the wrong thing, and no loss curve would show
it. So the checks here are on outputs *and* gradients, per agent, plus the
round trip back out to a plain `TractorNet` that evaluation depends on.

    PYTHONPATH=py python py/test_stacked.py
"""

from __future__ import annotations

import sys

import torch

from tractor_rl.features2 import ACTION_DIM, HISTORY_STEPS, HISTORY_TOKEN_DIM, STATE_DIM
from tractor_rl.model2 import NetConfig, TractorNet
from tractor_rl.stacked import StackedTractorNet
from tractor_rl import layout as L

FAILS: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def close(a: torch.Tensor, b: torch.Tensor, tol: float, name: str) -> None:
    diff = (a - b).abs()
    scale = b.abs().max().clamp(min=1e-6)
    check(diff.max().item() <= tol,
          f"{name}: max |diff| {diff.max().item():.3e} "
          f"(tol {tol:.0e}, values up to {scale.item():.3f})")


def build(n: int, cfg: NetConfig, dtype: torch.dtype):
    nets = []
    for i in range(n):
        torch.manual_seed(1234 + 97 * i)
        nets.append(TractorNet(cfg).to(dtype))
    stacked = StackedTractorNet(n, cfg).to(dtype)
    stacked.load_from(nets)
    return nets, stacked


def inputs(n: int, b: int, cfg: NetConfig, dtype: torch.dtype, empty_rows: bool):
    torch.manual_seed(7)
    states = torch.randn(n, b, STATE_DIM, dtype=dtype)
    actions = torch.randn(n, b, ACTION_DIM, dtype=dtype)
    roles = torch.randint(0, L.NUM_ROLES, (n, b))
    tokens = mask = None
    if cfg.history == "attn":
        tokens = torch.randn(n, b, HISTORY_STEPS, HISTORY_TOKEN_DIM, dtype=dtype)
        mask = (torch.rand(n, b, HISTORY_STEPS) > 0.3).to(dtype)
        # Guarantee at least one row with no history at all: that path is
        # special-cased in both encoders and is exactly where a rewrite
        # goes wrong quietly.
        if empty_rows:
            mask[:, 0] = 0.0
            mask[0, 1] = 0.0
    return states, actions, roles, tokens, mask


def test_forward(dtype: torch.dtype, tol: float) -> None:
    name = str(dtype).replace("torch.", "")
    print(f"forward parity ({name})")
    n, b = 4, 12
    cfg = NetConfig()
    nets, stacked = build(n, cfg, dtype)
    states, actions, roles, tokens, mask = inputs(n, b, cfg, dtype, empty_rows=True)

    got = stacked(states, actions, roles, tokens, mask, want="all")
    owner = torch.arange(b)
    for i, net in enumerate(nets):
        want = net(states[i], actions[i], owner, roles[i], tokens[i], mask[i], want="all")
        for key in ("q", "q_level", "v", "belief"):
            close(got[key][i], want[key], tol, f"agent {i} {key}")


def test_backward(dtype: torch.dtype, tol: float) -> None:
    name = str(dtype).replace("torch.", "")
    print(f"gradient parity ({name})")
    n, b = 3, 10
    cfg = NetConfig()
    nets, stacked = build(n, cfg, dtype)
    states, actions, roles, tokens, mask = inputs(n, b, cfg, dtype, empty_rows=True)

    # Distinct per-agent targets, so a bug that averages across the agent
    # axis cannot pass by symmetry.
    torch.manual_seed(11)
    tgt_q = torch.randn(n, b, dtype=dtype)
    tgt_v = torch.randn(n, b, dtype=dtype)

    out = stacked(states, actions, roles, tokens, mask, want="all")
    loss = ((out["q"] - tgt_q).square().mean(dim=1)
            + (out["v"] - tgt_v).square().mean(dim=1)).sum()
    stacked.zero_grad(set_to_none=True)
    loss.backward()

    owner = torch.arange(b)
    for i, net in enumerate(nets):
        o = net(states[i], actions[i], owner, roles[i], tokens[i], mask[i], want="all")
        li = ((o["q"] - tgt_q[i]).square().mean()
              + (o["v"] - tgt_v[i]).square().mean())
        net.zero_grad(set_to_none=True)
        li.backward()

    # Compare a representative parameter from every distinct kind of layer.
    pairs = [
        ("stem.weight", nets[0].stem.weight, stacked.stem.weight, True),
        ("stem.bias", nets[0].stem.bias, stacked.stem.bias, False),
        ("block0.fc1", nets[0].blocks[0].fc1.weight, stacked.blocks[0].fc1.weight, True),
        ("block0.norm", nets[0].blocks[0].norm.weight, stacked.blocks[0].norm.weight, False),
        ("fuse.lin", nets[0].fuse[1].weight, stacked.fuse_lin.weight, True),
        ("fuse.norm", nets[0].fuse[0].weight, stacked.fuse_norm.weight, False),
        ("hist.proj", nets[0].history.proj.weight, stacked.history.proj.weight, True),
        ("hist.in_proj",
         nets[0].history.layers[0]["attn"].in_proj_weight,
         stacked.history.in_proj[0].weight, True),
        ("hist.out_proj",
         nets[0].history.layers[0]["attn"].out_proj.weight,
         stacked.history.out_proj[0].weight, True),
        ("hist.ff1", nets[0].history.layers[0]["ff"][0].weight,
         stacked.history.ff1[0].weight, True),
        ("action_enc", nets[0].action_encoder[0].weight,
         stacked.action_encoder.layers[0].weight, True),
        ("q_points[1]", nets[0].q_points[1][0].weight,
         stacked.q_points[1].layers[0].weight, True),
        ("v_head", nets[0].v_head[0].weight, stacked.v_head.layers[0].weight, True),
    ]
    for label, ref, stk, transposed in pairs:
        g_ref, g_stk = ref.grad, stk.grad[0]
        if transposed:
            g_stk = g_stk.t()
        elif g_stk.dim() == 2:
            g_stk = g_stk.squeeze(0)
        close(g_stk, g_ref, tol, f"grad {label}")

    # And the agent axis really is independent: agent 1's gradient must
    # match net 1's, not net 0's.
    g1 = stacked.stem.weight.grad[1].t()
    close(g1, nets[1].stem.weight.grad, tol, "grad stem agent 1 (axis not mixed)")
    mismatch = (g1 - nets[0].stem.weight.grad).abs().max().item()
    check(mismatch > tol * 10,
          f"agents carry different gradients (agent1 vs net0 differs by {mismatch:.3e})")


def test_pos_and_history_grad() -> None:
    print("history positional embedding gets its own gradient per agent")
    n, b = 3, 12
    cfg = NetConfig()
    nets, stacked = build(n, cfg, torch.float64)
    states, actions, roles, tokens, mask = inputs(n, b, cfg, torch.float64, True)
    out = stacked(states, actions, roles, tokens, mask, want="q")
    stacked.zero_grad(set_to_none=True)
    (out["q"] * torch.arange(1, n + 1, dtype=torch.float64)[:, None]).sum().backward()
    owner = torch.arange(b)
    for i, net in enumerate(nets):
        o = net(states[i], actions[i], owner, roles[i], tokens[i], mask[i], want="q")
        net.zero_grad(set_to_none=True)
        (o["q"] * (i + 1)).sum().backward()
        close(stacked.history.pos.grad[i], net.history.pos.grad, 1e-9,
              f"agent {i} history pos grad")


def test_want_modes() -> None:
    print("want= modes return the same keys as the unstacked net")
    n, b = 2, 8
    cfg = NetConfig()
    nets, stacked = build(n, cfg, torch.float64)
    states, actions, roles, tokens, mask = inputs(n, b, cfg, torch.float64, False)
    owner = torch.arange(b)
    for want, keys in (("q", {"q"}), ("state", {"v", "belief"}),
                       ("all", {"q", "q_level", "v", "belief"})):
        got = stacked(states, actions, roles, tokens, mask, want=want)
        ref = nets[0](states[0], actions[0], owner, roles[0], tokens[0], mask[0], want=want)
        check(set(got) == keys == set(ref), f"want={want} keys {sorted(got)}")


def test_no_history() -> None:
    print("history=none configuration")
    n, b = 3, 16
    cfg = NetConfig(history="none")
    nets, stacked = build(n, cfg, torch.float64)
    states, actions, roles, _, _ = inputs(n, b, cfg, torch.float64, False)
    got = stacked(states, actions, roles, want="all")
    owner = torch.arange(b)
    for i, net in enumerate(nets):
        want = net(states[i], actions[i], owner, roles[i], want="all")
        close(got["q"][i], want["q"], 1e-9, f"agent {i} q (no history)")


def test_round_trip() -> None:
    print("write_into round-trips a slice back to a plain TractorNet")
    n, b = 3, 16
    cfg = NetConfig()
    nets, stacked = build(n, cfg, torch.float64)
    # Perturb the stacked weights so the round trip has something to carry
    # that the original nets do not already hold.
    with torch.no_grad():
        for p in stacked.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    states, actions, roles, tokens, mask = inputs(n, b, cfg, torch.float64, True)
    got = stacked(states, actions, roles, tokens, mask, want="all")
    owner = torch.arange(b)
    for i in range(n):
        out_net = TractorNet(cfg).to(torch.float64)
        stacked.write_into(i, out_net)
        want = out_net(states[i], actions[i], owner, roles[i],
                       tokens[i], mask[i], want="all")
        for key in ("q", "q_level", "v", "belief"):
            close(want[key], got[key][i], 1e-9, f"agent {i} {key} after write_into")


def test_parameter_count() -> None:
    print("parameter count matches N separate nets")
    for n in (1, 4):
        cfg = NetConfig()
        net = TractorNet(cfg)
        stacked = StackedTractorNet(n, cfg)
        check(stacked.num_parameters() == n * net.num_parameters(),
              f"N={n}: {stacked.num_parameters():,} == {n} x {net.num_parameters():,}")


def main() -> int:
    torch.set_num_threads(4)
    # float64 first: it is the tight check, and anything that fails there is
    # a real bug rather than accumulated rounding.
    test_forward(torch.float64, 1e-9)
    test_backward(torch.float64, 1e-9)
    test_pos_and_history_grad()
    test_want_modes()
    test_no_history()
    test_round_trip()
    test_parameter_count()
    # float32 is what actually runs; the looser tolerance is rounding, and
    # it is worth seeing how much of it there is.
    test_forward(torch.float32, 3e-4)
    test_backward(torch.float32, 3e-4)

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed")
        return 1
    print("stacked net is numerically identical to N separate nets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
