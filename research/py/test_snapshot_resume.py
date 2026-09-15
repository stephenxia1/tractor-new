"""Round-trips the opponent pool and the optimiser through a checkpoint.

Two things were previously dropped on every save, and both cost real
training.

The **opponent pool** went first: a resumed run restarted self-play against
copies of the current net and needed 100k steps to rebuild its diversity.

The **optimiser state** went next, and cost more. RMSprop's second moments
were rebuilt from zero on every resume, so the first steps after each
reclaim took uncalibrated sizes. One 12-hour arm of the history A/B was
reclaimed seven times; its counterpart's checkpoint, taken 35k steps after
such a reset, measured *worse* than the checkpoint 100k steps before it on
two independent opponents.

Checked here: both survive a round trip, snapshots come back as usable
float weights, and a checkpoint predating either field still loads.
"""

from __future__ import annotations

import sys
import types
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train2
from tractor_rl import features2
from tractor_rl.model2 import NetConfig, TractorNet
from tractor_rl.dmc2 import DMC2Config, Learner2


def build(run_name: str):
    # "attn", not True: `history` is a str compared against "attn", so a
    # bool silently builds a net with no encoder and the optimiser
    # round-trip below would cover 93% of the parameters while
    # reporting a pass on all of them.
    net_cfg = NetConfig(history="attn")
    net = TractorNet(net_cfg)
    critic = train2.PrivilegedCritic()
    cfg = DMC2Config()
    # A real Learner2, not a stand-in: `save` now reaches for its optimisers,
    # and a stub would let a broken payload pass.
    learner = Learner2(net, critic, cfg, torch.device("cpu"))
    learner.steps = 1234
    args = types.SimpleNamespace(run_name=run_name)
    return net, critic, learner, cfg, net_cfg, args


def main() -> int:
    torch.manual_seed(0)
    net, critic, learner, cfg, net_cfg, args = build("snaptest")

    # Three snapshots with deliberately distinct weights.
    snapshots: deque[dict] = deque(maxlen=10)
    originals = []
    for step in (10_000, 20_000, 30_000):
        other = TractorNet(net_cfg)
        with torch.no_grad():
            for p in other.parameters():
                p.add_(float(step) / 10_000)
        state = {k: v.detach().cpu().clone() for k, v in other.state_dict().items()}
        snapshots.append({"steps": step, "state": state})
        originals.append(state)

    train2.save(net, critic, learner, cfg, net_cfg, args, hands=99, snapshots=snapshots)
    path = train2.CHECKPOINT_DIR / "snaptest_latest.pt"
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    assert "snapshots" in loaded, "checkpoint carries no snapshots"
    assert len(loaded["snapshots"]) == 3, f"expected 3, got {len(loaded['snapshots'])}"
    assert [s["steps"] for s in loaded["snapshots"]] == [10_000, 20_000, 30_000]
    print(f"pool survives the save: {len(loaded['snapshots'])} snapshots")

    # Stored half, restored float, and close enough that an opponent net
    # built from it plays the same game.
    for stored, original in zip(loaded["snapshots"], originals):
        halves = [v.dtype for v in stored["state"].values() if v.is_floating_point()]
        assert all(d == torch.float16 for d in halves), "snapshots not stored at half precision"
        restored = train2._to_float(stored["state"])
        probe = TractorNet(net_cfg)
        probe.load_state_dict(restored)  # raises if a key or shape is wrong
        worst = max(
            (original[k] - restored[k]).abs().max().item()
            for k in original
            if original[k].is_floating_point() and original[k].numel()
        )
        assert worst < 1e-2, f"half-precision drift too large: {worst}"
    print("restored to float, loadable into a net, max drift < 1e-2")

    # A pre-fix checkpoint must still resume.
    legacy = dict(loaded)
    legacy.pop("snapshots")
    legacy_path = train2.CHECKPOINT_DIR / "snaptest_legacy.pt"
    torch.save(legacy, legacy_path)
    old = torch.load(legacy_path, map_location="cpu", weights_only=False)
    assert old.get("snapshots", []) == [], "legacy fallback is not empty"
    print("checkpoint without a pool still loads (empty pool)")

    # --- optimiser state ---
    # Take real steps first: a freshly built RMSprop has empty state, and a
    # round trip of nothing would pass while proving nothing.
    opt_net, opt_critic, opt_learner, ocfg, onet_cfg, oargs = build("opttest")
    for _ in range(5):
        opt_learner.opt.zero_grad(set_to_none=True)
        out = opt_net(
            torch.randn(8, features2.STATE_DIM),
            torch.randn(8, features2.ACTION_DIM),
            torch.arange(8),
            torch.zeros(8, dtype=torch.long),
            torch.randn(8, features2.HISTORY_STEPS, features2.HISTORY_TOKEN_DIM),
            torch.ones(8, features2.HISTORY_STEPS),
            want="q")
        out["q"].square().mean().backward()
        opt_learner.opt.step()
    before = opt_learner.opt.state_dict()
    n_moments = sum(1 for v in before["state"].values() if "square_avg" in v)
    assert n_moments > 0, "optimiser has no state to round-trip"

    train2.save(opt_net, opt_critic, opt_learner, ocfg, onet_cfg, oargs,
                hands=7, snapshots=())
    blob = torch.load(train2.CHECKPOINT_DIR / "opttest_latest.pt",
                      map_location="cpu", weights_only=False)
    assert "opt" in blob and "critic_opt" in blob, "optimiser state not saved"

    # Restore into a fresh optimiser and compare the moments element-wise.
    fresh_net = TractorNet(onet_cfg)
    fresh_net.load_state_dict(blob["model"])
    fresh = Learner2(fresh_net, train2.PrivilegedCritic(), ocfg, torch.device("cpu"))
    assert not fresh.opt.state_dict()["state"], "fresh optimiser should start empty"
    fresh.opt.load_state_dict(blob["opt"])
    after = fresh.opt.state_dict()
    assert set(after["state"]) == set(before["state"]), "optimiser state keys differ"
    worst = 0.0
    for k, v in before["state"].items():
        for name, t in v.items():
            if torch.is_tensor(t) and t.is_floating_point():
                worst = max(worst, (t - after["state"][k][name]).abs().max().item())
    assert worst == 0.0, f"optimiser moments changed across the round trip: {worst}"
    print(f"optimiser state round-trips exactly ({n_moments} parameter moments)")

    # And a checkpoint predating the field must still resume.
    old_style = {k: v for k, v in blob.items() if k not in ("opt", "critic_opt")}
    cold = Learner2(TractorNet(onet_cfg), train2.PrivilegedCritic(),
                    ocfg, torch.device("cpu"))
    assert old_style.get("opt") is None, "legacy payload still carries opt"
    assert not cold.opt.state_dict()["state"], "legacy path should leave it cold"
    print("checkpoint without optimiser state still loads (starts cold)")

    # The write is atomic, so no .tmp is left behind.
    assert not (train2.CHECKPOINT_DIR / "snaptest_latest.tmp").exists(), "temp file left behind"
    print("atomic save left no temp file")

    for f in ("snaptest_latest.pt", "snaptest_1234.pt", "snaptest_legacy.pt"):
        (train2.CHECKPOINT_DIR / f).unlink(missing_ok=True)
    print("\nall snapshot round-trip checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
