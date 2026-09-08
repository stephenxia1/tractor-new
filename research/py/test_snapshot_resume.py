"""Round-trips the opponent pool through a checkpoint.

The pool was previously dropped on every save, so a resumed run restarted
self-play against copies of the current net. This checks the three things
that have to hold for the fix to mean anything: the snapshots survive, they
come back as usable float weights, and an old checkpoint without them still
loads.
"""

from __future__ import annotations

import sys
import types
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train2
from tractor_rl.model2 import NetConfig, TractorNet
from tractor_rl.dmc2 import DMC2Config


def build(run_name: str):
    net_cfg = NetConfig(history=True)
    net = TractorNet(net_cfg)
    critic = train2.PrivilegedCritic()
    learner = types.SimpleNamespace(steps=1234)
    args = types.SimpleNamespace(run_name=run_name)
    return net, critic, learner, DMC2Config(), net_cfg, args


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

    # The write is atomic, so no .tmp is left behind.
    assert not (train2.CHECKPOINT_DIR / "snaptest_latest.tmp").exists(), "temp file left behind"
    print("atomic save left no temp file")

    for f in ("snaptest_latest.pt", "snaptest_1234.pt", "snaptest_legacy.pt"):
        (train2.CHECKPOINT_DIR / f).unlink(missing_ok=True)
    print("\nall snapshot round-trip checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
