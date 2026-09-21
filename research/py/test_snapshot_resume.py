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
from tractor_rl import layout as L
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
    # want="all", roles spanning all three, and a loss that touches every
    # output: RMSprop only files state for a parameter it has seen a
    # gradient for, so under the old want="q" with every role 0 the level,
    # value and belief heads and two of the three points heads -- 26 of the
    # net's 81 parameter tensors -- were never in the round trip at all.
    roles = torch.arange(8) % L.NUM_ROLES
    for _ in range(5):
        opt_learner.opt.zero_grad(set_to_none=True)
        out = opt_net(
            torch.randn(8, features2.STATE_DIM),
            torch.randn(8, features2.ACTION_DIM),
            torch.arange(8),
            roles,
            torch.randn(8, features2.HISTORY_STEPS, features2.HISTORY_TOKEN_DIM),
            torch.ones(8, features2.HISTORY_STEPS),
            want="all")
        loss = (out["q"].square().mean() + out["q_level"].square().mean()
                + out["v"].square().mean() + out["belief"].square().mean())
        loss.backward()
        opt_learner.opt.step()
        # The teacher's optimiser has to be stepped too, or its state stays
        # empty and the round trip below covers one of the two optimisers
        # while reporting a pass on both -- exactly the "round trip of
        # nothing" this section exists to rule out.
        opt_learner.critic_opt.zero_grad(set_to_none=True)
        q_priv = opt_critic(
            torch.randn(8, features2.STATE_DIM),
            torch.randn(8, features2.PRIVILEGED_DIM),
            torch.randn(8, features2.ACTION_DIM),
            torch.arange(8))
        q_priv.square().mean().backward()
        opt_learner.critic_opt.step()
    before = {"opt": opt_learner.opt.state_dict(),
              "critic_opt": opt_learner.critic_opt.state_dict()}
    n_moments = {k: sum(1 for v in sd["state"].values() if "square_avg" in v)
                 for k, sd in before.items()}
    # Every parameter tensor, not merely one of them. A tensor no gradient
    # reached carries no RMSprop state, is therefore absent from the
    # payload, and the element-wise comparison below would never look at
    # it. An equality rather than a floor, so a head added to either
    # network later cannot quietly sit the round trip out: nothing in
    # either forward pass is unreachable, so the counts must match exactly.
    n_params = {"opt": sum(1 for _ in opt_net.parameters()),
                "critic_opt": sum(1 for _ in opt_critic.parameters())}
    assert n_moments == n_params, \
        f"optimiser state covers {n_moments}, but the nets have {n_params}"

    train2.save(opt_net, opt_critic, opt_learner, ocfg, onet_cfg, oargs,
                hands=7, snapshots=())
    blob = torch.load(train2.CHECKPOINT_DIR / "opttest_latest.pt",
                      map_location="cpu", weights_only=False)
    assert "opt" in blob and "critic_opt" in blob, "optimiser state not saved"

    # Restore into fresh optimisers and compare the moments element-wise.
    fresh_net = TractorNet(onet_cfg)
    fresh_net.load_state_dict(blob["model"])
    fresh = Learner2(fresh_net, train2.PrivilegedCritic(), ocfg, torch.device("cpu"))
    for key, opt in (("opt", fresh.opt), ("critic_opt", fresh.critic_opt)):
        assert not opt.state_dict()["state"], f"fresh {key} should start empty"
        opt.load_state_dict(blob[key])
        after = opt.state_dict()
        assert set(after["state"]) == set(before[key]["state"]), \
            f"{key} state keys differ"
        worst = 0.0
        for k, v in before[key]["state"].items():
            for name, t in v.items():
                if torch.is_tensor(t) and t.is_floating_point():
                    worst = max(worst, (t - after["state"][k][name]).abs().max().item())
        assert worst == 0.0, f"{key} moments changed across the round trip: {worst}"
        print(f"{key} state round-trips exactly ({n_moments[key]} parameter moments)")

    # And a checkpoint predating the field must still resume -- through
    # train2's own resume path, not a restatement of it here. This block
    # used to assert that a dict built by dropping "opt" no longer had
    # "opt", and that a freshly constructed Learner2 had empty optimisers.
    # Both hold by construction, of any payload and any RMSprop, and
    # neither ever reached train2.py. What actually has to hold is that
    # train2 tolerates the missing keys, so run train2.
    old_style = {k: v for k, v in blob.items() if k not in ("opt", "critic_opt")}
    old_path = train2.CHECKPOINT_DIR / "opttest_legacy.pt"
    torch.save(old_style, old_path)

    class _StopBeforePlay(Exception):
        """Stands in for the worker pool main() is about to build."""

    built: list[Learner2] = []
    real_learner, real_pool, real_argv = train2.Learner2, train2.WorkerPool, sys.argv

    def _record(*a, **kw):
        built.append(real_learner(*a, **kw))
        return built[-1]

    def _no_pool(*a, **kw):
        raise _StopBeforePlay

    # main() restores the optimisers and only then stands the pool up, so a
    # pool that refuses to exist stops it a few lines past the code under
    # test -- and before it opens a log file, installs signal handlers or
    # writes a checkpoint of its own.
    train2.Learner2, train2.WorkerPool = _record, _no_pool
    sys.argv = ["train2.py", "--resume", str(old_path), "--device", "cpu",
                "--threads", "1", "--run-name", "opttest_legacy"]
    try:
        train2.main()
        raise AssertionError("train2.main() ran past the worker pool")
    except _StopBeforePlay:
        pass
    finally:
        train2.Learner2, train2.WorkerPool = real_learner, real_pool
        sys.argv = real_argv
    assert built, "train2.main() never reached the learner"
    resumed = built[-1]
    assert resumed.steps == 1234, f"the checkpoint was not read back: {resumed.steps}"
    # This is what the round trip above cannot check: `resumed_state.get(key)`
    # and the `if blob is None: continue` behind it. Make either one a plain
    # subscript and main() raises before it ever gets here.
    assert not resumed.opt.state_dict()["state"], "legacy resume filled opt anyway"
    assert not resumed.critic_opt.state_dict()["state"], \
        "legacy resume filled critic_opt anyway"
    print("checkpoint without optimiser state still loads (starts cold)")

    # The write is atomic, so no .tmp is left behind.
    assert not (train2.CHECKPOINT_DIR / "snaptest_latest.tmp").exists(), "temp file left behind"
    print("atomic save left no temp file")

    # The opttest save archives as well, so it leaves two 26 MB files of its
    # own. Unlisted here they accumulate every run, and on Modal this
    # directory is a symlink onto the persistent volume, so the leak is
    # billed -- while opttest_latest.pt has the exact <run>_latest.pt shape
    # that train_segment's `resume == "auto" and latest.exists()` keys on,
    # so a run named opttest would quietly resume from a five-step artifact.
    for f in ("snaptest_latest.pt", "snaptest_1234.pt", "snaptest_legacy.pt",
              "opttest_latest.pt", "opttest_1234.pt", "opttest_legacy.pt"):
        (train2.CHECKPOINT_DIR / f).unlink(missing_ok=True)
    print("\nall snapshot round-trip checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
