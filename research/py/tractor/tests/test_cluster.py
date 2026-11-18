"""tractor.cluster maps launch's flags to the overrides modal_train.py builds."""

import argparse

from tractor import cluster


def _parse(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    cluster._train_parser(sub)
    return ap.parse_args(["train", "--name", "x", *argv])


def test_zero2_dst2_flags():
    a = _parse(["--steps", "5000", "--lr", "1e-4", "--length-buckets", "4",
                "--fixed-shapes", "--cuda-graphs", "--rollout-forwards", "compiled",
                "--bury-policy", "--bury-samples", "4", "--no-bury-scale-minibatch",
                "--bury-reward", "playout", "--bury-landlords", "512",
                "--distill-paths", "/d/*.npz", "--distill-lr", "1e-4", "--distill-steps", "8"])
    o = cluster.train_overrides(a)
    assert o["ppo"]["lr"] == 1e-4 and o["ppo"]["length_buckets"] == 4
    assert o["ppo"]["cuda_graphs"] and o["ppo"]["fixed_shapes"] and not o["ppo"]["compile"]
    assert o["ppo"]["target_kl"] is None and "minibatch" not in o["ppo"]
    assert o["rollout_forwards"] == "compiled"
    assert o["declare"] == {"hands_per_step": 1024, "bury_samples": 4,
                            "bury_scale_minibatch": False, "bury_reward": "playout",
                            "bury_landlords": 512, "declare_reward": "critic"}
    assert o["distill"] == {"paths": "/d/*.npz", "coef": 1.0, "lr": 1e-4, "steps": 8,
                            "minibatch": 512}
    assert o["bury_policy"] and o["buffer_capacity"] == 1024 * 100


def test_defaults_and_v6():
    o = cluster.train_overrides(_parse(["--arch", "v6"]))
    assert o["rollout_forwards"] == "eager" and o["distill"] == {} and o["bury_train"] == {}
    assert o["model"]["ffn"] == "swiglu"


def test_milestones(tmp_path):
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    for s in (125, 250):
        (ck / f"step_{s:07d}.pt").write_bytes(b"x")
        cluster._keep_milestone(ck / f"step_{s:07d}.pt", tmp_path / "milestones", 250)
    assert [p.name for p in (tmp_path / "milestones").iterdir()] == ["step_0000250.pt"]
