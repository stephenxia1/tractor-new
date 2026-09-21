"""The outer training loop, end to end on CPU.

These tests are deliberately small but not smoke tests: each one would
fail on a specific defect rather than merely on an exception. The loop
is the piece where the rollout, the credit assignment and the update
meet, so the failures worth catching here are the ones that live in the
seams -- an anneal that is logged but never applied, a checkpoint that
does not restore the optimiser, a resumed run that silently restarts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from dataclasses import asdict

from ..nn.model import ActorCritic, ModelConfig
from ..train.loop import (
    TrainConfig,
    load_checkpoint,
    reward_mix,
    save_checkpoint,
    train,
)
from ..train.ppo import PPOConfig, PPOLearner


def _tiny() -> ModelConfig:
    """A model small enough to train a few steps in seconds.

    `max_history` stays at the default: `features.encode` always emits
    that many rows, so the buffer rejects a config that disagrees, and
    it is right to -- a shrunk value would silently store a truncated
    history. Only the widths that cost time are reduced.
    """
    return ModelConfig(
        d_model=64,
        n_layers_history=1,
        n_layers_query=1,
        n_heads=4,
        d_ff=128,
        critic_d=32,
        critic_layers=1,
        max_candidates=16,
    )


def _cfg(tmp: str, **kw) -> TrainConfig:
    base = dict(
        total_steps=2,
        hands_per_step=2,
        buffer_capacity=512,
        checkpoint_every=0,
        checkpoint_dir=tmp,
        model=asdict(_tiny()),
        ppo={"epochs": 1, "minibatch": 32, "bf16": False},
    )
    base.update(kw)
    return TrainConfig(**base)


def test_reward_mix_runs_from_zero_to_the_final_weight() -> None:
    cfg = TrainConfig(anneal_steps=10, final_level_weight=1.0)
    assert reward_mix(0, cfg) == 0.0
    assert reward_mix(5, cfg) == 0.5
    assert reward_mix(10, cfg) == 1.0
    # Past the anneal it saturates rather than continuing to climb.
    assert reward_mix(50, cfg) == 1.0

    # A zero-length anneal means the level reward is on from the start,
    # which is what a fine-tuning run from a points-trained checkpoint
    # wants.
    instant = TrainConfig(anneal_steps=0, final_level_weight=1.0)
    assert reward_mix(0, instant) == 1.0


def test_the_anneal_reaches_the_rollout_and_is_not_merely_logged() -> None:
    """The weight the loop logs must be the weight collect() was given.

    An earlier revision computed the mix, wrote it into the metrics row
    and then called collect() without it, so the charts showed an anneal
    that never happened.
    """
    import tractor.train.loop as L

    seen: list[float] = []
    real = L.collect

    def spy(*args, **kwargs):
        seen.append(kwargs.get("level_weight"))
        return real(*args, **kwargs)

    L.collect = spy
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(tmp, total_steps=3, anneal_steps=4,
                       final_level_weight=1.0)
            train(cfg, device="cpu")
    finally:
        L.collect = real

    assert seen == [0.0, 0.25, 0.5], seen


def test_two_steps_run_and_report_finite_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = train(_cfg(tmp), device="cpu")
    assert out["steps"] == 2
    final = out["final"]
    assert final["decisions"] > 0
    for key, value in final.items():
        if isinstance(value, float):
            assert value == value, f"{key} is NaN"
            assert abs(value) != float("inf"), f"{key} is infinite"


def test_checkpoint_round_trips_weights_and_optimiser_state() -> None:
    """Resuming must restore the optimiser, not only the weights.

    AdamW carries first and second moments; dropping them restarts the
    schedule and shows up as a loss spike at every resume.
    """
    cfg_model = _tiny()
    net = ActorCritic(cfg_model)
    learner = PPOLearner(net, PPOConfig(), "cpu")

    # Take a step so the optimiser has non-trivial state to restore.
    loss = sum(p.sum() for p in net.parameters())
    loss.backward()
    learner.opt.step()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ck.pt"
        with tempfile.TemporaryDirectory() as tmp2:
            save_checkpoint(path, net, learner, 7, _cfg(tmp2))

        fresh = ActorCritic(cfg_model)
        fresh_learner = PPOLearner(fresh, PPOConfig(), "cpu")
        step = load_checkpoint(path, fresh, fresh_learner)

    assert step == 7
    for a, b in zip(net.parameters(), fresh.parameters()):
        assert torch.equal(a, b)

    got = fresh_learner.opt.state_dict()["state"]
    want = learner.opt.state_dict()["state"]
    assert set(got) == set(want) and got, "optimiser state was not restored"
    for k in want:
        for field in ("exp_avg", "exp_avg_sq"):
            if field in want[k]:
                assert torch.equal(got[k][field], want[k][field]), field


def test_resume_continues_from_the_saved_step() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        first = _cfg(tmp, total_steps=2, checkpoint_every=2)
        train(first, device="cpu")
        saved = Path(tmp) / "step_0000002.pt"
        assert saved.exists(), sorted(p.name for p in Path(tmp).iterdir())

        # Four total steps resuming from two must run two, not four.
        second = _cfg(tmp, total_steps=4)
        out = train(second, device="cpu", resume=saved)
    assert out["steps"] == 2


def test_prune_keeps_only_the_most_recent_checkpoints() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp, total_steps=4, checkpoint_every=1, keep_last=2)
        train(cfg, device="cpu")
        kept = sorted(p.name for p in Path(tmp).glob("step_*.pt"))
    assert kept == ["step_0000003.pt", "step_0000004.pt"], kept


def test_evaluation_runs_on_schedule_and_lands_in_one_row() -> None:
    """`eval_every` must reach the loop, and log once, not twice.

    The field existed as configuration for a while before anything read
    it, which is the quiet kind of wrong: runs were configured to
    evaluate and simply did not.
    """
    rows: list[dict] = []

    class _Sink:
        def log(self, row, step=None):
            rows.append(dict(row))

        def summary(self, values):
            pass

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp, total_steps=4, eval_every=2, eval_deals=2)
        train(cfg, device="cpu", run=_Sink())

    assert len(rows) == 4, f"expected one row per step, got {len(rows)}"
    scored = [i for i, r in enumerate(rows) if "eval/levels" in r]
    assert scored == [1, 3], scored
    for i in scored:
        assert rows[i]["eval/deals"] == 2.0
        assert "span/evaluate" in rows[i]
        for key in ("eval/levels", "eval/points", "eval/levels_ci_low"):
            value = rows[i][key]
            assert value == value and abs(value) != float("inf"), key


def test_evaluation_leaves_the_network_in_training_mode() -> None:
    """A measurement must not silently disable dropout or batchnorm.

    There is neither in this model today, but `evaluate` flips the
    module into eval mode and a restore that did not happen would be
    invisible until something stateful was added.
    """
    from ..train.evaluate import evaluate as run_eval

    cfg_model = _tiny()
    net = ActorCritic(cfg_model)
    net.train()
    run_eval(net, "cpu", cfg_model, num_deals=1, seed=3)
    assert net.training, "evaluate() left the network in eval mode"


def test_behaviour_logprob_is_computed_in_the_learners_precision() -> None:
    """At identical weights the importance ratio must be exactly 1.

    Collection and the update run the same network over the same rows.
    If collection computes its log-probabilities in float32 while the
    learner recomputes them under bfloat16 autocast, the ratio is not 1
    at epoch zero, and the difference is not noise: the same rows round
    the same way every time, so the clip silences a fixed, input-
    correlated subset of the batch on every update. The failure grows
    with logit magnitude, so it is mildest exactly when a test is most
    likely to be written -- at initialisation.
    """
    import numpy as np

    from ..train.buffer import RolloutBuffer
    from ..train.ppo import PPOConfig, PPOLearner
    from ..train.rollout import collect

    model_cfg = _tiny()
    net = ActorCritic(model_cfg)
    # Sharpened deliberately. The rounding error is proportional to
    # logit magnitude and vanishes at initialisation, so an untrained
    # net passes this check whether or not the bug is present: measured
    # at initialisation the mismatched path gives approx_kl 3.7e-07 and
    # clips nothing, and at sixteen times the scorer weights the same
    # path gives 3.0e-03 and clips 3.6% of rows. A policy that has
    # learned anything is nearer the second.
    with torch.no_grad():
        for param in net.actor.scorer.parameters():
            param.mul_(16.0)
    learner = PPOLearner(net, PPOConfig(bf16=True), "cpu")
    assert learner.amp, "this test is meaningless unless autocast is on"

    buf = RolloutBuffer(4096, model_cfg)
    collect(
        net,
        np.random.default_rng(5),
        buf,
        n_hands=1,
        device="cpu",
        cfg=model_cfg,
        amp=learner.amp,
    )
    assert len(buf) > 0

    idx = np.arange(min(len(buf), 96))
    obs, privileged, batch = buf.as_torch(idx, "cpu")
    _loss, parts = learner.losses(obs, privileged, batch)

    kl = abs(float(parts["approx_kl"]))
    clipped = float(parts["clip_fraction"])
    # Exact, not approximate: the same computation run twice over the
    # same rows gives the same answer, so any deviation is the two
    # halves disagreeing about precision.
    assert kl < 1e-9, f"approx_kl {kl:.3e} at identical weights"
    assert clipped == 0.0, f"{clipped:.1%} of rows clipped before any step"


def test_a_nonfinite_loss_never_reaches_the_weights() -> None:
    """One poisoned row must not end the run.

    `clip_grad_norm_` is not a guard: a NaN total norm makes the clip
    coefficient NaN, every gradient is scaled by it, and the optimiser
    writes NaN into every parameter. On an unattended run that is a dead
    network and a checkpoint of it.
    """
    import numpy as np

    from ..train.buffer import RolloutBuffer
    from ..train.gae import compute_gae
    from ..train.ppo import PPOConfig, PPOLearner
    from ..train.rollout import collect

    model_cfg = _tiny()
    net = ActorCritic(model_cfg)
    learner = PPOLearner(net, PPOConfig(bf16=False, epochs=1), "cpu")

    buf = RolloutBuffer(4096, model_cfg)
    collect(net, np.random.default_rng(11), buf, n_hands=1,
            device="cpu", cfg=model_cfg)
    compute_gae(buf)

    # Poison one row the way an overflow would.
    buf.ret[3] = float("nan")

    before = [p.detach().clone() for p in net.parameters()]
    metrics = learner.update(buf)
    assert all(
        torch.isfinite(p).all() for p in net.parameters()
    ), "a NaN row reached the weights"
    assert metrics["nonfinite_minibatches"] >= 1.0, metrics

    # And the guard must not be silently refusing everything.
    buf.ret[3] = 0.0
    learner.update(buf)
    moved = any(
        not torch.equal(a, b)
        for a, b in zip(before, [p.detach() for p in net.parameters()])
    )
    assert moved, "the learner stopped taking steps entirely"


if __name__ == "__main__":
    import sys
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception:  # noqa: BLE001 - reporting harness
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    sys.exit(1 if failures else 0)
