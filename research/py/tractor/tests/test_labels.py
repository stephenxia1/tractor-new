"""Labels made beside training (``train/labels.py``): a follower writes
shards from a run's newest checkpoint, and a distiller rereads the
freshest of them while it trains."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..eval.search import SearchConfig
from ..nn.model import ActorCritic, ModelConfig
from ..train.distill import Distiller, DistillRows
from ..train.labels import (FreshRows, follow, initial_checkpoint, newest_checkpoint,
                            shard_name, shard_step)
from ..train.loop import TrainConfig, train

SMALL = ModelConfig(d_model=48, n_layers_history=1, n_layers_query=1, n_heads=4, d_ff=96,
                    critic_d=32, critic_layers=1, max_candidates=16)
SEARCH = SearchConfig(worlds=2, top_k=2, confident=0.99)


def _cfg(ckpt: Path, **kw) -> TrainConfig:
    base = dict(
        total_steps=2, hands_per_step=2, buffer_capacity=512, checkpoint_every=1,
        checkpoint_dir=str(ckpt), model=asdict(SMALL),
        ppo={"epochs": 1, "minibatch": 32, "bf16": False},
        warmup_steps=1, play_phase_steps=1, declare_phase_steps=1, bury_policy=True,
        declare={"hands_per_step": 4, "epochs": 1, "minibatch": 16},
    )
    base.update(kw)
    return TrainConfig(**base)


def test_shard_names_carry_the_checkpoint_step() -> None:
    name = shard_name(4250, 3, 7, 12)
    assert shard_step(name) == 4250 and shard_step(f"/x/{name}") == 4250


def test_follow_labels_the_newest_checkpoint_and_a_live_distiller_reads_it() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        train(_cfg(run / "checkpoints"))
        newest = newest_checkpoint(run)
        assert newest is not None and newest[1] == 2
        out = Path(tmp) / "labels"
        # The run has finished, so the follower writes one shard and stops.
        written = follow(run, out, hands=2, search=SEARCH, seed=1, keep=5, poll_s=0.1,
                         log=lambda *_: None)
        shards = sorted(out.glob("s*.npz"))
        assert written == 1 and len(shards) == 1 and shard_step(shards[0]) == 2

        summary = train(_cfg(
            Path(tmp) / "b" / "checkpoints", total_steps=2, warmup_steps=2,
            declare_phase_steps=0,
            distill={"paths": str(out / "s*.npz"), "reload_every": 1, "window_rows": 0,
                     "steps": 1, "minibatch": 8, "lr": 1e-4},
        ))
        last = summary["final"]  # the learner's metrics carry the "ppo/" prefix
        assert last["ppo/distill/rows"] > 0
        assert last["ppo/distill/label_step_max"] == 2.0
        assert last["ppo/distill/label_lag"] == 1.0 - 2.0  # step 1 of a fresh run
        assert "ppo/distill/ce" in last


def _fake_rows(n: int, value: int) -> DistillRows:
    z = lambda *s: np.zeros(s, dtype=np.float16)
    return DistillRows(
        history=z(n, 2, 3), history_len=np.full(n, value, dtype=np.int64), context=z(n, 2),
        unseen=z(n, 2), planes=z(n, 2), cards=z(n, 2), candidates=z(n, 2, 2),
        candidate_mask=np.ones((n, 2), dtype=bool), privileged=z(n, 2),
        target=np.zeros(n, dtype=np.int64), prior_top=np.zeros(n, dtype=np.int64),
    )


def test_fresh_rows_keeps_the_newest_shards_within_the_window() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for step in (100, 200, 300):
            _fake_rows(10, step).save(str(d / shard_name(step, 0, 0, step)))
        fresh = FreshRows(str(d / "s*.npz"), window_rows=15)
        rows, metrics = fresh.load()
        # The two newest shards reach the window; the oldest is left out.
        assert set(np.unique(rows.history_len)) == {200, 300}
        assert metrics["distill/label_step_max"] == 300.0
        assert metrics["distill/label_step_min"] == 200.0
        time.sleep(0.01)
        _fake_rows(10, 400).save(str(d / shard_name(400, 0, 0, 400)))
        rows, metrics = fresh.load()
        assert set(np.unique(rows.history_len)) == {300, 400}


def _three_shards(d: Path) -> None:
    for step in (100, 200, 300):
        _fake_rows(10, step).save(str(d / shard_name(step, 0, 0, step)))


def test_fresh_rows_leaves_out_shards_past_max_age_and_reports_label_age() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _three_shards(d)
        rows, metrics = FreshRows(str(d / "s*.npz"), max_age=150).load(step=350)
        assert set(np.unique(rows.history_len)) == {200, 300}
        assert metrics["distill/shards_too_old"] == 1.0
        assert metrics["distill/label_age_max"] == 150.0
        assert metrics["distill/label_age_mean"] == (50 * 10 + 150 * 10) / 20
        # Without max_age every shard stays and nothing is counted too old.
        rows, metrics = FreshRows(str(d / "s*.npz")).load(step=350)
        assert len(rows) == 30 and "distill/shards_too_old" not in metrics


def test_a_requeued_distiller_rereads_the_same_window_and_drops_stale_labels() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _three_shards(d)
        net = ActorCritic(SMALL)
        a = Distiller(net, None, "cpu", source=str(d / "s*.npz"), max_age=150)
        a.refresh(350)
        # A requeue builds a new distiller in a new process; the window
        # lives on disk, so its first refresh holds the same rows.
        b = Distiller(net, None, "cpu", source=str(d / "s*.npz"), max_age=150)
        b.refresh(350)
        assert np.array_equal(np.sort(a.rows.history_len), np.sort(b.rows.history_len))
        # Long after the newest shard, nothing is fresh: it stops distilling.
        metrics = b.refresh(1000)
        assert b.rows is None and metrics["distill/shards_too_old"] == 3.0
        assert b.step(1, np.random.default_rng(0)) == {"distill/rows": 0.0}


def _ckpt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def test_follow_starts_from_the_resume_checkpoint_and_live_distill_resumes_its_optimiser() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src" / "checkpoints"
        train(_cfg(src))
        ck = src / "step_0000002.pt"
        assert "distill" not in _ckpt(ck)  # no distillation: the checkpoint as before

        # A run resumed from ``ck`` under Slurm, before its first checkpoint.
        run = Path(tmp) / "run"
        run.mkdir()
        (run / "launch.json").write_text(json.dumps({"call_id": "slurm-1", "resume": str(ck)}))
        assert initial_checkpoint(run, "") is None
        assert initial_checkpoint(run, "auto") == (ck, 2)
        out = Path(tmp) / "labels"
        written = follow(run, out, hands=2, search=SEARCH, seed=1, keep=5, poll_s=0.1,
                         max_shards=1, initial="auto", log=lambda *_: None)
        shards = sorted(out.glob("s*.npz"))
        assert written == 1 and len(shards) == 1 and shard_step(shards[0]) == 2

        distill = {"paths": str(out / "s*.npz"), "steps": 1, "minibatch": 8, "lr": 1e-4}
        # Fixed shards: the checkpoint is as before.
        fixed = Path(tmp) / "fixed" / "checkpoints"
        train(_cfg(fixed, warmup_steps=2, declare_phase_steps=0, distill=distill))
        assert "distill" not in _ckpt(fixed / "step_0000002.pt")

        # Labels read live: the distiller's optimiser is checkpointed, and a
        # resume (a requeue) carries on from its moments, not from zero.
        live_kw = dict(warmup_steps=2, declare_phase_steps=0,
                       distill={**distill, "reload_every": 1, "window_rows": 0})
        live = Path(tmp) / "live" / "checkpoints"
        train(_cfg(live, **live_kw))
        adam = lambda p: max(float(s["step"]) for s in
                             _ckpt(p)["distill"]["optimizer"]["state"].values())
        assert adam(live / "step_0000002.pt") == 2.0
        train(_cfg(live, total_steps=3, **live_kw), resume=str(live / "step_0000002.pt"))
        assert adam(live / "step_0000003.pt") == 3.0
