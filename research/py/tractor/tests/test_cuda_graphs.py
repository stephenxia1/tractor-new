"""The update step captured as a CUDA graph (``train.graphs``). GPU only.

A captured step replays the kernels it recorded at the addresses it
recorded, so the failures worth fearing are silent: a replay that reads
last step's rows, a stale optimiser state after a reload, a refusal of a
non-finite step that the capture skipped. Each test holds the captured
update to the uncaptured one on the same minibatches.

Run on a GPU with ``modal run modal_perf.py::gpu_tests``.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from ..nn.model import relative_seat_agents
from ..train.buffer import DeviceRows, RolloutBuffer
from ..train.critic_learner import CriticLearner, CriticTrainConfig
from ..train.gae import compute_gae
from ..train.population import hand_agents
from ..train.ppo import PPOConfig, PPOLearner
from ..train.rollout import collect
from .test_population import ACTOR_ONLY, SMALL, _actors, _shared, _table

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA")
DEV = "cuda"


def _round(hands: int = 8, seed: int = 3):
    nets = [n.to(DEV) for n in _actors()]
    critic = _shared().to(DEV)
    seat_of = hand_agents(_table(9), hands)
    buf = RolloutBuffer(800, SMALL)
    collect(nets, np.random.default_rng(seed), buf, n_hands=hands, device=DEV, cfg=ACTOR_ONLY,
            pool=8, agents=seat_of, critic=critic)
    compute_gae(buf, gamma=1.0, lam=0.95)
    return nets, critic, seat_of, buf


def _max_diff(a: torch.nn.Module, b: torch.nn.Module) -> float:
    return max(float((p - q).abs().max()) for p, q in zip(a.parameters(), b.parameters()))


CFG = PPOConfig(minibatch=32, length_buckets=4, epochs=2, fixed_shapes=True)


@pytest.mark.parametrize("compiled", [False, True])
def test_a_captured_update_is_the_update(compiled: bool) -> None:
    nets, _c, _s, buf = _round()
    rows = DeviceRows(buf, DEV)
    cfg = replace(CFG, compile=compiled)
    plain = PPOLearner(copy.deepcopy(nets[0]), cfg, DEV)
    graphed = PPOLearner(copy.deepcopy(nets[0]), replace(cfg, cuda_graphs=True), DEV)
    for u in range(4):
        torch.manual_seed(u)
        want = plain.update(buf, device_rows=rows)
        torch.manual_seed(u)
        got = graphed.update(buf, device_rows=rows)
        for key, v in want.items():
            np.testing.assert_allclose(got[key], v, rtol=1e-4, atol=1e-6, err_msg=key)
    graphs = graphed._graphs
    assert graphs.captures > 0 and graphs.replays > graphs.captures
    assert _max_diff(plain.net, graphed.net) < 1e-5
    assert plain.steps == graphed.steps


def test_a_captured_update_stops_at_the_kl_as_the_plain_one() -> None:
    nets, _c, _s, buf = _round()
    rows = DeviceRows(buf, DEV)
    cfg = replace(CFG, lr=3e-2, target_kl=0.04)
    plain = PPOLearner(copy.deepcopy(nets[0]), cfg, DEV)
    graphed = PPOLearner(copy.deepcopy(nets[0]), replace(cfg, cuda_graphs=True), DEV)
    stops = 0
    for u in range(4):
        torch.manual_seed(u)
        want = plain.update(buf, device_rows=rows)
        torch.manual_seed(u)
        got = graphed.update(buf, device_rows=rows)
        for key, v in want.items():
            np.testing.assert_allclose(got[key], v, rtol=1e-4, atol=1e-6, err_msg=key)
        stops += int(want["early_stop"])
    assert stops > 0
    assert graphed._graphs.captures > 0 and graphed._graphs.replays > 0
    assert _max_diff(plain.net, graphed.net) < 1e-5
    assert plain.steps == graphed.steps


def test_a_captured_step_still_refuses_a_nonfinite_one() -> None:
    nets, _c, _s, buf = _round()
    rows = DeviceRows(buf, DEV)
    learner = PPOLearner(nets[0], replace(CFG, epochs=1, cuda_graphs=True), DEV)
    for _ in range(4):
        learner.update(buf, device_rows=rows)
    assert learner._graphs.captures > 0
    buf.ret[: len(buf)] = np.nan  # the actor's own value head: a NaN loss
    rows.load(buf)  # in place, so the graphs stand and replay
    before = [p.detach().clone() for p in learner.net.parameters()]
    first = next(iter(learner.opt.state.values()))
    state = {k: v.clone() for k, v in first.items()}
    replays = learner._graphs.replays
    m = learner.update(buf, device_rows=rows)
    assert learner._graphs.replays > replays
    assert m["n_minibatches"] == 0 and m["nonfinite_minibatches"] > 0
    assert all(torch.equal(p, q) for p, q in zip(learner.net.parameters(), before))
    assert all(torch.equal(first[k], v) for k, v in state.items())


def test_the_graphs_follow_a_reloaded_optimiser_and_new_rows() -> None:
    nets, _c, _s, buf = _round()
    rows = DeviceRows(buf, DEV)
    cfg = replace(CFG, epochs=1)
    plain = PPOLearner(copy.deepcopy(nets[0]), cfg, DEV)
    graphed = PPOLearner(copy.deepcopy(nets[0]), replace(cfg, cuda_graphs=True), DEV)

    def both(u: int, rows_: DeviceRows) -> None:
        torch.manual_seed(u)
        plain.update(buf, device_rows=rows_)
        torch.manual_seed(u)
        graphed.update(buf, device_rows=rows_)

    for u in range(3):
        both(u, rows)
    saved = (copy.deepcopy(plain.opt.state_dict()), copy.deepcopy(graphed.opt.state_dict()))
    for u in range(3, 5):
        both(u, rows)
    # A reload replaces the optimiser's state tensors the graphs point at.
    plain.opt.load_state_dict(saved[0])
    graphed.opt.load_state_dict(saved[1])
    captures = graphed._graphs.captures
    for u in range(5, 8):
        both(u, rows)
    assert graphed._graphs.captures > captures
    # So does a new set of device rows.
    captures = graphed._graphs.captures
    for u in range(8, 11):
        both(u, DeviceRows(buf, DEV) if u == 8 else rows)
    assert graphed._graphs.captures > captures
    assert _max_diff(plain.net, graphed.net) < 1e-5


def test_a_captured_critic_update_is_the_update() -> None:
    _n, critic, seat_of, buf = _round()
    n = len(buf)
    seats = relative_seat_agents(seat_of[buf.episode[:n]], buf.seat[:n])
    rows = DeviceRows(buf, DEV)
    cfg = CriticTrainConfig(minibatch=32, fixed_shapes=True)
    runs = []
    for graphs in (False, False, True):
        learner = CriticLearner(copy.deepcopy(critic), replace(cfg, cuda_graphs=graphs), DEV)
        for u in range(4):
            torch.manual_seed(u)
            learner.update(buf, seats, device_rows=rows)
        runs.append(learner)
    assert runs[2]._graphs.captures > 0 and runs[2]._graphs.replays > 0
    # Two uncaptured runs set the floor: attention's backward may
    # accumulate in a different order each run.
    noise = _max_diff(runs[0].critic, runs[1].critic)
    assert _max_diff(runs[0].critic, runs[2].critic) <= max(10 * noise, 1e-5)
