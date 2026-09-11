"""DMC v2: points-primary targets, privileged distillation, auxiliary heads.

Two changes to what is being regressed.

**Points, not levels.** The old target ended in the §8.3 level bracket,
which is a step function of points: 79 and 81 points produce different
integers from near-identical play, and the term fires once per hand where
points arrive every trick. The primary target is now the discounted point
return -- dense and smooth -- and the level outcome becomes an auxiliary
head so the 80-point cliff stays represented, since it is real strategy
rather than noise.

**A privileged teacher.** The Monte-Carlo return conditions on one
sampled world out of an enormous number, so its variance is dominated by
the deal rather than by the decision. A critic that sees every hand
predicts the same expectation with far less noise, and
E[Q_priv | policy view] = E[G | policy view], so distilling it into the
policy network is unbiased in expectation while being much quieter. The
blend weight `lambda_mc` keeps some raw return in the mix, because the
teacher is itself only as good as its own training.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layout as L
from .encoding import NUM_ENC_SLOTS
from . import features_torch as tfeat
from .features2 import (
    Episode,
    action_features,
    belief_targets,
    history_features,
    privileged_features,
    state_features,
)


@dataclass(slots=True)
class DMC2Config:
    gamma: float = 0.995
    point_scale: float = 100.0
    epsilon: float = 0.02
    buffer_capacity: int = 400_000
    batch_size: int = 512
    lr: float = 1e-4
    critic_lr: float = 3e-4
    grad_clip: float = 5.0
    train_every: int = 2
    steps_per_train: int = 1
    min_buffer: int = 20_000
    # How much raw Monte-Carlo return survives in the target. 1.0 is the
    # old behaviour (no teacher); 0.0 trusts the critic completely.
    #
    # The blend is a regulariser written as a target. Since
    #   MSE(q, l*G + (1-l)*P) == l*MSE(q, G) + (1-l)*MSE(q, P) + const,
    # l = 0.3 penalises disagreement with the teacher at 7:3 *against the
    # data*. That weight is right early, when the return's variance is
    # dominated by which deal came up rather than by the decision being
    # scored, and wrong late, when what binds is the teacher's own error --
    # which correlates with hidden information the policy cannot see, so
    # the student is pulled toward a target it structurally cannot reach.
    #
    # Annealing l upward trades the teacher out as the policy sharpens.
    # `lambda_mc_final = None` keeps the constant blend, which is what every
    # run before this one used.
    lambda_mc: float = 0.3
    lambda_mc_final: float | None = None
    lambda_anneal_start: int = 0    # learner steps; both 0 disables
    lambda_anneal_end: int = 0
    # Auxiliary loss weights.
    w_level: float = 0.3
    w_value: float = 0.3
    w_belief: float = 0.5
    # Warm-up: the teacher is useless until it has trained a little, so
    # ignore it for this many learner steps.
    critic_warmup_steps: int = 2000
    # The teacher trains on every Nth learner step; it converges fast and
    # costs as much per step as the policy network does.
    critic_every: int = 2
    # How often the loss scalars are pulled off the device. Every step
    # on CPU costs nothing; on GPU each pull is a pipeline stall.
    stats_every: int = 25

    def lambda_at(self, step: int) -> float:
        """Blend weight on the raw return at this learner step.

        Linear between the two step bounds and clamped outside them, so a
        resume lands on the right value without replaying history -- the
        step counter is cumulative and restored from the checkpoint.
        """
        if self.lambda_mc_final is None or self.lambda_anneal_end <= self.lambda_anneal_start:
            return self.lambda_mc
        span = self.lambda_anneal_end - self.lambda_anneal_start
        frac = min(1.0, max(0.0, (step - self.lambda_anneal_start) / span))
        return self.lambda_mc + frac * (self.lambda_mc_final - self.lambda_mc)


def hand_returns(episode: Episode, cfg: DMC2Config) -> tuple[np.ndarray, float]:
    """(discounted point return per trick, signed level outcome).

    Both are from the attackers' point of view; the caller flips the sign
    for defending seats.
    """
    n = episode.num_tricks
    level = float(episode.level_delta) * (-1.0 if episode.defenders_won else 1.0)
    if n == 0:
        return np.zeros(0, dtype=np.float32), level
    captured = np.where(episode.trick_attacker_won, episode.trick_points, 0)
    rewards = captured.astype(np.float64) / cfg.point_scale
    # Whatever the attackers finished with beyond the tricks is the kitty
    # multiplier, won or lost on the final trick.
    rewards[-1] += (float(episode.attacker_points) - float(captured.sum())) / cfg.point_scale
    out = np.zeros(n, dtype=np.float32)
    running = 0.0
    for t in range(n - 1, -1, -1):
        running = rewards[t] + cfg.gamma * running
        out[t] = running
    return out, level


class ReplayBuffer2:
    """Raw wire bytes plus both targets.

    Records are kept in wire form (~1 KB with the privileged block) rather
    than as features: it is smaller, and a change to the featuriser does
    not invalidate everything already collected.
    """

    def __init__(self, capacity: int, state_bytes: int) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, state_bytes), dtype=np.uint8)
        self.actions = np.zeros((capacity, L.CANDIDATE_BYTES), dtype=np.uint8)
        self.g_points = np.zeros(capacity, dtype=np.float32)
        self.g_level = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.cursor = 0
        self.lock = threading.Lock()

    def add(self, states, actions, g_points, g_level) -> None:
        n = states.shape[0]
        if n == 0:
            return
        with self.lock:
            self._add_locked(states, actions, g_points, g_level)

    def _add_locked(self, states, actions, g_points, g_level) -> None:
        n = states.shape[0]
        if n > self.capacity:
            states, actions = states[-self.capacity:], actions[-self.capacity:]
            g_points, g_level = g_points[-self.capacity:], g_level[-self.capacity:]
            n = self.capacity
        end = self.cursor + n
        if end <= self.capacity:
            sl = slice(self.cursor, end)
            self.states[sl], self.actions[sl] = states, actions
            self.g_points[sl], self.g_level[sl] = g_points, g_level
        else:
            first = self.capacity - self.cursor
            self.states[self.cursor:], self.actions[self.cursor:] = states[:first], actions[:first]
            self.g_points[self.cursor:], self.g_level[self.cursor:] = g_points[:first], g_level[:first]
            rest = n - first
            self.states[:rest], self.actions[:rest] = states[first:], actions[first:]
            self.g_points[:rest], self.g_level[:rest] = g_points[first:], g_level[first:]
        self.cursor = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        # Copies under the lock. The prefetch thread samples while the actor
        # thread is inserting, and the ring buffer overwrites in place, so an
        # unguarded read can return half of an old record and half of a new
        # one -- a corrupt training example that nothing would ever flag.
        with self.lock:
            idx = rng.integers(0, self.size, size=batch_size)
            return (self.states[idx], self.actions[idx],
                    self.g_points[idx], self.g_level[idx])


@dataclass(slots=True)
class RawBatch:
    """Records straight out of the buffer, not yet featurised.

    ~980 bytes a record against ~10 KB of features, so this is what crosses
    PCIe when the device does its own featurisation.
    """
    states: np.ndarray
    actions: np.ndarray
    g_points: np.ndarray
    g_level: np.ndarray


@dataclass(slots=True)
class FeaturisedBatch:
    """One training batch, already out of numpy and ready for the device."""
    sf: np.ndarray
    af: np.ndarray
    tokens: np.ndarray
    mask: np.ndarray
    priv: np.ndarray
    belief: np.ndarray
    roles: np.ndarray
    g_points: np.ndarray
    g_level: np.ndarray


def featurise_batch(states, actions, g_points, g_level) -> FeaturisedBatch:
    owner = np.arange(states.shape[0], dtype=np.int64)
    priv = privileged_features(states)
    tokens, mask = history_features(states)
    return FeaturisedBatch(
        sf=state_features(states),
        af=action_features(actions, states, owner),
        tokens=tokens,
        mask=mask,
        priv=priv,
        belief=belief_targets(states, priv),
        roles=states[:, L.H_ROLE].astype(np.int64),
        g_points=g_points,
        g_level=g_level,
    )


class AsyncLearner:
    """Runs gradient steps on a background thread.

    The actor loop and the learner are strictly serial today: every round
    stops to take its gradient steps, and the game workers idle through
    them. Since torch releases the GIL for the forward and backward, the
    two can overlap instead -- the loop becomes bounded by whichever half is
    slower rather than by their sum. On the measured split (roughly half
    actor, half learner) that is worth up to ~1.9x.

    The cost is that actions are chosen with weights a fraction of a second
    old. DMC tolerates this well: it regresses onto the return the
    *behaviour* policy achieved, so slightly stale weights are an ordinary
    on-policy dataset for the policy that generated them, not an off-policy
    correction problem.

    `pending` bounds how far the learner may run ahead; without it a fast
    learner would drain the queue and train repeatedly on the same buffer
    contents while the actors fall behind.
    """

    def __init__(self, learner, buffer, cfg, seed: int, max_pending: int = 2) -> None:
        self.learner = learner
        self.buffer = buffer
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self._requests: queue.Queue = queue.Queue(maxsize=max_pending)
        self._stop = threading.Event()
        self.steps_done = 0
        self.dropped = 0
        self.error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                n = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            for _ in range(n):
                if self._stop.is_set():
                    break
                try:
                    if self.learner.step(self.buffer, self._rng) is not None:
                        self.steps_done += 1
                except Exception as err:  # noqa: BLE001
                    # A thread that dies quietly is indistinguishable from a
                    # buffer that has not filled yet. Record it and stop.
                    self.error = err
                    traceback.print_exc()
                    self._stop.set()
                    return

    def request(self, n: int) -> None:
        """Ask for n gradient steps; skip if the learner is already behind."""
        try:
            self._requests.put_nowait(n)
        except queue.Full:
            # Actors outrunning the learner. Dropping the request is right:
            # the alternative is blocking the game loop, which is the thing
            # this exists to avoid.
            self.dropped += n

    @property
    def steps(self) -> int:
        return self.learner.steps

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


class BatchPrefetcher:
    """Samples and featurises on a background thread.

    Featurisation is numpy, which releases the GIL for the array work, so it
    genuinely overlaps with the forward and backward passes rather than
    time-slicing against them. That matters most on a GPU: the device step
    shrinks to a few milliseconds while featurisation does not, so without
    overlap the accelerator spends most of its time waiting for the host.

    Depth 2 is enough to cover one step of jitter without letting batches go
    stale -- a deep queue would train on samples drawn long before they are
    used, quietly lowering the recency the replay ratio is tuned around.
    """

    def __init__(self, buffer, batch_size: int, seed: int, depth: int = 2,
                 featurise: bool = True) -> None:
        self.buffer = buffer
        self.batch_size = batch_size
        # When the learner featurises on its own device there is nothing for
        # this thread to precompute, so it only samples.
        self.featurise = featurise
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._rng = np.random.default_rng(seed)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.buffer.size < self.batch_size:
                time.sleep(0.01)
                continue
            sample = self.buffer.sample(self.batch_size, self._rng)
            batch = featurise_batch(*sample) if self.featurise else RawBatch(*sample)
            while not self._stop.is_set():
                try:
                    self.q.put(batch, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def next(self, timeout: float = 30.0) -> FeaturisedBatch:
        return self.q.get(timeout=timeout)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class Learner2:
    def __init__(
        self,
        net: nn.Module,
        critic: nn.Module,
        cfg: DMC2Config,
        device: torch.device,
    ) -> None:
        self.net = net
        self.critic = critic
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.RMSprop(net.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.RMSprop(critic.parameters(), lr=cfg.critic_lr)
        self.steps = 0
        self._last_critic_loss_t: torch.Tensor | None = None
        # Featurising on the device only pays when the device is fast
        # relative to numpy, which in practice means CUDA.
        self.device_featurise = device.type == "cuda"
        self.amp_dtype: torch.dtype | None = None
        self.stats: dict[str, float] = {}

    def step(self, buffer: ReplayBuffer2, rng: np.random.Generator) -> dict[str, float] | None:
        cfg = self.cfg
        if buffer.size < cfg.min_buffer:
            return None
        sample = buffer.sample(cfg.batch_size, rng)
        if self.device_featurise:
            return self.step_batch(self.featurise_on_device(RawBatch(*sample)))
        return self.step_batch(featurise_batch(*sample))

    def featurise_on_device(self, raw: RawBatch) -> FeaturisedBatch:
        """Ship the raw records over and derive features where we train.

        On a GPU this is the difference between a 12 ms step and a 12 ms step
        that waits 18-35 ms for numpy first, and it moves ~980 bytes a record
        instead of ~10 KB of features.
        """
        st = torch.from_numpy(raw.states).to(self.device, non_blocking=True)
        cd = torch.from_numpy(raw.actions).to(self.device, non_blocking=True)
        owner = torch.arange(st.shape[0], device=self.device)
        priv = tfeat.privileged_features(st)
        tokens, mask = tfeat.history_features(st)
        return FeaturisedBatch(
            sf=tfeat.state_features(st),
            af=tfeat.action_features(cd, st, owner),
            tokens=tokens,
            mask=mask,
            priv=priv,
            belief=tfeat.belief_targets(st, priv),
            roles=st[:, L.H_ROLE].long(),
            g_points=torch.from_numpy(raw.g_points).to(self.device, non_blocking=True),
            g_level=torch.from_numpy(raw.g_level).to(self.device, non_blocking=True),
        )

    def step_batch(self, batch: FeaturisedBatch) -> dict[str, float] | None:
        """The device half of a step, on a batch someone else featurised.

        Split out so a prefetch thread can do the numpy work for step N+1
        while this runs step N. On a GPU that is the difference between the
        device working and the device waiting.
        """
        cfg = self.cfg
        n = batch.sf.shape[0]
        owner = torch.arange(n, device=self.device)

        def dev(x):
            # Already on the device when featurisation happened there;
            # otherwise a non-blocking upload, which pays off once the source
            # is pinned and is harmless when it is not.
            if torch.is_tensor(x):
                return x
            return torch.from_numpy(x).to(self.device, non_blocking=True)

        sf = dev(batch.sf)
        af = dev(batch.af)
        roles = dev(batch.roles)
        tokens = dev(batch.tokens)
        mask = dev(batch.mask)
        tgt_points = dev(batch.g_points)
        tgt_level = dev(batch.g_level)

        # --- privileged teacher, trained on the raw return ---
        priv = dev(batch.priv)
        train_critic = self.steps % cfg.critic_every == 0
        # bf16 rather than fp16: no loss scaling to get wrong, and the
        # Monte-Carlo returns this regresses onto vary enough in magnitude
        # across a hand that fp16's narrow exponent is a real risk.
        amp = torch.autocast(self.device.type, dtype=self.amp_dtype,
                             enabled=self.amp_dtype is not None)
        if train_critic:
            with amp:
                q_priv = self.critic(sf, priv, af, owner)
                critic_loss = F.mse_loss(q_priv.float(), tgt_points)
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
            self.critic_opt.step()
            q_priv = q_priv.detach()
            self._last_critic_loss_t = critic_loss.detach()
        else:
            with torch.no_grad(), amp:
                q_priv = self.critic(sf, priv, af, owner).float()

        # --- policy network ---
        with amp:
            out = self.net(sf, af, owner, roles, tokens, mask, want="all")
        # Losses in fp32: the critic residual is a small difference of two
        # similar numbers (~0.19 against a target spread of ~0.59), which is
        # exactly where reduced precision would bite first.
        out = {k: v.float() for k, v in out.items()}
        if self.steps < cfg.critic_warmup_steps:
            target = tgt_points
            lam = 1.0
        else:
            lam = cfg.lambda_at(self.steps)
            target = lam * tgt_points + (1.0 - lam) * q_priv

        loss_q = F.mse_loss(out["q"], target)
        loss_level = F.mse_loss(out["q_level"], tgt_level)
        loss_v = F.mse_loss(out["v"], tgt_points)
        belief = dev(batch.belief)
        loss_belief = F.binary_cross_entropy_with_logits(out["belief"], belief)
        loss = (
            loss_q
            + cfg.w_level * loss_level
            + cfg.w_value * loss_v
            + cfg.w_belief * loss_belief
        )
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
        self.opt.step()

        self.steps += 1

        # Reading a scalar off a tensor forces a device synchronisation, and
        # there are nine of them here. On CPU that is free; on GPU it drains
        # the queue on every step for numbers that are only ever read once
        # per `log_every` rounds. So collect them on a cadence and keep the
        # previous values in between -- the logger samples, it does not
        # average, so nothing is lost.
        if self.steps % cfg.stats_every != 0:
            return self.stats
        with torch.no_grad():
            # One transfer instead of nine: stack first, sync once.
            packed = torch.stack([
                loss.detach(), loss_q.detach(), loss_level.detach(),
                loss_v.detach(), loss_belief.detach(),
                tgt_points.std(), (tgt_points - q_priv).std(),
                ((out["belief"] > 0).float() == belief).float().mean(),
            ]).cpu().numpy()
        self.stats = {
            # A python float, so it costs no device sync and can go in
            # unconditionally rather than on the stats cadence.
            "lambda_mc": lam,
            "loss": float(packed[0]),
            "loss_q": float(packed[1]),
            "loss_level": float(packed[2]),
            "loss_value": float(packed[3]),
            "loss_belief": float(packed[4]),
            "critic_loss": (float(self._last_critic_loss_t)
                            if self._last_critic_loss_t is not None else float("nan")),
            "target_std": float(packed[5]),
            "critic_residual_std": float(packed[6]),
            "belief_acc": float(packed[7]),
        }
        return self.stats
