"""Tests for GAE and the PPO learner.

These are written against the two failure modes that do not announce
themselves.

*Credit assignment across the wrong axis.* A buffer row's neighbours are
other seats and other hands, so a GAE that walks the buffer in row order
produces finite, plausible, completely meaningless advantages. Every
shape test and every is-it-finite test passes. So the fast path is
checked against a slow reference that groups with a dictionary and
recurses in Python -- bit for bit, not approximately -- and there is a
separate test that constructs a bootstrap across a ``done`` boundary and
asserts the number that would result does *not* appear.

*Entropy over padding.* The candidate axis is 128 wide and mostly empty.
An entropy bonus computed over all 128 slots pays the policy to hold
mass on slots that do not exist, and it reads as a larger, healthier
entropy while it does so. ``test_entropy_ignores_padded_slots`` feeds
the helper log-probs that deliberately put most of their mass on
padding, so a version without the mask returns a number about 2.5 nats
too high.

*An auxiliary target that asks a question the network can already
answer.* The belief head predicts where the cards a seat cannot see
are. Point any of its five slots at something the observation already
holds -- the acting seat's own hand, say -- and the head scores well on
a free answer while the category that carries real hidden-information
signal goes untrained. The loss falls either way.
``test_belief_target_construction`` scrambles the acting seat's own
holding and demands the target back bit-identical, and
``test_belief_target_partitions_the_unseen_cards`` un-normalises the
target on 500-odd real positions and compares every slot against
ground truth taken from the ``PlayState`` rather than from the planes,
so it cross-checks the slice the target is built from instead of
restating it.

Where a test could pass vacuously it is asserted in both directions: the
buggy value is named and excluded, and the perturbation that *should*
move the answer is shown to move it.

Run directly; there is no pytest in this environment.

    python -m tractor.tests.test_ppo
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from tractor.gen.episode import generate
from tractor.nn.features import (
    CONTEXT_DIM,
    D_CARD,
    HISTORY_DIM,
    NUM_PLANES,
    PLANE_HAND,
    PLANE_PLAYED,
    PLANE_TRICK,
    encode,
)
from tractor.nn.model import (
    MASK_LOGIT,
    NUM_BELIEF_LOCATIONS,
    NUM_OUTCOMES,
    NUM_PRIVILEGED_ROWS,
    NUM_REL_SEATS,
    ActorCritic,
    ModelConfig,
    ObsBatch,
    privileged_view,
)
from tractor.rules.cards import NUM_TYPES
from tractor.train.gae import (
    compute_gae,
    compute_gae_reference,
    group_trajectories,
)
from tractor.train.ppo import (
    PPOConfig,
    PPOLearner,
    belief_target,
    explained_variance,
    masked_entropy,
    outcome_class,
)

DEVICE = torch.device("cpu")

# Small enough that a full PPO update over a few dozen rows runs in
# under a second on a laptop CPU, and structurally identical to the
# default: same towers, same masking, same heads.
TINY = ModelConfig(
    d_model=32,
    n_layers_history=1,
    n_layers_query=1,
    n_heads=2,
    d_ff=64,
    critic_d=16,
    critic_layers=1,
    critic_heads=2,
    critic_d_ff=32,
    belief_d=16,
    scalar_d=8,
    head_d=16,
    max_history=8,
    max_candidates=6,
)


# --- the stand-in buffer --------------------------------------------------


class _FakeBuffer:
    """The documented ``RolloutBuffer`` contract and nothing else.

    Deliberately not an import of the real buffer: the production code
    must depend only on the field names and ``as_torch``, so that a
    change to the real buffer's internals cannot silently couple to the
    learner. The generated contents are nonsense but structurally
    honest -- history is a prefix, candidates are a prefix, every stored
    action indexes a legal slot, and the counts are small non-negative
    integers.
    """

    def __init__(
        self,
        capacity: int,
        cfg: ModelConfig = TINY,
        *,
        seed: int = 0,
        fill: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.capacity = capacity
        self.cfg = cfg
        self.n = capacity if fill is None else int(fill)
        hist, cands = cfg.max_history, cfg.max_candidates

        self.history = np.zeros((capacity, hist, HISTORY_DIM), dtype=np.float16)
        self.history_len = rng.integers(0, hist + 1, size=capacity).astype(np.int64)
        for row, length in enumerate(self.history_len):
            self.history[row, :length] = rng.standard_normal(
                (length, HISTORY_DIM)
            ).astype(np.float16)

        self.context = rng.standard_normal((capacity, CONTEXT_DIM)).astype(np.float16)
        self.unseen = rng.integers(0, 3, (capacity, NUM_TYPES)).astype(np.float16)
        self.planes = rng.integers(
            0, 2, (capacity, NUM_PLANES, NUM_TYPES)
        ).astype(np.float16)
        self.cards = rng.standard_normal((capacity, NUM_TYPES, D_CARD)).astype(
            np.float16
        )

        self.candidates = np.zeros((capacity, cands, NUM_TYPES), dtype=np.float16)
        self.candidate_mask = np.zeros((capacity, cands), dtype=bool)
        self.action = np.zeros(capacity, dtype=np.int64)
        for row in range(capacity):
            legal = int(rng.integers(1, cands + 1))
            self.candidate_mask[row, :legal] = True
            for slot in range(legal):
                picks = rng.integers(0, NUM_TYPES, size=int(rng.integers(1, 4)))
                self.candidates[row, slot, picks] = 1.0
            self.action[row] = int(rng.integers(0, legal))

        self.privileged = rng.integers(
            0, 3, (capacity, NUM_PRIVILEGED_ROWS, NUM_TYPES)
        ).astype(np.float16)

        self.logp = np.zeros(capacity, dtype=np.float32)
        self.value = rng.standard_normal(capacity).astype(np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        # Non-zero only where ``done`` is, exactly as the rollout writes
        # it, and always a third of an integer in [-3, 3]. The outcome
        # head's label is read off this field, so a fake that filled it
        # everywhere would hide the very thing the restriction to
        # terminal rows exists for.
        self.reward_level = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        self.seat = np.zeros(capacity, dtype=np.int64)
        self.episode = np.zeros(capacity, dtype=np.int64)
        self.advantage = np.zeros(capacity, dtype=np.float32)
        self.ret = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.n

    def as_torch(self, idx, device):
        idx = np.asarray(idx, dtype=np.int64)

        def f32(field: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(field[idx].astype(np.float32)).to(device)

        def raw(field: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(field[idx]).to(device)

        obs = ObsBatch(
            history=f32(self.history),
            history_len=raw(self.history_len),
            context=f32(self.context),
            unseen=f32(self.unseen),
            planes=f32(self.planes),
            cards=f32(self.cards),
            candidates=f32(self.candidates),
            candidate_mask=raw(self.candidate_mask),
        )
        scalars = {
            name: raw(getattr(self, name))
            for name in (
                "action",
                "logp",
                "value",
                "reward",
                "reward_level",
                "done",
                "advantage",
                "ret",
            )
        }
        return obs, f32(self.privileged), scalars


def _net(seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(TINY)


def _fill_logp(net: ActorCritic, buf: _FakeBuffer, idx: np.ndarray) -> None:
    """Write the behaviour log-probs the current weights would produce."""
    obs, privileged, scalars = buf.as_torch(idx, DEVICE)
    with torch.no_grad():
        out, _ = net(obs, privileged)
    chosen = out.log_probs.gather(1, scalars["action"].reshape(-1, 1)).squeeze(1)
    buf.logp[idx] = chosen.cpu().numpy().astype(np.float32)


def _chosen_logp(net: ActorCritic, buf: _FakeBuffer, idx: np.ndarray) -> np.ndarray:
    obs, privileged, scalars = buf.as_torch(idx, DEVICE)
    with torch.no_grad():
        out, _ = net(obs, privileged)
    return (
        out.log_probs.gather(1, scalars["action"].reshape(-1, 1))
        .squeeze(1)
        .cpu()
        .numpy()
    )


# --- laying out realistic, interleaved trajectories -----------------------


def _interleaved(specs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Buffer row order for several hands played concurrently.

    ``specs`` is ``(episode, plays)``. Within a hand the seats act
    0, 1, 2, 3, 0, 1, ...; hands advance a trick at a time in round
    robin, which is what a batched rollout writing into one buffer
    produces. The result is that no two adjacent rows share a seat and
    most adjacent rows do not even share a hand.
    """
    streams = [[(ep, i % 4) for i in range(plays)] for ep, plays in specs]
    cursor = [0] * len(streams)
    rows: list[tuple[int, int]] = []
    while any(c < len(s) for c, s in zip(cursor, streams)):
        for h, stream in enumerate(streams):
            rows.extend(stream[cursor[h] : cursor[h] + 4])
            cursor[h] += 4
    return rows


def _lay_out(
    buf: _FakeBuffer, rows: list[tuple[int, int]], seed: int, terminal: set[int]
) -> None:
    """Write episode/seat/reward/value/done for the given row plan.

    ``terminal`` names the episodes that finished inside the buffer;
    every other one is truncated, which is the case the GAE has to
    bootstrap rather than terminate.
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    buf.episode[:n] = [ep for ep, _ in rows]
    buf.seat[:n] = [seat for _, seat in rows]
    buf.value[:n] = rng.standard_normal(n).astype(np.float32)
    # Small shaping rewards everywhere, a real outcome at the end.
    buf.reward[:n] = (rng.standard_normal(n) * 0.05).astype(np.float32)
    buf.done[:] = False

    last: dict[tuple[int, int], int] = {}
    for row, key in enumerate(rows):
        last[key] = row
    for (episode, _), row in last.items():
        if episode in terminal:
            buf.done[row] = True
            buf.reward[row] = float(rng.uniform(-1.0, 1.0))
            buf.reward_level[row] = float(rng.integers(-3, 4)) / 3.0


def _both_paths(buf: _FakeBuffer, gamma: float, lam: float):
    """Run the fast path, stash it, run the reference, hand back both.

    The fields are poisoned with NaN between the two so that a row the
    reference fails to write shows up as a mismatch rather than as a
    leftover from the fast path.
    """
    compute_gae(buf, gamma=gamma, lam=lam)
    fast = (buf.advantage.copy(), buf.ret.copy())
    buf.advantage[:] = np.nan
    buf.ret[:] = np.nan
    compute_gae_reference(buf, gamma=gamma, lam=lam)
    return fast, (buf.advantage.copy(), buf.ret.copy())


# --- 1. GAE: fast path against the reference ------------------------------


def test_gae_fast_path_matches_reference_exactly() -> None:
    """Interleaved hands, uneven lengths, one truncated. Bit for bit."""
    specs = [(11, 26), (12, 18), (13, 23), (14, 9)]
    rows = _interleaved(specs)
    buf = _FakeBuffer(len(rows), seed=1)
    _lay_out(buf, rows, seed=2, terminal={11, 12, 13})

    groups = group_trajectories(buf.episode, buf.seat)
    assert groups.index.shape[0] == 16, groups.index.shape
    assert set(groups.length.tolist()) == {7, 6, 5, 4, 3, 2}, groups.length.tolist()
    # The premise of the whole file: neighbours in the buffer are not
    # neighbours in any trajectory.
    same = sum(
        1
        for a, b in zip(rows, rows[1:])
        if a == b
    )
    assert same == 0, f"{same} adjacent rows share an (episode, seat)"

    for gamma, lam in ((1.0, 0.95), (1.0, 1.0), (0.99, 0.5), (0.9, 0.0)):
        fast, slow = _both_paths(buf, gamma, lam)
        assert np.array_equal(fast[0], slow[0]), f"advantage differs at {gamma=} {lam=}"
        assert np.array_equal(fast[1], slow[1]), f"ret differs at {gamma=} {lam=}"
        assert np.isfinite(fast[0]).all() and np.isfinite(fast[1]).all()
    print(
        f"    {len(rows)} rows, {groups.index.shape[0]} trajectories, "
        f"longest {groups.index.shape[1]}"
    )


def test_gae_fast_path_matches_reference_on_random_structure() -> None:
    """The same claim without the tidy layout.

    The interleaved case above is what a rollout really produces, which
    is also what makes it a weak test of the grouping: every group is a
    contiguous stride and every terminal sits at the end of one. Here
    the episode and seat ids are drawn at random, so groups come out of
    order and at length one, and ``done`` lands anywhere -- including on
    a group's first row, and several times in one group.
    """
    trials = 0
    for trial in range(12):
        rng = np.random.default_rng(100 + trial)
        rows = int(rng.integers(40, 400))
        buf = _FakeBuffer(rows, seed=trial)
        buf.episode[:] = rng.integers(0, max(2, rows // 20), rows)
        buf.seat[:] = rng.integers(0, 4, rows)
        buf.reward[:] = rng.standard_normal(rows).astype(np.float32)
        buf.value[:] = (rng.standard_normal(rows) * 3.0).astype(np.float32)
        buf.done[:] = rng.random(rows) < 0.25
        for gamma, lam in ((1.0, 0.95), (1.0, 1.0), (0.97, 0.63)):
            fast, slow = _both_paths(buf, gamma, lam)
            assert np.array_equal(fast[0], slow[0]), (trial, gamma, lam)
            assert np.array_equal(fast[1], slow[1]), (trial, gamma, lam)
            trials += 1
    print(f"    {trials} random (rows, gamma, lam) combinations, exact")


def test_gae_reference_and_fast_agree_on_a_partially_filled_buffer() -> None:
    """``len(buffer) < capacity`` must bound both paths identically."""
    rows = _interleaved([(1, 12), (2, 12)])
    buf = _FakeBuffer(len(rows) + 9, seed=3)
    _lay_out(buf, rows, seed=4, terminal={1, 2})
    buf.n = len(rows)

    fast, slow = _both_paths(buf, 1.0, 0.95)
    assert np.array_equal(fast[0], slow[0])
    assert np.array_equal(fast[1], slow[1])
    # Rows past the fill point belong to no trajectory and must be left
    # at zero rather than credited with whatever was in the arrays.
    assert not fast[0][len(rows) :].any()
    assert not fast[1][len(rows) :].any()


# --- 2. GAE: the done boundary --------------------------------------------


def test_gae_never_bootstraps_across_a_done() -> None:
    """Two hands, adjacent rows, one huge value. Name the wrong answer."""
    buf = _FakeBuffer(2, seed=5)
    buf.episode[:] = [0, 1]
    buf.seat[:] = [0, 0]
    buf.value[:] = [0.0, 100.0]
    buf.reward[:] = [1.0, 0.0]
    buf.done[:] = [True, True]

    compute_gae(buf, gamma=1.0, lam=1.0)
    # Correct: the hand ended, so there is nothing to bootstrap from.
    assert buf.advantage[0] == 1.0, buf.advantage[0]
    # What a row-order recursion that ignores `done` would have said.
    assert buf.advantage[0] != 101.0
    assert buf.ret[0] == 1.0

    # And the same again, this time with both rows in one (episode, seat)
    # group so only the `done` -- not the grouping -- can stop the leak.
    buf.episode[:] = [7, 7]
    buf.advantage[:] = 0.0
    compute_gae(buf, gamma=1.0, lam=1.0)
    assert buf.advantage[0] == 1.0, buf.advantage[0]


def test_gae_resets_at_a_mid_group_done() -> None:
    """Rows after a terminal must not reach the rows before it."""
    buf = _FakeBuffer(4, seed=6)
    buf.episode[:] = 3
    buf.seat[:] = 2
    buf.value[:] = [0.5, 0.25, -1.0, 2.0]
    buf.reward[:] = [0.125, 0.5, 0.25, 0.75]
    buf.done[:] = [False, True, False, True]

    compute_gae(buf, gamma=1.0, lam=1.0)
    before = buf.advantage.copy()

    # Move everything after the boundary. The first two rows may not
    # notice; the last two must.
    buf.value[2:] = [8.0, -3.0]
    buf.reward[2:] = [-2.0, 4.0]
    compute_gae(buf, gamma=1.0, lam=1.0)
    assert np.array_equal(before[:2], buf.advantage[:2]), (before, buf.advantage)
    assert not np.array_equal(before[2:], buf.advantage[2:]), "test is vacuous"

    # And the surviving numbers are the hand-computed ones: with the
    # terminal at index 1, adv[1] = r1 - v1 and adv[0] telescopes onto it.
    assert before[1] == 0.5 - 0.25
    assert before[0] == (0.125 + 0.25 - 0.5) + before[1]


# --- 3. GAE: gamma = lam = 1 is reward-to-go ------------------------------


def test_gae_with_gamma_lam_one_is_reward_to_go() -> None:
    """advantage + value == the rewards still to come, within a trajectory."""
    rows = _interleaved([(1, 22), (2, 15), (3, 19)])
    buf = _FakeBuffer(len(rows), seed=7)
    _lay_out(buf, rows, seed=8, terminal={1, 2, 3})
    # Quarters are exact in binary, so the telescoping sum is exact and
    # the assertion can be tight.
    rng = np.random.default_rng(9)
    buf.reward[: len(rows)] = (rng.integers(-8, 9, len(rows)) / 4.0).astype(np.float32)
    buf.value[: len(rows)] = (rng.integers(-8, 9, len(rows)) / 4.0).astype(np.float32)

    compute_gae(buf, gamma=1.0, lam=1.0)

    groups = group_trajectories(buf.episode, buf.seat)
    checked = 0
    for g in range(groups.index.shape[0]):
        members = groups.index[g][groups.index[g] >= 0]
        assert buf.done[members[-1]], "every hand here finished inside the buffer"
        for position, row in enumerate(members):
            to_go = float(buf.reward[members[position:]].sum())
            got = float(buf.advantage[row] + buf.value[row])
            assert abs(got - to_go) < 1e-6, (row, got, to_go)
            assert abs(float(buf.ret[row]) - to_go) < 1e-6
            checked += 1
    print(f"    {checked} rows equal their own remaining reward sum")


def test_gae_bootstraps_a_truncated_tail_off_itself() -> None:
    """No terminal in the group: the tail has zero TD error by construction."""
    buf = _FakeBuffer(3, seed=10)
    buf.episode[:] = 4
    buf.seat[:] = 1
    buf.value[:] = [1.0, 2.0, 3.0]
    buf.reward[:] = 0.0
    buf.done[:] = False

    compute_gae(buf, gamma=1.0, lam=1.0)
    # Every return collapses onto the bootstrap value; nothing claims the
    # cut-off hand scored zero.
    assert list(buf.ret) == [3.0, 3.0, 3.0], list(buf.ret)
    assert list(buf.advantage) == [2.0, 1.0, 0.0], list(buf.advantage)

    # Treating truncation as termination would have said this instead.
    assert buf.ret[2] != 0.0

    # "Zero TD error, so advantage = reward" is shorthand that holds at
    # gamma = 1 and nowhere else: bootstrapping off its own value leaves
    # the tail at delta = reward - (1 - gamma) * value. Pinned here
    # because it is a claim the module docstring makes.
    buf.advantage[:] = 0.0
    buf.ret[:] = 0.0
    compute_gae(buf, gamma=0.5, lam=1.0)
    assert abs(float(buf.advantage[2]) - (0.0 - 0.5 * 3.0)) < 1e-9, buf.advantage[2]
    assert abs(float(buf.ret[2]) - (0.0 + 0.5 * 3.0)) < 1e-9, buf.ret[2]


# --- 4. entropy over the masked distribution ------------------------------


def test_entropy_ignores_padded_slots() -> None:
    """Log-probs that spread mass over padding must not inflate entropy."""
    torch.manual_seed(0)
    slots = 128
    legal = 6
    logits = torch.full((1, slots), 0.0)
    # The legal slots are made *unattractive* so that a distribution over
    # all 128 slots keeps most of its mass, and most of its entropy, on
    # padding. This is the case where the two answers diverge most.
    logits[0, :legal] = torch.tensor([-2.0, -1.0, -3.0, -2.5, -1.5, -4.0])
    mask = torch.zeros(1, slots, dtype=torch.bool)
    mask[0, :legal] = True

    spread = F.log_softmax(logits, dim=-1)  # mass on padding: the bad input
    got = masked_entropy(spread, mask)

    reference = torch.distributions.Categorical(logits=logits[:, :legal]).entropy()
    naive = -(spread.exp() * spread).sum(dim=-1)

    assert torch.allclose(got, reference, atol=1e-6), (got, reference)
    assert float(got) <= math.log(legal) + 1e-6
    # The number an unmasked implementation returns: 2.5 nats too high,
    # and larger than the largest entropy the real action space admits.
    assert float(naive) - float(got) > 2.0, (float(naive), float(got))
    assert float(naive) > math.log(legal)
    print(f"    masked {float(got):.4f} vs unmasked {float(naive):.4f} nats")


def test_entropy_matches_the_models_own_masking() -> None:
    """On real network output the renormalisation is a no-op."""
    slots, legal = 128, 3
    logits = torch.randn(4, slots)
    mask = torch.zeros(4, slots, dtype=torch.bool)
    mask[:, :legal] = True
    masked_logits = torch.where(mask, logits, torch.full_like(logits, MASK_LOGIT))
    log_probs = F.log_softmax(masked_logits, dim=-1)

    got = masked_entropy(log_probs, mask)
    reference = torch.distributions.Categorical(logits=logits[:, :legal]).entropy()
    assert torch.allclose(got, reference, atol=1e-5), (got, reference)
    assert torch.isfinite(got).all()

    # One legal move is 14.8% of real decisions; it has to come out at
    # exactly zero entropy, not at a small negative epsilon.
    single = torch.zeros(1, slots, dtype=torch.bool)
    single[0, 0] = True
    solo = F.log_softmax(
        torch.where(single, logits[:1], torch.full_like(logits[:1], MASK_LOGIT)), dim=-1
    )
    assert float(masked_entropy(solo, single)) == 0.0


def test_learner_entropy_is_bounded_by_the_legal_count() -> None:
    """The bonus inside ``losses`` is the masked one, not a 128-slot one."""
    buf = _FakeBuffer(8, seed=11)
    buf.candidate_mask[:] = False
    buf.candidate_mask[:, :2] = True
    buf.action[:] = 0
    net = _net(1)
    learner = PPOLearner(net, PPOConfig(bf16=False), DEVICE)
    idx = np.arange(len(buf))
    obs, privileged, scalars = buf.as_torch(idx, DEVICE)
    _, parts = learner.losses(obs, privileged, scalars)
    assert float(parts["entropy"]) <= math.log(2.0) + 1e-6, float(parts["entropy"])
    assert float(parts["entropy"]) > 0.0

    # And the bonus is differentiable: an entropy term that is only ever
    # reported, never backpropagated, would satisfy every check above.
    parts["entropy"].backward()
    scorer = net.actor.scorer.score_q.weight
    assert scorer.grad is not None and float(scorer.grad.abs().max()) > 0.0
    learner.opt.zero_grad(set_to_none=True)


# --- 5. the belief target -------------------------------------------------


def _obs_with_planes(planes: torch.Tensor) -> ObsBatch:
    """An ObsBatch in which only ``planes`` carries information.

    ``belief_target`` reads nothing else. Saying so with zeros is better
    than fabricating a whole observation and leaving a reader to work
    out which parts of it the function actually looked at.
    """
    b = int(planes.shape[0])
    return ObsBatch(
        history=torch.zeros(b, 1, HISTORY_DIM),
        history_len=torch.zeros(b, dtype=torch.int64),
        context=torch.zeros(b, CONTEXT_DIM),
        unseen=torch.zeros(b, NUM_TYPES),
        planes=planes,
        cards=torch.zeros(b, NUM_TYPES, D_CARD),
        candidates=torch.zeros(b, 1, NUM_TYPES),
        candidate_mask=torch.ones(b, 1, dtype=torch.bool),
    )


def test_belief_target_construction() -> None:
    """Each of the five slots reads from its documented source.

    Built by hand so every card type in it exercises one rule: a slot's
    source, the exclusion of the acting seat's own hand, the open trick
    counting as played, or the zero-weight case.
    """
    privileged = torch.zeros(1, NUM_PRIVILEGED_ROWS, NUM_TYPES)
    planes = torch.zeros(1, NUM_PLANES, NUM_TYPES)

    # 0: both copies in my own hand. Known, not believed -- it must not
    #    reach the target at all, at any slot.
    privileged[0, 0, 0] = 2.0
    planes[0, PLANE_HAND, 0] = 2.0
    # 1: one with the seat after me, one with the seat before me.
    privileged[0, 1, 1] = 1.0
    privileged[0, 3, 1] = 1.0
    # 2: one with my partner, one in the kitty.
    privileged[0, 2, 2] = 1.0
    privileged[0, 4, 2] = 1.0
    # 3: both on the table -- one in a completed trick, one in the trick
    #    still in progress. Both are PLAYED.
    planes[0, PLANE_PLAYED + 2, 3] = 1.0
    planes[0, PLANE_TRICK + 1, 3] = 1.0
    # 4: one in my hand, one I played earlier. Dropping the hand copy
    #    makes this a one-hot on PLAYED rather than a 50/50.
    privileged[0, 0, 4] = 1.0
    planes[0, PLANE_HAND, 4] = 1.0
    planes[0, PLANE_PLAYED + 0, 4] = 1.0
    # 5: left empty everywhere -- a degenerate input, weight 0.

    obs = _obs_with_planes(planes)
    target, weight = belief_target(obs, privileged)
    assert target.shape == (1, NUM_TYPES, NUM_BELIEF_LOCATIONS)
    assert weight.shape == (1, NUM_TYPES)

    assert list(target[0, 0]) == [0.0, 0.0, 0.0, 0.0, 0.0], "own hand leaked in"
    assert list(target[0, 1]) == [0.5, 0.0, 0.5, 0.0, 0.0]
    assert list(target[0, 2]) == [0.0, 0.5, 0.0, 0.5, 0.0]
    assert list(target[0, 3]) == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert list(target[0, 4]) == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert list(target[0, 5]) == [0.0, 0.0, 0.0, 0.0, 0.0]

    assert float(weight[0, 0]) == 0.0, "a card I hold every copy of is not a belief"
    assert float(weight[0, 5]) == 0.0
    assert float(weight.sum()) == 4.0, "exactly types 1, 2, 3 and 4 are believed"
    assert [float(weight[0, c]) for c in range(6)] == [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]

    rows = target[0][weight[0] > 0]
    assert torch.allclose(rows.sum(dim=-1), torch.ones(rows.shape[0]))
    assert torch.isfinite(target).all()

    # The structural version of "own-hand counts do not appear": scramble
    # privileged row 0 into nonsense and demand the target back
    # bit-identical. An implementation that reads row 0 fails this even
    # when every hand-checked number above happens to be right.
    scrambled = privileged.clone()
    scrambled[0, 0] = (torch.arange(NUM_TYPES) % 3).float()
    same_target, same_weight = belief_target(obs, scrambled)
    assert torch.equal(target, same_target), "the acting seat's own hand is being read"
    assert torch.equal(weight, same_weight)

    # Not vacuous: scrambling a row that *is* a source does move it.
    moved = privileged.clone()
    moved[0, 1] = 1.0
    assert not torch.equal(belief_target(obs, moved)[0], target)

    # And a card type nobody holds must not pull the loss.
    from tractor.train.ppo import belief_cross_entropy

    log_probs = F.log_softmax(torch.randn(1, NUM_TYPES, NUM_BELIEF_LOCATIONS), dim=-1)
    with_absent = belief_cross_entropy(log_probs, target, weight)
    shrunk = belief_cross_entropy(log_probs[:, :5], target[:, :5], weight[:, :5])
    assert torch.allclose(with_absent, shrunk, atol=1e-6)


def test_belief_target_partitions_the_unseen_cards() -> None:
    """On real positions the five slots account for every unseen card.

    The invariant is that the normaliser is exactly ``num_decks minus
    what I hold``; if it is, the five slots partition the unseen cards
    and no copy is unaccounted for. Checked the strong way -- each slot
    is un-normalised by that factor and compared against ground truth
    taken from the ``PlayState`` rather than from the planes, so the test
    is an independent cross-check of the slice ``belief_target`` uses
    rather than a restatement of it.
    """
    decks = 2
    rng = np.random.default_rng(7)
    deck = torch.full((NUM_TYPES,), float(decks))
    checked = following = 0
    open_trick_cards = 0

    for _ in range(6):
        state, _ = generate(rng, decks)
        while not state.done:
            seat = state.to_act
            encoded = encode(state, seat)
            planes = torch.from_numpy(encoded.planes.astype(np.float32))[None]
            privileged = torch.from_numpy(
                privileged_view(state.hands, state.kitty, seat)
            )[None]
            target, weight = belief_target(_obs_with_planes(planes), privileged)

            hands = torch.from_numpy(state.hands.astype(np.float32))
            kitty = torch.from_numpy(state.kitty.astype(np.float32))
            truth = torch.stack(
                [hands[(seat + rel) % 4] for rel in (1, 2, 3)]
                + [kitty, deck - hands.sum(0) - kitty],
                dim=-1,
            )  # [54, 5], ground truth counts per location
            unseen = deck - hands[seat]  # what the five slots must hold

            assert torch.equal(weight[0], (unseen > 0).float())
            # Un-normalising by `unseen` must recover the true counts.
            recovered = target[0] * unseen.unsqueeze(-1)
            assert torch.allclose(recovered, truth, atol=1e-5), (
                seat,
                (recovered - truth).abs().max(),
            )

            in_trick = planes[0, PLANE_TRICK : PLANE_TRICK + NUM_REL_SEATS].sum()
            if in_trick > 0:
                following += 1
                open_trick_cards += int(in_trick)
            checked += 1

            moves = state.legal_moves(cap=64)
            state.step(moves.moves[int(rng.integers(len(moves.moves)))])

    assert checked > 400, checked
    # The reason the played slice spans planes 1..8. Counting only the
    # completed tricks would lose exactly these cards, on this many
    # decisions, and would renormalise the remaining slots to make up
    # the difference -- a wrong label, not just a missing one.
    assert following > checked // 2, "no follower positions; test is weak"
    print(
        f"    {checked} real decisions exact; {following} "
        f"({100 * following / checked:.0f}%) had an open trick holding "
        f"{open_trick_cards} cards the completed-trick planes omit"
    )


# --- 6. the first minibatch of the first epoch ----------------------------


def test_first_minibatch_is_a_no_op() -> None:
    """Stored logp from the current weights: ratio 1, no KL, no clipping.

    The gradient claim needs care. At ratio exactly 1 the *value* of the
    clipped surrogate is ``-mean(advantage)``, which normalisation makes
    zero, but its gradient is ``-mean(advantage * dlogp/dtheta)`` and
    that is emphatically not zero -- it is the policy gradient, which is
    the entire point. The gradient is zero exactly when the advantages
    are, so that is the case asserted here, with the nonzero case
    asserted alongside it so the test cannot pass by computing nothing.
    """
    buf = _FakeBuffer(16, seed=12)
    buf.advantage[:] = np.tile([1.0, -1.0], len(buf) // 2)
    buf.ret[:] = 0.25
    net = _net(2)
    learner = PPOLearner(net, PPOConfig(bf16=False), DEVICE)

    idx = np.arange(len(buf))
    _fill_logp(net, buf, idx)
    obs, privileged, scalars = buf.as_torch(idx, DEVICE)
    _, parts = learner.losses(obs, privileged, scalars)

    ratio = parts["ratio"]
    assert float((ratio - 1.0).abs().max()) < 1e-6, float((ratio - 1.0).abs().max())
    assert float(parts["approx_kl"]) < 1e-10, float(parts["approx_kl"])
    assert float(parts["clip_fraction"]) == 0.0
    # -mean(normalised advantage) is zero to fp error.
    assert abs(float(parts["policy_loss"])) < 1e-6, float(parts["policy_loss"])

    def policy_grad_norm() -> float:
        obs, privileged, scalars = buf.as_torch(idx, DEVICE)
        _, parts = learner.losses(obs, privileged, scalars)
        learner.opt.zero_grad(set_to_none=True)
        parts["policy_loss"].backward()
        return max(
            float(p.grad.abs().max())
            for p in net.parameters()
            if p.grad is not None
        )

    nonzero_advantage = policy_grad_norm()
    buf.advantage[:] = 0.0
    zero_advantage = policy_grad_norm()

    assert zero_advantage == 0.0, zero_advantage
    assert nonzero_advantage > 1e-8, "no policy gradient at all; test is vacuous"
    learner.opt.zero_grad(set_to_none=True)
    print(f"    |grad| {nonzero_advantage:.3e} with advantages, {zero_advantage} without")


# --- 7. a real update -----------------------------------------------------


def test_update_decreases_the_loss() -> None:
    """A trivially learnable batch: fixed returns, fixed advantage signs.

    The critic is regressing 24 fixed targets and the policy is being
    pushed up on half the rows and down on the other half, so both parts
    of the loss have somewhere to go. Behaviour log-probs are refreshed
    between updates, as a real loop would by collecting again; without
    that the clip pins the ratio at 1.2 after the first update and the
    policy stops moving for reasons that have nothing to do with the
    code under test.
    """
    rows = 24
    buf = _FakeBuffer(rows, seed=13)
    rng = np.random.default_rng(14)
    buf.advantage[:] = np.tile([1.0, -1.0], rows // 2).astype(np.float32)
    buf.ret[:] = rng.uniform(-1.0, 1.0, rows).astype(np.float32)
    buf.value[:] = 0.0

    net = _net(3)
    cfg = PPOConfig(lr=3e-3, epochs=2, minibatch=rows, ent_coef=0.0, bf16=False)
    learner = PPOLearner(net, cfg, DEVICE)
    idx = np.arange(rows)

    up = buf.advantage > 0
    logp_before = _chosen_logp(net, buf, idx)

    first: dict[str, float] = {}
    last: dict[str, float] = {}
    for step in range(30):
        _fill_logp(net, buf, idx)
        metrics = learner.update(buf)
        if step == 0:
            first = metrics
        last = metrics

    logp_after = _chosen_logp(net, buf, idx)
    delta = logp_after - logp_before

    for key, value in last.items():
        assert math.isfinite(value), f"{key} is {value}"
    assert set(
        [
            "policy_loss",
            "value_loss",
            "entropy",
            "belief_loss",
            "approx_kl",
            "clip_fraction",
            "grad_norm",
            "explained_variance",
            "lr",
            "n_minibatches",
        ]
    ) <= set(last), sorted(last)

    assert last["total_loss"] < first["total_loss"], (first, last)
    assert last["value_loss"] < 0.5 * first["value_loss"], (
        first["value_loss"],
        last["value_loss"],
    )
    assert last["belief_loss"] < first["belief_loss"]
    assert delta[up].mean() > 0.0, delta[up].mean()
    assert delta[~up].mean() < 0.0, delta[~up].mean()
    # 30 updates x 2 epochs x one whole-buffer minibatch each.
    assert learner.steps == 60
    assert last["n_minibatches"] == 2.0
    print(
        f"    total {first['total_loss']:.4f} -> {last['total_loss']:.4f}, "
        f"value {first['value_loss']:.4f} -> {last['value_loss']:.4f}, "
        f"dlogp up {delta[up].mean():+.3f} down {delta[~up].mean():+.3f}"
    )


def test_update_runs_under_bf16_autocast() -> None:
    """The default path. Different numbers, same structure, all finite."""
    rows = 20
    buf = _FakeBuffer(rows, seed=15)
    buf.episode[:] = np.arange(rows) // 8
    buf.seat[:] = np.arange(rows) % 4
    buf.reward[:] = 0.1
    buf.done[7] = buf.done[15] = buf.done[19] = True
    compute_gae(buf)

    net = _net(4)
    learner = PPOLearner(net, PPOConfig(minibatch=8, bf16=True), DEVICE)
    assert learner.amp, "bf16 autocast should be on for CPU"
    metrics = learner.update(buf)
    for key, value in metrics.items():
        assert math.isfinite(value), f"{key} is {value}"
    assert metrics["n_minibatches"] == 4.0 * math.ceil(rows / 8)
    assert metrics["grad_norm"] > 0.0
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    assert metrics["approx_kl"] >= 0.0
    assert metrics["lr"] == 3e-4
    print(f"    bf16 metrics: { {k: round(v, 4) for k, v in metrics.items()} }")


def test_explained_variance_is_the_real_thing() -> None:
    rows = 16
    buf = _FakeBuffer(rows, seed=16)
    buf.ret[:] = np.linspace(-1.0, 1.0, rows).astype(np.float32)
    buf.value[:] = buf.ret  # a perfect critic
    net = _net(5)
    learner = PPOLearner(net, PPOConfig(epochs=1, minibatch=rows, bf16=False), DEVICE)
    assert abs(learner.update(buf)["explained_variance"] - 1.0) < 1e-6

    buf.value[:] = buf.ret.mean()  # predicts the mean: zero skill
    assert abs(learner.update(buf)["explained_variance"]) < 1e-6
    assert math.isnan(explained_variance(np.zeros(4), np.ones(4)))


def test_target_kl_discards_the_offending_minibatch() -> None:
    """Early stop takes no step, so the trust region is never left."""
    rows = 12
    buf = _FakeBuffer(rows, seed=17)
    buf.advantage[:] = np.tile([1.0, -1.0], rows // 2)
    buf.ret[:] = 0.0
    net = _net(6)
    idx = np.arange(rows)
    _fill_logp(net, buf, idx)
    buf.logp -= 0.5  # a behaviour policy half a nat away: KL is real

    stopped = PPOLearner(
        net, PPOConfig(minibatch=rows, target_kl=1e-4, bf16=False), DEVICE
    )
    before = [p.detach().clone() for p in net.parameters()]
    metrics = stopped.update(buf)
    assert metrics["early_stop"] == 1.0
    assert metrics["n_minibatches"] == 0.0
    assert stopped.steps == 0
    for p, q in zip(net.parameters(), before):
        assert torch.equal(p.detach(), q), "a discarded minibatch still stepped"

    # Without the threshold the same buffer trains normally.
    free = PPOLearner(net, PPOConfig(minibatch=rows, bf16=False), DEVICE)
    metrics = free.update(buf)
    assert metrics["early_stop"] == 0.0
    assert metrics["n_minibatches"] == 4.0
    assert metrics["approx_kl"] > 1e-4


def test_illegal_stored_action_is_rejected() -> None:
    """An action off the end of the move list would otherwise train silently."""
    buf = _FakeBuffer(4, seed=18)
    net = _net(7)
    learner = PPOLearner(net, PPOConfig(bf16=False), DEVICE)
    idx = np.arange(len(buf))
    buf.action[2] = int(buf.candidate_mask[2].sum())  # first padded slot
    obs, privileged, scalars = buf.as_torch(idx, DEVICE)
    try:
        learner.losses(obs, privileged, scalars)
    except ValueError as exc:
        assert "illegal candidate slot" in str(exc), exc
    else:
        raise AssertionError("an illegal action slot was accepted")

    # The same row, made legal again, goes through.
    buf.action[2] = 0
    obs, privileged, scalars = buf.as_torch(idx, DEVICE)
    loss, _ = learner.losses(obs, privileged, scalars)
    assert torch.isfinite(loss)


def test_critic_trains_when_the_learner_asks_it_to() -> None:
    """``detach_critic=False`` during the update, or the critic never learns.

    The model detaches by default; if the learner inherits that default
    the value loss cannot reach the history encoder and the privileged
    critic is trained on a frozen trunk. Checked by taking the value
    loss alone and looking for a gradient on an actor parameter.
    """
    buf = _FakeBuffer(8, seed=19)
    buf.ret[:] = 0.5
    net = _net(8)
    learner = PPOLearner(net, PPOConfig(bf16=False), DEVICE)
    obs, privileged, scalars = buf.as_torch(np.arange(len(buf)), DEVICE)
    _, parts = learner.losses(obs, privileged, scalars)
    learner.opt.zero_grad(set_to_none=True)
    parts["value_loss"].backward()

    trunk = net.actor.history.in_proj.weight
    assert trunk.grad is not None and float(trunk.grad.abs().max()) > 0.0
    critic = net.critic.head[0].weight
    assert critic.grad is not None and float(critic.grad.abs().max()) > 0.0
    learner.opt.zero_grad(set_to_none=True)


# --- 8. every head is actually connected to the loss ----------------------


def _buffer_with_terminal_rows(seed: int) -> _FakeBuffer:
    """A batch that exercises every head: full history, a hand, an outcome.

    ``history_len`` is maxed and ``PLANE_HAND`` filled so that no part of
    the encoder or the card embedding is masked out of the graph for
    reasons that have nothing to do with the loss, and two rows carry a
    terminal level swing so the outcome head has a label to learn from.
    """
    buf = _FakeBuffer(8, seed=seed)
    rng = np.random.default_rng(seed)
    buf.history_len[:] = TINY.max_history
    buf.history[:] = rng.standard_normal(buf.history.shape).astype(np.float16)
    buf.planes[:, PLANE_HAND] = 1.0
    buf.advantage[:] = np.tile([1.0, -1.0], len(buf) // 2)
    buf.ret[:] = rng.uniform(-1.0, 1.0, len(buf)).astype(np.float32)
    buf.done[3] = buf.done[7] = True
    buf.reward_level[3] = 1.0  # a three-level win for the acting team
    buf.reward_level[7] = -2.0 / 3.0  # two levels to the other team
    return buf


def test_every_parameter_gets_a_gradient() -> None:
    """No head may be computed on every forward and trained by nothing.

    The actor's own value and outcome heads are 149,384 parameters --
    1.34% of the net -- hanging off the summary vector. For as long as
    the total loss mentioned neither of them, every forward paid for
    them and every backward left them at their initial values, and
    nothing said so: the loss was finite, the metrics were plausible,
    and ``test_gradients_reach_every_actor_parameter`` passed because it
    builds its own loss out of all four outputs rather than using the
    one the learner optimises.

    So the claim asserted here is about the learner's loss specifically:
    take ``losses`` as the learner runs it, back-propagate the total,
    and demand a finite gradient on every parameter of ``ActorCritic``.
    Reverting either auxiliary term leaves its head's parameters at
    ``grad is None`` and fails this outright.
    """
    buf = _buffer_with_terminal_rows(20)
    net = _net(9)
    learner = PPOLearner(net, PPOConfig(bf16=False), DEVICE)
    idx = np.arange(len(buf))
    _fill_logp(net, buf, idx)

    obs, privileged, scalars = buf.as_torch(idx, DEVICE)
    total, _parts = learner.losses(obs, privileged, scalars)
    learner.opt.zero_grad(set_to_none=True)
    total.backward()

    ungraded = [n for n, p in net.named_parameters() if p.grad is None]
    assert not ungraded, (
        f"{len(ungraded)} parameters got no gradient from the total loss: "
        f"{ungraded[:8]}"
    )
    nonfinite = [
        n for n, p in net.named_parameters() if not torch.isfinite(p.grad).all()
    ]
    assert not nonfinite, f"non-finite gradient at: {nonfinite[:8]}"

    # Stronger where it matters. A gradient of exactly zero is legal for
    # a parameter the batch happens not to exercise, but not for the two
    # heads this test exists for: their loss terms are unconditional
    # (own value) and have a labelled row in this batch (outcome).
    for head in ("actor.heads.value", "actor.heads.outcome"):
        params = [
            (n, p) for n, p in net.named_parameters() if n.startswith(head + ".")
        ]
        assert params, f"{head} has no parameters; the test is addressing nothing"
        for name, p in params:
            assert float(p.grad.abs().sum()) > 0.0, f"{name} got a zero gradient"

    learner.opt.zero_grad(set_to_none=True)
    counted = sum(
        p.numel()
        for n, p in net.named_parameters()
        if n.startswith(("actor.heads.value.", "actor.heads.outcome."))
    )
    print(
        f"    {len(list(net.parameters()))} tensors all graded; "
        f"{counted:,} of them in the two heads that used to get nothing"
    )


def test_outcome_class_is_the_signed_level_swing() -> None:
    """The label derivation, on values written out by hand.

    ``reward_level`` is ``(attacker_gain - defender_gain) / 3`` signed
    for the acting seat's team, so times three it is the swing and plus
    three it is a class index. Every one of the seven is checked, both
    as the exact third a float32 can hold and as the value the rollout
    actually stores.
    """
    exact = torch.tensor([-1.0, -2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3, 1.0])
    got = outcome_class(exact)
    assert got.tolist() == [0, 1, 2, 3, 4, 5, 6], got.tolist()
    assert got.dtype == torch.int64

    # The same values as the rollout produces them: an integer gain
    # divided by MAX_LEVEL_GAIN in float32, which is not the same bit
    # pattern as the literals above.
    for swing in range(-3, 4):
        stored = torch.tensor([float(swing)], dtype=torch.float32) / 3.0
        assert int(outcome_class(stored)) == swing + 3, (swing, stored)

    # Out of range cannot happen from ``hand_result``, but the function
    # is handed whole minibatches and must return a legal index for
    # every row of them rather than an index error deep in the loss.
    wild = outcome_class(torch.tensor([-5.0, 5.0, float("nan")]))
    assert wild.min() >= 0 and wild.max() < NUM_OUTCOMES, wild.tolist()
    print("    seven classes, exact and as stored, plus the clamp")


def test_outcome_head_is_trained_on_terminal_rows_only() -> None:
    """A non-terminal row has no label, and must not be given class 3.

    ``reward_level`` is 0.0 on every row but the one carrying ``done``,
    and 0.0 is a real outcome -- the hand ended level -- not a marker
    for "unknown". Letting those rows through would train the head that
    nineteen decisions in twenty end in a draw. The loss weights them
    out instead, which this pins down from both sides: with no terminal
    row the term is exactly zero and reports zero labelled rows, and
    flipping a single ``done`` flag makes it non-zero.
    """
    buf = _buffer_with_terminal_rows(21)
    net = _net(10)
    learner = PPOLearner(net, PPOConfig(bf16=False), DEVICE)
    idx = np.arange(len(buf))
    _fill_logp(net, buf, idx)

    buf.done[:] = False
    obs, privileged, scalars = buf.as_torch(idx, DEVICE)
    _, parts = learner.losses(obs, privileged, scalars)
    assert float(parts["outcome_loss_done"]) == 0.0, float(parts["outcome_loss_done"])
    assert float(parts["outcome_rows"]) == 0.0

    # Not vacuous: the rows are there, they simply carry no label. Give
    # one of them a label and the same batch produces a real loss.
    buf.done[3] = True
    obs, privileged, scalars = buf.as_torch(idx, DEVICE)
    _, parts = learner.losses(obs, privileged, scalars)
    labelled = float(parts["outcome_loss_done"])
    assert labelled > 0.0, labelled
    assert float(parts["outcome_rows"]) == 1.0

    # And the unlabelled rows are not quietly contributing class 3
    # underneath: rewriting every one of them to a different swing
    # leaves the loss bit-identical.
    buf.reward_level[~buf.done[: len(buf)]] = -1.0
    obs, privileged, scalars = buf.as_torch(idx, DEVICE)
    _, parts = learner.losses(obs, privileged, scalars)
    assert float(parts["outcome_loss_done"]) == labelled, (
        labelled,
        float(parts["outcome_loss_done"]),
    )
    print(f"    0 labelled rows -> 0.0; 1 labelled row -> {labelled:.4f}")


def test_an_aborted_update_does_not_report_zero_losses() -> None:
    """An update stopped by ``target_kl`` on its first minibatch.

    Nothing was measured, so nothing may be reported as a number. The
    trap is that zero is a value every one of these can legitimately
    take: an ``entropy`` of 0.0 is the dashboard signature of a
    collapsed policy, and an ``approx_kl`` of 0.0 is the exact opposite
    of what happened -- the update aborted *because* the KL was large.
    """
    rows = 12
    buf = _FakeBuffer(rows, seed=22)
    buf.advantage[:] = np.tile([1.0, -1.0], rows // 2)
    # Spread, not flat: a flat return makes ``explained_variance``
    # legitimately NaN, and this test needs to be able to say that every
    # metric of a *completed* update is finite.
    buf.ret[:] = np.linspace(-1.0, 1.0, rows).astype(np.float32)
    net = _net(11)
    idx = np.arange(rows)
    _fill_logp(net, buf, idx)
    buf.logp -= 0.5  # half a nat of behaviour drift: the KL is real

    learner = PPOLearner(
        net, PPOConfig(minibatch=rows, target_kl=1e-4, bf16=False), DEVICE
    )
    metrics = learner.update(buf)

    assert metrics["early_stop"] == 1.0
    assert metrics["n_minibatches"] == 0.0 and learner.steps == 0
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "belief_loss",
        "own_value_loss",
        "outcome_loss_done",
        "total_loss",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
    ):
        assert math.isnan(metrics[key]), f"{key} reported as {metrics[key]}"

    # The two that would be read as a result rather than as an absence.
    assert not metrics["approx_kl"] == 0.0
    assert not metrics["entropy"] == 0.0

    # Counts stay counts. "No rows were clamped" and "no labelled rows
    # were seen" are both true of an update that took no step, so NaN
    # there would be the mirror-image lie.
    assert metrics["ratio_clamped"] == 0.0
    assert metrics["outcome_rows"] == 0.0
    assert metrics["rows"] == float(rows)
    assert metrics["nonfinite_minibatches"] == 0.0

    # The same buffer without the threshold reports real numbers, so the
    # NaNs above are the abort and not a broken metrics path.
    ran = PPOLearner(net, PPOConfig(minibatch=rows, bf16=False), DEVICE).update(buf)
    assert all(math.isfinite(v) for v in ran.values()), ran
    assert ran["approx_kl"] > 1e-4
    print(f"    aborted approx_kl {metrics['approx_kl']}, ran {ran['approx_kl']:.4f}")


def test_a_huge_log_ratio_is_clamped_before_it_overflows() -> None:
    """``exp`` of an unbounded log-ratio is +inf, and inf ends the step.

    float32 ``exp`` overflows at 88.7. One row past that makes the
    surrogate mean infinite, the gradient non-finite, and the whole
    minibatch a write-off -- and it is reachable, because ``old_logp``
    comes from a policy several updates old and the new policy is free
    to put a candidate's probability anywhere.
    """
    assert not torch.isfinite(torch.tensor(200.0).exp()), "premise wrong"

    buf = _FakeBuffer(8, seed=23)
    buf.advantage[:] = np.tile([1.0, -1.0], len(buf) // 2)
    buf.ret[:] = 0.0
    net = _net(12)
    learner = PPOLearner(net, PPOConfig(bf16=False), DEVICE)
    idx = np.arange(len(buf))
    _fill_logp(net, buf, idx)
    buf.logp[:3] -= 200.0  # log ratio +200 on three rows

    obs, privileged, scalars = buf.as_torch(idx, DEVICE)
    loss, parts = learner.losses(obs, privileged, scalars)
    assert torch.isfinite(parts["ratio"]).all(), parts["ratio"]
    bounded = float(parts["ratio"].max())
    assert bounded <= math.exp(20.0) * (1.0 + 1e-6)
    assert float(parts["ratio_clamped"]) == 3.0
    assert torch.isfinite(loss), float(loss)
    assert math.isfinite(float(parts["approx_kl"]))

    learner.opt.zero_grad(set_to_none=True)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None
    ), "a clamped ratio still produced a non-finite gradient"
    learner.opt.zero_grad(set_to_none=True)

    # A legitimate ratio is untouched. The clip range is 1 +- 0.2, a log
    # ratio of +-0.182, so nothing PPO does in anger comes near +-20.
    buf.logp[:] = 0.0
    _fill_logp(net, buf, idx)
    buf.logp -= 0.18
    obs, privileged, scalars = buf.as_torch(idx, DEVICE)
    _, parts = learner.losses(obs, privileged, scalars)
    assert float(parts["ratio_clamped"]) == 0.0
    assert abs(float(parts["ratio"].mean()) - math.exp(0.18)) < 1e-4
    print(
        f"    3 rows at log-ratio 200 bounded to {bounded:.3e} instead of inf; "
        f"a ratio of {float(parts['ratio'].mean()):.3f} at the clip edge untouched"
    )


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        started = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - reporting harness
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}"[:1500])
        else:
            print(f"ok    {fn.__name__}  ({time.time() - started:.1f}s)")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


# --- 12. fitting a subset of the buffer -----------------------------------


def test_update_rows_restricts_the_fit_to_the_selection() -> None:
    """An excluded row is never read, so its poison never fires.

    ``check_actions`` raises on an action that indexes a padded slot, so
    an illegal action makes a sharp tracer: put one in a row, exclude
    that row, and an update that still touches it says so. Checking
    ``rows`` by looking at the weights instead would prove nothing --
    every row shares every parameter, so all of them move whichever
    subset was fitted.
    """
    buf = _FakeBuffer(12, seed=41)
    poisoned = 5
    buf.action[poisoned] = int(buf.candidate_mask[poisoned].sum())  # padded slot

    keep = np.array([i for i in range(len(buf)) if i != poisoned], dtype=np.int64)
    learner = PPOLearner(_net(11), PPOConfig(bf16=False, epochs=2, minibatch=4), DEVICE)
    metrics = learner.update(buf, rows=keep)
    assert metrics["rows"] == float(keep.size)

    # The same buffer, with the poisoned row back in the selection.
    learner = PPOLearner(_net(11), PPOConfig(bf16=False, epochs=2, minibatch=4), DEVICE)
    try:
        learner.update(buf, rows=np.arange(len(buf), dtype=np.int64))
    except ValueError as exc:
        assert "illegal candidate slot" in str(exc), exc
    else:
        raise AssertionError("the poisoned row was not read back")


def test_update_rows_of_everything_is_the_unrestricted_update() -> None:
    """``rows=arange(n)`` and ``rows=None`` are the same update.

    Both build the same selection and shuffle it with the same draw, so
    the minibatches are identical and every metric should agree to the
    bit. A difference here would mean the restricted path takes a
    different route through the buffer than the one every run so far
    has used.
    """

    def run(rows: np.ndarray | None) -> dict[str, float]:
        buf = _FakeBuffer(24, seed=42)
        learner = PPOLearner(
            _net(12), PPOConfig(bf16=False, epochs=2, minibatch=8), DEVICE
        )
        torch.manual_seed(99)
        return learner.update(buf, rows=rows)

    everything = run(None)
    explicit = run(np.arange(24, dtype=np.int64))
    assert set(everything) == set(explicit)
    for key, a in everything.items():
        b = explicit[key]
        if math.isnan(a) and math.isnan(b):
            continue
        assert a == b, (key, a, b)


def test_update_rows_outside_the_buffer_is_refused() -> None:
    """Reading an unwritten row would train on a zeroed observation."""
    buf = _FakeBuffer(8, seed=43)
    learner = PPOLearner(_net(13), PPOConfig(bf16=False, epochs=1), DEVICE)
    for bad in (np.array([0, 8], dtype=np.int64), np.array([-1, 2], dtype=np.int64)):
        try:
            learner.update(buf, rows=bad)
        except IndexError as exc:
            assert "outside" in str(exc), exc
        else:
            raise AssertionError(f"selection {bad} was accepted")

    try:
        learner.update(buf, rows=np.array([], dtype=np.int64))
    except ValueError as exc:
        assert "empty" in str(exc), exc
    else:
        raise AssertionError("an empty selection was accepted")


def test_detach_critic_severs_the_shared_trunk() -> None:
    """The flag's whole job: no value gradient reaches the actor's encoder.

    The companion of ``test_critic_trains_when_the_learner_asks_it_to``,
    which pins the default. Together they say the flag is wired to
    something rather than merely stored.
    """
    buf = _FakeBuffer(8, seed=19)
    buf.ret[:] = 0.5
    net = _net(8)
    learner = PPOLearner(net, PPOConfig(bf16=False, detach_critic=True), DEVICE)
    obs, privileged, scalars = buf.as_torch(np.arange(len(buf)), DEVICE)
    _, parts = learner.losses(obs, privileged, scalars)
    learner.opt.zero_grad(set_to_none=True)
    parts["value_loss"].backward()

    trunk = net.actor.history.in_proj.weight
    assert trunk.grad is None or float(trunk.grad.abs().max()) == 0.0, (
        "the value loss reached the shared encoder despite detach_critic"
    )
    # The critic itself still learns; detaching cuts the trunk, not the head.
    critic = net.critic.head[0].weight
    assert critic.grad is not None and float(critic.grad.abs().max()) > 0.0
    learner.opt.zero_grad(set_to_none=True)
