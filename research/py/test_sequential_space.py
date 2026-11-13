"""The sequential (card-by-card) action space, end to end through the wire.

    PYTHONPATH=py python py/test_sequential_space.py

Runs real workers with `--action-space sequential` under a random policy
and checks what the learner relies on:

* every record sent offers at least two children (one-option sub-steps are
  taken by the actor, never sent);
* the extended action features are the old 144 columns plus two, so a
  network trained on enumerated candidates reads exactly what it read
  before, and the two new columns are the complete flag and the cards a
  partial follow still needs;
* within a follow, every partial child needs exactly as many cards as the
  lead has, and complete children have that many;
* the sub-steps of one decision share (hand, trick, position in trick) and
  arrive in order, with the partial play growing every time -- the grouping
  the learner uses to give each sub-step weight 1/m.

When torch is importable it also checks that widening a network's action
inputs leaves its outputs unchanged, and that a weighted learner step runs.
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from tractor_rl import layout as L
from tractor_rl.features2 import (
    ACTION_DIM,
    ACTION_DIM_SEQ,
    action_features,
    field_u8,
    field_u16,
    field_u32,
    merge_batches,
    parse_episodes,
    split_per_worker,
)
from tractor_rl.protocol import WorkerConfig, WorkerPool

ROUNDS = 400


def check_wire() -> None:
    pool = WorkerPool(2, WorkerConfig(games=16, seed=5, action_space="sequential",
                                      preplay="smart", baseline="smart"))
    rng = np.random.default_rng(0)
    hands = 0
    decisions = 0
    group_sizes: Counter = Counter()
    # (worker, game) -> (decision key, smallest child size at the last sub-step, sub-steps so far)
    last: dict[tuple[int, int], tuple[tuple[int, int, int], int, int]] = {}
    try:
        for _ in range(ROUNDS):
            batches = pool.read_round()
            merged = merge_batches(batches)
            for b in batches:
                hands += len(parse_episodes(b.episodes))
            if merged.num_decisions == 0:
                pool.write_round(batches, [np.zeros(0, dtype=np.int64)] * len(batches))
                continue
            decisions += merged.num_decisions
            counts = merged.cand_counts
            assert (counts >= 2).all(), f"a record with {counts.min()} children was sent"

            old = action_features(merged.cands, merged.states, merged.owner, width=ACTION_DIM)
            new = action_features(merged.cands, merged.states, merged.owner, width=ACTION_DIM_SEQ)
            assert new.shape == (merged.cands.shape[0], ACTION_DIM_SEQ)
            assert np.array_equal(new[:, :ACTION_DIM], old), "extended features changed old columns"
            flags = merged.cands[:, 2].astype(np.int64)
            complete = (flags & L.FLAG_COMPLETE) != 0
            remaining = merged.cands[:, L.C_REMAINING].astype(np.int64)
            n_cards = merged.cands[:, 0].astype(np.int64)
            assert np.array_equal(new[:, ACTION_DIM], complete.astype(np.float32))
            assert np.allclose(new[:, ACTION_DIM + 1], remaining / 8.0)
            assert (remaining[complete] == 0).all(), "a complete child still needs cards"

            is_lead = field_u8(merged.states, L.H_IS_LEAD)
            game_ids = field_u16(merged.states, L.H_GAME_ID)
            hand_ids = field_u32(merged.states, L.H_HAND_ID)
            trick_idx = field_u8(merged.states, L.H_TRICK_INDEX)
            trick_pos = field_u8(merged.states, L.H_TRICK_POS)
            worker_of = np.searchsorted(merged.bounds[1:], np.arange(merged.num_decisions), side="right")
            for d in range(merged.num_decisions):
                rows = np.nonzero(merged.owner == d)[0]
                if not is_lead[d]:
                    partial = rows[~complete[rows]]
                    full = rows[complete[rows]]
                    sizes = set((n_cards[partial] + remaining[partial]).tolist()) | set(n_cards[full].tolist())
                    assert len(sizes) == 1, f"follow children disagree on the follow size: {sizes}"
                else:
                    assert (remaining[rows] == 0).all(), "a lead child carries a remaining count"
                slot = (int(worker_of[d]), int(game_ids[d]))
                key = (int(hand_ids[d]), int(trick_idx[d]), int(trick_pos[d]))
                smallest = int(n_cards[rows].min())
                prev = last.get(slot)
                if prev is not None and prev[0] == key:
                    # Same decision, next sub-step: the play under
                    # construction must have grown. A submit child keeps the
                    # prefix size, so compare against the extension children.
                    grow = int(n_cards[rows[~complete[rows]]].min()) if (~complete[rows]).any() else smallest
                    assert grow > prev[1] or smallest >= prev[1], (prev, key, smallest)
                    last[slot] = (key, smallest, prev[2] + 1)
                else:
                    if prev is not None:
                        group_sizes[prev[2]] += 1
                    last[slot] = (key, smallest, 1)
            picks = (rng.random(counts.shape[0]) * counts).astype(np.int64)
            pool.write_round(batches, split_per_worker(picks, merged.bounds))
    finally:
        pool.close()
    assert hands > 0, "no hand finished"
    total = sum(group_sizes.values())
    dist = {m: round(c / total, 3) for m, c in sorted(group_sizes.items())}
    print(f"ok  wire: {decisions} sub-step records, {hands} hands finished")
    print(f"    learned sub-steps per decision (random policy): {dist}")


def check_torch() -> None:
    try:
        import torch
    except ImportError:
        print("skip torch checks: torch not importable")
        return
    from tractor_rl.dmc2 import DMC2Config, Learner2, ReplayBuffer2
    from tractor_rl.model2 import NetConfig, PrivilegedCritic, TractorNet, widen_action_inputs
    from tractor_rl.features2 import HISTORY_STEPS, HISTORY_TOKEN_DIM, STATE_DIM

    torch.manual_seed(0)
    old = TractorNet(NetConfig(action_in=ACTION_DIM)).eval()
    new = TractorNet(NetConfig(action_in=ACTION_DIM_SEQ)).eval()
    new.load_state_dict(widen_action_inputs(old.state_dict(), ACTION_DIM_SEQ))
    b, k = 6, 5
    states = torch.randn(b, STATE_DIM)
    acts = torch.randn(b * k, ACTION_DIM)
    extra = torch.rand(b * k, 2)
    owner = torch.arange(b).repeat_interleave(k)
    roles = torch.randint(0, 3, (b,))
    tokens = torch.randn(b, HISTORY_STEPS, HISTORY_TOKEN_DIM)
    mask = torch.ones(b, HISTORY_STEPS)
    with torch.no_grad():
        q_old = old(states, acts, owner, roles, tokens, mask)["q"]
        q_new = new(states, torch.cat([acts, extra], 1), owner, roles, tokens, mask)["q"]
    # Zero-initialised columns ignore their inputs entirely.
    assert torch.allclose(q_old, q_new, atol=1e-6), (q_old - q_new).abs().max()
    critic = PrivilegedCritic(action_in=ACTION_DIM)
    PrivilegedCritic(action_in=ACTION_DIM_SEQ).load_state_dict(
        widen_action_inputs(critic.state_dict(), ACTION_DIM_SEQ))
    print("ok  widening the action inputs preserves the network's outputs")

    # A weighted learner step on real sequential records.
    pool = WorkerPool(1, WorkerConfig(games=8, seed=9, action_space="sequential",
                                      preplay="smart", baseline="smart", privileged=True))
    buf = ReplayBuffer2(4096, L.STATE_BYTES_PRIVILEGED)
    rng = np.random.default_rng(1)
    try:
        while buf.size < 600:
            batches = pool.read_round()
            merged = merge_batches(batches)
            if merged.num_decisions:
                picks = (rng.random(merged.cand_counts.shape[0]) * merged.cand_counts).astype(np.int64)
                chosen = merged.cands[merged.cand_offsets + picks]
                n = merged.num_decisions
                buf.add(merged.states, chosen, rng.normal(size=n).astype(np.float32),
                        rng.normal(size=n).astype(np.float32),
                        rng.choice([1.0, 0.5, 1 / 3], size=n).astype(np.float32))
                pool.write_round(batches, split_per_worker(picks, merged.bounds))
            else:
                pool.write_round(batches, [np.zeros(0, dtype=np.int64)] * len(batches))
    finally:
        pool.close()
    net = TractorNet(NetConfig(action_in=ACTION_DIM_SEQ))
    learner = Learner2(net, PrivilegedCritic(action_in=ACTION_DIM_SEQ),
                       DMC2Config(min_buffer=100, batch_size=64), torch.device("cpu"))
    for _ in range(3):
        stats = learner.step(buf, rng)
        assert stats is not None and all(np.isfinite(v) for v in stats.values()), stats
    print(f"ok  weighted learner step: loss {stats['loss']:.4f}")


def main() -> int:
    check_wire()
    check_torch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
