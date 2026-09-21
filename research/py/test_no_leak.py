"""Proves the policy cannot see the privileged block.

The centralised critic trains on the other seats' hands. If those bytes
ever reached the policy's features, every result would be worthless and
nothing would visibly break -- the agent would simply look brilliant in
training and fall apart against anyone. So this is checked rather than
asserted in a comment:

  1. featurise a real frame, then zero the privileged bytes and featurise
     again; the policy and history views must be bit-identical
  2. the privileged view must *change* when those bytes are zeroed, or
     step 1 proves nothing
  3. asking for the privileged view without the block must raise, rather
     than quietly returning zeros

    PYTHONPATH=py python py/test_no_leak.py
"""

from __future__ import annotations

import numpy as np

from tractor_rl import layout as L
from tractor_rl import features2 as F
from tractor_rl.protocol import WorkerConfig, WorkerPool


def one_frame(privileged: bool) -> F.Batch:
    pool = WorkerPool(
        1,
        WorkerConfig(games=24, seed=99, preplay="lowball", privileged=privileged),
    )
    try:
        # Enough rounds to fill the history window, and a guard that waits
        # for it. Full self-play advances every game by exactly one play per
        # round, so the frame read on round i carries min(i, HISTORY_STEPS)
        # history steps and a 16-slot window is only full on round 16 --
        # hence the bound. The guard used to be H_TRICK_INDEX > 0, which is
        # already true on round 4 and returned a frame with 12 of its 16
        # slots still empty: a leak confined to later history steps would
        # have gone unseen. Every game moves in lockstep, so the round that
        # fills the window fills it for every decision in the frame, not
        # just for the one `max` finds.
        for _ in range(L.HISTORY_STEPS + 1):
            batches = pool.read_round()
            batch = batches[0]
            picks = np.zeros(batch.num_decisions, dtype=np.int64)
            pool.write_round(batches, [picks])
            if batch.num_decisions > 0 and int(
                    F.field_u8(batch.states, L.H_HISTORY_LEN).max()) >= L.HISTORY_STEPS:
                return batch
        raise AssertionError("never reached a frame with a full history window")
    finally:
        pool.close()


def main() -> None:
    batch = one_frame(privileged=True)
    assert batch.has_privileged, "worker did not emit the privileged block"
    assert batch.states.shape[1] == L.STATE_BYTES_PRIVILEGED, batch.states.shape
    print(f"frame: {batch.num_decisions} decisions, {batch.states.shape[1]} bytes each")

    policy = F.state_features(batch.states)
    hist, mask = F.history_features(batch.states)
    actions = F.action_features(batch.cands, batch.states, batch.owner)
    priv = F.privileged_features(batch.states)
    beliefs = F.belief_targets(batch.states)

    # 2 (first, so step 1 cannot be vacuous): scrambling the block must move
    # the privileged view.
    base = L.HEADER_BYTES + L.PLANES_BYTES + L.HISTORY_BYTES
    scrubbed = batch.states.copy()
    scrubbed[:, base : base + L.PRIVILEGED_BYTES] = 0
    priv_after = F.privileged_features(scrubbed)
    assert not np.array_equal(priv, priv_after), "privileged view ignored its own block"
    assert priv.sum() > 0, "privileged view is empty -- nothing was being tested"
    print(f"ok   privileged view responds to its block (sum {priv.sum():.0f} -> {priv_after.sum():.0f})")

    # 1: the policy and history views must not move at all.
    policy_after = F.state_features(scrubbed)
    hist_after, mask_after = F.history_features(scrubbed)
    actions_after = F.action_features(batch.cands, scrubbed, batch.owner)
    assert np.array_equal(policy, policy_after), "POLICY VIEW READS PRIVILEGED BYTES"
    assert np.array_equal(hist, hist_after), "HISTORY VIEW READS PRIVILEGED BYTES"
    assert np.array_equal(mask, mask_after), "HISTORY MASK READS PRIVILEGED BYTES"
    assert np.array_equal(actions, actions_after), "ACTION VIEW READS PRIVILEGED BYTES"
    print("ok   policy, history and action views are bit-identical without the block")

    # 3: a run without the block must refuse, not silently return zeros.
    plain = one_frame(privileged=False)
    assert not plain.has_privileged
    assert plain.states.shape[1] == L.STATE_BYTES, plain.states.shape
    try:
        F.privileged_features(plain.states)
        raise AssertionError("privileged_features accepted a record without the block")
    except ValueError:
        print("ok   privileged view refuses records that lack the block")

    # Sanity on the supervision itself: a seat holds at most 25 cards, and
    # early in a hand it should hold a lot of them.
    per_seat = beliefs.reshape(beliefs.shape[0], 3, F.NUM_ENC_SLOTS).sum(axis=2)
    assert per_seat.max() <= 25, per_seat.max()
    print(f"ok   belief targets in range (distinct positions per opponent: "
          f"mean {per_seat.mean():.1f}, max {per_seat.max():.0f})")

    # And the policy view is actually populated.
    assert np.isfinite(policy).all() and policy.shape[1] == F.STATE_DIM
    assert np.isfinite(hist).all()
    print(f"ok   policy view {policy.shape}, history {hist.shape}, "
          f"mean history steps {mask.sum(axis=1).mean():.1f}")
    print("\nno-leak checks passed")


if __name__ == "__main__":
    main()
