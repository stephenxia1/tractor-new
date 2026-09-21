"""The torch featuriser must agree with the numpy one exactly.

Both derive the same quantities from the same bytes with integer-exact
arithmetic, so there is no tolerance to spend: a disagreement is a bug, and
one that would show up only as a slightly worse agent.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from tractor_rl import features2 as npf
from tractor_rl import features_torch as tf
from tractor_rl import layout as L
from tractor_rl.features2 import merge_batches
from tractor_rl.protocol import WorkerConfig, WorkerPool


def load_fixture():
    """Real records captured by py/make_fixture.py.

    A benchmark box has no game engine, and installing Node on a GPU box to
    generate records it uses once is not worth it. Synthetic bytes would not
    do -- the featuriser indexes by real card slots and trump contexts.
    """
    import numpy as _np
    from pathlib import Path as _Path
    q = _Path(__file__).resolve().parents[1] / "fixtures" / "records.npz"
    if not q.exists():
        raise SystemExit(f"missing {q}; run py/make_fixture.py where the engine runs")
    return _np.load(q)


def collect(rounds: int, workers: int = 3, games: int = 64):
    """Every round's decisions joined into one batch.

    Keeping the single largest round instead always kept round 0: with no
    search, games * workers is a hard ceiling on one round's decisions and
    every round hits it exactly, so a strict ">" never displaces the first.
    Round 0 is the one round where no card has been played yet -- its
    H_HISTORY_LEN is uniformly 0, both featurisers finish with
    `tokens *= mask`, and the token comparison collapses to two all-zero
    arrays agreeing with each other. The join spans the whole hand instead.
    """
    pool = WorkerPool(workers, WorkerConfig(games=games, seed=17, learner_seats=15,
                                            privileged=True), num_groups=1)
    states, cands, owners = [], [], []
    decisions = 0
    try:
        for _ in range(rounds):
            bs = pool.read_round()
            m = merge_batches(bs)
            if m.num_decisions:
                states.append(m.states)
                cands.append(m.cands)
                # Owners index into their own round's states, so each round's
                # have to move past every decision joined ahead of them.
                owners.append(m.owner + decisions)
                decisions += m.num_decisions
            pool.write_round(bs, [np.zeros(b.num_decisions, dtype=np.int64) for b in bs])
    finally:
        pool.close()
    if not states:
        return None
    return np.concatenate(states), np.concatenate(cands), np.concatenate(owners)


def main() -> int:
    ap = argparse.ArgumentParser()
    # Rounds 0..16. Full self-play advances every game by exactly one play
    # per round, so the frame collected on round i carries min(i, 16)
    # history steps and only round 16 fills the 16-slot window; anything
    # shorter leaves the tail of the window zeroed on both sides by
    # `tokens *= mask`. Four players, so it also takes until round 5 before
    # the remembered plays span more than trick 0.
    ap.add_argument("--rounds", type=int, default=L.HISTORY_STEPS + 1)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--fixture", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device)

    if args.fixture:
        z = load_fixture()
        st_np, cd_np, ow_np = z["states"], z["cands"], z["owner"]
        if "n_cand_states" not in z.files:
            # A fixture from before make_fixture kept every round's
            # candidates carries trick-0 leads only, and the two flag
            # assertions below would reject it a few lines later with a
            # message about collecting more rounds, which is not the
            # problem and not the fix. Say the real thing here instead.
            # `*.npz` is gitignored, so a checkout never regenerates one
            # and a stale fixture can sit on a machine indefinitely.
            raise SystemExit(
                f"fixtures/records.npz predates make_fixture keeping every "
                f"round's candidates, so it holds trick-0 leads only and cannot "
                f"exercise the led-suit or beats-winner action features; "
                f"regenerate it with `PYTHONPATH=py python py/make_fixture.py`"
            )
        n_cand = int(z["n_cand_states"])
        print(f"batch: {st_np.shape[0]} decisions ({n_cand} carrying candidates), "
              f"{cd_np.shape[0]} candidates (fixture), device {args.device}")
    else:
        collected = collect(args.rounds)
        if collected is None:
            raise SystemExit("no decisions collected")
        st_np, cd_np, ow_np = collected
        n_cand = st_np.shape[0]
        print(f"batch: {st_np.shape[0]} decisions, {cd_np.shape[0]} candidates, "
              f"device {args.device}")

    # A header byte that never varies makes every check reading it vacuous:
    # two featurisers agree on a constant for free. History length is the
    # worst of them, since both implementations end on `tokens *= mask` and
    # the mask comes from this byte -- a batch where it is uniformly 0
    # compares two all-zero token arrays and calls that a match.
    for name, offset in (("H_HISTORY_LEN", L.H_HISTORY_LEN), ("H_ROLE", L.H_ROLE)):
        assert np.unique(st_np[:, offset]).size > 1, (
            f"{name} is single-valued across this batch, so every check that "
            "reads it proves nothing; collect more rounds or regenerate the fixture")

    # "Not constant" is too weak for the history, because the mask zeroes
    # both sides above the batch's longest H_HISTORY_LEN: at the old
    # default of five rounds it reached step 4, and steps 4..15 were an
    # all-zero comparison on every row. This fires when the batch never
    # fills the window.
    deepest = int(st_np[:, L.H_HISTORY_LEN].max())
    assert deepest >= L.HISTORY_STEPS, (
        f"history reaches only step {deepest} of {L.HISTORY_STEPS}, so the "
        "rest of the window is zero on both sides and proves nothing; "
        "collect more rounds or regenerate the fixture")
    # And the trick index the tokens carry has to move. features2.py and
    # features_torch.py hold a `trick_idx / 25.0` literal each, so a batch
    # whose remembered plays all sit in trick 0 -- any batch that never
    # gets past the first four plays -- makes that channel identically
    # zero on both sides and lets the two literals drift apart unseen.
    # Read from the raw bytes, so this stays a statement about the batch
    # rather than about the featurisers it is meant to keep honest.
    hist_base = L.HEADER_BYTES + L.PLANES_BYTES
    steps = st_np[:, hist_base : hist_base + L.HISTORY_BYTES].reshape(
        st_np.shape[0], L.HISTORY_STEPS, L.HISTORY_STEP_BYTES)
    live = np.arange(L.HISTORY_STEPS)[None, :] < st_np[:, L.H_HISTORY_LEN][:, None]
    assert np.unique(steps[:, :, 2][live]).size > 1, (
        "every remembered play in this batch belongs to the same trick, so "
        "the token trick-index channel is single-valued; collect more rounds "
        "or regenerate the fixture")
    # Two of the action-feature flag columns can only be set away from the
    # opening play of a trick: candidates.ts leaves FLAG_ALL_LED_SUIT unset
    # when there is no led suit yet and FLAG_BEATS_WINNER unset when nobody
    # is winning yet, and both hold for every candidate of a lead. A
    # candidate set drawn from leads alone -- which is what a fixture
    # keeping only round 0 carries -- compares two all-zero columns.
    cand_flags = cd_np[:, 2].astype(np.int64)
    for name, bit in (("FLAG_ALL_LED_SUIT", L.FLAG_ALL_LED_SUIT),
                      ("FLAG_BEATS_WINNER", L.FLAG_BEATS_WINNER)):
        assert (cand_flags & bit).any(), (
            f"no candidate in this batch has {name} set, so its action-feature "
            "column is zero on both sides; collect more rounds, or regenerate "
            "the fixture with py/make_fixture.py -- one written before it kept "
            "every round's candidates holds trick-0 leads only")

    print()
    st_t = torch.from_numpy(np.ascontiguousarray(st_np)).to(dev)
    cd_t = torch.from_numpy(np.ascontiguousarray(cd_np)).to(dev)
    ow_t = torch.from_numpy(np.ascontiguousarray(ow_np)).to(dev)
    # A fixture regenerated since make_fixture started keeping every round's
    # candidates carries one for every state, so this slice is a no-op. It
    # still bites on an older one, whose candidates hang off the leading
    # block alone; the states it drops are checked by every other view.
    act_np, act_t = st_np[:n_cand], st_t[:n_cand]

    failures = 0

    def check(name, a: np.ndarray, b: torch.Tensor):
        nonlocal failures
        got = b.detach().cpu().numpy()
        if a.shape != got.shape:
            print(f"FAIL {name}: shape {a.shape} vs {got.shape}")
            failures += 1
            return
        bad = np.argwhere(a != got)
        if len(bad):
            failures += 1
            print(f"FAIL {name}: {len(bad)} of {a.size} differ")
            for idx in bad[:5]:
                k = tuple(int(i) for i in idx)
                print(f"     at {k}: numpy={a[k]!r} torch={got[k]!r}")
        else:
            print(f"ok   {name}: {a.size:,} values bit-identical")

    check("state_features", npf.state_features(st_np), tf.state_features(st_t))
    tok_np, msk_np = npf.history_features(st_np)
    tok_t, msk_t = tf.history_features(st_t)
    check("history tokens", tok_np, tok_t)
    check("history mask", msk_np, msk_t)
    check("action_features",
          npf.action_features(cd_np, act_np, ow_np),
          tf.action_features(cd_t, act_t, ow_t))
    priv_np = npf.privileged_features(st_np)
    priv_t = tf.privileged_features(st_t)
    check("privileged_features", priv_np, priv_t)
    check("belief_targets",
          npf.belief_targets(st_np, priv_np),
          tf.belief_targets(st_t, priv_t))

    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("torch featuriser matches numpy exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
