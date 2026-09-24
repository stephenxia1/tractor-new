"""Tests for the rollout buffer and the self-play actor.

The one that matters most is
``test_reward_points_account_for_every_captured_point``. It rebuilds each
collected hand from the buffer alone -- the four hands and the kitty out
of the privileged view, the trump context out of the one-hot block of the
context vector, the landlord out of the first row's seat -- replays the
recorded actions through the real simulator, and checks the stored reward
against the trick records that come back. Nothing in that path shares
code with the crediting it is testing, so a misattributed winner, an
inverted sign, a trick credited twice or a seat-to-team mapping off by
one all show up as a mismatch in points rather than as a slightly worse
training curve. The per-team totals are then checked against the points
each team actually captured, which is the statement the credit rule is
supposed to make.

``test_logp_matches_a_recomputed_log_softmax`` is tight -- 1e-5 -- and
that is only possible because the actor casts its observations to float16
*before* the forward it samples from. Sampling from float32 and storing
float16 would put a 1e-3 gap here, and that gap would be a silent bias in
every importance ratio a learner computed.

The remaining tests check the things that fail quietly: an action outside
the candidate mask, a save that does not come back byte for byte, a
``done`` flag on the wrong row, and a combined reward that is not exactly
its two terms.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tractor.nn.features import (
    CTX_LEVEL,
    CTX_TRUMP,
    MAX_HISTORY,
)
from tractor.nn.model import (
    MASK_LOGIT,
    NUM_BELIEF_LOCATIONS,
    NUM_OUTCOMES,
    ActorCritic,
    ModelConfig,
)
from tractor.rules.cards import (
    NUM_RANKS,
    NUM_TRUMP_CHOICES,
    NUM_TYPES,
    Ctx,
    total_points,
)
from tractor.sim.state import NUM_SEATS, PlayState, hand_result, team_of
from tractor.train import buffer as buffer_module
from tractor.train.buffer import OBS_FIELDS, SCALAR_FIELDS, RolloutBuffer
from tractor.train.rollout import DECKS, MAX_LEVEL_GAIN, collect
from tractor.telemetry import Spans

TOTAL_POINTS = 200
HANDS = 3
CAPACITY = 512
SEED = 20260921

# Small enough to run a few hundred forwards in seconds, but the padded
# extents have to stay at the real ones: the encoder always emits 128
# history tokens and the buffer stores 128 candidate slots.
TINY = ModelConfig(
    d_model=64,
    n_layers_history=1,
    n_layers_query=1,
    n_heads=2,
    d_ff=128,
    critic_d=32,
    critic_layers=1,
    critic_heads=2,
    critic_d_ff=64,
    belief_d=32,
    scalar_d=16,
    head_d=32,
)

_CACHED: tuple[ActorCritic, RolloutBuffer, dict] | None = None


def _net() -> ActorCritic:
    torch.manual_seed(SEED)
    return ActorCritic(TINY).eval()


def _collected() -> tuple[ActorCritic, RolloutBuffer, dict]:
    """One rollout, shared by every test that only reads it."""
    global _CACHED
    if _CACHED is None:
        net = _net()
        buffer = RolloutBuffer(CAPACITY, TINY)
        stats = collect(
            net,
            np.random.default_rng(SEED),
            buffer,
            n_hands=HANDS,
            device="cpu",
            cfg=TINY,
            pool=HANDS,
        )
        _CACHED = (net, buffer, stats)
    return _CACHED


def _episode_rows(buffer: RolloutBuffer, episode: int) -> np.ndarray:
    """Rows of one hand, in the order they were decided."""
    return np.nonzero(buffer.episode[: len(buffer)] == episode)[0]


def _replay(buffer: RolloutBuffer, episode: int):
    """Rebuild a hand from its rows and play it through the simulator.

    Returns the finished state, the hand's rows in decision order, and
    one ``(points, winner, rows)`` entry per trick. Everything needed to
    reconstruct the starting position is already in the buffer: the
    privileged view of the first decision holds the four hands in
    relative order plus the kitty, the context vector holds the level
    rank and the trump suit as one-hots, and the landlord is whoever
    made the first decision, because the landlord leads.
    """
    rows = _episode_rows(buffer, episode)
    first = int(rows[0])
    landlord = int(buffer.seat[first])

    context = buffer.context[first].astype(np.float32)
    level_rank = int(np.argmax(context[CTX_LEVEL : CTX_LEVEL + NUM_RANKS]))
    trump_suit = int(
        np.argmax(context[CTX_TRUMP : CTX_TRUMP + NUM_TRUMP_CHOICES])
    )

    view = buffer.privileged[first].astype(np.int8)
    hands = np.zeros((NUM_SEATS, NUM_TYPES), dtype=np.int8)
    for seat in range(NUM_SEATS):
        hands[seat] = view[(seat - landlord) % NUM_SEATS]
    kitty = view[NUM_SEATS].copy()

    state = PlayState.new(
        Ctx(level_rank, trump_suit), hands, kitty, landlord, DECKS
    )
    tricks: list[tuple[int, int, list[int]]] = []
    open_rows: list[int] = []
    for row in rows:
        row = int(row)
        assert state.to_act == int(buffer.seat[row]), (
            f"episode {episode} row {row}: replay has seat {state.to_act} "
            f"to act, buffer recorded {int(buffer.seat[row])}"
        )
        cards = buffer.candidates[row, int(buffer.action[row])].astype(np.int8)
        before = len(state.tricks)
        state.step(cards)
        open_rows.append(row)
        if len(state.tricks) > before:
            trick = state.tricks[-1]
            tricks.append((int(trick.points), int(trick.winner), open_rows))
            open_rows = []
    assert state.done, f"episode {episode} did not finish on replay"
    assert not open_rows, "the hand ended with a trick still open"
    return state, rows, tricks


# --- the buffer -----------------------------------------------------------


def test_buffer_roundtrips_through_npz() -> None:
    _net, buffer, _stats = _collected()
    # The learner writes these two; filling them here means the round
    # trip covers every field rather than only the actor's.
    rng = np.random.default_rng(1)
    n = len(buffer)
    buffer.advantage[:n] = rng.standard_normal(n, dtype=np.float32)
    buffer.ret[:n] = rng.standard_normal(n, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = buffer.save_npz(Path(tmp) / "round")
        assert path.exists()
        back = RolloutBuffer.load_npz(path, TINY)

    assert back.capacity == buffer.capacity
    assert len(back) == len(buffer)
    for name in buffer.fields:
        mine = buffer.array(name)
        theirs = back.array(name)
        assert theirs.dtype == mine.dtype, f"{name} came back as {theirs.dtype}"
        assert theirs.shape == mine.shape, f"{name} came back as {theirs.shape}"
        # Whole array, not only the filled prefix: the unwritten tail is
        # zeros on both sides and a save that quietly kept junk there
        # would show up here.
        np.testing.assert_array_equal(theirs, mine, f"{name} did not round-trip")

    fields = set(buffer.fields)
    assert fields == set(OBS_FIELDS) | {"privileged"} | set(SCALAR_FIELDS)
    print(f"    {len(fields)} fields, {len(buffer)} rows, bit-identical")


def test_buffer_refuses_to_overfill() -> None:
    small = RolloutBuffer(2, TINY)
    _net, source, _stats = _collected()
    row = {
        name: source.array(name)[0]
        for name in OBS_FIELDS + ("privileged",)
        if name != "history_len"
    }
    kwargs = dict(
        row,
        history_len=int(source.history_len[0]),
        action=int(source.action[0]),
        logp=float(source.logp[0]),
        value=float(source.value[0]),
        seat=0,
        episode=0,
    )
    assert small.add(**kwargs) == 0
    assert not small.full
    assert small.add(**kwargs) == 1
    assert small.full
    try:
        small.add(**kwargs)
    except RuntimeError as exc:
        assert "full" in str(exc)
    else:
        raise AssertionError("a full buffer accepted another row")

    illegal = dict(kwargs)
    illegal["action"] = int(np.nonzero(~source.candidate_mask[0])[0][0])
    try:
        RolloutBuffer(2, TINY).add(**illegal)
    except ValueError as exc:
        assert "legal candidate" in str(exc)
    else:
        raise AssertionError("an action outside the mask was accepted")

    small.reset()
    assert len(small) == 0 and not small.full
    assert not small.candidate_mask.any(), "reset left a stale tail"
    print("    full, illegal-action and reset paths all behave")


def _one_row_kwargs() -> dict:
    """``add`` arguments for one real collected row."""
    _net_, source, _stats = _collected()
    row = {
        name: source.array(name)[0]
        for name in OBS_FIELDS + ("privileged",)
        if name != "history_len"
    }
    return dict(
        row,
        history_len=int(source.history_len[0]),
        action=int(source.action[0]),
        logp=float(source.logp[0]),
        value=float(source.value[0]),
        seat=0,
        episode=0,
    )


def test_buffer_rejects_a_nonfinite_scalar() -> None:
    """A NaN reward is a NaN advantage on every earlier row of the hand.

    The reduced fields are the ones the GAE recursion runs over, so a
    single non-finite value does not stay on its own row: it propagates
    backwards through the whole trajectory and arrives at the optimiser
    as a NaN gradient. The learner's own guard catches it a minibatch at
    a time and cannot say which row was responsible. Here it can.
    """
    base = _one_row_kwargs()
    fields = ("logp", "value", "reward", "reward_points", "reward_level")
    for field in fields:
        for bad in (float("nan"), float("inf"), float("-inf")):
            kwargs = dict(base)
            kwargs[field] = bad
            buffer = RolloutBuffer(2, TINY)
            try:
                buffer.add(**kwargs)
            except ValueError as exc:
                assert field in str(exc), f"{field}={bad} rejected as {exc}"
            else:
                raise AssertionError(f"{field}={bad} was accepted")
            assert len(buffer) == 0, f"{field}={bad} was rejected but still stored"

    # The same row with every field finite goes in, so the check is
    # rejecting the value and not the shape of the call.
    clean = RolloutBuffer(2, TINY)
    assert clean.add(**base) == 0
    print(f"    {len(fields)} reduced fields x 3 non-finite values, all refused")


def test_load_npz_rejects_a_corrupted_file() -> None:
    """The crash-recovery path is the one nobody is watching.

    ``load_npz`` checks dtype, shape and capacity, all of which a
    truncated write or a flipped bit preserves. What it has to check is
    the two things ``add`` checks per row and a file bypasses entirely:
    that every action indexes a legal slot, and that no float is
    non-finite. Both corruptions here survive every structural check.
    """
    _net_, source, _stats = _collected()
    n = min(8, len(source))

    def saved(mutate) -> Path:
        buffer = RolloutBuffer(n, TINY)
        for row in range(n):
            buffer.add(
                **{
                    name: source.array(name)[row]
                    for name in OBS_FIELDS + ("privileged",)
                    if name != "history_len"
                },
                history_len=int(source.history_len[row]),
                action=int(source.action[row]),
                logp=float(source.logp[row]),
                value=float(source.value[row]),
                seat=int(source.seat[row]),
                episode=int(source.episode[row]),
            )
        mutate(buffer)
        return buffer.save_npz(Path(tmp) / f"corrupt{next(counter)}")

    counter = iter(range(100))
    with tempfile.TemporaryDirectory() as tmp:
        # An intact save comes back, so the checks are not simply
        # refusing everything.
        good = saved(lambda _buffer: None)
        assert len(RolloutBuffer.load_npz(good, TINY)) == n

        def point_at_padding(buffer: RolloutBuffer) -> None:
            padded = np.nonzero(~buffer.candidate_mask[3])[0]
            assert padded.size, "row 3 has no padded slot to point at"
            buffer.action[3] = int(padded[0])

        def poison_logp(buffer: RolloutBuffer) -> None:
            buffer.logp[5] = np.nan

        def poison_history(buffer: RolloutBuffer) -> None:
            buffer.history[2, 0, 0] = np.float16("inf")

        cases = (
            (point_at_padding, "legal candidate slot", "3"),
            (poison_logp, "logp", "5"),
            (poison_history, "history", "2"),
        )
        for mutate, wanted, row in cases:
            path = saved(mutate)
            try:
                RolloutBuffer.load_npz(path, TINY)
            except ValueError as exc:
                message = str(exc)
                assert wanted in message, message
                assert f"row {row}" in message, message
            else:
                raise AssertionError(f"a file corrupted by {mutate.__name__} loaded")

        # The finiteness scan runs in row blocks so its working set does
        # not scale with the buffer, which means the reported row is a
        # block offset plus a within-block index. Shrinking the block to
        # two rows puts the poisoned row in the fourth block and pins
        # that arithmetic; at the real block size every test above lands
        # in the first one and an offset bug would be invisible.
        poisoned = saved(poison_logp)
        original = buffer_module._VALIDATE_ROWS
        buffer_module._VALIDATE_ROWS = 2
        try:
            RolloutBuffer.load_npz(poisoned, TINY)
        except ValueError as exc:
            assert "row 5" in str(exc), str(exc)
        else:
            raise AssertionError("a poisoned row survived a chunked scan")
        finally:
            buffer_module._VALIDATE_ROWS = original

    print(f"    {len(cases)} corruptions refused, each naming its row and field")


def test_as_torch_feeds_the_real_network() -> None:
    net, buffer, _stats = _collected()
    rows = np.arange(0, min(24, len(buffer)))
    obs, privileged, scalars = buffer.as_torch(rows, "cpu")

    assert obs._fields == OBS_FIELDS
    assert obs.history.shape == (len(rows), MAX_HISTORY, obs.history.shape[2])
    assert obs.history.dtype == torch.float32
    assert obs.candidate_mask.dtype == torch.bool
    assert obs.history_len.dtype == torch.int64
    assert privileged.shape == (len(rows), 5, NUM_TYPES)
    assert privileged.dtype == torch.float32
    for name in SCALAR_FIELDS:
        assert name in scalars, f"{name} missing from the scalars dict"
        assert scalars[name].shape == (len(rows),)

    with torch.no_grad():
        out, value = net(obs, privileged)

    assert out.logits.shape == (len(rows), TINY.max_candidates)
    assert out.log_probs.shape == (len(rows), TINY.max_candidates)
    assert out.value.shape == (len(rows),)
    assert out.outcome_logits.shape == (len(rows), NUM_OUTCOMES)
    assert out.belief_logits.shape == (len(rows), NUM_TYPES, NUM_BELIEF_LOCATIONS)
    assert value.shape == (len(rows),)
    assert torch.isfinite(out.log_probs).all(), "log_probs are not all finite"
    assert torch.isfinite(value).all()

    probs = out.log_probs.exp().sum(dim=-1)
    torch.testing.assert_close(probs, torch.ones_like(probs), atol=1e-5, rtol=0)
    # The padding must carry no probability at all, not merely little.
    masked = out.log_probs[~obs.candidate_mask].exp()
    assert float(masked.max()) == 0.0, "a padded slot has non-zero probability"
    print(f"    forward on {len(rows)} stored rows, every head the right shape")


# --- the rollout ----------------------------------------------------------


def test_collected_actions_are_legal() -> None:
    _net, buffer, _stats = _collected()
    n = len(buffer)
    rows = np.arange(n)
    actions = buffer.action[:n]
    assert actions.min() >= 0 and actions.max() < TINY.max_candidates
    chosen = buffer.candidate_mask[rows, actions]
    assert chosen.all(), (
        f"{int((~chosen).sum())} of {n} decisions chose a slot outside the "
        f"candidate mask"
    )
    # Every chosen slot holds a real play, and a legal one never plays a
    # card the seat does not hold: the privileged view's own row is that
    # seat's hand.
    played = buffer.candidates[rows, actions].astype(np.int16)
    own_hand = buffer.privileged[:n, 0].astype(np.int16)
    assert (played <= own_hand).all(), "a decision played a card it did not hold"
    assert (played.sum(axis=1) > 0).all(), "a decision played nothing"
    print(f"    {n} decisions, all inside the mask and inside the hand")


def test_logp_matches_a_recomputed_log_softmax() -> None:
    net, buffer, _stats = _collected()
    rows = np.arange(len(buffer))
    obs, _privileged, scalars = buffer.as_torch(rows, "cpu")
    with torch.no_grad():
        out = net.actor(obs)

    masked = torch.where(
        obs.candidate_mask, out.logits, torch.full_like(out.logits, MASK_LOGIT)
    )
    log_probs = F.log_softmax(masked, dim=-1)
    got = log_probs[torch.arange(len(rows)), scalars["action"]].numpy()
    error = np.abs(got - buffer.logp[rows])
    assert error.max() < 1e-5, f"recorded log-probs drifted by {error.max():.2e}"
    assert np.isfinite(buffer.logp[rows]).all()
    assert (buffer.logp[rows] <= 0.0).all(), "a log-probability above zero"
    print(f"    max |recorded - recomputed| = {error.max():.2e} over {len(rows)} rows")


def test_reward_points_account_for_every_captured_point() -> None:
    """The credit rule, checked against a replay through the simulator."""
    _net, buffer, stats = _collected()
    checked = 0
    for episode in range(int(stats["hands"])):
        state, rows, tricks = _replay(buffer, episode)
        where = {int(row): i for i, row in enumerate(rows)}

        # The replay itself has to be sound before its trick records can
        # be used as the reference.
        trick_points = sum(points for points, _w, _r in tricks)
        assert trick_points + total_points(state.kitty) == TOTAL_POINTS, (
            f"episode {episode}: {trick_points} in tricks + "
            f"{total_points(state.kitty)} in the kitty"
        )
        assert len(rows) == NUM_SEATS * len(tricks)

        expected_points = np.zeros(len(rows), dtype=np.float64)
        expected_level = np.zeros(len(rows), dtype=np.float64)
        captured = {0: 0, 1: 0}
        # The kitty bonus rides on the last trick: nonzero only when the
        # attackers took it, and computed here from the replay's own
        # totals rather than from anything the crediting code touched.
        bonus = state.attacker_points - sum(
            p for p, w, _r in tricks if team_of(w) != state.defenders
        )
        assert bonus >= 0
        if bonus:
            assert team_of(tricks[-1][1]) != state.defenders
        tricks = [
            (p + (bonus if i == len(tricks) - 1 else 0), w, r)
            for i, (p, w, r) in enumerate(tricks)
        ]
        for points, winner, trick_rows in tricks:
            assert len(trick_rows) == NUM_SEATS
            captured[team_of(winner)] += points
            for row in trick_rows:
                seat = int(buffer.seat[row])
                sign = 1.0 if team_of(seat) == team_of(winner) else -1.0
                expected_points[where[row]] = sign * points / (2.0 * TOTAL_POINTS)

        defender_gain, attacker_gain = hand_result(state.attacker_points)
        swing = (attacker_gain - defender_gain) / MAX_LEVEL_GAIN
        for seat in range(NUM_SEATS):
            seat_rows = [r for r in rows if int(buffer.seat[r]) == seat]
            assert seat_rows, f"seat {seat} never acted"
            last = int(seat_rows[-1])
            sign = -1.0 if team_of(seat) == state.defenders else 1.0
            expected_level[where[last]] = sign * swing

        np.testing.assert_allclose(
            buffer.reward_points[rows],
            expected_points,
            atol=1e-7,
            err_msg=f"episode {episode}: trick credit does not match the replay",
        )
        np.testing.assert_allclose(
            buffer.reward_level[rows],
            expected_level,
            atol=1e-7,
            err_msg=f"episode {episode}: terminal credit does not match",
        )

        # The statement the rule is meant to make: a team's positive
        # trick credits, in points, are the points that team captured.
        for team in (0, 1):
            won_rows = [
                where[row]
                for points, winner, trick_rows in tricks
                for row in trick_rows
                if team_of(winner) == team and team_of(int(buffer.seat[row])) == team
            ]
            gained = buffer.reward_points[rows][won_rows].astype(np.float64).sum()
            assert abs(gained * TOTAL_POINTS - captured[team]) < 1e-3, (
                f"episode {episode} team {team}: credited "
                f"{gained * TOTAL_POINTS:.4f} points, captured {captured[team]}"
            )
            mine = [
                where[int(row)]
                for row in rows
                if team_of(int(buffer.seat[row])) == team
            ]
            net_credit = buffer.reward_points[rows][mine].astype(np.float64).sum()
            expected_net = (captured[team] - captured[1 - team]) / TOTAL_POINTS
            assert abs(net_credit - expected_net) < 1e-5, (
                f"episode {episode} team {team}: net credit {net_credit:.6f} "
                f"against a point differential of {expected_net:.6f}"
            )
        checked += 1
        print(
            f"    episode {episode}: {len(tricks)} tricks, captured "
            f"{captured[0]}/{captured[1]}, attacker points "
            f"{state.attacker_points}, levels {(defender_gain, attacker_gain)}"
        )
    assert checked == HANDS


def test_reward_is_exactly_its_two_terms() -> None:
    _net, buffer, _stats = _collected()
    n = len(buffer)
    combined = buffer.reward_points[:n] + buffer.reward_level[:n]
    np.testing.assert_array_equal(
        buffer.reward[:n],
        combined,
        "reward is not exactly reward_points + reward_level at weight 1",
    )
    assert (buffer.reward_level[:n][~buffer.done[:n]] == 0.0).all(), (
        "the terminal term landed on a row that is not a seat's last"
    )
    assert np.abs(buffer.reward_level[:n][buffer.done[:n]]).max() <= 1.0
    print(
        f"    {n} rows, level term only on the {int(buffer.done[:n].sum())} "
        f"done rows"
    )


def test_level_weight_zero_drops_the_terminal_term() -> None:
    """The anneal knob moves the combined reward and nothing else."""
    weighted = RolloutBuffer(CAPACITY, TINY)
    collect(
        _net(),
        np.random.default_rng(SEED + 5),
        weighted,
        n_hands=2,
        device="cpu",
        cfg=TINY,
        pool=2,
    )
    plain = RolloutBuffer(CAPACITY, TINY)
    collect(
        _net(),
        np.random.default_rng(SEED + 5),
        plain,
        n_hands=2,
        device="cpu",
        cfg=TINY,
        pool=2,
        level_weight=0.0,
    )

    n = len(weighted)
    assert len(plain) == n, "the weight changed the trajectory"
    np.testing.assert_array_equal(plain.action[:n], weighted.action[:n])
    np.testing.assert_array_equal(
        plain.reward_points[:n],
        weighted.reward_points[:n],
        "level_weight disturbed the point reward",
    )
    np.testing.assert_array_equal(
        plain.reward_level[:n],
        weighted.reward_level[:n],
        "level_weight changed the stored terminal term instead of its weight",
    )
    np.testing.assert_array_equal(
        plain.reward[:n],
        plain.reward_points[:n],
        "at weight 0 the combined reward is not the point term alone",
    )
    # Non-vacuous: at weight 1 the two disagree exactly on the done rows.
    differs = plain.reward[:n] != weighted.reward[:n]
    assert differs.any(), "the terminal term was zero, so nothing was tested"
    np.testing.assert_array_equal(differs, weighted.reward_level[:n] != 0.0)
    print(f"    {int(differs.sum())} rows carry a terminal term; only they moved")


def test_done_is_one_row_per_episode_and_seat() -> None:
    _net, buffer, stats = _collected()
    n = len(buffer)
    hands = int(stats["hands"])
    assert int(buffer.done[:n].sum()) == hands * NUM_SEATS

    for episode in range(hands):
        rows = _episode_rows(buffer, episode)
        for seat in range(NUM_SEATS):
            mine = rows[buffer.seat[rows] == seat]
            flagged = mine[buffer.done[mine]]
            assert len(flagged) == 1, (
                f"episode {episode} seat {seat}: {len(flagged)} done flags"
            )
            assert flagged[0] == mine[-1], (
                f"episode {episode} seat {seat}: done is on row {flagged[0]}, "
                f"not on the seat's last decision {mine[-1]}"
            )
    print(f"    {hands} hands x {NUM_SEATS} seats, one done flag each")


def test_collect_fills_the_buffer_consistently() -> None:
    spans = Spans(enabled=True, device="cpu")
    net = _net()
    buffer = RolloutBuffer(CAPACITY, TINY)
    stats = collect(
        net,
        np.random.default_rng(SEED + 1),
        buffer,
        n_hands=HANDS,
        device="cpu",
        cfg=TINY,
        pool=HANDS,
        spans=spans,
    )

    n = len(buffer)
    assert stats["hands"] == HANDS
    assert stats["decisions"] == n
    assert n > 0 and not buffer.full
    # The point of the pool: far fewer forwards than decisions.
    assert stats["forwards"] < stats["decisions"]
    assert stats["mean_batch"] > 1.0
    assert stats["candidates_per_decision"] >= 1.0
    assert 0.0 <= stats["forced_fraction"] <= 1.0
    assert stats["seconds"] > 0.0
    for name in ("encode", "forward", "sample", "step"):
        assert name in spans.summary(), f"span {name} was never recorded"

    assert set(np.unique(buffer.episode[:n])) == set(range(HANDS))
    assert buffer.seat[:n].min() >= 0 and buffer.seat[:n].max() < NUM_SEATS
    assert np.isfinite(buffer.value[:n]).all()
    assert buffer.candidate_mask[:n].any(axis=1).all()
    assert (buffer.history_len[:n] >= 0).all()
    assert (buffer.history_len[:n] <= MAX_HISTORY).all()
    # Unwritten rows stay exactly as allocated.
    assert not buffer.candidate_mask[n:].any()
    assert buffer.reward[n:].sum() == 0.0

    for episode in range(HANDS):
        rows = _episode_rows(buffer, episode)
        assert len(rows) % NUM_SEATS == 0
        lengths = buffer.history_len[rows]
        assert (np.diff(lengths) >= 0).all(), "history shrank inside a hand"
        for start in range(0, len(rows), NUM_SEATS):
            seats = buffer.seat[rows[start : start + NUM_SEATS]]
            assert len(set(seats.tolist())) == NUM_SEATS, (
                f"episode {episode}: a trick did not have all four seats"
            )
            steps = (np.diff(seats) % NUM_SEATS)
            assert (steps == 1).all(), "seats did not go round the table"

    # A second collection into the same buffer must not reuse episode ids.
    more = collect(
        net,
        np.random.default_rng(SEED + 2),
        buffer,
        n_hands=1,
        device="cpu",
        cfg=TINY,
        pool=1,
    )
    assert more["hands"] == 1
    assert set(np.unique(buffer.episode[: len(buffer)])) == set(range(HANDS + 1))
    print(
        f"    {stats['hands']} hands, {stats['decisions']} decisions, "
        f"{stats['forwards']} forwards (mean batch {stats['mean_batch']:.2f}), "
        f"{stats['candidates_per_decision']:.1f} candidates/decision, "
        f"{stats['forced_fraction'] * 100:.1f}% forced, "
        f"{stats['seconds']:.1f}s"
    )


def main() -> int:
    import time

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
