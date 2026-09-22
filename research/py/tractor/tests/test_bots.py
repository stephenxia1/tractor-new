"""The bridge between the Python network and the TypeScript bots.

Four of these tests are about the bridge rather than about any policy,
because a measurement harness that is wrong does not announce itself: it
returns a plausible margin with a plausible error bar, and the margin is
wrong by a sign or by a factor.

The one that matters most is ``test_wire_round_trip_is_bit_identical``.
The whole design rests on a claim -- that the documented wire fields are
enough to rebuild a position the encoder cannot tell from the original
-- and if that claim were false the network would be scored on
observations it was never trained on, at every decision, invisibly. So
it is checked at every decision of several real hands, from all four
seats, array by array and dtype by dtype.

The rest cover the seams. The protocol is driven against a canned host
so a failure is the client's; the child is then killed mid-stream, which
must produce an error rather than a wait. The digest check is exercised
with two halves that disagree. The statistics are checked against a
score worked out by hand, in both directions, because a sign error here
would read as a policy result.

    python -m tractor.tests.test_bots
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tractor.eval.bots import (
    HostError,
    NET_TEAM,
    _Half,
    play_against_bot,
    state_from_wire,
    summarise,
)
from tractor.eval.duplicate import Z95
from tractor.gen.episode import generate
from tractor.nn.features import Cache, encode
from tractor.nn.model import ActorCritic, ModelConfig
from tractor.sim.state import NUM_SEATS, PlayState

DECKS = 2
SEED = 20260922
# Hands walked end to end by the round-trip test. Ten is about four
# hundred decisions and sixteen hundred seat-encodings, which is enough
# to reach every shape the encoder has: leads and follows, open and
# closed tricks, the landlord's burial and the three seats that may not
# see it, and a history long enough to exercise the token window.
ROUND_TRIP_HANDS = 10
MOVE_CAP = 16


def _tiny() -> ModelConfig:
    """A model small enough that a match is a few seconds, not a few
    minutes. Only the widths that cost time are reduced; ``max_history``
    stays at the default because the encoder always emits that many
    rows."""
    return ModelConfig(
        d_model=64,
        n_layers_history=1,
        n_layers_query=1,
        n_heads=4,
        d_ff=128,
        critic_d=32,
        critic_layers=1,
        max_candidates=MOVE_CAP,
    )


# --- the wire format ------------------------------------------------------


def to_wire(state: PlayState) -> dict:
    """A position in exactly the fields the host sends, and no others.

    This is the contract, written down as code. Adding a field here to
    make a test pass would be moving the goalposts: the host emits these
    and nothing else, so anything the rebuild needs and this omits is a
    bug in the contract rather than in the test.
    """
    return {
        "level_rank": int(state.ctx.level_rank),
        "trump_suit": int(state.ctx.trump_suit),
        "num_decks": int(state.num_decks),
        "landlord": int(state.landlord),
        "to_act": int(state.to_act),
        "trick_leader": int(state.trick_leader),
        "hands": [[int(v) for v in row] for row in state.hands],
        "kitty": [int(v) for v in state.kitty],
        "tricks": [
            {
                "leader": int(t.leader),
                "winner": int(t.winner),
                "points": int(t.points),
                "plays": [[int(v) for v in p] for p in t.plays],
            }
            for t in state.tricks
        ],
        "trick_plays": [[int(v) for v in p] for p in state.trick_plays],
    }


def test_wire_round_trip_is_bit_identical() -> None:
    """A rebuilt position must encode to the same bits as the original.

    Every decision of every hand, from all four seats. The comparison is
    exact rather than approximate: the two states are supposed to be the
    same position, so any difference at all is the wire format failing
    to carry something the encoder reads. Dtypes are compared too --
    a float64 history that happened to hold the same values would still
    change what the network is fed.

    The round trip goes through ``json`` on purpose. The host writes
    NDJSON, so a field that only survives in memory has not survived.
    """
    rng = np.random.default_rng(SEED)
    original_cache, rebuilt_cache = Cache(), Cache()
    compared = 0
    positions = 0

    for _ in range(ROUND_TRIP_HANDS):
        state, _info = generate(rng, DECKS)
        while not state.done:
            rebuilt = state_from_wire(json.loads(json.dumps(to_wire(state))))
            positions += 1
            for seat in range(NUM_SEATS):
                want = encode(state, seat, original_cache)
                got = encode(rebuilt, seat, rebuilt_cache)
                assert int(got.history_len) == int(want.history_len), (
                    f"history_len {int(got.history_len)} != "
                    f"{int(want.history_len)} at seat {seat}"
                )
                rebuilt_arrays = got.arrays()
                for name, a in want.arrays().items():
                    b = rebuilt_arrays[name]
                    assert a.dtype == b.dtype, f"{name} dtype {b.dtype} != {a.dtype}"
                    assert a.shape == b.shape, f"{name} shape {b.shape} != {a.shape}"
                    assert np.array_equal(a, b), (
                        f"{name} differs at seat {seat}, trick "
                        f"{len(state.tricks)}: "
                        f"{int(np.count_nonzero(a != b))} of {a.size} entries"
                    )
                compared += 1
            moves = state.legal_moves(cap=MOVE_CAP).moves
            state.step(moves[int(rng.integers(len(moves)))])

    print(
        f"    {positions} positions, {compared} seat-encodings, 0 mismatches "
        f"({ROUND_TRIP_HANDS} hands)"
    )
    assert compared >= 1000, f"only {compared} encodings compared"


# --- the protocol ---------------------------------------------------------

# A host that says exactly what it is told to say. It takes the plan as
# its first argument and ignores the flags the client appends, so the
# client's own argv construction is exercised rather than bypassed.
#
# ``__die__`` is the interesting entry: the process leaves mid-stream,
# which is what a crashed host looks like from the other end.
FAKE_HOST = '''
import json
import sys

plan = json.load(open(sys.argv[1]))
for message in plan:
    if message.get("t") == "__die__":
        sys.stderr.write("fake host: falling over on purpose\\n")
        sys.stderr.flush()
        sys.exit(3)
    sys.stdout.write(json.dumps(message) + "\\n")
    sys.stdout.flush()
    if message.get("t") != "pending":
        continue
    reply = sys.stdin.readline()
    if reply == "":
        sys.stderr.write("fake host: client closed the stream\\n")
        sys.exit(4)
    got = json.loads(reply)
    if got.get("t") != "choose":
        sys.stderr.write(f"fake host: expected choose, got {got!r}\\n")
        sys.exit(5)
    index = got.get("index")
    if not isinstance(index, int) or not 0 <= index < len(message["candidates"]):
        sys.stderr.write(
            f"fake host: index {index!r} outside 0..{len(message['candidates']) - 1}\\n"
        )
        sys.exit(6)
'''


def _canned_pending(game: int, half: int, seed: int) -> dict:
    """A real position, dressed as a ``pending`` message."""
    state, _info = generate(np.random.default_rng(seed), DECKS)
    moves = state.legal_moves(cap=MOVE_CAP).moves[:MOVE_CAP]
    message = to_wire(state)
    message.update(
        t="pending",
        game=game,
        half=half,
        seat=int(state.to_act),
        candidates=[[int(v) for v in m] for m in moves],
    )
    return message


def _hand(game: int, half: int, **fields: int) -> dict:
    return {"t": "hand", "game": game, "half": half, **fields}


def _run_fake(plan: list[dict], **kwargs: Any) -> dict[str, float]:
    """Drive the real client against the canned host."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "fake_host.py"
        script.write_text(FAKE_HOST, encoding="utf-8")
        plan_path = Path(tmp) / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        cfg = _tiny()
        net = ActorCritic(cfg)
        return play_against_bot(
            net,
            "fake",
            cfg=cfg,
            node=[sys.executable, str(script), str(plan_path)],
            **kwargs,
        )


def _with_deadline(fn: Callable[[], Any], seconds: float) -> Any:
    """Call `fn`, failing loudly if it has not returned in `seconds`.

    A client that deadlocks would otherwise hang the suite until someone
    noticed, which is the failure mode this whole file exists to rule
    out. The worker is a daemon so a hung call cannot keep the process
    alive either.
    """
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise AssertionError(f"the client did not return within {seconds:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def test_the_client_speaks_the_protocol() -> None:
    """A canned two-deal match must come back with the hand-computed score.

    The fake host validates every reply -- it must be a ``choose`` whose
    index is inside the candidate list it offered -- and exits non-zero
    otherwise, so an off-by-one or a stray message fails here rather than
    becoming a slightly wrong number.

    Deal 0, landlord 1, so the defenders are team 1.
      half 0: the network is team 0, attacking; attackers gain 2, so the
              network gains 2, and it extracted 170 points.
      half 1: the network is team 1, defending; defenders gain 2, so the
              network gains 2 again, and conceded 35 points.
      paired: (2 + 2) / 2 = 2 levels, (170 - 35) / 2 = 67.5 points.
    Deal 1, landlord 0, so the defenders are team 0. Both halves are
    drawn, 0 levels either way, and the points are 120 conceded against
    80 taken: (-120 + 80) / 2 = -20.
    """
    plan = [
        _canned_pending(0, 0, 1),
        _hand(0, 0, landlord=1, attacker_points=170, defender_gain=0,
              attacker_gain=2, digest=111),
        _canned_pending(0, 1, 2),
        _hand(0, 1, landlord=1, attacker_points=35, defender_gain=2,
              attacker_gain=0, digest=111),
        _canned_pending(1, 0, 3),
        _hand(1, 0, landlord=0, attacker_points=120, defender_gain=0,
              attacker_gain=0, digest=222),
        _canned_pending(1, 1, 4),
        _hand(1, 1, landlord=0, attacker_points=80, defender_gain=0,
              attacker_gain=0, digest=222),
        {"t": "done", "deals": 2},
    ]
    stats = _with_deadline(lambda: _run_fake(plan, num_deals=2, seed=7), 120)

    assert stats["deals"] == 2.0
    assert stats["decisions"] == 4.0, stats["decisions"]
    assert stats["levels"] == 1.0, stats["levels"]
    assert stats["points"] == 23.75, stats["points"]
    # Two deals scoring 2 and 0: sample sd sqrt(2), stderr sqrt(2)/sqrt(2).
    assert abs(stats["levels_stderr"] - 1.0) < 1e-12, stats["levels_stderr"]
    assert abs(stats["levels_ci_low"] - (1.0 - Z95)) < 1e-12
    assert abs(stats["levels_ci_high"] - (1.0 + Z95)) < 1e-12
    assert stats["ahead"] == 0.0, "an interval straddling zero is not ahead"
    print(f"    levels {stats['levels']:+.2f}  points {stats['points']:+.2f}")


def test_a_host_that_dies_mid_stream_raises_rather_than_hanging() -> None:
    """The child leaves after one decision. The client must say so.

    Both halves of this matter. A client that blocked on a reply that
    will never come would wedge a training run at its first evaluation,
    and one that raised without the child's stderr would leave no way to
    find out why.
    """
    plan = [_canned_pending(0, 0, 11), {"t": "__die__"}]
    started = time.perf_counter()
    try:
        _with_deadline(lambda: _run_fake(plan, num_deals=1, seed=7), 120)
    except HostError as exc:
        elapsed = time.perf_counter() - started
        message = str(exc)
        assert "falling over on purpose" in message, message
        assert "exit 3" in message, message
        print(f"    raised in {elapsed:.1f}s: {message.splitlines()[0]}")
    else:
        raise AssertionError("a dead host was not reported")


def test_halves_dealt_different_cards_are_rejected() -> None:
    """A pair whose digests disagree is not a duplicate pair.

    Scoring it anyway would compare two policies on two different deals
    and call the difference a margin, which is the one thing the whole
    duplicate design exists to prevent. The host checks this too; the
    client checks it again because the statistics are computed here,
    from what the host said rather than from what it did.
    """
    plan = [
        _canned_pending(0, 0, 21),
        _hand(0, 0, landlord=0, attacker_points=100, defender_gain=0,
              attacker_gain=0, digest=1234),
        _canned_pending(0, 1, 22),
        _hand(0, 1, landlord=0, attacker_points=100, defender_gain=0,
              attacker_gain=0, digest=5678),
        {"t": "done", "deals": 1},
    ]
    try:
        _with_deadline(lambda: _run_fake(plan, num_deals=1, seed=7), 120)
    except HostError as exc:
        assert "different cards" in str(exc), str(exc)
        print(f"    rejected: {str(exc).splitlines()[0]}")
    else:
        raise AssertionError("mismatched digests were scored anyway")


# --- the statistics -------------------------------------------------------


def _loss_for_the_network(game: int, landlord: int) -> dict[int, _Half]:
    """One deal the network loses by two levels in both halves.

    The landlord seat is the same in both halves -- it is a property of
    the deal -- so the network defends in one and attacks in the other,
    and losing both means the *other* side gained in both.
    """
    halves: dict[int, _Half] = {}
    defenders = landlord % 2
    for half in (0, 1):
        network_defends = NET_TEAM[half] == defenders
        halves[half] = _Half(
            landlord=landlord,
            # Kept consistent with ``hand_result``: 160 points and up is
            # two levels to the attackers, under 40 is two to the
            # defenders. Whichever side the network is not on takes them.
            attacker_points=170 if network_defends else 20,
            defender_gain=0 if network_defends else 2,
            attacker_gain=2 if network_defends else 0,
            digest=1000 + game,
            decisions=25,
        )
    return halves


def test_a_policy_that_always_loses_scores_below_zero() -> None:
    """And is never reported as ahead.

    ``ahead`` is the flag a training chart reads, so it has to be a
    statement about the interval rather than about the mean. Both
    directions are checked: a policy that always wins by the same margin
    has a zero-width interval above zero and must come back ahead, or
    the flag could simply be off.
    """
    losing = {game: _loss_for_the_network(game, game % NUM_SEATS) for game in range(8)}
    stats = summarise(losing)
    assert stats["deals"] == 8.0
    assert stats["levels"] == -2.0, stats["levels"]
    assert stats["points"] == -75.0, stats["points"]
    assert stats["ahead"] == 0.0
    assert stats["levels_ci_high"] <= 0.0, stats["levels_ci_high"]
    assert stats["decisions"] == 8 * 2 * 25.0

    # The same hands with the two seatings exchanged is the same match
    # seen from the other side of the table, so the margin must be its
    # exact negative. An error in the sign convention survives the test
    # above -- a policy that loses is still reported as losing if both
    # halves are negated -- but it cannot survive this one.
    swapped = {
        game: {0: halves[1], 1: halves[0]} for game, halves in losing.items()
    }
    flipped = summarise(swapped)
    assert flipped["levels"] == 2.0, flipped["levels"]
    assert flipped["points"] == 75.0, flipped["points"]
    assert flipped["ahead"] == 1.0, "a policy winning every deal is not ahead"
    print(
        f"    always loses {stats['levels']:+.1f} levels / "
        f"{stats['points']:+.1f} points (ahead {stats['ahead']:.0f}), "
        f"swapped {flipped['levels']:+.1f} / {flipped['points']:+.1f} "
        f"(ahead {flipped['ahead']:.0f})"
    )


# --- end to end -----------------------------------------------------------


def test_a_real_match_against_the_engine() -> None:
    """The whole bridge, against the real host. NEEDS NODE.

    Skipped with a message when ``npx`` is not on PATH, because the
    failure would otherwise look like a bug in this stack rather than a
    missing toolchain.

    The opponent is ``lowball`` rather than one of the smart bots: the
    bot is the host's business and costs about two hundred times as much
    per deal when it searches, which would turn this into a two-minute
    test of somebody else's code. What is under test here is that a real
    position comes across the wire, is rebuilt, is scored, and that the
    move chosen is one the engine accepts -- and the engine rejecting a
    move is a crash, not a wrong number.
    """
    if shutil.which("npx") is None:
        print("    SKIPPED: npx is not on PATH, so the TypeScript host cannot run")
        return

    cfg = _tiny()
    net = ActorCritic(cfg)
    net.train()
    started = time.perf_counter()
    stats = _with_deadline(
        lambda: play_against_bot(
            net,
            "lowball",
            cfg=cfg,
            num_deals=2,
            seed=4242,
            max_candidates=MOVE_CAP,
        ),
        600,
    )
    elapsed = time.perf_counter() - started

    assert stats["deals"] == 2.0
    assert stats["decisions"] > 0.0, "the network was never asked to move"
    for key, value in stats.items():
        assert value == value, f"{key} is NaN"
        assert abs(value) != float("inf"), f"{key} is infinite"
    # A measurement must not leave the network in eval mode; a run whose
    # dropout quietly stayed off after the first evaluation would train
    # a different model from then on.
    assert net.training, "play_against_bot left the network in eval mode"
    print(
        f"    untrained vs lowball: {stats['levels']:+.2f} levels/deal, "
        f"{int(stats['decisions'])} decisions, {elapsed:.1f}s "
        f"({elapsed / 2:.1f}s/deal)"
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
            print(f"FAIL  {fn.__name__}: {exc}"[:2000])
        else:
            print(f"ok    {fn.__name__}  ({time.time() - started:.1f}s)")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


def test_the_ladder_resolves_and_dedupes() -> None:
    """`ladder` expands to the rungs; anything else passes through."""
    from ..eval.bots import LADDER, resolve_bots

    assert resolve_bots(["ladder"]) == LADDER
    # An explicit name keeps its place and is not repeated by the ladder.
    got = resolve_bots(["lowball", "ladder"])
    assert got[0] == "lowball"
    assert len(got) == len(LADDER)
    assert len(set(got)) == len(got)
    # A name outside the ladder is still measured.
    assert "smart-v31" in resolve_bots(["ladder", "smart-v31"])
    print(f"    {len(LADDER)} rungs, resolution dedupes and preserves order")


def test_the_ladder_runs_cheapest_first() -> None:
    """The order is what makes a truncated measurement useful.

    Measured per decision: the heuristic rungs cost 12 to 18
    milliseconds, smart-v9 costs 93, and the rungs that search cost 560
    to 670. A budget that runs out must therefore leave the expensive
    end unmeasured, not the cheap end.
    """
    from ..eval.bots import LADDER

    cheap = ("random", "lowball", "point-hoarder", "aggressive")
    dear = ("smart-v9", "smart-v18", "smart-v28", "smart-v34")
    last_cheap = max(LADDER.index(b) for b in cheap if b in LADDER)
    first_dear = min(LADDER.index(b) for b in dear if b in LADDER)
    assert last_cheap < first_dear, LADDER
    print("    cheap rungs precede every rung that searches")


def test_a_budget_leaves_the_expensive_end_unmeasured() -> None:
    """A truncated ladder says so, rather than reading as a full one."""
    import time as _time

    from ..train.evaluate import evaluate_against_bots

    cfg = _tiny()
    net = ActorCritic(cfg)
    started = _time.perf_counter()
    out = evaluate_against_bots(
        net, ["ladder"], "cpu", cfg, num_deals=2, seed=5, workers=4,
        budget_s=20.0,
    )
    elapsed = _time.perf_counter() - started
    run = out["ladder/rungs_run"]
    skipped = out["ladder/rungs_skipped"]
    assert run >= 1, out
    assert skipped >= 1, "the budget was too generous to test truncation"
    from ..eval.bots import resolve_bots as _resolve
    assert run + skipped == len(_resolve(["ladder"]))
    # Whatever ran must be from the cheap end.
    measured = {k.split("/")[0][3:] for k in out if k.startswith("vs_")}
    assert "smart-v34" not in measured, measured
    print(f"    ran {run:.0f} rungs, skipped {skipped:.0f}, in {elapsed:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())
