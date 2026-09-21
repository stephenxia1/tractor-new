"""Whole hands replayed against the TypeScript engine.

``test_differential.py`` pins the rules primitives -- card ordering,
combo classification, follow legality -- one call at a time. None of
that reaches the behaviour that only exists across a whole hand: who
took each trick, which team the points landed on, what a blocked throw
actually put on the table, and what the hand finally scored. A
simulator can agree on every primitive and still disagree on all four.

So this replays the engine's own games. ``research/src/dumpGames.ts``
plays hands out with a uniform-random policy and records the position at
the first trick, every play in order, and the outcomes. Here the same
position is rebuilt and the same *requested* plays are pushed through
``PlayState.step``; the two sides are compared at every step.

Requested, not played: the engine shortens an unsafe throw to its
smallest beatable component and returns the rest to hand (rules.md
§7.6). Feeding back what the engine played would hide exactly that. The
comparison of request against commitment is what tests the penalty.

The two move *enumerators* deliberately differ -- the Python one offers
more throws -- so candidate lists are never compared. Only legality is:
a recorded follow must be a move Python would allow, and a recorded lead
must at least be a well-formed single-suit play.

    python -m tractor.tests.test_game_differential
    python -m tractor.tests.test_game_differential dump.jsonl --survey
    python -m tractor.tests.test_game_differential dump.jsonl --survey --resync

Regenerate the fixture with (from ``research/``):

    npx tsx src/dumpGames.ts 500 20260920 py/tractor/tests/fixtures/games.jsonl
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tractor.rules.cards import (
    NUM_TYPES,
    RANK_NAMES,
    SUIT_NAMES,
    Ctx,
    card_name,
    counts_from_ids,
    ids_from_counts,
)
from tractor.rules.combos import decompose, play_suit
from tractor.rules.follow import is_legal_follow, lead_obligations
from tractor.sim.state import PlayState, hand_result

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "games.jsonl"

REGENERATE = (
    "regenerate from research/ with:  npx tsx src/dumpGames.ts 500 20260920 "
    "py/tractor/tests/fixtures/games.jsonl"
)

# Cap for the follow enumeration used in the membership check. Well above
# the 64 training uses, because a move dropped by the cap is not an
# illegal move and must not read as one. Where the cap still bites, the
# check falls back to `is_legal_follow` and the fallback is counted.
# Measured: 256 costs ~30% less than 512 and needs the fallback on
# roughly one follow in 4000 either way.
MOVE_CAP = 256


class Skipped(Exception):
    """The fixture this test needs is not present."""


class Mismatch(Exception):
    """The two implementations disagreed. Carries the full position."""


@dataclass
class Tally:
    hands: int = 0
    plays: int = 0
    leads: int = 0
    follows: int = 0
    tricks: int = 0
    forced_leads: int = 0
    throw_leads: int = 0
    # Follows the Python enumerator did not list because its cap cut the
    # enumeration short; legality confirmed directly instead.
    follow_fallbacks: int = 0
    # Plays where --resync forced Python back onto the engine's cards.
    resyncs: int = 0
    mismatches: list[str] = field(default_factory=list)

    def summary(self) -> str:
        tail = f"  resyncs {self.resyncs}" if self.resyncs else ""
        return (
            f"hands {self.hands}  plays {self.plays} "
            f"(leads {self.leads}, follows {self.follows})  "
            f"tricks {self.tricks}  forced leads {self.forced_leads}  "
            f"throw leads {self.throw_leads}  "
            f"enumerator fallbacks {self.follow_fallbacks}{tail}"
        )


# --- reporting ------------------------------------------------------------


def _names(ids: "list[int] | np.ndarray") -> str:
    out = [card_name(int(t)) for t in ids]
    return " ".join(out) if out else "(none)"


def _counts_names(counts: np.ndarray) -> str:
    return _names(ids_from_counts(counts))


def _ctx_name(rec: dict) -> str:
    suit = rec["trumpSuit"]
    label = "no-trump" if suit == 4 else SUIT_NAMES[suit]
    return f"level {RANK_NAMES[rec['levelRank']]}, trump {label}"


def _report(rec: dict, source: Path, upto: int, message: str) -> str:
    """The full position, the plays that led to it, and the disagreement.

    Printed whole rather than summarised: the point of a differential
    failure is that someone has to adjudicate it, and that needs the
    deal, the sequence and both verdicts in one place.
    """
    lines: list[str] = []
    lines.append(
        f"hand {rec['hand']} (deal seed {rec['seed']}) of {source}"
    )
    lines.append(f"  context   {_ctx_name(rec)}, {rec['decks']} decks")
    lines.append(f"  landlord  seat {rec['landlord']}")
    for seat, cards in enumerate(rec["hands"]):
        mark = " (landlord)" if seat == rec["landlord"] else ""
        lines.append(f"  hand {seat}    {_names(cards)}{mark}")
    lines.append(f"  kitty     {_names(rec['kitty'])}")
    lines.append("  plays:")
    trick = 0
    in_trick = 0
    for i, play in enumerate(rec["plays"][: upto + 1]):
        tag = "LEAD  " if in_trick == 0 else "follow"
        shown = f"    {i:3d} t{trick:<2d} seat {play['seat']} {tag} {_names(play['req'])}"
        if play["req"] != play["got"]:
            shown += f"   -> engine played {_names(play['got'])}"
        if i == upto:
            shown += "   <<< HERE"
        lines.append(shown)
        in_trick += 1
        if in_trick == 4:
            t = rec["tricks"][trick] if trick < len(rec["tricks"]) else None
            if t is not None:
                lines.append(
                    f"        trick {trick}: engine winner seat {t['winner']}, "
                    f"{t['points']} points"
                )
            trick += 1
            in_trick = 0
    lines.append(f"  --> {message}")
    lines.append(f"  repro json: {json.dumps(rec, separators=(',', ':'))}")
    return "\n".join(lines)


# --- replay ---------------------------------------------------------------


def replay(rec: dict, source: Path, tally: Tally, resync: bool = False) -> None:
    """Replay one recorded hand. Raises Mismatch on the first disagreement.

    With `resync`, a play the two sides commit differently is recorded and
    then overridden with the engine's cards instead of aborting the hand.
    That is an investigation tool, not a relaxation: once the two states
    diverge every later comparison is meaningless, so a single known
    disagreement would otherwise hide everything downstream of it. Resync
    keeps the trick winners, the points and the final score under test
    while the disagreement itself is still reported.
    """
    decks = int(rec["decks"])
    ctx = Ctx(int(rec["levelRank"]), int(rec["trumpSuit"]))
    hands = np.stack([counts_from_ids(h) for h in rec["hands"]])
    kitty = counts_from_ids(rec["kitty"])

    def fail(index: int, message: str) -> Mismatch:
        return Mismatch(_report(rec, source, index, message))

    # A dump that lost or duplicated a card would make every later
    # disagreement meaningless, so check the deal before trusting it.
    total = hands.sum(axis=0).astype(np.int16) + kitty.astype(np.int16)
    if not np.array_equal(total, np.full(NUM_TYPES, decks, dtype=np.int16)):
        raise fail(0, f"recorded deal is not a full {decks}-deck pack")

    state = PlayState.new(ctx, hands, kitty, int(rec["landlord"]), decks)
    trick_index = 0

    for index, play in enumerate(rec["plays"]):
        seat = int(play["seat"])
        req = counts_from_ids(play["req"])
        want = counts_from_ids(play["got"])

        if state.done:
            raise fail(index, "python ended the hand while the engine played on")
        if state.to_act != seat:
            raise fail(
                index,
                f"engine has seat {seat} to play, python has seat {state.to_act}",
            )
        if np.any(req > state.hands[seat]):
            raise fail(
                index,
                f"seat {seat} does not hold {_counts_names(req)} in python; "
                f"holds {_counts_names(state.hands[seat])}",
            )

        is_lead = state.is_lead
        if is_lead:
            tally.leads += 1
            _check_lead_legal(state, req, decks, index, fail)
            if len(decompose(req, ctx, decks) or []) > 1:
                tally.throw_leads += 1
        else:
            tally.follows += 1
            _check_follow_legal(state, req, decks, index, fail, tally)

        before = state.clone() if resync else None
        got = state.step(req)
        tally.plays += 1
        returned = (req - want).astype(np.int8)
        if int(returned.sum()) > 0:
            tally.forced_leads += 1

        if not np.array_equal(got, want):
            kind = "lead" if is_lead else "follow"
            problem = fail(
                index,
                f"{kind} committed different cards: engine played "
                f"{_counts_names(want)}, python played {_counts_names(got)} "
                f"(requested {_names(play['req'])}, in that order)",
            )
            if not resync or before is None:
                raise problem
            tally.mismatches.append(str(problem))
            tally.resyncs += 1
            state = before
            state.step(want)
        # The penalty and the shortened play are the same event; if the
        # engine shortened, python must say which cards came back. Skipped
        # after a resync: the override is not a throw, so nothing to say.
        elif int(returned.sum()) > 0:
            if state.last_forced_lead is None:
                raise fail(index, "engine blocked the throw, python did not")
            if not np.array_equal(state.last_forced_lead, returned):
                raise fail(
                    index,
                    f"engine returned {_counts_names(returned)} to hand, "
                    f"python returned {_counts_names(state.last_forced_lead)}",
                )
        elif state.last_forced_lead is not None:
            raise fail(
                index,
                f"python blocked the throw (returning "
                f"{_counts_names(state.last_forced_lead)}), engine did not",
            )

        # A trick closed on this play iff the record grew.
        if len(state.tricks) > trick_index:
            if trick_index >= len(rec["tricks"]):
                raise fail(index, "python closed more tricks than the engine")
            want_trick = rec["tricks"][trick_index]
            got_trick = state.tricks[trick_index]
            tally.tricks += 1
            if got_trick.leader != int(want_trick["leader"]):
                raise fail(
                    index,
                    f"trick {trick_index} led by seat {want_trick['leader']} in "
                    f"the engine, seat {got_trick.leader} in python",
                )
            if got_trick.winner != int(want_trick["winner"]):
                raise fail(
                    index,
                    f"trick {trick_index} won by seat {want_trick['winner']} in "
                    f"the engine, seat {got_trick.winner} in python",
                )
            if got_trick.points != int(want_trick["points"]):
                raise fail(
                    index,
                    f"trick {trick_index} worth {want_trick['points']} points in "
                    f"the engine, {got_trick.points} in python",
                )
            trick_index += 1

    last = len(rec["plays"]) - 1
    if not state.done:
        raise fail(last, "engine finished the hand, python did not")
    if len(state.tricks) != len(rec["tricks"]):
        raise fail(
            last,
            f"engine played {len(rec['tricks'])} tricks, python played "
            f"{len(state.tricks)}",
        )
    if state.attacker_points != int(rec["attackerPoints"]):
        raise fail(
            last,
            f"attacker points: engine {rec['attackerPoints']}, python "
            f"{state.attacker_points} (the kitty bonus is the usual "
            f"difference; kitty {_counts_names(kitty)})",
        )

    # computeHandResult returns a winning side and a level gain; the
    # Python table returns the gain for each side. Only one ever moves.
    delta = int(rec["levelDelta"])
    want_result = (delta, 0) if rec["winner"] == "defenders" else (0, delta)
    got_result = hand_result(state.attacker_points)
    if got_result != want_result:
        raise fail(
            last,
            f"level outcome: engine {rec['winner']} +{delta} = {want_result}, "
            f"python {got_result} on {state.attacker_points} attacker points",
        )
    tally.hands += 1


def _check_lead_legal(
    state: PlayState,
    req: np.ndarray,
    decks: int,
    index: int,
    fail: Callable[[int, str], Mismatch],
) -> None:
    """A lead must be a well-formed play in one effective suit.

    Membership in ``legal_moves`` is the wrong test here: the two lead
    enumerators deliberately differ, so a lead the engine offered may
    legitimately be absent from the Python list (and vice versa). What
    both sides must agree on is that the cards form a play at all.
    """
    if int(req.sum()) == 0:
        raise fail(index, "empty lead")
    suit = play_suit(req, state.ctx)
    if suit is None:
        raise fail(
            index,
            f"lead {_counts_names(req)} spans more than one effective suit "
            f"in python",
        )
    if decompose(req, state.ctx, decks) is None:
        raise fail(index, f"lead {_counts_names(req)} does not decompose in python")


def _check_follow_legal(
    state: PlayState,
    req: np.ndarray,
    decks: int,
    index: int,
    fail: Callable[[int, str], Mismatch],
    tally: Tally,
) -> None:
    """A follow the engine accepted must be one Python offers."""
    moves = state.legal_moves(cap=MOVE_CAP)
    key = req.tobytes()
    if any(m.tobytes() == key for m in moves.moves):
        return

    # Absent from the list. That is only a disagreement if the list was
    # complete; where the cap or the sub-multiset limit cut it short, ask
    # the legality predicate directly rather than reporting a phantom.
    capped = moves.truncated > 0 or moves.unenumerated
    lead_info = state.lead_components()
    if lead_info is None:
        raise fail(index, "python cannot decompose the lead of this trick")
    parts, suit = lead_info
    ob = lead_obligations(parts, decks)
    lead_count = int(state.trick_plays[0].sum())
    legal = is_legal_follow(
        ob, lead_count, suit, req, state.hands[state.to_act], state.ctx, decks
    )
    if legal and capped:
        tally.follow_fallbacks += 1
        return
    raise fail(
        index,
        f"follow {_counts_names(req)} was accepted by the engine but "
        f"{'is rejected by is_legal_follow' if not legal else 'is missing from a complete enumeration'} "
        f"in python (lead {_counts_names(state.trick_plays[0])}, "
        f"{len(moves.moves)} moves offered, truncated {moves.truncated}, "
        f"unenumerated {moves.unenumerated})",
    )


# --- driver ---------------------------------------------------------------


def compare(
    path: Path,
    limit: int | None = None,
    survey: bool = False,
    resync: bool = False,
) -> Tally:
    """Replay a dump. Raises on the first mismatch unless surveying."""
    if not path.exists():
        raise Skipped(f"no dump at {path}; {REGENERATE}")
    tally = Tally()
    seen = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if limit is not None and seen >= limit:
                break
            seen += 1
            rec = json.loads(line)
            try:
                replay(rec, path, tally, resync=resync)
            except Mismatch as exc:
                if not survey:
                    raise
                tally.mismatches.append(str(exc))
    if seen == 0:
        raise Skipped(f"{path} held no hands; {REGENERATE}")
    return tally


def test_recorded_hands_match_the_engine() -> None:
    tally = compare(FIXTURE)
    print(f"    {tally.summary()}")
    assert tally.plays > 0
    assert tally.tricks > 0
    # A dump with no blocked throw in it would leave the forced-lead
    # penalty untested while still passing, so say so rather than pass
    # quietly on a fixture that lost its coverage.
    assert tally.forced_leads > 0, (
        "no forced-lead penalty in the fixture: the dumper's throw rate "
        "must have changed, and §7.6 is going untested"
    )


def main() -> int:
    """Run every check. The repository's other Python tests are plain
    scripts, so this one is too; pytest collects it either way.

    Positional argument: a dump to use instead of the checked-in
    fixture. ``--survey`` reports every mismatching hand rather than
    stopping at the first, ``--resync`` carries on inside a hand after a
    play the two sides commit differently (see `replay`), and
    ``--limit=N`` caps the hand count.
    """
    import time

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = {a for a in sys.argv[1:] if a.startswith("-")}
    limit: int | None = None
    for f in list(flags):
        if f.startswith("--limit="):
            limit = int(f.split("=", 1)[1])
            flags.discard(f)

    if args or flags or limit is not None:
        path = Path(args[0]) if args else FIXTURE
        survey = "--survey" in flags
        started = time.time()
        try:
            tally = compare(
                path, limit=limit, survey=survey, resync="--resync" in flags
            )
        except Skipped as exc:
            print(f"skip  {path}: {exc}")
            return 0
        except Mismatch as exc:
            print(f"FAIL  {path}:\n{exc}")
            return 1
        print(f"      {tally.summary()}  ({time.time() - started:.1f}s)")
        for report in tally.mismatches:
            print(f"FAIL\n{report}")
        if tally.mismatches:
            # One entry per hand when surveying (each hand stops at its
            # first disagreement), one per play when resyncing.
            print(f"{len(tally.mismatches)} disagreements reported")
            return 1
        print(f"ok    {path}")
        return 0

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    skipped = 0
    for fn in fns:
        started = time.time()
        try:
            fn()
        except Skipped as exc:
            skipped += 1
            print(f"skip  {fn.__name__}: {exc}")
        except Mismatch as exc:
            # Never truncated: the position is the whole point of the
            # failure, and it is what someone has to adjudicate from.
            failed += 1
            print(f"FAIL  {fn.__name__}:\n{exc}")
        except Exception as exc:  # noqa: BLE001 - reporting harness
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}"[:1500])
        else:
            print(f"ok    {fn.__name__}  ({time.time() - started:.1f}s)")
    tail = f" ({skipped} skipped)" if skipped else ""
    print(f"{len(fns) - failed - skipped}/{len(fns)} passed{tail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
