"""Duplicate-deal evaluation: the same cards, played both ways round.

One hand of Tractor is mostly luck. The spread of level outcomes between
two policies of genuinely different strength is small next to the spread
between two deals, so a head-to-head that deals fresh cards for every
match has to run a very long time before the difference clears its own
noise.

Duplicate scoring removes most of that. Each deal is played twice from
the identical position: once with team X on seats (0, 2) and team Y on
(1, 3), then again with the two swapped. Whatever the cards were worth,
both teams are handed them, and X's score on the deal is the mean of its
two signed results. The part of the outcome that came from the deal
cancels in that mean; what is left came from the play.

Three things then have to be right, and each is a place a harness can
quietly go wrong.

*The position must be identical.* It is cloned rather than regenerated,
and a digest of the pre-play state is taken in both replays and compared,
mirroring the card-for-card check ``research/src/dupcheck.ts`` makes on
the TypeScript side. Regenerating from a seed would also work, but only
as long as nothing downstream ever consumes the generator differently in
the two halves -- exactly the assumption a duplicate harness exists to
avoid making.

*The randomness must be attached to the seats, not to the policies.*
Each seat pair draws from its own stream, derived from the deal and from
the pair, so the two replays of a deal differ in exactly one respect:
who sits where. Common random numbers, on top of the shared deal. It is
also what makes ``compare(x, y)`` and ``compare(y, x)`` exact negatives
rather than merely consistent, because the four hands played are then
the same four hands in the other order.

*The standard error must be clustered by deal.* Seats 0 and 2 always
score the same thing and seats 1 and 3 always score its negation, so a
per-seat error bar counts one observation four times and comes out half
the size it should be. The two replays of a deal are a matched pair, not
two draws, so a per-hand error bar is wrong too -- and wrong in the
pessimistic direction, since the pairing is what the design bought. The
independent unit is the deal, and nothing below it.

    python -m tractor.eval.duplicate --deals 400
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..gen.episode import generate
from ..sim.state import NUM_SEATS, PlayState, hand_result, team_of

# A policy sees a position and the seat it is acting for, and returns one
# of that seat's legal moves as an int8[54] count vector.
Policy = Callable[[PlayState, int], np.ndarray]

# Policies reach the harness as factories rather than as bare callables,
# because the harness -- not the caller -- owns the seeding. A
# deterministic policy ignores the generator it is handed.
PolicyFactory = Callable[[np.random.Generator], Policy]

DECKS = 2
MOVE_CAP = 64

# Normal quantile. At the deal counts this harness is meant for, the
# t-correction is well under a percent; below about thirty deals the
# interval is optimistic, and the deal count on the result says so.
Z95 = 1.959963984540054

# Seating 0 puts X on the first pair, seating 1 on the second. Each pair
# has to be one whole team, or a "team's" signed result would not be a
# single number.
SEAT_PAIRS: tuple[tuple[int, int], tuple[int, int]] = ((0, 2), (1, 3))
assert sorted(s for pair in SEAT_PAIRS for s in pair) == list(range(NUM_SEATS))
assert all(len({team_of(s) for s in pair}) == 1 for pair in SEAT_PAIRS)

# Child stream ids under one deal's seed sequence.
_DEAL_STREAM = 0
_PAIR_STREAM = (1, 2)  # indexed by seat pair, never by replay


def _seq(seed: int, index: int, child: int) -> np.random.SeedSequence:
    """Sub-stream ``child`` of deal ``index`` under a run's seed.

    Spelled with an explicit spawn key rather than with ``spawn()`` so
    that any one deal is reproducible on its own, without replaying the
    deals before it.
    """
    return np.random.SeedSequence(entropy=seed, spawn_key=(index, child))


def _rng(seed: int, index: int, child: int) -> np.random.Generator:
    return np.random.default_rng(_seq(seed, index, child))


# --- policies -------------------------------------------------------------


def uniform_random_policy(rng: np.random.Generator, cap: int = MOVE_CAP) -> Policy:
    """The reference opponent: uniform over the enumerated legal moves.

    Uniform over *moves*, which is not uniform over anything the game
    cares about -- a hand with one long suit offers more distinct throws
    than a balanced one does -- but it is the policy with no judgement in
    it, which is what a reference needs to be.
    """

    def play(state: PlayState, seat: int) -> np.ndarray:
        moves = state.legal_moves(cap=cap).moves
        return moves[int(rng.integers(len(moves)))]

    return play


def lowest_index_policy(
    rng: np.random.Generator | None = None, cap: int = MOVE_CAP
) -> Policy:
    """Always the first move the enumerator emits. Deliberately weak.

    The enumeration order is not arbitrary: leads are built suit by suit
    with the singles first in card order, and follows are built taking as
    many of the lowest card type as will fit. So this leads the lowest
    card of the lowest suit it holds, always, and never spends a high
    card to win a trick it could win. Its purpose is to be a difference
    large enough that failing to resolve it would indict the harness.
    """
    del rng

    def play(state: PlayState, seat: int) -> np.ndarray:
        return state.legal_moves(cap=cap).moves[0]

    return play


def stateless(policy: Policy) -> PolicyFactory:
    """Adapt a policy that carries no randomness to the factory protocol."""

    def build(rng: np.random.Generator) -> Policy:
        del rng
        return policy

    return build


def salted(make: PolicyFactory, salt: int) -> PolicyFactory:
    """The same policy, drawing from a different stream.

    Two differently salted copies of one stochastic policy are equal in
    strength by construction and independent in their choices, which is
    what a self-comparison needs: a true difference of exactly zero, but
    a real sampling distribution around it.
    """

    def build(rng: np.random.Generator) -> Policy:
        child = np.random.SeedSequence([int(rng.integers(1 << 62)), int(salt)])
        return make(np.random.default_rng(child))

    return build


_SELF_SALT_X = 0x5851F42D
_SELF_SALT_Y = 0x14057B7E


# --- one hand -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HandOutcome:
    """What one played-out hand settled, before a perspective is taken."""

    attacker_points: int
    defender_levels: int
    attacker_levels: int
    defenders: int
    tricks: int
    decisions: int


def deal_digest(state: PlayState) -> int:
    """A 64-bit fingerprint of a starting position.

    Cards, kitty, landlord and trump context -- everything that makes the
    deal what it is. Two replays whose digests agree were dealt the same
    position, and a duplicate score computed across two different
    positions is not a duplicate score at all.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(state.hands.tobytes())
    h.update(state.kitty.tobytes())
    h.update(
        bytes(
            (
                state.landlord & 0xFF,
                state.ctx.level_rank & 0xFF,
                state.ctx.trump_suit & 0xFF,
                state.num_decks & 0xFF,
            )
        )
    )
    return int.from_bytes(h.digest(), "little")


def _check_legal(state: PlayState, move: np.ndarray, cap: int) -> None:
    moves = state.legal_moves(cap=cap)
    if moves.unenumerated:
        # The exact set was too large to build, so absence proves nothing.
        return
    key = np.asarray(move, dtype=np.int8).tobytes()
    if not any(m.tobytes() == key for m in moves.moves):
        raise ValueError(f"seat {state.to_act} returned a move outside the legal set")


def play_hand(
    state: PlayState,
    seats: Sequence[Policy],
    *,
    cap: int = MOVE_CAP,
    validate: bool = False,
) -> HandOutcome:
    """Play `state` to the end with one policy per seat, in place.

    `validate` re-enumerates the legal moves to confirm the returned move
    is among them. It roughly doubles the cost of a hand, so it is off by
    default and worth turning on whenever a policy is new.
    """
    decisions = 0
    guard = 0
    while not state.done:
        guard += 1
        if guard >= 400:
            raise RuntimeError("hand did not terminate")
        seat = state.to_act
        move = seats[seat](state, seat)
        if validate:
            _check_legal(state, move, cap)
        state.step(move)
        decisions += 1

    defender_levels, attacker_levels = hand_result(state.attacker_points)
    return HandOutcome(
        attacker_points=int(state.attacker_points),
        defender_levels=defender_levels,
        attacker_levels=attacker_levels,
        defenders=state.defenders,
        tricks=len(state.tricks),
        decisions=decisions,
    )


def signed_levels(outcome: HandOutcome, team: int) -> int:
    """The hand's result in level units, positive when `team` gained.

    ``hand_result`` reports the two sides separately and only ever moves
    one of them, so ``defender_levels - attacker_levels`` is the whole
    settlement seen from the defending side, and its negation is the same
    settlement seen from the attacking side.
    """
    gain = outcome.defender_levels - outcome.attacker_levels
    return gain if team == outcome.defenders else -gain


def signed_points(outcome: HandOutcome, team: int) -> int:
    """The hand's attacker points, signed the same way.

    Points only ever accrue to the attackers, so the paired mean of this
    across the two replays is half the difference between what X
    extracted on offence and what Y extracted on offence from the same
    cards. It is the finer metric: levels are a six-step staircase and
    cannot see a twenty-point swing that stays inside one step.
    """
    ap = outcome.attacker_points
    return -ap if team == outcome.defenders else ap


# --- one deal -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DealScore:
    """One deal played both ways round, scored for X."""

    index: int
    digest: int
    score: float  # paired level score: the mean of the two signed results
    point_score: float  # the same, in attacker points
    hand_levels: tuple[int, int]  # X's signed level result, per seating
    hand_points: tuple[int, int]  # X's signed point result, per seating
    attacker_points: tuple[int, int]  # raw, per seating
    landlord: int
    decisions: int


def play_deal(
    policy_x: PolicyFactory,
    policy_y: PolicyFactory,
    seed: int,
    index: int,
    *,
    num_decks: int = DECKS,
    cap: int = MOVE_CAP,
    validate: bool = False,
) -> DealScore:
    """Generate deal `index` and play it under both seatings."""
    base, _info = generate(_rng(seed, index, _DEAL_STREAM), num_decks)
    digest = deal_digest(base)

    levels: list[int] = []
    points: list[int] = []
    raw: list[int] = []
    decisions = 0

    for seating in (0, 1):
        state = base.clone()
        if deal_digest(state) != digest:
            raise RuntimeError(f"deal {index} differed between seatings")

        # Fresh generators per replay, seeded from the seat pair alone:
        # one pair of seats sees the same stream in both replays, and so
        # does the other. Only the occupant changes.
        by_seat: dict[int, Policy] = {}
        for pair_index, pair in enumerate(SEAT_PAIRS):
            build = policy_x if pair_index == seating else policy_y
            instance = build(_rng(seed, index, _PAIR_STREAM[pair_index]))
            for s in pair:
                by_seat[s] = instance
        seats = [by_seat[s] for s in range(NUM_SEATS)]

        outcome = play_hand(state, seats, cap=cap, validate=validate)
        x_team = team_of(SEAT_PAIRS[seating][0])
        levels.append(signed_levels(outcome, x_team))
        points.append(signed_points(outcome, x_team))
        raw.append(outcome.attacker_points)
        decisions += outcome.decisions

    return DealScore(
        index=index,
        digest=digest,
        score=0.5 * (levels[0] + levels[1]),
        point_score=0.5 * (points[0] + points[1]),
        hand_levels=(levels[0], levels[1]),
        hand_points=(points[0], points[1]),
        attacker_points=(raw[0], raw[1]),
        landlord=base.landlord,
        decisions=decisions,
    )


# --- many deals -----------------------------------------------------------


def _interval(x: np.ndarray) -> tuple[float, float, float, float]:
    """Mean, standard error and 95% interval over the per-deal scores."""
    n = int(x.size)
    if n == 0:
        raise ValueError("no deals to summarise")
    mean = float(x.mean())
    if n == 1:
        return mean, float("nan"), float("-inf"), float("inf")
    stderr = float(x.std(ddof=1) / np.sqrt(n))
    return mean, stderr, mean - Z95 * stderr, mean + Z95 * stderr


@dataclass(frozen=True, slots=True, eq=False)
class Result:
    """X's measured edge over Y, clustered by deal.

    `mean` is in level units per deal and `point_mean` in attacker points
    per deal; both are positive when X is ahead. `scores` holds the
    per-deal paired scores the interval was computed from, and
    `hand_levels` the two unpaired signed results behind each of them, so
    a caller can see for itself what the pairing bought.
    """

    num_deals: int
    mean: float
    stderr: float
    ci_low: float
    ci_high: float
    scores: np.ndarray  # float[num_deals]
    point_mean: float
    point_stderr: float
    point_ci_low: float
    point_ci_high: float
    point_scores: np.ndarray  # float[num_deals]
    hand_levels: np.ndarray  # int32[num_deals, 2]
    hand_points: np.ndarray  # int32[num_deals, 2]
    attacker_points: np.ndarray  # int32[num_deals, 2]
    digests: np.ndarray  # uint64[num_deals]
    decisions: int

    @property
    def significant(self) -> bool:
        """Does the 95% interval exclude zero?"""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def hand_stdev(self) -> float:
        """Spread of the raw per-hand signed result, pairing ignored."""
        flat = self.hand_levels.reshape(-1).astype(np.float64)
        return float(flat.std(ddof=1))

    @property
    def deal_stdev(self) -> float:
        """Spread of the per-deal paired score."""
        return float(self.scores.std(ddof=1))

    def summary(self, label: str = "X - Y") -> str:
        pad = " " * len(label)
        return (
            f"{label}  {self.mean:+.4f} levels/deal  "
            f"se {self.stderr:.4f}  "
            f"95% [{self.ci_low:+.4f}, {self.ci_high:+.4f}]\n"
            f"{pad}  {self.point_mean:+.2f} points/deal  "
            f"se {self.point_stderr:.2f}  "
            f"95% [{self.point_ci_low:+.2f}, {self.point_ci_high:+.2f}]  "
            f"n={self.num_deals} deals"
        )

    def __str__(self) -> str:
        return self.summary()


def compare(
    policy_x: PolicyFactory,
    policy_y: PolicyFactory,
    num_deals: int,
    seed: int,
    *,
    num_decks: int = DECKS,
    cap: int = MOVE_CAP,
    validate: bool = False,
) -> Result:
    """Duplicate-score `policy_x` against `policy_y` over `num_deals` deals.

    Both arguments are policy factories: each is handed a generator
    belonging to the seat pair it is about to occupy. A policy that needs
    no randomness ignores it; wrap a bare callable with `stateless`.
    """
    if num_deals < 1:
        raise ValueError("num_deals must be at least 1")

    deals = [
        play_deal(
            policy_x,
            policy_y,
            seed,
            i,
            num_decks=num_decks,
            cap=cap,
            validate=validate,
        )
        for i in range(num_deals)
    ]

    scores = np.array([d.score for d in deals], dtype=np.float64)
    point_scores = np.array([d.point_score for d in deals], dtype=np.float64)
    mean, stderr, lo, hi = _interval(scores)
    pmean, pstderr, plo, phi = _interval(point_scores)

    return Result(
        num_deals=num_deals,
        mean=mean,
        stderr=stderr,
        ci_low=lo,
        ci_high=hi,
        scores=scores,
        point_mean=pmean,
        point_stderr=pstderr,
        point_ci_low=plo,
        point_ci_high=phi,
        point_scores=point_scores,
        hand_levels=np.array([d.hand_levels for d in deals], dtype=np.int32),
        hand_points=np.array([d.hand_points for d in deals], dtype=np.int32),
        attacker_points=np.array([d.attacker_points for d in deals], dtype=np.int32),
        digests=np.array([d.digest for d in deals], dtype=np.uint64),
        decisions=sum(d.decisions for d in deals),
    )


def self_check(
    policy: PolicyFactory,
    num_deals: int,
    seed: int,
    **kwargs,
) -> Result:
    """Duplicate-score a policy against an independently seeded copy of itself.

    The true difference is exactly zero, so the interval this returns is a
    statement about the harness rather than about the policy: a mean that
    misses zero by more than its own error bar means the scoring, the
    seating or the clustering is wrong.

    Handing `compare` the same factory twice would be a weaker check, not
    a stronger one. The seat-pair streams would then make the two replays
    of every deal byte-identical and every deal score exactly zero, and a
    quantity that is zero by construction cannot test the arithmetic that
    is supposed to produce it.
    """
    return compare(
        salted(policy, _SELF_SALT_X),
        salted(policy, _SELF_SALT_Y),
        num_deals,
        seed,
        **kwargs,
    )


# --- cli ------------------------------------------------------------------


def main() -> int:
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Duplicate-deal policy comparison.")
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--cap", type=int, default=MOVE_CAP)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    started = time.perf_counter()
    zero = self_check(
        uniform_random_policy,
        args.deals,
        args.seed,
        cap=args.cap,
        validate=args.validate,
    )
    print("harness self-check (random vs an independent copy of random)")
    print("  " + zero.summary("null").replace("\n", "\n  "))
    print(f"  {abs(zero.mean) / max(zero.stderr, 1e-12):.2f} se from zero")

    edge = compare(
        uniform_random_policy,
        lowest_index_policy,
        args.deals,
        args.seed,
        cap=args.cap,
        validate=args.validate,
    )
    print("\nrandom vs lowest-index")
    print("  " + edge.summary("rand - low").replace("\n", "\n  "))
    print(
        f"  per-hand sd {edge.hand_stdev:.3f}  "
        f"per-deal sd {edge.deal_stdev:.3f}  "
        f"({edge.deal_stdev / max(edge.hand_stdev, 1e-12):.2f}x)"
    )

    elapsed = time.perf_counter() - started
    hands = 4 * args.deals
    print(f"\n{hands} hands in {elapsed:.1f}s ({1000 * elapsed / hands:.0f} ms/hand)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
