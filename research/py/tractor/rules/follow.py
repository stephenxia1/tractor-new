"""The follow rule: what a lead demands and whether a play supplies it.

This module is the single source of legality for the play phase. Move
enumeration derives its structure from `lead_obligations` and
`obligatory_core` rather than re-deriving it, so an enumerated move
cannot be rejected by `is_legal_follow` for a structural reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cards import NUM_TYPES, Ctx
from .combos import NKIND, TRACTOR, Component, _runs, paired_rungs


@dataclass(frozen=True, slots=True)
class Obligations:
    """What a lead demands of a follower who can fill it from the suit.

    ``pairs`` is the total number of pair slots; a tractor of length L
    supplies L of them. ``tractors`` holds every tractor component's
    length, descending -- a throw may carry more than one, and a
    follower may not withhold structure they hold, so the whole multiset
    is required rather than just the longest.
    """

    pairs: int
    tractors: tuple[int, ...]


def lead_obligations(
    components: list[Component], num_decks: int
) -> Obligations:
    pairs = 0
    tractors: list[int] = []
    for c in components:
        if c.kind == NKIND:
            if c.n >= num_decks:
                pairs += 1
        elif c.kind == TRACTOR:
            pairs += c.length
            tractors.append(c.length)
    tractors.sort(reverse=True)
    return Obligations(pairs, tuple(tractors))


def restrict(counts: np.ndarray, ctx: Ctx, suit: int) -> np.ndarray:
    """The part of a holding that lies in one effective suit."""
    return counts * ctx.suit_mask[suit]


def count_pairs(counts: np.ndarray, num_decks: int) -> int:
    return int(np.count_nonzero(counts >= num_decks))


def _profile_from_runs(runs: list[int], lengths: tuple[int, ...]) -> list[int]:
    """The most a holding's runs can supply against these tractor slots.

    Cutting a length-t piece off one end of a run of length r leaves a
    contiguous run of r - t, so only run lengths matter and no search
    over which window to take is needed. Which run feeds which slot
    does matter: runs [4, 3] against slots (3, 2, 2) yields (3, 2, 2)
    by sending the 3-run to the first slot and splitting the 4-run,
    while always drawing from the longest remaining run stops at
    (3, 2, 0).

    The assignment is therefore searched rather than taken greedily,
    maximising the profile in slot order. The state is the multiset of
    remaining run lengths, which is small: a holding of one suit yields
    at most a handful of runs and a lead at most a handful of slots.
    """
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def best(pool: tuple[int, ...], index: int) -> tuple[int, ...]:
        if index == len(lengths):
            return ()
        want = lengths[index]
        top: tuple[int, ...] | None = None
        # Taking nothing is always available and is the answer when no
        # run can supply a pair.
        for take_from in range(len(pool)):
            run = pool[take_from]
            if run < 2:
                continue
            t = min(want, run)
            rest = list(pool)
            rest[take_from] = run - t
            tail = best(tuple(sorted((r for r in rest if r), reverse=True)), index + 1)
            cand = (t,) + tail
            if top is None or cand > top:
                top = cand
        if top is None:
            return (0,) + best(pool, index + 1)
        return top

    return list(best(tuple(sorted((r for r in runs if r), reverse=True)), 0))


def hand_tractor_profile(
    ob: Obligations, counts: np.ndarray, ctx: Ctx, num_decks: int
) -> list[int]:
    """Tractor lengths `counts` must supply against this lead, slot by slot."""
    if not ob.tractors:
        return []
    rungs = paired_rungs(counts, ctx, num_decks)
    return _profile_from_runs([ln for _, ln in _runs(rungs)], ob.tractors)


def is_legal_follow(
    ob: Obligations,
    lead_count: int,
    led_suit: int,
    candidate: np.ndarray,
    hand: np.ndarray,
    ctx: Ctx,
    num_decks: int,
) -> bool:
    if int(candidate.sum()) != lead_count:
        return False
    if np.any(candidate > hand):
        return False

    hand_in = restrict(hand, ctx, led_suit)
    cand_in = restrict(candidate, ctx, led_suit)
    hand_in_total = int(hand_in.sum())

    # Spend as many led-suit cards as held, up to the lead's count.
    if int(cand_in.sum()) < min(hand_in_total, lead_count):
        return False

    # With no more led-suit cards than slots, all of them are committed
    # by the rule above and no structural choice remains.
    if hand_in_total <= lead_count:
        return True

    if ob.tractors:
        need = hand_tractor_profile(ob, hand_in, ctx, num_decks)
        have = hand_tractor_profile(ob, cand_in, ctx, num_decks)
        for i, want in enumerate(need):
            if have[i] < want:
                return False

    if ob.pairs > 0:
        want_pairs = min(count_pairs(hand_in, num_decks), ob.pairs)
        if count_pairs(cand_in, num_decks) < want_pairs:
            return False

    return True


def _take_run(
    out: np.ndarray,
    remaining: np.ndarray,
    ctx: Ctx,
    start: int,
    length: int,
    num_decks: int,
) -> None:
    for rung in range(start, start + length):
        hit = np.nonzero((remaining >= num_decks) & (ctx.strength == rung))[0]
        out[hit[0]] += num_decks
        remaining[hit[0]] -= num_decks


def obligatory_core(
    ob: Obligations,
    lead_count: int,
    in_led: np.ndarray,
    ctx: Ctx,
    num_decks: int,
) -> np.ndarray:
    """The led-suit cards a follow must contain, built greedily.

    Every tractor the profile demands, then pairs, then whatever led-suit
    cards are left. `in_led` must already be restricted to the led suit.
    """
    if int(in_led.sum()) <= lead_count:
        return in_led.copy()

    out = np.zeros(NUM_TYPES, dtype=np.int8)
    remaining = in_led.copy()
    have_pairs = 0

    for t in hand_tractor_profile(ob, in_led, ctx, num_decks):
        if t < 2:
            continue
        if int(out.sum()) + num_decks * t > lead_count:
            continue
        rungs = paired_rungs(remaining, ctx, num_decks)
        runs = [(s, ln) for s, ln in _runs(rungs) if ln >= t]
        if not runs:
            continue
        _take_run(out, remaining, ctx, runs[0][0], t, num_decks)
        have_pairs += t

    want_pairs = min(count_pairs(in_led, num_decks), ob.pairs)
    for t in np.nonzero(remaining >= num_decks)[0]:
        if have_pairs >= want_pairs:
            break
        if int(out.sum()) + num_decks > lead_count:
            continue
        out[t] += num_decks
        remaining[t] -= num_decks
        have_pairs += 1

    for t in np.nonzero(remaining)[0]:
        while remaining[t] > 0 and int(out.sum()) < lead_count:
            out[t] += 1
            remaining[t] -= 1

    return out


def canonical_follow(
    ob: Obligations,
    lead_count: int,
    led_suit: int,
    hand: np.ndarray,
    ctx: Ctx,
    num_decks: int,
) -> np.ndarray:
    """One follow that satisfies every obligation. Always legal."""
    out = obligatory_core(
        ob, lead_count, restrict(hand, ctx, led_suit), ctx, num_decks
    )
    if int(out.sum()) >= lead_count:
        return out
    spare = (hand - out).astype(np.int8)
    for t in np.nonzero(spare)[0]:
        while spare[t] > 0 and int(out.sum()) < lead_count:
            out[t] += 1
            spare[t] -= 1
        if int(out.sum()) >= lead_count:
            break
    return out
