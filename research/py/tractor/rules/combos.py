"""Combination structure: pairs, tractors, throws, and comparison.

A play is a count vector over card types. Every combination lives inside
one effective suit, so the functions here take the suit as given and
operate on the counts restricted to it.

Two notions of position are kept apart because they differ:

* a *pair* is ``num_decks`` copies of one card **type**;
* a *tractor* is a run of consecutive **strength rungs** each of which
  holds a pair.

The three off-suit level cards share a single rung, so two off-suit level
pairs occupy one rung between them and never chain with each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cards import NUM_TYPES, TRUMP, Ctx

SINGLE = 0
NKIND = 1
TRACTOR = 2

KIND_NAMES = {SINGLE: "single", NKIND: "nkind", TRACTOR: "tractor"}


@dataclass(frozen=True, slots=True)
class Component:
    """One structural piece of a play."""

    kind: int
    top: int  # strength of the highest rung in the piece
    length: int  # rungs spanned: 1 for a single or a pair
    n: int  # copies per rung: 1 for a single, num_decks for the rest

    @property
    def num_cards(self) -> int:
        return self.length * self.n

    @property
    def shape(self) -> tuple[int, int, int]:
        """Identity for shape matching, independent of rank."""
        return (self.kind, self.length, self.n)

    def __repr__(self) -> str:
        return f"{KIND_NAMES[self.kind]}(top={self.top}, len={self.length})"


def play_suit(counts: np.ndarray, ctx: Ctx) -> int | None:
    """The single effective suit of a play, or None if it spans several."""
    present = np.nonzero(counts)[0]
    if present.size == 0:
        return None
    suits = ctx.eff[present]
    first = int(suits[0])
    if np.any(suits != first):
        return None
    return first


def paired_rungs(counts: np.ndarray, ctx: Ctx, num_decks: int) -> np.ndarray:
    """Sorted strengths at which the play holds at least one full pair."""
    idx = np.nonzero(counts >= num_decks)[0]
    if idx.size == 0:
        return np.empty(0, dtype=np.int8)
    return np.unique(ctx.strength[idx])


def _runs(sorted_rungs: np.ndarray) -> list[tuple[int, int]]:
    """Maximal consecutive runs as ``(start_strength, length)``."""
    out: list[tuple[int, int]] = []
    if sorted_rungs.size == 0:
        return out
    start = int(sorted_rungs[0])
    prev = start
    for v in sorted_rungs[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
            continue
        out.append((start, prev - start + 1))
        start = prev = v
    out.append((start, prev - start + 1))
    return out


def classify(counts: np.ndarray, ctx: Ctx, num_decks: int) -> Component | None:
    """The play as a single combination, or None if it is not one."""
    total = int(counts.sum())
    if total == 0:
        return None
    if play_suit(counts, ctx) is None:
        return None

    present = np.nonzero(counts)[0]
    if total == 1:
        return Component(SINGLE, int(ctx.strength[present[0]]), 1, 1)

    # Every present type must be held exactly `num_decks` times for the
    # play to be a pair or a tractor.
    if np.any(counts[present] != num_decks):
        return None
    if present.size == 1:
        return Component(NKIND, int(ctx.strength[present[0]]), 1, num_decks)

    rungs = np.unique(ctx.strength[present])
    if rungs.size != present.size:
        # Two types on one rung: off-suit level cards, which do not chain.
        return None
    runs = _runs(rungs)
    if len(runs) != 1:
        return None
    start, length = runs[0]
    return Component(TRACTOR, start + length - 1, length, num_decks)


def decompose_with_cards(
    counts: np.ndarray, ctx: Ctx, num_decks: int
) -> list[tuple[Component, np.ndarray]] | None:
    """Greedy decomposition, each component paired with its own cards.

    A ``Component`` records strength, not identity, and several card
    types can share one strength rung -- the three off-suit level cards
    all do. Recovering a component's cards afterwards by searching for
    types at its rung therefore picks an arbitrary one of them, which is
    wrong wherever the specific cards matter rather than their rank.
    Callers that need the cards take them from here.
    """
    if play_suit(counts, ctx) is None:
        return None

    remaining = counts.copy()
    out: list[tuple[Component, np.ndarray]] = []

    def take(t: int, n: int) -> np.ndarray:
        piece = np.zeros(NUM_TYPES, dtype=np.int8)
        piece[t] = n
        remaining[t] -= n
        return piece

    while True:
        rungs = paired_rungs(remaining, ctx, num_decks)
        runs = [r for r in _runs(rungs) if r[1] >= 2]
        if not runs:
            break
        # Longest first; ties to the higher run.
        runs.sort(key=lambda r: (r[1], r[0]), reverse=True)
        start, length = runs[0]
        cards = np.zeros(NUM_TYPES, dtype=np.int8)
        for s in range(start, start + length):
            hit = np.nonzero((remaining >= num_decks) & (ctx.strength == s))[0]
            cards += take(int(hit[0]), num_decks)
        out.append(
            (Component(TRACTOR, start + length - 1, length, num_decks), cards)
        )

    for t in np.nonzero(remaining >= num_decks)[0]:
        comp = Component(NKIND, int(ctx.strength[t]), 1, num_decks)
        out.append((comp, take(int(t), num_decks)))

    for t in np.nonzero(remaining)[0]:
        for _ in range(int(remaining[t])):
            comp = Component(SINGLE, int(ctx.strength[t]), 1, 1)
            out.append((comp, take(int(t), 1)))

    return out


def decompose(counts: np.ndarray, ctx: Ctx, num_decks: int) -> list[Component] | None:
    """Greedy decomposition: longest tractors, then pairs, then singles.

    Returns None if the play spans more than one effective suit. A play
    that is a single combination comes back as a one-element list.
    """
    got = decompose_with_cards(counts, ctx, num_decks)
    return None if got is None else [c for c, _cards in got]


def is_throw(components: list[Component]) -> bool:
    return len(components) > 1


def component_beats(a: Component, b: Component, a_suit: int, b_suit: int) -> bool:
    """Does `a` outrank `b`? Both must be the same shape to compete."""
    if a.shape != b.shape:
        return False
    a_trump = a_suit == TRUMP
    b_trump = b_suit == TRUMP
    if a_trump != b_trump:
        return a_trump
    if not a_trump and a_suit != b_suit:
        return False
    return a.top > b.top


def _bipartite_all_beat(
    cands: list[Component],
    incs: list[Component],
    cand_suit: int,
    inc_suit: int,
) -> bool:
    """Every incumbent gets a distinct candidate that outranks it."""
    if len(cands) != len(incs):
        return False
    m = len(incs)
    edges = [
        [component_beats(c, i, cand_suit, inc_suit) for i in incs] for c in cands
    ]
    match = [-1] * m

    def augment(ci: int, seen: list[bool]) -> bool:
        for j in range(m):
            if not edges[ci][j] or seen[j]:
                continue
            seen[j] = True
            if match[j] == -1 or augment(match[j], seen):
                match[j] = ci
                return True
        return False

    matched = 0
    for ci in range(len(cands)):
        if augment(ci, [False] * m):
            matched += 1
    return matched == m


def _tightness(c: Component) -> int:
    if c.kind == TRACTOR:
        return 1000 + c.length
    if c.kind == NKIND:
        return 100 + c.n
    return 1


def _pieces_of_shape(
    counts: np.ndarray, inc: Component, ctx: Ctx, num_decks: int
) -> list[np.ndarray]:
    """Card groups in `counts` matching `inc`'s shape. Shape only."""
    out: list[np.ndarray] = []
    if inc.kind == SINGLE:
        for t in np.nonzero(counts)[0]:
            piece = np.zeros(NUM_TYPES, dtype=np.int8)
            piece[t] = 1
            out.append(piece)
        return out
    if inc.kind == NKIND:
        for t in np.nonzero(counts >= inc.n)[0]:
            piece = np.zeros(NUM_TYPES, dtype=np.int8)
            piece[t] = inc.n
            out.append(piece)
        return out
    rungs = paired_rungs(counts, ctx, num_decks)
    for start, run_len in _runs(rungs):
        for off in range(run_len - inc.length + 1):
            piece = np.zeros(NUM_TYPES, dtype=np.int8)
            for rung in range(start + off, start + off + inc.length):
                hit = np.nonzero((counts >= num_decks) & (ctx.strength == rung))[0]
                piece[hit[0]] = num_decks
            out.append(piece)
    return out


def _singles_cover(
    counts: np.ndarray,
    comps: list[Component],
    cand_suit: int,
    inc_suit: int,
    ctx: Ctx,
) -> bool:
    """Every remaining card covers one remaining single slot."""
    if int(counts.sum()) != len(comps):
        return False
    singles = [
        Component(SINGLE, int(ctx.strength[t]), 1, 1)
        for t in np.nonzero(counts)[0]
        for _ in range(int(counts[t]))
    ]
    return _bipartite_all_beat(singles, comps, cand_suit, inc_suit)


def _cover_partition(
    counts: np.ndarray,
    comps: list[Component],
    cand_suit: int,
    inc_suit: int,
    ctx: Ctx,
    num_decks: int,
) -> bool:
    if not comps:
        return int(counts.sum()) == 0
    # Components arrive tightest-first, so once the head is a single the
    # whole tail is singles and the rest is an exact assignment rather
    # than a search.
    if comps[0].kind == SINGLE:
        return _singles_cover(counts, comps, cand_suit, inc_suit, ctx)
    inc = comps[0]
    rest = comps[1:]
    for piece in _pieces_of_shape(counts, inc, ctx, num_decks):
        got = classify(piece, ctx, num_decks)
        if got is None or not component_beats(got, inc, cand_suit, inc_suit):
            continue
        if _cover_partition(
            (counts - piece).astype(np.int8), rest, cand_suit, inc_suit, ctx, num_decks
        ):
            return True
    return False


def covers(
    cand_counts: np.ndarray,
    inc_components: list[Component],
    inc_suit: int,
    ctx: Ctx,
    num_decks: int,
) -> bool:
    """Does the candidate play cover every component of the incumbent?

    The incumbent's structure is fixed; only the candidate is
    re-decomposed to match it. A tractor may stand in for pairs and a
    pair for singles, but singles never merge into a pair.
    """
    cand_suit = play_suit(cand_counts, ctx)
    if cand_suit is None:
        return False
    cand_components = decompose(cand_counts, ctx, num_decks)
    if cand_components is None:
        return False

    if len(cand_components) == len(inc_components) and _bipartite_all_beat(
        cand_components, inc_components, cand_suit, inc_suit
    ):
        return True

    need = sum(c.num_cards for c in inc_components)
    if int(cand_counts.sum()) != need:
        return False
    ordered = sorted(inc_components, key=_tightness, reverse=True)
    return _cover_partition(
        cand_counts, ordered, cand_suit, inc_suit, ctx, num_decks
    )


def play_rank(components: list[Component], suit: int) -> tuple[int, int]:
    """Ordering key among plays that cover the lead.

    Trump beats non-trump; within one effective suit the higher play by
    its highest card. Ties go to whoever played first, which the caller
    enforces with a strict comparison.
    """
    top = -1
    for c in components:
        if c.top > top:
            top = c.top
    return (1 if suit == TRUMP else 0, top)
