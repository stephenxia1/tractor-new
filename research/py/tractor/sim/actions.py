"""Legal move enumeration for the play phase.

Every move this module emits passes ``is_legal_follow``; legality is
never re-derived here. Where a cap stops the enumeration short, the
count is reported rather than dropped silently, because a silent cap
reads downstream as full coverage.

Throws are built from *live components* rather than from a fixed set of
shapes. A throw fails if any one of its components can be beaten by a
single other hand, so liveness is a per-component property and pruning
there removes whole families of doomed throws before they are formed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..rules.cards import NUM_TYPES, TRUMP, Ctx
from ..rules.combos import NKIND, SINGLE, TRACTOR, Component, _runs, paired_rungs
from ..rules.follow import (
    Obligations,
    canonical_follow,
    count_pairs,
    hand_tractor_profile,
    is_legal_follow,
    restrict,
)

# Provably safe, provably doomed, or neither. A component with no unseen
# beater at all is safe; one whose single-card beaters outnumber the
# hiding places in the kitty has at least one beater sitting in a hand.
SAFE = 0
UNCERTAIN = 1
DOOMED = 2


@dataclass(slots=True)
class MoveSet:
    """Legal moves at one decision, with what the caps discarded."""

    moves: list[np.ndarray] = field(default_factory=list)
    truncated: int = 0  # candidates dropped for exceeding the cap
    unenumerated: bool = False  # the exact set was too large to build

    def __len__(self) -> int:
        return len(self.moves)


def sub_multisets(counts: np.ndarray, k: int, cap: int) -> list[np.ndarray] | None:
    """Every k-card sub-multiset of `counts`, or None past `cap`."""
    if k == 0:
        return [np.zeros(NUM_TYPES, dtype=np.int8)]
    types = np.nonzero(counts)[0]
    if int(counts.sum()) < k:
        return []
    # Suffix capacities let the recursion prune branches that cannot
    # reach k rather than discovering it at the leaf.
    suffix = np.zeros(len(types) + 1, dtype=np.int32)
    for i in range(len(types) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + int(counts[types[i]])

    out: list[np.ndarray] = []
    cur = np.zeros(NUM_TYPES, dtype=np.int8)
    overflow = False

    def rec(i: int, left: int) -> None:
        nonlocal overflow
        if overflow:
            return
        if left == 0:
            if len(out) >= cap:
                overflow = True
                return
            out.append(cur.copy())
            return
        if i >= len(types) or suffix[i] < left:
            return
        t = int(types[i])
        for take in range(min(int(counts[t]), left), -1, -1):
            cur[t] = take
            rec(i + 1, left - take)
            cur[t] = 0
            if overflow:
                return

    rec(0, k)
    return None if overflow else out


def _finalize(moves: list[np.ndarray], cap: int, out: MoveSet) -> None:
    """Fit a move list into `cap`, losing coverage rather than one end.

    Candidates are generated in card order, so keeping the first `cap`
    of them drops the high cards systematically. An evenly spaced
    subsample keeps the choice deterministic while spreading what
    survives across the whole space.
    """
    if len(moves) <= cap:
        out.moves = moves
        return
    out.truncated = len(moves) - cap
    idx = sorted(set(np.linspace(0, len(moves) - 1, cap).astype(int).tolist()))
    out.moves = [moves[i] for i in idx]


# --- follows --------------------------------------------------------------


def legal_follows(
    ob: Obligations,
    lead_count: int,
    led_suit: int,
    hand: np.ndarray,
    ctx: Ctx,
    num_decks: int,
    cap: int = 64,
) -> MoveSet:
    out = MoveSet()
    in_led = restrict(hand, ctx, led_suit)
    in_led_total = int(in_led.sum())

    if in_led_total <= lead_count:
        # Every led-suit card is forced; only the filler is a choice.
        spare = (hand - in_led).astype(np.int8)
        fills = sub_multisets(spare, lead_count - in_led_total, cap * 8)
        if fills is None:
            out.unenumerated = True
            out.moves.append(
                canonical_follow(ob, lead_count, led_suit, hand, ctx, num_decks)
            )
            return out
        _finalize([(in_led + f).astype(np.int8) for f in fills], cap, out)
        return out

    subs = sub_multisets(in_led, lead_count, cap * 32)
    if subs is None:
        # Too many led-suit selections to build exactly. The canonical
        # follow is always legal and keeps the seat playable; Phase 2
        # measures how often this path is taken.
        out.unenumerated = True
        out.moves.append(
            canonical_follow(ob, lead_count, led_suit, hand, ctx, num_decks)
        )
        return out

    legal = [
        s
        for s in subs
        if is_legal_follow(ob, lead_count, led_suit, s, hand, ctx, num_decks)
    ]
    _finalize(legal, cap, out)
    return out


# --- leads ----------------------------------------------------------------


def _components_in_suit(
    in_suit: np.ndarray, ctx: Ctx, num_decks: int
) -> list[tuple[Component, np.ndarray]]:
    """Every single, pair and tractor window available in one suit."""
    out: list[tuple[Component, np.ndarray]] = []

    for t in np.nonzero(in_suit)[0]:
        cards = np.zeros(NUM_TYPES, dtype=np.int8)
        cards[t] = 1
        out.append((Component(SINGLE, int(ctx.strength[t]), 1, 1), cards))

    for t in np.nonzero(in_suit >= num_decks)[0]:
        cards = np.zeros(NUM_TYPES, dtype=np.int8)
        cards[t] = num_decks
        out.append((Component(NKIND, int(ctx.strength[t]), 1, num_decks), cards))

    rungs = paired_rungs(in_suit, ctx, num_decks)
    for start, run_len in _runs(rungs):
        for length in range(2, run_len + 1):
            for off in range(run_len - length + 1):
                cards = np.zeros(NUM_TYPES, dtype=np.int8)
                for rung in range(start + off, start + off + length):
                    hit = np.nonzero((in_suit >= num_decks) & (ctx.strength == rung))[0]
                    cards[hit[0]] = num_decks
                top = start + off + length - 1
                out.append((Component(TRACTOR, top, length, num_decks), cards))
    return out


def component_status(
    comp: Component,
    suit: int,
    unseen: np.ndarray,
    ctx: Ctx,
    num_decks: int,
    kitty_slots: int,
) -> tuple[int, int]:
    """Is this component safe to throw, and how many beaters are unseen?

    A component is beaten by a same-shape, higher group held by one other
    player. Counting single-card beaters is enough to prove doom: each
    can only be neutralised by sitting in the kitty, so more of them than
    the kitty can hold puts at least one in a hand. Multi-card beaters
    can also be neutralised by splitting across hands, so they never
    prove doom and only make the component uncertain.
    """
    higher = (ctx.eff == suit) & (ctx.strength > comp.top)
    if comp.kind == SINGLE:
        beaters = int(unseen[higher].sum())
        single_card_beaters = beaters
    elif comp.kind == NKIND:
        beaters = int(np.count_nonzero(unseen[higher] >= num_decks))
        single_card_beaters = 0
    else:
        # A tractor outranks this one when its *top* rung is higher; its
        # lower rungs may sit at or below `comp.top`. Masking the unseen
        # cards to strictly higher rungs first would hide exactly those,
        # and two holdings can share a rung on the off-suit level cards.
        in_suit = np.where(ctx.eff == suit, unseen, 0).astype(np.int8)
        beaters = 0
        for start, run_len in _runs(paired_rungs(in_suit, ctx, num_decks)):
            for off in range(run_len - comp.length + 1):
                if start + off + comp.length - 1 > comp.top:
                    beaters += 1
        single_card_beaters = 0

    if beaters == 0:
        return SAFE, 0
    if single_card_beaters > kitty_slots:
        return DOOMED, beaters
    return UNCERTAIN, beaters


def legal_leads(
    hand: np.ndarray,
    unseen: np.ndarray,
    ctx: Ctx,
    num_decks: int,
    kitty_slots: int,
    cap: int = 64,
    max_throw_components: int = 4,
) -> MoveSet:
    out = MoveSet()
    # Generation ceiling, well above any cap. Reaching it is recorded
    # separately from capping, since it means the exact set was never
    # built in the first place.
    hard = max(cap * 8, 1024)
    found: list[np.ndarray] = []
    # A tractor's card multiset is reachable twice: once as a tractor
    # component, once as a throw of the pairs it is built from. Both
    # describe the same play, and a duplicate would take two slots in
    # the padded tensor and twice the weight under sampling.
    emitted: set[bytes] = set()

    def offer(cards: np.ndarray) -> bool:
        key = cards.tobytes()
        if key in emitted:
            return True
        if len(found) >= hard:
            out.unenumerated = True
            return False
        emitted.add(key)
        found.append(cards)
        return True

    for suit in range(5):
        in_suit = restrict(hand, ctx, suit)
        if int(in_suit.sum()) == 0:
            continue
        pieces = _components_in_suit(in_suit, ctx, num_decks)

        for _comp, cards in pieces:
            if not offer(cards):
                break

        # Throws: subsets of two or more components that are not
        # provably beaten, and that use each card at most as often as
        # the hand holds it.
        live = [
            (comp, cards)
            for comp, cards in pieces
            if component_status(comp, suit, unseen, ctx, num_decks, kitty_slots)[0]
            != DOOMED
        ]
        if len(live) < 2:
            continue
        # Strongest first, so a truncated enumeration keeps the throws
        # most likely to survive the block check.
        live.sort(key=lambda pc: pc[0].top, reverse=True)
        _emit_throws(live, in_suit, num_decks, max_throw_components, offer, out)

    _finalize(found, cap, out)
    return out


def _emit_throws(
    live: list[tuple[Component, np.ndarray]],
    in_suit: np.ndarray,
    num_decks: int,
    max_components: int,
    offer,
    out: MoveSet,
) -> None:
    seen: set[bytes] = set()
    chosen: list[np.ndarray] = []

    def rec(start: int, used: np.ndarray) -> None:
        if len(chosen) >= 2:
            cards = used.copy()
            key = cards.tobytes()
            if key not in seen:
                seen.add(key)
                if not offer(cards):
                    return
        if len(chosen) >= max_components:
            # Deeper subsets exist and are legal leads; record that the
            # enumeration stopped rather than that it finished.
            if len(live) > len(chosen):
                out.unenumerated = True
            return
        for i in range(start, len(live)):
            cards = live[i][1]
            if np.any(used + cards > in_suit):
                continue
            chosen.append(cards)
            rec(i + 1, (used + cards).astype(np.int8))
            chosen.pop()

    rec(0, np.zeros(NUM_TYPES, dtype=np.int8))
