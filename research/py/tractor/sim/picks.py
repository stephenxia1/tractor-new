"""Card-by-card play selection: which next pick is legal (E7).

A play is chosen as an ascending sequence of card-type picks
``t_1 <= t_2 <= ...`` (the 54-slot index of ``rules/cards.py``) ended by
``SUBMIT``. The order is fixed, so every play has exactly one pick
sequence -- its types ascending, each repeated by its count -- and its
probability under a policy over picks is exact.

This module is the legality oracle for that action space. Given the
cards picked so far ``P`` (counts) and the last type picked ``l`` (-1
before the first pick):

* **Lead.** With ``P`` empty, every held type. After that, any held type
  ``t >= l`` with a copy left, in the effective suit of ``P``; SUBMIT is
  legal once ``P`` is non-empty. Every one-suit multiset is a legal lead
  (a multi-component one is a throw, and a blocked throw is resolved by
  ``PlayState._resolve_throw``).
* **Follow** (lead of ``L`` cards in led suit ``s``). A type ``t`` is
  legal iff ``t >= l``, a copy is left, ``|P| < L``, and ``P + t`` can
  still be completed, using only types ``>= t``, to a play that
  ``rules/follow.py::is_legal_follow`` accepts. SUBMIT is legal iff
  ``|P| = L``, and is then the only legal action.

Follow feasibility is decided in closed form or by a small cached
search rather than by enumerating completions:

(a) The hand holds no more led-suit cards than the lead (``H_in <= L``).
    Every led-suit card must be played. A prefix is completable iff no
    led-suit type below the frontier has copies left behind (the
    ascending order can never return to it), the off-suit cards picked
    plus all led-suit cards fit in ``L``, and enough cards remain at or
    above the frontier to reach ``L``.

(b) ``H_in > L``. The play must be all led-suit. Without pair
    obligations (``want_pairs = 0``) the only question is whether enough
    led-suit cards remain. With them, the structure of a play depends
    only on its set ``S`` of paired types: ``count_pairs`` is ``|S|`` and
    the tractor profile is a function of the rungs ``S`` occupies. The
    structural test of ``is_legal_follow`` -- at least ``want_pairs``
    pairs and a profile at least ``need`` slot by slot, both measured on
    the whole hand -- is **upward closed** over subsets of the hand:
    the profile is the lexicographic maximum over assignments of
    disjoint rung segments to the lead's tractor slots, a superset of
    rungs offers a superset of assignments, and a candidate drawn from
    the hand can never exceed the hand's own maximum, so "at least
    ``need`` slot by slot" is the same as "equal to ``need``" and holds
    for every superset once it holds for a set. (The profile itself is
    *not* monotone slot by slot: against slots (4, 2), runs [2, 2]
    give (2, 2) but filling the gap to make [5] gives (4, 0). The
    tests pin both facts.)

    So a prefix ``Q`` with frontier ``f`` is completable iff some set
    ``T`` of extra pairs among the pairable led-suit types ``>= f`` not
    yet paired in ``Q`` makes ``paired(Q) | T`` structurally legal, fits
    (``|Q| + sum(num_decks - Q[x]) <= L``), and leaves enough unpaired
    led-suit cards ``>= f`` to pad to exactly ``L`` without completing
    another pair. When only the pair count is demanded that is a closed
    form (the cheapest pairs that meet both the count and the padding);
    with a tractor demand it is a search over a handful of pairable
    types, pruned by cost and by upward closure. Answers are cached per
    rules object.

Masks are 55 bits (bit ``i`` is action ``i``) and are stored packed in
a ``uint64`` per step.
"""

from __future__ import annotations

from bisect import bisect_left

import numpy as np

from ..rules.cards import NUM_TYPES, Ctx, ids_from_counts
from ..rules.combos import decompose, play_suit
from ..rules.follow import (
    _profile_from_runs,
    count_pairs,
    hand_tractor_profile,
    lead_obligations,
    restrict,
)

SUBMIT = NUM_TYPES  # 54
NUM_PICK_ACTIONS = NUM_TYPES + 1  # 55
# A hand of 25 cards (two decks) is the longest play, plus SUBMIT.
MAX_PICKS = 26

_SUBMIT_BIT = 1 << SUBMIT
_SHIFTS = np.arange(NUM_PICK_ACTIONS, dtype=np.uint64)
_ONE = np.uint64(1)

# (rung bitmask, tractor slots) -> profile. The profile of a rung set
# against a lead's slots does not depend on the position, so the cache
# is shared by every rules object.
_PROFILE_CACHE: dict[tuple[int, tuple[int, ...]], tuple[int, ...]] = {}


def _runs_of_mask(mask: int) -> list[int]:
    """Lengths of the maximal runs of set bits."""
    out: list[int] = []
    run = 0
    while mask:
        if mask & 1:
            run += 1
        elif run:
            out.append(run)
            run = 0
        mask >>= 1
    if run:
        out.append(run)
    return out


def _profile(rung_mask: int, tractors: tuple[int, ...]) -> tuple[int, ...]:
    key = (rung_mask, tractors)
    got = _PROFILE_CACHE.get(key)
    if got is None:
        got = tuple(_profile_from_runs(_runs_of_mask(rung_mask), tractors))
        _PROFILE_CACHE[key] = got
    return got


# --- mask packing -----------------------------------------------------------


def pack_mask(mask: np.ndarray) -> np.uint64:
    """A bool[55] mask as one uint64, bit i holding action i."""
    return pack_masks(np.asarray(mask, dtype=bool)[None])[0]


def unpack_mask(packed: "np.uint64 | int") -> np.ndarray:
    """The bool[55] mask a uint64 holds."""
    return ((np.uint64(packed) >> _SHIFTS) & _ONE).astype(bool)


def pack_masks(masks: np.ndarray) -> np.ndarray:
    """bool[..., 55] -> uint64[...]."""
    m = np.asarray(masks, dtype=bool)
    if m.shape[-1] != NUM_PICK_ACTIONS:
        raise ValueError(f"mask has {m.shape[-1]} entries, want {NUM_PICK_ACTIONS}")
    packed = np.packbits(m, axis=-1, bitorder="little")  # (..., 7)
    wide = np.zeros(m.shape[:-1] + (8,), dtype=np.uint8)
    wide[..., : packed.shape[-1]] = packed
    return wide.view("<u8")[..., 0].astype(np.uint64)


def unpack_masks(packed: np.ndarray) -> np.ndarray:
    """uint64[...] -> bool[..., 55]."""
    u = np.asarray(packed, dtype=np.uint64)
    return ((u[..., None] >> _SHIFTS) & _ONE).astype(bool)


def bits_to_mask(bits: int) -> np.ndarray:
    """A Python-int bitmask (as the rules compute it) as bool[55]."""
    return ((np.uint64(bits) >> _SHIFTS) & _ONE).astype(bool)


# --- pick sequences -----------------------------------------------------------


def ascending_picks(play: np.ndarray) -> list[int]:
    """The unique pick sequence of a play: its types ascending, each
    repeated by its count, then SUBMIT."""
    return ids_from_counts(np.asarray(play)) + [SUBMIT]


def play_from_picks(picks: "list[int] | np.ndarray") -> np.ndarray:
    """The play a pick sequence spells (SUBMIT and -1 padding ignored)."""
    out = np.zeros(NUM_TYPES, dtype=np.int8)
    for a in picks:
        a = int(a)
        if 0 <= a < NUM_TYPES:
            out[a] += 1
    return out


# --- the rules ----------------------------------------------------------------


class PickRules:
    """Legal next picks for one seat at one decision.

    Built from raw inputs -- the seat's hand, the trump context, the
    deck count and, for a follow, the counts of the trick's lead -- or
    from a ``PlayState`` with ``for_state``. Everything that depends only
    on the decision is computed here; ``mask`` then costs a pass over the
    held types, and feasibility answers for structured follows are cached
    on the object.
    """

    __slots__ = (
        "hand",
        "ctx",
        "num_decks",
        "is_lead",
        "lead",
        "lead_count",
        "led_suit",
        "obligations",
        "hand_in",
        "in_total",
        "all_in",
        "want_pairs",
        "need",
        "check_profile",
        "structured",
        "_held",
        "_held_bits",
        "_suit_bits",
        "_eff",
        "_is_in",
        "_in_types",
        "_pair_types",
        "_rung_bit",
        "_suffix_all",
        "_suffix_in",
        "_suffix_unpaired",
        "_feasible_cache",
        "_struct_cache",
    )

    def __init__(
        self,
        hand: np.ndarray,
        ctx: Ctx,
        num_decks: int = 2,
        lead: np.ndarray | None = None,
    ) -> None:
        hand = np.asarray(hand, dtype=np.int8)
        h = [int(v) for v in hand.tolist()]
        self.hand = h
        self.ctx = ctx
        self.num_decks = num_decks
        eff = [int(v) for v in ctx.eff.tolist()]
        self._eff = eff
        self._held = [t for t in range(NUM_TYPES) if h[t] > 0]
        self._held_bits = sum(1 << t for t in self._held)
        self._suit_bits = [
            sum(1 << t for t in self._held if eff[t] == s) for s in range(5)
        ]
        self._feasible_cache: dict[tuple[int, int, int, int], bool] = {}
        self._struct_cache: dict[int, bool] = {}

        if lead is None:
            self.is_lead = True
            self.lead = None
            self.lead_count = 0
            self.led_suit = -1
            self.obligations = None
            self.hand_in = None
            self.in_total = 0
            self.all_in = False
            self.want_pairs = 0
            self.need = ()
            self.check_profile = False
            self.structured = False
            return

        lead = np.asarray(lead, dtype=np.int8)
        suit = play_suit(lead, ctx)
        comps = decompose(lead, ctx, num_decks)
        if suit is None or comps is None:
            raise ValueError("lead spans more than one effective suit")
        ob = lead_obligations(comps, num_decks)
        L = int(lead.sum())
        hand_in = restrict(hand, ctx, suit).astype(np.int8)
        h_in = [int(v) for v in hand_in.tolist()]
        in_total = sum(h_in)

        self.is_lead = False
        self.lead = lead.copy()
        self.lead_count = L
        self.led_suit = suit
        self.obligations = ob
        self.hand_in = h_in
        self.in_total = in_total
        self.all_in = in_total <= L

        # Both measured on the whole led-suit holding, exactly as
        # ``is_legal_follow`` measures them -- not on what the ascending
        # order still allows.
        self.want_pairs = (
            min(count_pairs(hand_in, num_decks), ob.pairs) if ob.pairs > 0 else 0
        )
        self.need = (
            tuple(hand_tractor_profile(ob, hand_in, ctx, num_decks))
            if ob.tractors
            else ()
        )
        self.check_profile = any(v > 0 for v in self.need)
        # want_pairs == 0 means the holding has no pair, hence no tractor
        # either, and both structural tests pass for any candidate.
        self.structured = (not self.all_in) and self.want_pairs > 0

        self._is_in = [v > 0 for v in h_in]
        self._in_types = [t for t in range(NUM_TYPES) if h_in[t] > 0]
        self._pair_types = [t for t in self._in_types if h_in[t] >= num_decks]
        strength = ctx.strength.tolist()
        self._rung_bit = [1 << int(strength[t]) for t in range(NUM_TYPES)]

        suffix_all = [0] * (NUM_TYPES + 2)
        suffix_in = [0] * (NUM_TYPES + 2)
        suffix_unpaired = [0] * (NUM_TYPES + 2)
        for t in range(NUM_TYPES - 1, -1, -1):
            suffix_all[t] = suffix_all[t + 1] + h[t]
            suffix_in[t] = suffix_in[t + 1] + h_in[t]
            suffix_unpaired[t] = suffix_unpaired[t + 1] + min(h_in[t], num_decks - 1)
        self._suffix_all = suffix_all
        self._suffix_in = suffix_in
        self._suffix_unpaired = suffix_unpaired

    @classmethod
    def for_state(cls, state) -> "PickRules":
        """Rules for the seat to act in a ``PlayState``."""
        hand = state.hands[state.to_act]
        lead = None if state.is_lead else state.trick_plays[0]
        return cls(hand, state.ctx, state.num_decks, lead)

    # --- public queries -----------------------------------------------------

    def mask(self, picked: np.ndarray, last: int | None = None) -> np.ndarray:
        """bool[55]: the legal next actions after picking ``picked``.

        ``last`` is the last type picked, -1 before the first pick; left
        out, it is the highest type in ``picked``.
        """
        return bits_to_mask(self.mask_bits(picked, last))

    def mask_bits(self, picked: np.ndarray, last: int | None = None) -> int:
        """The legal next actions as a Python int, bit i for action i."""
        P = [int(v) for v in np.asarray(picked).tolist()]
        n = sum(P)
        top = -1
        if n:
            for t in range(NUM_TYPES - 1, -1, -1):
                if P[t]:
                    top = t
                    break
        if last is None or last < top:
            last = top
        return self._bits(P, int(last), n)

    def cursor(self) -> "PickCursor":
        return PickCursor(self)

    def trace(self, play: np.ndarray) -> tuple[list[int], list[int]]:
        """The pick sequence of ``play`` and the mask before each pick.

        Raises ValueError if any pick of the sequence is not legal, i.e.
        if the play is not a legal play here.
        """
        picks = ascending_picks(play)
        P = [0] * NUM_TYPES
        last, n = -1, 0
        masks: list[int] = []
        for a in picks:
            bits = self._bits(P, last, n)
            if not (bits >> a) & 1:
                raise ValueError(f"pick {a} is not legal after {n} cards")
            masks.append(bits)
            if a != SUBMIT:
                P[a] += 1
                last = a
                n += 1
        return picks, masks

    def encode(self, play: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """A play as a rollout row: picks int8[MAX_PICKS] (-1 padded),
        packed masks uint64[MAX_PICKS] (0 padded), and the pick count."""
        picks, masks = self.trace(play)
        if len(picks) > MAX_PICKS:
            raise ValueError(f"{len(picks)} picks exceed MAX_PICKS={MAX_PICKS}")
        out_p = np.full(MAX_PICKS, -1, dtype=np.int8)
        out_m = np.zeros(MAX_PICKS, dtype=np.uint64)
        out_p[: len(picks)] = picks
        out_m[: len(masks)] = np.array(masks, dtype=np.uint64)
        return out_p, out_m, len(picks)

    # --- internals ----------------------------------------------------------

    def _bits(self, P: list[int], last: int, n: int) -> int:
        if self.is_lead:
            return self._lead_bits(P, last, n)
        return self._follow_bits(P, last, n)

    def _lead_bits(self, P: list[int], last: int, n: int) -> int:
        if n == 0:
            return self._held_bits
        bits = self._suit_bits[self._eff[last]] & ~((1 << last) - 1)
        if P[last] >= self.hand[last]:
            bits &= ~(1 << last)
        return bits | _SUBMIT_BIT

    def _follow_bits(self, P: list[int], last: int, n: int) -> int:
        L = self.lead_count
        if n >= L:
            return _SUBMIT_BIT if n == L else 0
        h = self.hand
        lo = last if last > 0 else 0
        bits = 0

        if self.all_in:
            # Every led-suit card is committed. The first led-suit type
            # with copies still unpicked bounds the next pick: going past
            # it would leave it behind for good.
            frontier = NUM_TYPES
            picked_in = 0
            for x in self._in_types:
                picked_in += P[x]
                if frontier == NUM_TYPES and P[x] < h[x]:
                    frontier = x
            committed = n - picked_in + self.in_total  # off-suit picked + all led
            is_in = self._is_in
            suffix_all = self._suffix_all
            held = self._held
            for i in range(bisect_left(held, lo), len(held)):
                t = held[i]
                if t > frontier:
                    break
                if P[t] >= h[t]:
                    continue
                if committed + (0 if is_in[t] else 1) > L:
                    continue
                if n - P[t] + suffix_all[t] < L:
                    if t > last:
                        break  # P[t] is 0 from here on and the suffix only shrinks
                    continue
                bits |= 1 << t
            return bits

        # The play must lie wholly in the led suit.
        in_types = self._in_types
        picked_in = 0
        for x in in_types:
            picked_in += P[x]
        if picked_in != n:
            return 0
        suffix_in = self._suffix_in
        start = bisect_left(in_types, lo)

        if not self.structured:
            for i in range(start, len(in_types)):
                t = in_types[i]
                if P[t] >= h[t]:
                    continue
                if n - P[t] + suffix_in[t] < L:
                    if t > last:
                        break
                    continue
                bits |= 1 << t
            return bits

        nd = self.num_decks
        base = 0
        for x in self._pair_types:
            if P[x] >= nd:
                base |= 1 << x
        for i in range(start, len(in_types)):
            t = in_types[i]
            if P[t] >= h[t]:
                continue
            if n - P[t] + suffix_in[t] < L:
                if t > last:
                    break
                continue
            qf = P[t] + 1
            b = base | (1 << t) if qf >= nd else base
            if self._feasible(b, t, qf, n + 1):
                bits |= 1 << t
        return bits

    def _struct_ok(self, paired: int) -> bool:
        """Do these paired types meet the lead's pair and tractor demands?"""
        got = self._struct_cache.get(paired)
        if got is not None:
            return got
        ok = paired.bit_count() >= self.want_pairs
        if ok and self.check_profile:
            rungs = 0
            m = paired
            rung_bit = self._rung_bit
            while m:
                low = m & -m
                rungs |= rung_bit[low.bit_length() - 1]
                m ^= low
            have = _profile(rungs, self.obligations.tractors)
            for want, got_len in zip(self.need, have):
                if got_len < want:
                    ok = False
                    break
        self._struct_cache[paired] = ok
        return ok

    def _feasible(self, base: int, f: int, qf: int, nq: int) -> bool:
        """Can a led-suit prefix be completed to a legal structured follow?

        ``base`` holds the types paired in the prefix, ``f`` is its last
        (highest) type, held ``qf`` times in it, and ``nq`` is its size.
        Only types ``>= f`` may still be added.
        """
        key = (base, f, qf, nq)
        got = self._feasible_cache.get(key)
        if got is not None:
            return got

        nd = self.num_decks
        L = self.lead_count
        budget = L - nq
        extra = [x for x in self._pair_types if x >= f and not (base >> x) & 1]
        f_cost = nd - qf  # cheaper than nd only for f itself
        f_extra = bool(extra) and extra[0] == f

        # Most cards reachable without pairing anything beyond `base`:
        # every unpaired led-suit type >= f filled to num_decks - 1.
        most = nq + self._suffix_unpaired[f + 1]
        if not (base >> f) & 1:
            most += min(self.hand_in[f], nd - 1) - qf
        # Each extra pair raises that ceiling by one card, so padding to
        # exactly L without forming a pair beyond the chosen set needs at
        # least this many extra pairs. (Given the caller's count check --
        # enough led-suit cards >= f to reach L -- this never binds: an
        # extra pair costs one card more than the single it replaces.
        # It is kept so the function is right on its own.)
        k_min = L - most

        def cheapest(p: int, f_free: bool) -> int:
            """Cost of the p cheapest extra pairs; f is the cheap one."""
            if p <= 0:
                return 0
            return f_cost + (p - 1) * nd if f_free else p * nd

        want_pairs = self.want_pairs
        if not self.check_profile:
            # Only the pair count matters: take the cheapest pairs that
            # both meet it and allow padding to exactly L.
            k = max(want_pairs - base.bit_count(), k_min, 0)
            got = k <= len(extra) and cheapest(k, f_extra) <= budget
            self._feasible_cache[key] = got
            return got

        rest = [0] * (len(extra) + 1)
        for i in range(len(extra) - 1, -1, -1):
            rest[i] = rest[i + 1] | (1 << extra[i])
        struct_ok = self._struct_ok

        def rec(i: int, chosen: int, k: int, cost: int) -> bool:
            mask = base | chosen
            if struct_ok(mask):
                # Upward closure: every superset qualifies too, and the
                # cheapest superset with enough pairs to pad to L adds
                # the cheapest remaining extras.
                p = k_min - k
                if p <= 0:
                    return True
                if len(extra) - k < p:
                    return False
                f_free = f_extra and not (chosen >> f) & 1
                return cost + cheapest(p, f_free) <= budget
            if i == len(extra):
                return False
            # Below this node only extra[i:] can join. Prune when even all
            # of them leave the structure short, or when the pairs still
            # missing cannot be afforded.
            if not struct_ok(mask | rest[i]):
                return False
            short = want_pairs - mask.bit_count()
            if short > 0 and (
                short > len(extra) - i
                or cost + cheapest(short, f_extra and i == 0) > budget
            ):
                return False
            x = extra[i]
            c = f_cost if x == f else nd
            if cost + c <= budget and rec(i + 1, chosen | (1 << x), k + 1, cost + c):
                return True
            return rec(i + 1, chosen, k, cost)

        got = rec(0, 0, 0, 0)
        self._feasible_cache[key] = got
        return got


class PickCursor:
    """One play being picked: the cards so far and what may come next."""

    __slots__ = ("rules", "counts", "last", "n_cards", "picks", "done", "_bits")

    def __init__(self, rules: PickRules) -> None:
        self.rules = rules
        self.counts = [0] * NUM_TYPES
        self.last = -1
        self.n_cards = 0
        self.picks: list[int] = []
        self.done = False
        self._bits: int | None = None

    @property
    def n_picks(self) -> int:
        return len(self.picks)

    @property
    def play(self) -> np.ndarray:
        return np.array(self.counts, dtype=np.int8)

    def legal_bits(self) -> int:
        if self.done:
            return 0
        if self._bits is None:
            self._bits = self.rules._bits(self.counts, self.last, self.n_cards)
        return self._bits

    def legal(self) -> np.ndarray:
        return bits_to_mask(self.legal_bits())

    def take(self, action: int) -> None:
        action = int(action)
        if self.done:
            raise ValueError("play already submitted")
        if not (self.legal_bits() >> action) & 1:
            raise ValueError(f"action {action} is not legal here")
        self.picks.append(action)
        self._bits = None
        if action == SUBMIT:
            self.done = True
            return
        self.counts[action] += 1
        self.last = action
        self.n_cards += 1
