"""Card-by-card legality (``sim/picks.py``) against the follow rule.

The pick masks are an independent construction: they decide each next
pick from closed forms and a structural search, never by asking
``is_legal_follow``. These tests enumerate every play the masks can
reach and compare it with brute force -- every sub-multiset of the hand
of the lead's size that ``is_legal_follow`` accepts -- on positions
drawn from simulated hands, plus synthetic positions built to stress
the pair and tractor obligations. They also check that nothing reached
is a dead end, that every legal play's ascending sequence is accepted
pick by pick, and that the simulator accepts every reachable play.
"""

from __future__ import annotations

import itertools
import time
from collections import Counter

import numpy as np
import pytest

from tractor.gen.episode import generate
from tractor.rules import cards as C
from tractor.rules import combos as K
from tractor.rules import follow as F
from tractor.sim import actions as A
from tractor.sim import picks as PK
from tractor.sim.picks import SUBMIT, PickRules
from tractor.sim.state import PlayState

DECKS = 2

# Brute force over every L-card sub-multiset of the whole hand up to
# this many candidates; past it, over the sub-multisets the first clause
# of ``is_legal_follow`` leaves (every led-suit card when the hand holds
# no more than L of them, otherwise L led-suit cards), still filtered
# by ``is_legal_follow``.
FULL_BRUTE_CAP = 30_000
RESTRICTED_BRUTE_CAP = 400_000
LEAD_SET_CAP = 200_000


# --- helpers ----------------------------------------------------------------


def _key(counts) -> bytes:
    return bytes(int(v) for v in counts)


def _reachable(rules: PickRules, limit: int = 2_000_000, check_public_every: int = 0):
    """Every play the masks reach by DFS over picks.

    Returns (plays, dead_ends, submit_not_alone). A dead end is a prefix
    with no legal action at all; for a follow, SUBMIT must be the only
    legal action whenever it is legal.
    """
    plays: set[bytes] = set()
    dead = 0
    not_alone = 0
    P = [0] * C.NUM_TYPES
    visits = 0

    def rec(last: int, n: int) -> None:
        nonlocal dead, not_alone, visits
        bits = rules._bits(P, last, n)
        visits += 1
        if check_public_every and visits % check_public_every == 0:
            public = rules.mask(np.array(P, dtype=np.int8), last)
            assert np.array_equal(public, PK.bits_to_mask(bits))
            assert PK.pack_mask(public) == np.uint64(bits)
        if bits == 0:
            dead += 1
            return
        if (bits >> SUBMIT) & 1:
            if n == 0:
                dead += 1  # SUBMIT of an empty play is never legal
            plays.add(bytes(P))
            if not rules.is_lead and bits != (1 << SUBMIT):
                not_alone += 1
            if len(plays) > limit:
                raise RuntimeError("reachable set over limit")
        b = bits & ~(1 << SUBMIT)
        while b:
            low = b & -b
            t = low.bit_length() - 1
            b ^= low
            assert t >= max(last, 0), "pick below the last pick"
            assert P[t] < rules.hand[t], "pick of a type with no copy left"
            P[t] += 1
            rec(t, n + 1)
            P[t] -= 1

    rec(-1, 0)
    return plays, dead, not_alone


def _brute_follows(ob, L, suit, hand, ctx, nd) -> tuple[set[bytes], str]:
    """Every legal follow, by brute force filtered with is_legal_follow."""
    subs = A.sub_multisets(hand, L, FULL_BRUTE_CAP)
    mode = "full"
    if subs is None:
        mode = "restricted"
        in_led = F.restrict(hand, ctx, suit).astype(np.int8)
        n_in = int(in_led.sum())
        if n_in <= L:
            fills = A.sub_multisets((hand - in_led).astype(np.int8), L - n_in, RESTRICTED_BRUTE_CAP)
            assert fills is not None, "restricted brute force over its cap"
            subs = [(in_led + f).astype(np.int8) for f in fills]
        else:
            subs = A.sub_multisets(in_led, L, RESTRICTED_BRUTE_CAP)
            assert subs is not None, "restricted brute force over its cap"
    legal = {
        _key(s) for s in subs if F.is_legal_follow(ob, L, suit, s, hand, ctx, nd)
    }
    return legal, mode


def _one_suit_leads(hand: np.ndarray, ctx: C.Ctx) -> set[bytes] | None:
    """Every non-empty one-effective-suit sub-multiset of the hand."""
    total = 0
    per_suit = []
    for s in range(5):
        types = [t for t in range(C.NUM_TYPES) if hand[t] > 0 and ctx.eff[t] == s]
        size = 1
        for t in types:
            size *= int(hand[t]) + 1
        total += size - 1
        per_suit.append(types)
    if total > LEAD_SET_CAP:
        return None
    out: set[bytes] = set()
    for types in per_suit:
        if not types:
            continue
        for counts in itertools.product(*[range(int(hand[t]) + 1) for t in types]):
            if not any(counts):
                continue
            v = [0] * C.NUM_TYPES
            for t, c in zip(types, counts):
                v[t] = c
            out.add(bytes(v))
    return out


def _lead_kind(play: np.ndarray, ctx: C.Ctx, nd: int = DECKS) -> str:
    parts = K.decompose(play, ctx, nd)
    assert parts is not None
    if len(parts) > 1:
        if any(p.kind == K.TRACTOR for p in parts):
            return "throw+tractor"
        if any(p.kind == K.NKIND for p in parts):
            return "throw+pair"
        return "throw"
    return K.KIND_NAMES[parts[0].kind]


# Lead categories and how strongly the random policy prefers each, so
# that structured leads -- the ones that carry obligations -- turn up
# far more often than uniform play over the move list would give.
_LEAD_WEIGHTS = {
    "single": 1.0,
    "nkind": 2.0,
    "tractor": 4.0,
    "throw": 1.0,
    "throw+pair": 3.0,
    "throw+tractor": 4.0,
}


_BRANCH_ORDER = ("tractor", "throw+tractor", "throw+pair", "nkind", "throw")


def _random_lead(
    rng: np.random.Generator, cats: dict[str, list[np.ndarray]]
) -> np.ndarray:
    names = sorted(cats)
    w = np.array([_LEAD_WEIGHTS[n] for n in names])
    name = names[int(rng.choice(len(names), p=w / w.sum()))]
    pool = cats[name]
    return pool[int(rng.integers(len(pool)))]


def _new_game(rng: np.random.Generator, force_nt: bool) -> PlayState:
    state, _info = generate(rng, DECKS)
    if force_nt and state.ctx.trump_suit != C.NO_TRUMP:
        # The same deal under a no-trump declaration: only the level
        # cards and the jokers are trump, and the four level cards share
        # one rung.
        ctx = C.Ctx(state.ctx.level_rank, C.NO_TRUMP)
        state = PlayState.new(ctx, state.hands, state.kitty, state.landlord, DECKS)
    return state


def _decisions(
    seed: int,
    games: int,
    branch: int = 2,
    nt_every: int = 3,
    kinds: tuple[str, ...] = _BRANCH_ORDER,
    main: bool = True,
):
    """Decision points of random play on simulated deals.

    Yields ``(state, moves)`` with ``moves = state.legal_moves(cap=1e6)``;
    the state must not be mutated by the consumer. At each lead the
    walk also branches into up to ``branch`` extra leads of the given
    ``kinds`` (one per kind, in that order) and yields the three follows
    of each, so that tractor, pair and throw leads are well represented.
    With ``main=False`` only those branch positions are yielded.
    """
    rng = np.random.default_rng(seed)
    for g in range(games):
        state = _new_game(rng, force_nt=(g % nt_every == nt_every - 1))
        while not state.done:
            moves = state.legal_moves(cap=10**6)
            if main:
                yield state, moves
            if state.is_lead:
                by_kind: dict[str, list[np.ndarray]] = {}
                for m in moves.moves:
                    by_kind.setdefault(_lead_kind(m, state.ctx), []).append(m)
                for kind in [k for k in kinds if k in by_kind][:branch]:
                    pool = by_kind[kind]
                    side = state.clone()
                    side.step(pool[int(rng.integers(len(pool)))])
                    for _f in range(3):
                        side_moves = side.legal_moves(cap=10**6)
                        yield side, side_moves
                        side.step(
                            side_moves.moves[int(rng.integers(len(side_moves.moves)))]
                        )
                move = _random_lead(rng, by_kind)
            else:
                move = moves.moves[int(rng.integers(len(moves.moves)))]
            state.step(move)


def _follow_context(state: PlayState):
    parts, suit = state.lead_components()
    ob = F.lead_obligations(parts, state.num_decks)
    L = int(state.trick_plays[0].sum())
    return ob, L, suit


def _check_follow(rules, ob, L, suit, hand, ctx, nd) -> tuple[set[bytes], str]:
    got, dead, not_alone = _reachable(rules, check_public_every=97)
    assert dead == 0, f"{dead} dead ends in {ctx}, lead {L} {ob}"
    assert not_alone == 0, "SUBMIT legal beside a card pick"
    want, mode = _brute_follows(ob, L, suit, hand, ctx, nd)
    assert want, "no legal follow at all"
    missing = want - got
    extra = got - want
    assert not missing and not extra, (
        f"{ctx} lead {L} {ob} hand {C.ids_from_counts(hand)}: "
        f"{len(missing)} legal follows unreachable, {len(extra)} illegal reachable; "
        f"e.g. missing {[C.ids_from_counts(np.frombuffer(m, np.uint8)) for m in list(missing)[:2]]} "
        f"extra {[C.ids_from_counts(np.frombuffer(m, np.uint8)) for m in list(extra)[:2]]}"
    )
    return got, mode


# --- simulated positions ----------------------------------------------------


def _check_simulated_follow(state: PlayState, moves, tally: Counter) -> None:
    hand = state.hands[state.to_act]
    ob, L, suit = _follow_context(state)
    rules = PickRules.for_state(state)
    exact = not moves.unenumerated and moves.truncated == 0
    if not exact:
        # The existing enumerator gave up here; still require that
        # everything reachable is legal and nothing dead-ends.
        got, dead, _ = _reachable(rules)
        assert dead == 0
        for k in got:
            play = np.frombuffer(k, np.uint8).astype(np.int8)
            assert F.is_legal_follow(ob, L, suit, play, hand, state.ctx, DECKS)
        tally["inexact"] += 1
        return

    got, mode = _check_follow(rules, ob, L, suit, hand, state.ctx, DECKS)
    # The existing exact enumeration is the same set.
    assert {_key(m) for m in moves.moves} == got

    tally["positions"] += 1
    tally[f"brute:{mode}"] += 1
    tally[f"lead:{_lead_kind(state.trick_plays[0], state.ctx)}"] += 1
    if rules.all_in:
        tally["case:all-in"] += 1
    elif rules.structured:
        tally["case:structured"] += 1
    else:
        tally["case:count-only"] += 1
    if rules.structured and rules.check_profile:
        tally["profile-checked"] += 1
    if state.ctx.trump_suit == C.NO_TRUMP:
        tally["no-trump"] += 1
    if len(got) > 1:
        tally["choices>1"] += 1


def test_follow_masks_match_brute_force_on_simulated_positions() -> None:
    tally: Counter = Counter()
    for state, moves in _decisions(seed=20261119, games=36):
        if not state.is_lead:
            _check_simulated_follow(state, moves, tally)
    print("\n    follows:", dict(sorted(tally.items())))
    assert tally["positions"] > 2000
    assert tally["lead:nkind"] > 200
    assert tally["lead:throw+pair"] + tally["lead:throw+tractor"] > 100
    assert tally["case:structured"] > 100
    assert tally["no-trump"] > 300


def test_follow_masks_after_tractor_leads_on_simulated_hands() -> None:
    """Tractors are rare in a random deal, so this walk only branches
    into tractor leads and throws carrying one, and checks their
    follows."""
    tally: Counter = Counter()
    for state, moves in _decisions(
        seed=777, games=150, branch=2, kinds=("tractor", "throw+tractor"), main=False
    ):
        _check_simulated_follow(state, moves, tally)
    print("\n    tractor-led follows:", dict(sorted(tally.items())))
    assert tally["lead:tractor"] > 150
    assert tally["lead:throw+tractor"] > 20
    # A follower who must match a tractor from a tractor of their own is
    # rare in a random deal; the synthetic test below covers it in bulk.
    assert tally["profile-checked"] >= 3


def test_lead_masks_reach_every_one_suit_multiset() -> None:
    rng = np.random.default_rng(5)
    tally: Counter = Counter()
    for state, moves in _decisions(seed=77, games=12, branch=0):
        if not state.is_lead:
            continue
        hand = state.hands[state.to_act]
        rules = PickRules.for_state(state)
        assert rules.is_lead
        want = _one_suit_leads(hand, state.ctx)
        if want is not None:
            got, dead, _ = _reachable(rules, check_public_every=211)
            assert dead == 0
            assert got == want, f"{len(want - got)} missing, {len(got - want)} extra"
            tally["full"] += 1
            tally["plays"] += len(got)
        else:
            # Too many to list: random pick walks must end in a one-suit
            # play, and random one-suit plays must trace.
            for _ in range(200):
                cur = rules.cursor()
                while not cur.done:
                    legal = np.nonzero(cur.legal())[0]
                    assert legal.size, "dead end"
                    cur.take(int(rng.choice(legal)))
                play = cur.play
                assert play.sum() > 0 and K.play_suit(play, state.ctx) is not None
            tally["sampled"] += 1
        # Every lead the existing enumerator offers is reachable.
        for m in moves.moves:
            rules.trace(m)
        tally["positions"] += 1
    print("\n    leads:", dict(tally))
    assert tally["full"] > 200


def test_every_legal_play_traces_and_steps() -> None:
    """Ascending picks of every legal play pass the masks one by one, the
    row encoding round-trips, and the simulator accepts every reachable
    play."""
    rng = np.random.default_rng(11)
    tally: Counter = Counter()
    for state, moves in _decisions(seed=4242, games=6):
        hand = state.hands[state.to_act]
        rules = PickRules.for_state(state)
        if state.is_lead:
            legal = _one_suit_leads(hand, state.ctx)
            if legal is None:
                continue
            reach_cap = 60
        else:
            ob, L, suit = _follow_context(state)
            legal, _mode = _brute_follows(ob, L, suit, hand, state.ctx, DECKS)
            reach_cap = 300
        plays = [np.frombuffer(k, np.uint8).astype(np.int8) for k in legal]
        traced = plays if len(plays) <= 3000 else [
            plays[i] for i in rng.choice(len(plays), 3000, replace=False)
        ]
        for play in traced:
            picks, masks = rules.trace(play)
            assert picks == PK.ascending_picks(play)
            cur = rules.cursor()
            for a, bits in zip(picks, masks):
                assert np.array_equal(cur.legal(), PK.unpack_mask(np.uint64(bits)))
                cur.take(a)
            assert cur.done and np.array_equal(cur.play, play)
            assert cur.n_picks == int(play.sum()) + 1
            p8, m64, n = rules.encode(play)
            assert n == len(picks) and np.array_equal(PK.play_from_picks(p8), play)
            assert np.all(p8[n:] == -1) and np.all(m64[n:] == 0)
            assert np.array_equal(PK.unpack_masks(m64[:n]), np.stack([PK.bits_to_mask(b) for b in masks]))
            tally["traced"] += 1
        sample = plays if len(plays) <= reach_cap else [
            plays[i] for i in rng.choice(len(plays), reach_cap, replace=False)
        ]
        for play in sample:
            s = state.clone()
            played = s.step(play)
            assert int(played.sum()) >= 1
            if not state.is_lead:
                assert np.array_equal(played, play)
            tally["stepped"] += 1
    print("\n    trace/step:", dict(tally))
    assert tally["traced"] > 20000 and tally["stepped"] > 5000


# --- synthetic positions that stress the obligations ------------------------


def _stress_position(rng: np.random.Generator, nd: int):
    """A structured lead and a follower long in the led suit."""
    ts = int(rng.choice([0, 1, 2, 3, 4, 4]))  # extra weight on no-trump
    ctx = C.Ctx(int(rng.integers(13)), ts)
    suit = int(rng.choice([0, 1, 2, 3, 4, 4, 4]))
    in_types = [t for t in range(C.NUM_TYPES) if ctx.eff[t] == suit]
    if len(in_types) < 3:
        return None
    shoe = C.full_deck(nd).astype(np.int8)

    # Lead: each led-suit type absent, single, or a full pair, with
    # consecutive pairs common enough to form tractors and throws of
    # several tractors.
    lead = np.zeros(C.NUM_TYPES, np.int8)
    p_pair = float(rng.uniform(0.15, 0.6))
    for t in in_types:
        u = rng.random()
        if u < p_pair:
            lead[t] = nd
        elif u < p_pair + 0.12:
            lead[t] = 1
    while int(lead.sum()) > 12:
        lead[int(rng.choice(np.nonzero(lead)[0]))] = 0
    if int(lead.sum()) == 0:
        return None
    rest = (shoe - lead).astype(np.int8)

    # Follower: long in the led suit with many pairs (runs), plus a few
    # off-suit cards.
    hand = np.zeros(C.NUM_TYPES, np.int8)
    q_pair = float(rng.uniform(0.3, 0.9))
    for t in in_types:
        if rest[t] >= nd and rng.random() < q_pair:
            hand[t] = nd
        elif rest[t] >= 1 and rng.random() < 0.3:
            hand[t] = int(rng.integers(1, rest[t] + 1)) if rest[t] < nd else int(rng.integers(1, nd))
    while int(hand.sum()) > 14:
        hand[int(rng.choice(np.nonzero(hand)[0]))] -= 1
    L = int(lead.sum())
    off = [t for t in range(C.NUM_TYPES) if ctx.eff[t] != suit]
    n_off = int(rng.integers(0, 5))
    while int(hand.sum()) < L or n_off > 0:
        t = int(rng.choice(off))
        if hand[t] < rest[t]:
            hand[t] += 1
            n_off -= 1
    return ctx, suit, lead, hand


@pytest.mark.parametrize("nd,cases", [(2, 2000), (3, 600)])
def test_follow_masks_under_heavy_obligations(nd: int, cases: int) -> None:
    rng = np.random.default_rng(1000 + nd)
    tally: Counter = Counter()
    for _ in range(cases):
        got = _stress_position(rng, nd)
        if got is None:
            continue
        ctx, suit, lead, hand = got
        parts = K.decompose(lead, ctx, nd)
        ob = F.lead_obligations(parts, nd)
        L = int(lead.sum())
        rules = PickRules(hand, ctx, nd, lead=lead)
        assert rules.led_suit == suit and rules.lead_count == L
        _check_follow(rules, ob, L, suit, hand, ctx, nd)
        tally["positions"] += 1
        tally[f"slots={len(ob.tractors)}"] += 1
        if rules.structured:
            tally["structured"] += 1
            if rules.check_profile:
                tally["profile-checked"] += 1
                if sum(1 for v in rules.need if v) >= 2:
                    tally["need>=2 tractors"] += 1
        if ctx.trump_suit == C.NO_TRUMP and suit == C.TRUMP:
            tally["no-trump trump lead"] += 1
    print(f"\n    stress nd={nd}:", dict(sorted(tally.items())))
    assert tally["structured"] > cases // 4
    assert tally["profile-checked"] > (cases // 10 if nd == 2 else 20)
    if nd == 2:
        assert tally["need>=2 tractors"] > 20
        assert tally["no-trump trump lead"] > 20


def test_hand_written_obligation_cases() -> None:
    """Small positions with known answers."""
    ctx = C.Ctx(12, 0)  # level A, spades trump; hearts is a plain suit
    H = lambda r: C.card_id(1, r)  # noqa: E731 - hearts by rank index
    S = lambda r: C.card_id(0, r)  # noqa: E731

    # Lead: a hearts tractor 2-2-3-3. Follower: hearts 5-5-6-6, 8-8, 9, and
    # a spade. Must play the 5566 tractor.
    lead = C.counts_from_ids([H(0), H(0), H(1), H(1)])
    hand = C.counts_from_ids([H(3), H(3), H(4), H(4), H(6), H(6), H(7), S(2)])
    rules = PickRules(hand, ctx, 2, lead=lead)
    plays, dead, _ = _reachable(rules)
    assert dead == 0
    assert plays == {_key(C.counts_from_ids([H(3), H(3), H(4), H(4)]))}
    # The 8 (rank 6) is never a legal first pick, nor is the spade.
    m = rules.mask(np.zeros(C.NUM_TYPES, np.int8), -1)
    assert m[H(3)] and not m[H(6)] and not m[H(7)] and not m[S(2)] and not m[SUBMIT]

    # Same lead, follower holds 5-5, 8-8 and singles: two pairs, no tractor.
    hand = C.counts_from_ids([H(3), H(3), H(6), H(6), H(8), H(9), S(2)])
    plays, dead, _ = _reachable(PickRules(hand, ctx, 2, lead=lead))
    assert dead == 0
    assert plays == {_key(C.counts_from_ids([H(3), H(3), H(6), H(6)]))}

    # Short in the suit: every heart plus any filler.
    hand = C.counts_from_ids([H(3), H(9), S(2), S(2), S(5)])
    plays, dead, _ = _reachable(PickRules(hand, ctx, 2, lead=lead))
    assert dead == 0
    assert len(plays) == 2  # both hearts, then S2 S2 or S2 S5
    for k in plays:
        play = np.frombuffer(k, np.uint8)
        assert play[H(3)] == 1 and play[H(9)] == 1

    # After picking a 5, the cursor offers the second 5 but no 6.
    hand = C.counts_from_ids([H(3), H(3), H(4), H(4), H(6), H(6), H(7), S(2)])
    cur = PickRules(hand, ctx, 2, lead=lead).cursor()
    cur.take(H(3))
    legal = set(np.nonzero(cur.legal())[0].tolist())
    assert legal == {H(3)}
    with pytest.raises(ValueError):
        cur.take(H(6))
    for a in (H(3), H(4), H(4), SUBMIT):
        cur.take(a)
    assert cur.done and cur.n_picks == 5 and int(cur.play.sum()) == 4
    with pytest.raises(ValueError):
        cur.take(SUBMIT)

    # A lead: after a heart, only hearts at or above it, and SUBMIT.
    rules = PickRules(hand, ctx, 2)
    m0 = rules.mask(np.zeros(C.NUM_TYPES, np.int8), -1)
    assert set(np.nonzero(m0)[0].tolist()) == set(np.nonzero(hand)[0].tolist())
    p = np.zeros(C.NUM_TYPES, np.int8)
    p[H(4)] = 1
    m1 = rules.mask(p, H(4))
    assert set(np.nonzero(m1)[0].tolist()) == {H(4), H(6), H(7), SUBMIT}
    p[H(4)] = 2
    assert set(np.nonzero(rules.mask(p))[0].tolist()) == {H(6), H(7), SUBMIT}

    with pytest.raises(ValueError):
        PickRules(hand, ctx, 2, lead=C.counts_from_ids([H(0), S(0)]))


# --- the monotonicity behind the feasibility shortcut -------------------------


def test_structure_is_upward_closed_within_the_hand() -> None:
    """Adding cards never loses structure, in the sense the shortcut uses.

    The pair count is monotone. The tractor profile is monotone in the
    lexicographic order but *not* slot by slot (see the next test), so
    the claim the mask relies on is about the legality test itself:
    once a subset of the hand meets ``need`` and ``want_pairs``, so does
    every larger subset of the hand. That holds because a subset's
    profile can never lexicographically exceed the hand's, so meeting
    ``need`` slot by slot means equalling it.

    Checked along chains that grow one card at a time from nothing to
    the whole led-suit holding, which is where runs merge. The legality
    side asks ``is_legal_follow`` itself, with the lead's count set to
    the candidate's size: while that is below the holding's size, the
    function's verdict is exactly the structural test.
    """
    rng = np.random.default_rng(99)
    steps = 0
    slot_losses = 0
    became_legal = 0
    for _ in range(2500):
        nd = 2 if rng.random() < 0.85 else 3
        ctx = C.Ctx(int(rng.integers(13)), int(rng.integers(5)))
        suit = int(rng.integers(5))
        in_types = [t for t in range(C.NUM_TYPES) if ctx.eff[t] == suit]
        H = np.zeros(C.NUM_TYPES, np.int8)
        q = rng.uniform(0.3, 0.95)
        for t in in_types:
            if rng.random() < q:
                H[t] = nd if rng.random() < 0.7 else int(rng.integers(1, nd))
        total = int(H.sum())
        if total < 2:
            continue
        n_slots = int(rng.integers(1, 4))
        slots = tuple(sorted((int(rng.integers(2, 6)) for _ in range(n_slots)), reverse=True))
        ob = F.Obligations(sum(slots) + int(rng.integers(0, 3)), slots)
        need = tuple(F.hand_tractor_profile(ob, H, ctx, nd))

        order = rng.permutation(np.repeat(np.arange(C.NUM_TYPES), H.astype(np.int64)))
        cur = np.zeros(C.NUM_TYPES, np.int8)
        prev_pairs, prev_prof, prev_legal = 0, None, False
        for i, t in enumerate(order[:-1]):  # stop short of the whole holding
            cur[t] += 1
            pairs = F.count_pairs(cur, nd)
            prof = tuple(F.hand_tractor_profile(ob, cur, ctx, nd))
            legal = F.is_legal_follow(ob, i + 1, suit, cur, H, ctx, nd)
            assert pairs >= prev_pairs
            assert prof <= need, (prof, need)
            if prev_prof is not None:
                assert prof >= prev_prof, (prev_prof, prof)
                if any(a < b for a, b in zip(prof, prev_prof)):
                    slot_losses += 1
            if prev_legal:
                assert legal, "adding a card lost a legal structure"
            elif legal:
                became_legal += 1
            if legal:
                assert prof == need
            prev_pairs, prev_prof, prev_legal = pairs, prof, legal
            steps += 1
    print(
        f"\n    {steps} one-card steps; {became_legal} chains became legal and stayed so; "
        f"{slot_losses} steps lost a profile slot while gaining lexicographically"
    )
    assert steps > 20000 and became_legal > 500
    assert slot_losses > 0  # the slot-by-slot claim really is false


def test_profile_is_not_monotone_slot_by_slot() -> None:
    """Against tractor slots (4, 2), pairs at hearts 2 3 _ 5 6 give the
    profile (2, 2); filling the gap with the 4s gives (4, 0)."""
    ctx = C.Ctx(12, 0)
    ob = F.Obligations(6, (4, 2))
    gap = C.counts_from_ids([C.card_id(1, r) for r in (0, 0, 1, 1, 3, 3, 4, 4)])
    full = gap.copy()
    full[C.card_id(1, 2)] = 2
    assert F.hand_tractor_profile(ob, gap, ctx, 2) == [2, 2]
    assert F.hand_tractor_profile(ob, full, ctx, 2) == [4, 0]


# --- packing ----------------------------------------------------------------


def test_mask_packing_round_trips() -> None:
    rng = np.random.default_rng(3)
    masks = rng.random((7, 5, PK.NUM_PICK_ACTIONS)) < 0.4
    masks[0, 0] = False
    masks[0, 1] = True
    packed = PK.pack_masks(masks)
    assert packed.dtype == np.uint64 and packed.shape == (7, 5)
    assert np.array_equal(PK.unpack_masks(packed), masks)
    assert packed[0, 0] == 0 and packed[0, 1] == np.uint64((1 << 55) - 1)
    for i in range(PK.NUM_PICK_ACTIONS):
        one = np.zeros(PK.NUM_PICK_ACTIONS, bool)
        one[i] = True
        assert int(PK.pack_mask(one)) == 1 << i
        assert np.array_equal(PK.unpack_mask(np.uint64(1 << i)), one)
    for m in masks.reshape(-1, PK.NUM_PICK_ACTIONS):
        assert np.array_equal(PK.unpack_mask(PK.pack_mask(m)), m)
    assert PK.SUBMIT == 54 and PK.NUM_PICK_ACTIONS == 55 and PK.MAX_PICKS == 26
    assert PK.ascending_picks(C.counts_from_ids([7, 3, 7, 53])) == [3, 7, 7, 53, SUBMIT]


# --- timing -------------------------------------------------------------------


def test_mask_timing_report() -> None:
    """Average cost of one mask call over realistic decisions.

    Each decision builds its rules once and then asks for the mask at
    every pick of a random legal play, as a rollout worker would.
    Printed, not asserted strictly.
    """
    rng = np.random.default_rng(8)
    t_build = t_mask = t_bits = 0.0
    n_build = n_mask = 0
    worst = 0.0
    follow_calls = 0
    for state, _moves in _decisions(seed=31337, games=12, branch=1):
        hand = state.hands[state.to_act]
        lead = None if state.is_lead else state.trick_plays[0]
        t0 = time.perf_counter()
        rules = PickRules(hand, state.ctx, state.num_decks, lead)
        t_build += time.perf_counter() - t0
        n_build += 1
        picked = np.zeros(C.NUM_TYPES, np.int8)
        last = -1
        while True:
            t0 = time.perf_counter()
            m = rules.mask(picked, last)
            dt = time.perf_counter() - t0
            t_mask += dt
            worst = max(worst, dt)
            n_mask += 1
            follow_calls += not rules.is_lead
            legal = np.nonzero(m)[0]
            assert legal.size
            cards = legal[legal != SUBMIT]
            if m[SUBMIT] and (not rules.is_lead or cards.size == 0 or rng.random() < 0.5):
                break
            a = int(rng.choice(cards))
            picked[a] += 1
            last = a
        # The same walk again through the cursor's int masks, cache warm.
        cur = rules.cursor()
        for a in PK.ascending_picks(picked):
            t0 = time.perf_counter()
            cur.legal_bits()
            t_bits += time.perf_counter() - t0
            cur.take(a)
    mean_mask = 1e6 * t_mask / n_mask
    print(
        f"\n    mask(): {n_mask} calls ({follow_calls} in follows) over {n_build} decisions, "
        f"mean {mean_mask:.1f} us, worst {1e6 * worst:.0f} us; "
        f"cursor.legal_bits() warm mean {1e6 * t_bits / n_mask:.1f} us; "
        f"PickRules build mean {1e6 * t_build / n_build:.1f} us"
    )
    assert mean_mask < 5000  # a loose guard against pathologies only
