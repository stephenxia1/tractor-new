"""Move enumeration: soundness, completeness, and non-emptiness.

Completeness is checked against brute force over every sub-multiset of
the hand, filtered by ``is_legal_follow``. The enumerator builds its
moves structurally and never consults that filter for the follow case it
can enumerate exactly, so the two are independent constructions of the
same set.
"""

from __future__ import annotations

import numpy as np

from tractor.rules import cards as C
from tractor.rules import combos as K
from tractor.rules import follow as F
from tractor.sim import actions as A

DECKS = 2


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_hand(rng: np.random.Generator, size: int) -> np.ndarray:
    """A hand drawn without replacement from a full two-deck shoe."""
    pool = np.repeat(np.arange(C.NUM_TYPES), DECKS)
    picked = rng.choice(pool, size=size, replace=False)
    return C.counts_from_ids(picked.tolist())


def _random_lead(
    rng: np.random.Generator, hand: np.ndarray, ctx: Ctx_t, num_decks: int
) -> tuple[np.ndarray, int] | None:
    """A legal-looking lead drawn from a separate holding."""
    suits = [s for s in range(5) if int(F.restrict(hand, ctx, s).sum()) > 0]
    if not suits:
        return None
    suit = int(rng.choice(suits))
    in_suit = F.restrict(hand, ctx, suit)
    size = int(rng.integers(1, min(6, int(in_suit.sum())) + 1))
    subs = A.sub_multisets(in_suit, size, 4000)
    if not subs:
        return None
    return subs[int(rng.integers(len(subs)))], suit


Ctx_t = C.Ctx


def test_follows_are_sound_complete_and_nonempty() -> None:
    rng = _rng(20260920)
    positions = 0
    total_moves = 0
    for _ in range(4000):
        ctx = C.Ctx(int(rng.integers(13)), int(rng.integers(5)))
        leader = _random_hand(rng, int(rng.integers(6, 14)))
        picked = _random_lead(rng, leader, ctx, DECKS)
        if picked is None:
            continue
        lead, led_suit = picked
        components = K.decompose(lead, ctx, DECKS)
        if components is None:
            continue
        ob = F.lead_obligations(components, DECKS)
        lead_count = int(lead.sum())

        hand = _random_hand(rng, int(rng.integers(lead_count, 14)))
        if int(hand.sum()) < lead_count:
            continue

        got = A.legal_follows(ob, lead_count, led_suit, hand, ctx, DECKS, cap=4096)
        if got.unenumerated:
            continue
        positions += 1
        total_moves += len(got)

        # Soundness: everything emitted is legal and drawn from the hand.
        for mv in got.moves:
            assert int(mv.sum()) == lead_count
            assert not np.any(mv > hand)
            assert F.is_legal_follow(
                ob, lead_count, led_suit, mv, hand, ctx, DECKS
            ), f"emitted an illegal follow in {ctx}"

        # A legal follow always exists.
        assert len(got) > 0, f"no legal follow offered in {ctx}"

        # Completeness against brute force over the whole hand.
        brute = A.sub_multisets(hand, lead_count, 60000)
        if brute is None:
            continue
        want = {
            m.tobytes()
            for m in brute
            if F.is_legal_follow(ob, lead_count, led_suit, m, hand, ctx, DECKS)
        }
        have = {m.tobytes() for m in got.moves}
        assert have == want, (
            f"follow set differs in {ctx}: "
            f"missing {len(want - have)}, extra {len(have - want)}"
        )

    assert positions > 1500
    print(f"    follows: {positions} positions, mean {total_moves / positions:.1f} moves")


def test_leads_are_single_suit_and_nonempty() -> None:
    rng = _rng(7)
    positions = 0
    total = 0
    throws = 0
    for _ in range(2000):
        ctx = C.Ctx(int(rng.integers(13)), int(rng.integers(5)))
        hand = _random_hand(rng, int(rng.integers(4, 26)))
        seen = C.full_deck(DECKS) - hand
        unseen = np.maximum(seen, 0).astype(np.int8)
        kitty_slots = int(rng.integers(0, 2)) * 8

        got = A.legal_leads(hand, unseen, ctx, DECKS, kitty_slots, cap=4096)
        positions += 1
        total += len(got)
        assert len(got) > 0

        for mv in got.moves:
            assert not np.any(mv > hand)
            assert int(mv.sum()) >= 1
            assert K.play_suit(mv, ctx) is not None, "lead spans several suits"
            parts = K.decompose(mv, ctx, DECKS)
            assert parts is not None
            if len(parts) > 1:
                throws += 1

    print(
        f"    leads: {positions} positions, mean {total / positions:.1f} moves, "
        f"{throws} throws offered"
    )
    assert throws > 0


def test_doomed_components_are_never_thrown() -> None:
    """A component the kitty cannot hide is never offered inside a throw."""
    rng = _rng(99)
    checked = 0
    for _ in range(600):
        ctx = C.Ctx(int(rng.integers(13)), int(rng.integers(5)))
        hand = _random_hand(rng, int(rng.integers(8, 26)))
        unseen = (C.full_deck(DECKS) - hand).astype(np.int8)
        # The landlord knows the kitty, so nothing can hide there and the
        # pigeonhole test is exact.
        got = A.legal_leads(hand, unseen, ctx, DECKS, kitty_slots=0, cap=4096)
        for mv in got.moves:
            parts = K.decompose(mv, ctx, DECKS)
            assert parts is not None
            if len(parts) < 2:
                continue
            suit = K.play_suit(mv, ctx)
            assert suit is not None
            for comp in parts:
                status, _ = A.component_status(
                    comp, suit, unseen, ctx, DECKS, kitty_slots=0
                )
                assert status != A.DOOMED, (
                    f"threw a doomed {comp} in {ctx}"
                )
                checked += 1
    assert checked > 0
    print(f"    throw components checked: {checked}")


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
