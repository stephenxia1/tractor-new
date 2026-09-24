"""The deal-and-declare environment, against the engine and itself."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ..rules.cards import NO_TRUMP, SMALL_JOKER, full_deck, suit_of
from ..sim.deal import (
    NUM_ACTIONS,
    PASS,
    DealEnv,
    action_tier,
    declare_obs,
    legal_mask,
)

FIXTURE = Path(__file__).parent / "fixtures" / "declare.json"


def test_legal_declarations_match_the_engine() -> None:
    """Regenerate with ``npx tsx src/dumpDeclare.ts 600 20260924
    py/tractor/tests/fixtures/declare.json`` from research/."""
    spec = json.loads(FIXTURE.read_text())
    for i, sc in enumerate(spec["scenarios"]):
        want = np.array(sc["mask"], dtype=bool)
        got = legal_mask(np.array(sc["hand"], dtype=np.int8), sc["levelRank"], sc["standingTier"])
        assert np.array_equal(got, want), (i, sc["standingTier"], got.tolist(), want.tolist())
    assert len(spec["scenarios"]) >= 500


def _run(env: DealEnv, policy) -> list[tuple[int, int]]:
    taken = []
    while (seat := env.next_decision()) is not None:
        mask = env.mask(seat)
        assert mask[1:].any(), "asked a seat with nothing to declare"
        action = policy(seat, mask)
        taken.append((seat, action))
        env.act(seat, action)
    return taken


def test_a_hand_deals_everything_and_closes() -> None:
    rng = np.random.default_rng(0)
    for i in range(50):
        env = DealEnv(rng, landlord=i % 4, level_rank=int(rng.integers(13)))
        pick = np.random.default_rng(i)

        def policy(seat, mask):
            legal = np.nonzero(mask)[0]
            return int(pick.choice(legal))

        taken = _run(env, policy)
        assert env.done and env.dealt == 100
        assert np.array_equal(env.revealed, env.final)
        assert all(int(env.final[s].sum()) == 25 for s in range(4))
        assert int(env.kitty.sum()) == 8
        total = env.final.sum(axis=0).astype(np.int64) + env.kitty
        assert np.array_equal(total, full_deck(2))
        # Standing tiers only ever rise.
        tiers = [d.tier for d in env.log]
        assert tiers == sorted(tiers) and len(set(tiers)) == len(tiers)
        assert all(0 <= a < NUM_ACTIONS for _s, a in taken)
        assert int(env.pool.sum()) == 33


def test_eager_declarers_leave_nothing_to_override() -> None:
    """Declaring whenever possible, the closing round runs until no seat
    has a higher declaration left to make."""
    rng = np.random.default_rng(1)
    for i in range(30):
        env = DealEnv(rng, landlord=i % 4, level_rank=int(rng.integers(13)))
        _run(env, lambda seat, mask: int(np.nonzero(mask)[0][-1]))
        for s in range(4):
            assert not env.mask(s)[1:].any(), (i, s)


def test_no_declaration_flips_the_kitty() -> None:
    rng = np.random.default_rng(2)
    for i in range(20):
        env = DealEnv(rng, landlord=i % 4, level_rank=int(rng.integers(13)))
        _run(env, lambda seat, mask: PASS)
        assert env.standing is None
        top = int(env.stack[100])
        assert env.trump_suit == (NO_TRUMP if top >= SMALL_JOKER else suit_of(top))


def test_a_seat_sees_only_its_own_cards() -> None:
    rng = np.random.default_rng(3)
    for i in range(20):
        env = DealEnv(rng, landlord=i % 4, level_rank=int(rng.integers(13)))
        seat = env.next_decision()
        if seat is None:
            continue
        other = copy.deepcopy(env)
        # Swap every other seat's cards around; the asked seat's view,
        # the public state and the kitty stay.
        others = [s for s in range(4) if s != seat]
        other.revealed[others] = other.revealed[others][::-1]
        other.final[others] = other.final[others][::-1]
        assert np.array_equal(declare_obs(env, seat), declare_obs(other, seat))
        assert action_tier(10) == 4
