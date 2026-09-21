"""Play-phase state and transitions.

The state begins at the first trick: the trump context, the landlord and
the kitty are inputs, produced by the episode generator rather than by
play. Twenty-five tricks at most, fewer when leads carry more than one
card.

State is mutable and cloned explicitly. Search evaluates hundreds of
rollouts from one position, and rebuilding an immutable state per ply
costs more than a copy of five small arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..rules.cards import NUM_TYPES, TRUMP, Ctx, total_points
from ..rules.combos import (
    Component,
    covers,
    decompose,
    decompose_with_cards,
    play_rank,
    play_suit,
)
from ..rules.follow import lead_obligations, restrict
from .actions import MoveSet, legal_leads, legal_follows

NUM_SEATS = 4


def team_of(seat: int) -> int:
    return seat % 2


@dataclass(slots=True)
class TrickRecord:
    leader: int
    plays: list[np.ndarray]
    winner: int
    points: int


@dataclass(slots=True)
class PlayState:
    ctx: Ctx
    num_decks: int
    hands: np.ndarray  # int8[4, 54]
    kitty: np.ndarray  # int8[54]
    landlord: int
    to_act: int
    trick_leader: int
    trick_plays: list[np.ndarray]
    attacker_points: int
    tricks: list[TrickRecord]
    last_forced_lead: np.ndarray | None  # cards revealed by a blocked throw
    done: bool

    # --- construction -----------------------------------------------------

    @staticmethod
    def new(
        ctx: Ctx,
        hands: np.ndarray,
        kitty: np.ndarray,
        landlord: int,
        num_decks: int = 2,
    ) -> "PlayState":
        return PlayState(
            ctx=ctx,
            num_decks=num_decks,
            hands=hands.astype(np.int8).copy(),
            kitty=kitty.astype(np.int8).copy(),
            landlord=landlord,
            to_act=landlord,
            trick_leader=landlord,
            trick_plays=[],
            attacker_points=0,
            tricks=[],
            last_forced_lead=None,
            done=False,
        )

    def clone(self) -> "PlayState":
        return PlayState(
            ctx=self.ctx,
            num_decks=self.num_decks,
            hands=self.hands.copy(),
            kitty=self.kitty.copy(),
            landlord=self.landlord,
            to_act=self.to_act,
            trick_leader=self.trick_leader,
            trick_plays=[p.copy() for p in self.trick_plays],
            attacker_points=self.attacker_points,
            tricks=list(self.tricks),
            last_forced_lead=None
            if self.last_forced_lead is None
            else self.last_forced_lead.copy(),
            done=self.done,
        )

    # --- queries ----------------------------------------------------------

    @property
    def is_lead(self) -> bool:
        return len(self.trick_plays) == 0

    @property
    def defenders(self) -> int:
        return team_of(self.landlord)

    def unseen_from(self, seat: int) -> np.ndarray:
        """Cards `seat` has not seen: the deck less its hand, less every
        card played, less the kitty when the seat buried it."""
        seen = self.hands[seat].astype(np.int16).copy()
        for trick in self.tricks:
            for play in trick.plays:
                seen += play
        for play in self.trick_plays:
            seen += play
        if seat == self.landlord:
            seen += self.kitty
        out = np.full(NUM_TYPES, self.num_decks, dtype=np.int16) - seen
        return np.maximum(out, 0).astype(np.int8)

    def lead_components(self) -> tuple[list[Component], int] | None:
        if self.is_lead:
            return None
        lead = self.trick_plays[0]
        suit = play_suit(lead, self.ctx)
        parts = decompose(lead, self.ctx, self.num_decks)
        if suit is None or parts is None:
            return None
        return parts, suit

    def legal_moves(self, cap: int = 64) -> MoveSet:
        seat = self.to_act
        if self.is_lead:
            kitty_slots = 0 if seat == self.landlord else int(self.kitty.sum())
            return legal_leads(
                self.hands[seat],
                self.unseen_from(seat),
                self.ctx,
                self.num_decks,
                kitty_slots,
                cap=cap,
            )
        lead_info = self.lead_components()
        assert lead_info is not None
        parts, suit = lead_info
        ob = lead_obligations(parts, self.num_decks)
        lead_count = int(self.trick_plays[0].sum())
        return legal_follows(
            ob, lead_count, suit, self.hands[seat], self.ctx, self.num_decks, cap=cap
        )

    # --- transition -------------------------------------------------------

    def step(self, cards: np.ndarray) -> np.ndarray:
        """Play `cards` for the seat to act. Returns what was played,
        which differs from the request only when a throw is blocked."""
        if self.done:
            raise RuntimeError("hand is over")
        seat = self.to_act
        cards = cards.astype(np.int8)
        if np.any(cards > self.hands[seat]):
            raise ValueError(f"seat {seat} does not hold those cards")
        if int(cards.sum()) == 0:
            raise ValueError("empty play")

        self.last_forced_lead = None
        if self.is_lead:
            cards = self._resolve_throw(seat, cards)
        else:
            lead_count = int(self.trick_plays[0].sum())
            if int(cards.sum()) != lead_count:
                raise ValueError(
                    f"must play {lead_count} cards, got {int(cards.sum())}"
                )

        self.hands[seat] -= cards
        self.trick_plays.append(cards)
        if len(self.trick_plays) < NUM_SEATS:
            self.to_act = (seat + 1) % NUM_SEATS
            return cards

        self._close_trick()
        return cards

    def _resolve_throw(self, seat: int, cards: np.ndarray) -> np.ndarray:
        """Apply the block check. A throw no single other hand can break
        stands; otherwise only the smallest beatable component is led and
        the rest returns to hand, revealed."""
        pieces = decompose_with_cards(cards, self.ctx, self.num_decks)
        suit = play_suit(cards, self.ctx)
        if pieces is None or suit is None:
            raise ValueError("lead spans more than one effective suit")
        if len(pieces) < 2:
            return cards

        beatable: list[tuple[Component, np.ndarray]] = []
        for comp, comp_cards in pieces:
            for other in range(NUM_SEATS):
                if other == seat:
                    continue
                if self._hand_beats_component(self.hands[other], comp, suit):
                    beatable.append((comp, comp_cards))
                    break
        if not beatable:
            return cards

        # rules.md 7.6 step 4: the smallest component the opponent can
        # beat, ties between equal sizes going to the lower-ranked one
        # and, when rank ties too, to the lowest card type. Two
        # off-suit level cards share a rung under a no-trump
        # declaration and are interchangeable, so the last key is
        # arbitrary -- but it has to exist, or the answer depends on
        # the order the cards arrived in.
        # The cards come from the decomposition rather than being
        # recovered from the component's rung, which would pick the
        # wrong card type whenever two share a rung.
        _comp, forced = min(
            beatable,
            key=lambda cc: (
                cc[0].num_cards,
                cc[0].top,
                int(np.nonzero(cc[1])[0][0]),
            ),
        )
        self.last_forced_lead = (cards - forced).astype(np.int8)
        return forced

    def _hand_beats_component(
        self, hand: np.ndarray, comp: Component, suit: int
    ) -> bool:
        """Does `hand` hold a same-shape higher group in the led suit?

        The block check is led-suit only: a follower can be compelled to
        follow suit but never to trump in.
        """
        in_suit = restrict(hand, self.ctx, suit)
        from .actions import _components_in_suit

        for other, _cards in _components_in_suit(in_suit, self.ctx, self.num_decks):
            if other.shape == comp.shape and other.top > comp.top:
                return True
        return False

    def _close_trick(self) -> None:
        lead = self.trick_plays[0]
        lead_parts = decompose(lead, self.ctx, self.num_decks)
        lead_suit = play_suit(lead, self.ctx)
        assert lead_parts is not None and lead_suit is not None

        winner = self.trick_leader
        best = play_rank(lead_parts, lead_suit)
        # Every response is measured against the lead, not against the
        # running winner: once a response may be re-decomposed to the
        # lead's shape, two responses can each cover the lead while
        # neither covers the other, and chaining would hand the trick to
        # whichever came first.
        for i in range(1, NUM_SEATS):
            seat = (self.trick_leader + i) % NUM_SEATS
            play = self.trick_plays[i]
            if not covers(play, lead_parts, lead_suit, self.ctx, self.num_decks):
                continue
            parts = decompose(play, self.ctx, self.num_decks)
            suit = play_suit(play, self.ctx)
            assert parts is not None and suit is not None
            rank = play_rank(parts, suit)
            if rank > best:
                best = rank
                winner = seat

        points = sum(total_points(p) for p in self.trick_plays)
        if team_of(winner) != self.defenders:
            self.attacker_points += points
        self.tricks.append(
            TrickRecord(self.trick_leader, list(self.trick_plays), winner, points)
        )

        self.trick_plays = []
        self.trick_leader = winner
        self.to_act = winner
        if int(self.hands.sum()) == 0:
            self._finish()

    def _finish(self) -> None:
        self.done = True
        last = self.tricks[-1]
        if team_of(last.winner) != self.defenders:
            # rules.md §8.2: the multiplier is twice the number of cards
            # in the winning play, whatever its shape.
            cards = int(last.plays[
                (last.winner - last.leader) % NUM_SEATS
            ].sum())
            self.attacker_points += total_points(self.kitty) * 2 * cards


# rules.md §8.3. Attacker points to (defender level gain, attacker level
# gain). Only one side ever moves, and the table caps at three.
def hand_result(attacker_points: int) -> tuple[int, int]:
    p = attacker_points
    if p == 0:
        return 3, 0
    if p < 40:
        return 2, 0
    if p < 80:
        return 1, 0
    if p < 120:
        return 0, 0
    if p < 160:
        return 0, 1
    if p < 200:
        return 0, 2
    return 0, 3
