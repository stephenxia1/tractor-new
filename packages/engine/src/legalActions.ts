// Enumerates a (small, opinionated) set of legal events for `seat` at the
// current game state. Designed as a starting point for bot authors: a bot
// can pull a list of candidate moves from here and pick one, rather than
// reimplementing the rules. It is NOT exhaustive — in particular, it does
// not enumerate every possible follow choice in `play` (the space is huge);
// it returns a curated subset that's guaranteed legal.
//
// `applyEvent(state, action)` is the source of truth; anything returned
// here should round-trip through it without error.

import { availableDeclarationsFor } from './declareEnum';
import { classifyCombo, comboBeats } from './classifier';
import { findNKinds, findTractors } from './combos';
import { validateFollow } from './follow';
import { effectiveSuit, compareCards } from './ordering';
import { decomposeThrow, isThrowSafe } from './throws';
import { handFor, nextToPlay } from './reducer';
import { teamOf, type GameState, type Seat } from './state';
import type { GameEvent } from './events';
import { cardEquals, type Card, type TrumpContext } from './types';

export type LegalActionDescription =
  // Anyone may declare; bots choose from this list during declare phases.
  | {
      readonly kind: 'declare';
      readonly seat: Seat;
      readonly cards: ReadonlyArray<Card>;
      readonly tier: number;
    }
  // Landlord-only: pick up the kitty during declare_open.
  | { readonly kind: 'pick_up_kitty'; readonly seat: Seat }
  // Landlord-only: bury 8 cards from hand. We return one suggested
  // "lowest-by-trump-ordering" burial; bots may also construct their own.
  | {
      readonly kind: 'replace_kitty';
      readonly seat: Seat;
      readonly cards: ReadonlyArray<Card>;
    }
  // Play phase actions, only legal when nextToPlay(state) === seat.
  // `tags` is metadata to help bots reason about the action.
  | {
      readonly kind: 'play_cards';
      readonly seat: Seat;
      readonly cards: ReadonlyArray<Card>;
      readonly tags: ReadonlyArray<
        | 'lead-single'
        | 'lead-pair'
        | 'lead-tractor'
        | 'lead-throw-safe'
        | 'follow-led-suit'
        | 'follow-trump'
        | 'follow-discard'
      >;
    }
  // After a complete trick: anyone (any human; bots don't advance) may
  // send this. Listed for completeness.
  | { readonly kind: 'advance_trick' };

export function legalActions(
  state: GameState,
  seat: Seat,
): ReadonlyArray<LegalActionDescription> {
  const out: LegalActionDescription[] = [];
  const hand = state.currentHand;
  if (hand === null) return out;

  if (
    state.phase === 'declare_lockout' ||
    state.phase === 'declare_open'
  ) {
    const myCards = hand.hands[seat] ?? [];
    const R =
      state.landlord !== null
        ? hand.trumpRank
        : state.teamLevels[teamOf(state, seat)]!;
    if (R !== null) {
      for (const d of availableDeclarationsFor(myCards, R, hand.standingDeclaration)) {
        out.push({ kind: 'declare', seat, cards: d.cards, tier: d.tier });
      }
    }
    if (state.phase === 'declare_open' && hand.landlord === seat) {
      out.push({ kind: 'pick_up_kitty', seat });
    }
    return out;
  }

  if (state.phase === 'kitty_replace') {
    if (hand.landlord !== seat) return out;
    if (hand.trumpRank === null) return out;
    const ctx: TrumpContext = {
      trumpRank: hand.trumpRank,
      trumpSuit: hand.trumpSuit,
    };
    const myCards = hand.hands[seat] ?? [];
    const sorted = [...myCards].sort((a, b) => compareCards(a, b, ctx));
    out.push({
      kind: 'replace_kitty',
      seat,
      cards: sorted.slice(0, state.config.kittySize),
    });
    return out;
  }

  if (state.phase === 'play') {
    if (nextToPlay(state) !== seat) return out;
    if (hand.trumpRank === null) return out;
    const ctx: TrumpContext = {
      trumpRank: hand.trumpRank,
      trumpSuit: hand.trumpSuit,
    };
    const n = state.config.numDecks;
    const myCards = hand.hands[seat] ?? [];
    const sorted = [...myCards].sort((a, b) => compareCards(a, b, ctx));
    const trick = hand.currentTrick;

    if (trick === null) {
      // Leader: every single card is legal; pairs in same suit; same-suit
      // tractors of length ≥2; and any safe throw (2+ same-effective-suit
      // cards that are not a single combo and pass the safety check).
      for (const c of sorted) {
        out.push({
          kind: 'play_cards',
          seat,
          cards: [c],
          tags: ['lead-single'],
        });
      }
      // Pairs (and higher n-of-a-kinds — we report just pairs here).
      for (const nk of findNKinds(myCards, 2)) {
        out.push({
          kind: 'play_cards',
          seat,
          cards: nk.cards,
          tags: ['lead-pair'],
        });
      }
      // Tractors.
      for (const t of findTractors(myCards, ctx, n)) {
        out.push({
          kind: 'play_cards',
          seat,
          cards: t.cards,
          tags: ['lead-tractor'],
        });
      }
      // Safe throws across other seats' hands. The leader can't see
      // opponents in real play; this enumeration assumes the leader is
      // a bot driver that already has full state access (e.g. self-play
      // training). For inference under partial information, callers
      // should ignore this branch.
      for (const throwCards of candidateThrows(myCards, ctx, n)) {
        const decomp = decomposeThrow(throwCards, ctx, n);
        if (!decomp.ok) continue;
        const opponentHands = collectOpponentHands(state, seat);
        const safety = isThrowSafe(decomp.throw, opponentHands, ctx, n);
        if (!safety.safe) continue;
        out.push({
          kind: 'play_cards',
          seat,
          cards: throwCards,
          tags: ['lead-throw-safe'],
        });
      }
      return out;
    }

    // Follow.
    const leadCards = trick.plays[0]!.cards;
    const leadClass = classifyCombo(leadCards, ctx, n);
    const leadCount = leadCards.length;
    const ledSuit =
      leadClass.ok
        ? leadClass.combo.effectiveSuit
        : effectiveSuit(leadCards[0]!, ctx);
    const myInLed = myCards.filter((c) => effectiveSuit(c, ctx) === ledSuit);

    // Generate a small set of follow candidates: (1) play your lowest k
    // led-suit + lowest filler; (2) play your highest single led-suit +
    // lowest filler (only if it would beat current winner); (3) play a
    // matching pair in led suit if structurally required and you have one.
    const candidates: Card[][] = [];
    candidates.push(followCandidate(myCards, leadCount, ledSuit, ctx, 'low'));
    if (leadClass.ok) {
      const tryBeat = followCandidate(myCards, leadCount, ledSuit, ctx, 'high');
      candidates.push(tryBeat);
    }
    // Pair-matching candidate.
    if (
      leadClass.ok &&
      leadClass.combo.kind === 'nkind' &&
      leadClass.combo.n === 2
    ) {
      const pairs = findNKinds(myInLed, 2);
      for (const p of pairs) {
        const filled = [...p.cards];
        for (const c of sorted) {
          if (filled.length >= leadCount) break;
          if (filled.some((x) => cardEquals(x, c))) continue;
          filled.push(c);
        }
        candidates.push(filled);
      }
    }

    // Deduplicate by sorted-key signature and emit only those that pass
    // validateFollow.
    const seen = new Set<string>();
    for (const cards of candidates) {
      if (cards.length !== leadCount) continue;
      const key = cards
        .map((c) =>
          c.kind === 'joker'
            ? `J${c.rank}-${c.deckId}`
            : `S${c.suit}${c.rank}-${c.deckId}`,
        )
        .sort()
        .join('|');
      if (seen.has(key)) continue;
      seen.add(key);
      if (leadClass.ok) {
        const v = validateFollow(leadClass.combo, cards, myCards, ctx, n);
        if (!v.ok) continue;
      }
      const tags = classifyFollowTags(
        cards,
        leadClass.ok ? leadClass.combo : null,
        ctx,
      );
      out.push({ kind: 'play_cards', seat, cards, tags });
    }

    return out;
  }

  if (state.phase === 'trick_review') {
    out.push({ kind: 'advance_trick' });
    return out;
  }

  return out;
}

// --- helpers ---------------------------------------------------------------

function followCandidate(
  hand: ReadonlyArray<Card>,
  count: number,
  ledSuit: ReturnType<typeof effectiveSuit>,
  ctx: TrumpContext,
  preference: 'low' | 'high',
): Card[] {
  const sorted = [...hand].sort((a, b) => compareCards(a, b, ctx));
  const inLed = sorted.filter((c) => effectiveSuit(c, ctx) === ledSuit);
  const pool = preference === 'high' ? [...inLed].reverse() : inLed;
  const chosen: Card[] = [];
  for (const c of pool) {
    if (chosen.length >= count) break;
    chosen.push(c);
  }
  if (chosen.length < count) {
    for (const c of sorted) {
      if (chosen.length >= count) break;
      if (chosen.some((x) => cardEquals(x, c))) continue;
      chosen.push(c);
    }
  }
  return chosen;
}

function classifyFollowTags(
  cards: ReadonlyArray<Card>,
  lead: import('./classifier').ClassifiedCombo | null,
  ctx: TrumpContext,
): ReadonlyArray<'follow-led-suit' | 'follow-trump' | 'follow-discard'> {
  if (lead === null) return ['follow-discard'];
  const ledSuit = lead.effectiveSuit;
  const inLed = cards.filter((c) => effectiveSuit(c, ctx) === ledSuit);
  if (inLed.length === cards.length) return ['follow-led-suit'];
  const cls = classifyCombo(cards, ctx, 1);
  if (cls.ok && cls.combo.effectiveSuit === 'trump' && ledSuit !== 'trump') {
    if (comboBeats(cls.combo, lead) > 0) return ['follow-trump'];
  }
  return ['follow-discard'];
}

// All same-effective-suit subsets of `hand` with size ∈ [2, hand.length]
// that aren't classifiable as a single combo (i.e., candidate throws).
// Bounded: we cap on suit by suit and only consider pairs+singles
// combinations to keep the count reasonable.
function candidateThrows(
  hand: ReadonlyArray<Card>,
  ctx: TrumpContext,
  _n: number,
): ReadonlyArray<ReadonlyArray<Card>> {
  const bySuit = new Map<string, Card[]>();
  for (const c of hand) {
    const es = effectiveSuit(c, ctx);
    if (!bySuit.has(es)) bySuit.set(es, []);
    bySuit.get(es)!.push(c);
  }
  const out: Card[][] = [];
  for (const cards of bySuit.values()) {
    if (cards.length < 2) continue;
    // Try the whole same-suit set, plus pairs of size 2..min(5, length).
    out.push([...cards]);
    if (cards.length > 2) {
      for (let i = 0; i < cards.length - 1; i++) {
        out.push([cards[i]!, cards[i + 1]!]);
      }
    }
  }
  return out;
}

function collectOpponentHands(
  state: GameState,
  leaderSeat: Seat,
): ReadonlyArray<ReadonlyArray<Card>> {
  if (state.currentHand === null) return [];
  const leaderTeam = teamOf(state, leaderSeat);
  const out: ReadonlyArray<Card>[] = [];
  for (let s = 0; s < state.config.numPlayers; s++) {
    if (teamOf(state, s) === leaderTeam) continue;
    out.push(handFor(state, s));
  }
  return out;
}

// Convenience: legalActions → GameEvent[] without metadata.
export function legalEvents(
  state: GameState,
  seat: Seat,
): ReadonlyArray<GameEvent> {
  return legalActions(state, seat).map((a) => {
    switch (a.kind) {
      case 'declare':
        return { kind: 'declare', seat: a.seat, cards: a.cards };
      case 'pick_up_kitty':
        return { kind: 'pick_up_kitty', seat: a.seat };
      case 'replace_kitty':
        return { kind: 'replace_kitty', seat: a.seat, cards: a.cards };
      case 'play_cards':
        return { kind: 'play_cards', seat: a.seat, cards: a.cards };
      case 'advance_trick':
        return { kind: 'advance_trick' };
    }
  });
}
