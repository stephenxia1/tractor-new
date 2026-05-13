import { classifyCombo, type ClassifiedCombo } from './classifier';
import { buildDeck } from './deck';
import type { GameEvent } from './events';
import { validateFollow, validateFollowThrow } from './follow';
import { mulberry32, shuffle } from './prng';
import {
  decomposePlay,
  decomposeThrow,
  decomposedBeats,
  isThrowSafe,
  throwTotalCards,
  type ThrowDecomposition,
} from './throws';
import {
  advanceLevel,
  computeHandResult,
  kittyMultiplier,
  sumPoints,
} from './scoring';
import {
  declarationTier,
  partnerSeats,
  teamOf,
  type ActiveTrick,
  type CompletedTrick,
  type Declaration,
  type GameState,
  type HandState,
  type Seat,
  type TeamId,
} from './state';
import {
  cardEquals,
  containsAll,
  removeCards,
  type Card,
  type JokerRank,
  type NaturalRank,
  type Suit,
  type TrumpContext,
} from './types';

// Optional side-info attached to a successful play.
// `forced_lead` fires when a leader threw an unsafe multi-component combo:
// the engine plays only the smallest beatable component and returns the
// rest of the throw to the leader's hand. The server uses this to surface
// a chat-log line ("seat X can beat your throw") instead of an error.
export type ApplyNotice = {
  readonly kind: 'forced_lead';
  readonly leaderSeat: Seat;
  readonly beaterSeat: Seat;
  readonly playedCards: ReadonlyArray<Card>;
  readonly returnedCards: ReadonlyArray<Card>;
};

export type ApplyResult =
  | { readonly ok: true; readonly state: GameState; readonly notice?: ApplyNotice }
  | { readonly ok: false; readonly error: string; readonly state: GameState };

const ok = (state: GameState, notice?: ApplyNotice): ApplyResult =>
  notice === undefined ? { ok: true, state } : { ok: true, state, notice };
const err = (state: GameState, error: string): ApplyResult => ({ ok: false, error, state });

export function applyEvent(state: GameState, event: GameEvent): ApplyResult {
  switch (event.kind) {
    case 'start_hand':
      return startHand(state, event);
    case 'declare':
      return declare(state, event);
    case 'release_lockout':
      return releaseLockout(state, event);
    case 'pick_up_kitty':
      return pickUpKitty(state, event);
    case 'replace_kitty':
      return replaceKitty(state, event);
    case 'play_cards':
      return playCards(state, event);
    case 'advance_trick':
      return advanceTrick(state);
  }
}

// ---------------------------------------------------------------------------
// start_hand
// ---------------------------------------------------------------------------
function startHand(
  state: GameState,
  event: Extract<GameEvent, { kind: 'start_hand' }>,
): ApplyResult {
  if (state.phase !== 'between_hands') {
    return err(state, `cannot start a hand from phase ${state.phase}`);
  }

  const { config } = state;
  const deck = buildDeck(config.numDecks);
  const rng = mulberry32(event.seed);
  const shuffled = shuffle(deck, rng);

  const dealable = shuffled.length - config.kittySize;
  const dealCards = shuffled.slice(0, dealable);
  const kitty = shuffled.slice(dealable);

  const hands: Record<Seat, Card[]> = {};
  for (let s = 0; s < config.numPlayers; s++) hands[s] = [];
  for (let i = 0; i < dealCards.length; i++) {
    const seat = i % config.numPlayers;
    hands[seat]!.push(dealCards[i]!);
  }

  const trumpRank =
    state.landlord !== null ? state.teamLevels[teamOf(state, state.landlord)]! : null;

  const handState: HandState = {
    handNumber: state.handsCompleted + 1,
    trumpRank,
    trumpSuit: null,
    trumpFinalized: false,
    landlord: state.landlord,
    hands,
    kitty,
    declarationLog: [],
    standingDeclaration: null,
    tricks: [],
    currentTrick: null,
    attackerPoints: 0,
  };

  return ok({
    ...state,
    phase: 'declare_lockout',
    currentHand: handState,
  });
}

// ---------------------------------------------------------------------------
// declare
// ---------------------------------------------------------------------------
function declare(
  state: GameState,
  event: Extract<GameEvent, { kind: 'declare' }>,
): ApplyResult {
  if (state.phase !== 'declare_lockout' && state.phase !== 'declare_open') {
    return err(state, `cannot declare in phase ${state.phase}`);
  }
  const hand = state.currentHand;
  if (hand === null) return err(state, 'no current hand');

  const seatHand = hand.hands[event.seat];
  if (seatHand === undefined) return err(state, `unknown seat: ${event.seat}`);

  if (!containsAll(seatHand, event.cards)) {
    return err(state, 'declared cards not in declarer hand');
  }

  // Determine the rank R the declarer's declaration must conform to.
  // - Landlord set: R = landlord's team's level (the hand's trumpRank).
  // - Landlord unset: R = declarer's team's level (each declarer claims their
  //   own team's level as R; landlord assignment follows the standing declaration).
  const declarerLevel = state.teamLevels[teamOf(state, event.seat)]!;
  const requiredR: NaturalRank =
    state.landlord !== null
      ? hand.trumpRank! // non-null when landlord is set
      : declarerLevel;

  const classified = classifyDeclaration(event.cards, requiredR);
  if (classified === null) {
    return err(state, 'declared cards do not form a valid declaration');
  }

  const newDecl: Declaration =
    classified.kind === 'single'
      ? {
          kind: 'single',
          seat: event.seat,
          suit: classified.suit,
          cards: [event.cards[0]!],
        }
      : classified.kind === 'pair'
        ? {
            kind: 'pair',
            seat: event.seat,
            suit: classified.suit,
            cards: [event.cards[0]!, event.cards[1]!],
          }
        : {
            kind: 'joker-pair',
            seat: event.seat,
            jokerRank: classified.jokerRank,
            cards: [event.cards[0]!, event.cards[1]!],
          };

  // Override rules (rules.md §5.1, §5.2):
  // - Higher tier overrides lower tier.
  // - Same-tier re-declarations are NOT allowed.
  if (hand.standingDeclaration !== null) {
    const currentTier = declarationTier(hand.standingDeclaration);
    const newTier = declarationTier(newDecl);
    if (newTier <= currentTier) {
      return err(
        state,
        `cannot override tier ${currentTier} declaration with tier ${newTier}`,
      );
    }
  }

  // Resolve trump suit, trump rank, tentative landlord.
  const newTrumpSuit: Suit | null =
    newDecl.kind === 'joker-pair' ? null : newDecl.suit;
  const newTrumpRank: NaturalRank =
    state.landlord !== null ? hand.trumpRank! : declarerLevel;
  const newHandLandlord: Seat | null =
    state.landlord !== null ? hand.landlord : event.seat;

  const newHand: HandState = {
    ...hand,
    trumpRank: newTrumpRank,
    trumpSuit: newTrumpSuit,
    landlord: newHandLandlord,
    declarationLog: [...hand.declarationLog, newDecl],
    standingDeclaration: newDecl,
  };

  return ok({ ...state, currentHand: newHand });
}

// Validates the structural shape AND rank of a declaration's cards.
// Returns the declaration descriptor or null if invalid.
function classifyDeclaration(
  cards: ReadonlyArray<Card>,
  requiredRank: NaturalRank,
):
  | { kind: 'single'; suit: Suit }
  | { kind: 'pair'; suit: Suit }
  | { kind: 'joker-pair'; jokerRank: JokerRank }
  | null {
  if (cards.length === 1) {
    const c = cards[0]!;
    if (c.kind !== 'standard') return null;
    if (c.rank !== requiredRank) return null;
    return { kind: 'single', suit: c.suit };
  }
  if (cards.length === 2) {
    const a = cards[0]!;
    const b = cards[1]!;
    // Both same standard rank R, same suit, different deckIds.
    if (
      a.kind === 'standard' &&
      b.kind === 'standard' &&
      a.rank === requiredRank &&
      b.rank === requiredRank &&
      a.suit === b.suit &&
      a.deckId !== b.deckId
    ) {
      return { kind: 'pair', suit: a.suit };
    }
    // Both same joker rank, different deckIds.
    if (
      a.kind === 'joker' &&
      b.kind === 'joker' &&
      a.rank === b.rank &&
      a.deckId !== b.deckId
    ) {
      return { kind: 'joker-pair', jokerRank: a.rank };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// release_lockout
// ---------------------------------------------------------------------------
function releaseLockout(
  state: GameState,
  event: Extract<GameEvent, { kind: 'release_lockout' }>,
): ApplyResult {
  if (state.phase !== 'declare_lockout') {
    return err(state, `cannot release lockout from phase ${state.phase}`);
  }
  const hand = state.currentHand;
  if (hand === null) return err(state, 'no current hand');

  if (hand.standingDeclaration !== null) {
    // At least one declaration stands; open the kitty-pickup window.
    return ok({ ...state, phase: 'declare_open' });
  }
  // No declaration: kitty-flip fallback (rules.md §5.4).
  return resolveKittyFlip(state, event.seed);
}

function resolveKittyFlip(state: GameState, seed: number | undefined): ApplyResult {
  const hand = state.currentHand!;
  const topCard = hand.kitty[0];
  if (topCard === undefined) return err(state, 'cannot flip kitty: kitty empty');

  // Trump suit: flipped card's suit (or null = no-trump if a joker).
  const flippedSuit: Suit | null = topCard.kind === 'standard' ? topCard.suit : null;

  // Landlord: if unset, pick a random seat using the provided seed.
  let landlord: Seat | null = state.landlord;
  if (landlord === null) {
    const rng = mulberry32(seed ?? 0);
    landlord = Math.floor(rng() * state.config.numPlayers);
  }

  const trumpRank = state.teamLevels[teamOf(state, landlord)]!;

  // Move all kitty cards into the landlord's hand. Move directly into the
  // "kitty_replace" phase so the landlord picks 8 to set aside.
  const handsWithKitty = {
    ...hand.hands,
    [landlord]: [...(hand.hands[landlord] ?? []), ...hand.kitty],
  };

  const newHand: HandState = {
    ...hand,
    trumpRank,
    trumpSuit: flippedSuit,
    trumpFinalized: true,
    landlord,
    hands: handsWithKitty,
    kitty: [],
  };
  return ok({
    ...state,
    landlord,
    phase: 'kitty_replace',
    currentHand: newHand,
  });
}

// ---------------------------------------------------------------------------
// pick_up_kitty
// ---------------------------------------------------------------------------
function pickUpKitty(
  state: GameState,
  event: Extract<GameEvent, { kind: 'pick_up_kitty' }>,
): ApplyResult {
  if (state.phase !== 'declare_open') {
    return err(state, `cannot pick up kitty from phase ${state.phase}`);
  }
  const hand = state.currentHand;
  if (hand === null) return err(state, 'no current hand');
  if (hand.landlord === null) return err(state, 'no landlord set');
  if (event.seat !== hand.landlord) {
    return err(state, `only landlord (seat ${hand.landlord}) can pick up kitty`);
  }

  const handsWithKitty = {
    ...hand.hands,
    [hand.landlord]: [...(hand.hands[hand.landlord] ?? []), ...hand.kitty],
  };

  const newHand: HandState = {
    ...hand,
    hands: handsWithKitty,
    kitty: [],
    trumpFinalized: true,
  };
  return ok({
    ...state,
    landlord: hand.landlord, // commit the tentative landlord to game level
    phase: 'kitty_replace',
    currentHand: newHand,
  });
}

// ---------------------------------------------------------------------------
// replace_kitty
// ---------------------------------------------------------------------------
function replaceKitty(
  state: GameState,
  event: Extract<GameEvent, { kind: 'replace_kitty' }>,
): ApplyResult {
  if (state.phase !== 'kitty_replace') {
    return err(state, `cannot replace kitty from phase ${state.phase}`);
  }
  const hand = state.currentHand;
  if (hand === null) return err(state, 'no current hand');
  if (hand.landlord === null) return err(state, 'no landlord set');
  if (event.seat !== hand.landlord) {
    return err(state, `only landlord (seat ${hand.landlord}) can replace kitty`);
  }
  if (event.cards.length !== state.config.kittySize) {
    return err(
      state,
      `must set aside exactly ${state.config.kittySize} cards, got ${event.cards.length}`,
    );
  }

  const landlordHand = hand.hands[hand.landlord]!;
  if (!containsAll(landlordHand, event.cards)) {
    return err(state, 'kitty replacement cards not all in landlord hand');
  }

  // Cards must be distinct (no double-counting the same card).
  for (let i = 0; i < event.cards.length; i++) {
    for (let j = i + 1; j < event.cards.length; j++) {
      if (cardEquals(event.cards[i]!, event.cards[j]!)) {
        return err(state, 'kitty cards must be distinct');
      }
    }
  }

  const newLandlordHand = removeCards(landlordHand, event.cards);
  const newHand: HandState = {
    ...hand,
    hands: { ...hand.hands, [hand.landlord]: newLandlordHand },
    kitty: [...event.cards],
  };

  return ok({
    ...state,
    phase: 'play',
    currentHand: newHand,
  });
}

// ---------------------------------------------------------------------------
// play_cards
// ---------------------------------------------------------------------------
function playCards(
  state: GameState,
  event: Extract<GameEvent, { kind: 'play_cards' }>,
): ApplyResult {
  if (state.phase !== 'play') {
    return err(state, `cannot play cards from phase ${state.phase}`);
  }
  const hand = state.currentHand;
  if (hand === null) return err(state, 'no current hand');
  if (hand.landlord === null) return err(state, 'no landlord set');
  if (hand.trumpRank === null) return err(state, 'no trump rank');

  // Determine expected next player.
  const trick: ActiveTrick | null = hand.currentTrick;
  const numPlayers = state.config.numPlayers;
  let expectedSeat: Seat;
  if (trick === null) {
    expectedSeat =
      hand.tricks.length === 0
        ? hand.landlord
        : hand.tricks[hand.tricks.length - 1]!.winner;
  } else {
    expectedSeat = (trick.leader + trick.plays.length) % numPlayers;
  }
  if (event.seat !== expectedSeat) {
    return err(state, `expected seat ${expectedSeat} to play, got ${event.seat}`);
  }

  const seatHand = hand.hands[event.seat];
  if (seatHand === undefined) return err(state, `unknown seat ${event.seat}`);
  if (event.cards.length === 0) return err(state, 'must play at least one card');
  if (!containsAll(seatHand, event.cards)) {
    return err(state, 'played cards not in hand');
  }

  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;

  // The cards actually committed to the trick. Normally equal to
  // event.cards; reduced if a forced-lead penalty fires.
  let effectiveCards: ReadonlyArray<Card> = event.cards;
  let notice: ApplyNotice | undefined;

  // Validate count and (for leader) lead structure.
  if (trick === null) {
    // Leader: try simple combo first; else try a throw and check safety.
    const single = classifyCombo(event.cards, ctx, n);
    if (!single.ok) {
      const decomp = decomposeThrow(event.cards, ctx, n);
      if (!decomp.ok) {
        return err(state, `invalid lead: ${single.reason} / ${decomp.reason}`);
      }
      // Throw: safety check against opponent hands.
      const opponentSeats = collectOpponentSeats(state, event.seat);
      const opponentHands = opponentSeats.map((s) => hand.hands[s] ?? []);
      const safety = isThrowSafe(decomp.throw, opponentHands, ctx, n);
      if (!safety.safe) {
        // Forced-lead penalty: only the smallest beatable component is
        // played; the rest of the throw returns to the leader's hand.
        const beat = safety.smallestBeatable;
        effectiveCards = beat.kind === 'single' ? [beat.card] : beat.cards;
        const returned = removeCards(event.cards, effectiveCards);
        notice = {
          kind: 'forced_lead',
          leaderSeat: event.seat,
          beaterSeat: opponentSeats[safety.beaterIndex]!,
          playedCards: [...effectiveCards],
          returnedCards: returned,
        };
      }
    }
  } else {
    const leadCards = trick.plays[0]!.cards;
    const leadSingle = classifyCombo(leadCards, ctx, n);
    if (leadSingle.ok) {
      const v = validateFollow(leadSingle.combo, event.cards, seatHand, ctx, n);
      if (!v.ok) return err(state, `invalid follow: ${v.error}`);
    } else {
      // If the prior lead was a single-component combo, `leadSingle.ok`
      // would be true. If we get here, the prior lead was a throw — or
      // a forced-lead component (which is a single combo).
      const leadThrow = decomposeThrow(leadCards, ctx, n);
      if (!leadThrow.ok) {
        return err(state, `lead is malformed: ${leadThrow.reason}`);
      }
      const v = validateFollowThrow(leadThrow.throw, event.cards, seatHand, ctx);
      if (!v.ok) return err(state, `invalid follow: ${v.error}`);
    }
  }

  // Apply the play. Only `effectiveCards` leave the hand — on a
  // forced-lead penalty, the rest of the throw stays in the leader's hand.
  const newSeatHand = removeCards(seatHand, effectiveCards);
  const newHands = { ...hand.hands, [event.seat]: newSeatHand };

  const newPlay = { seat: event.seat, cards: [...effectiveCards] };
  const newTrick: ActiveTrick =
    trick === null
      ? { leader: event.seat, plays: [newPlay] }
      : { ...trick, plays: [...trick.plays, newPlay] };

  let updatedHand: HandState = {
    ...hand,
    hands: newHands,
    currentTrick: newTrick,
  };

  // If this play completes the trick, transition to trick_review so the
  // played cards stay on the table until someone advances the trick.
  // Resolution / points / hand-finalization are deferred to advance_trick.
  if (newTrick.plays.length === numPlayers) {
    return ok(
      { ...state, phase: 'trick_review', currentHand: updatedHand },
      notice,
    );
  }

  return ok({ ...state, currentHand: updatedHand }, notice);
}

// ---------------------------------------------------------------------------
// advance_trick
// ---------------------------------------------------------------------------
function advanceTrick(state: GameState): ApplyResult {
  if (state.phase !== 'trick_review') {
    return err(state, `cannot advance trick from phase ${state.phase}`);
  }
  const hand = state.currentHand;
  if (hand === null) return err(state, 'no current hand');
  if (hand.landlord === null) return err(state, 'no landlord set');
  if (hand.trumpRank === null) return err(state, 'no trump rank');
  const trick = hand.currentTrick;
  if (trick === null) return err(state, 'no current trick to advance');
  const numPlayers = state.config.numPlayers;
  if (trick.plays.length !== numPlayers) {
    return err(state, 'cannot advance an incomplete trick');
  }
  const ctx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const n = state.config.numDecks;
  const resolution = resolveTrick(trick, ctx, n);
  const trickPoints = sumPoints(trick.plays.flatMap((p) => p.cards));

  const completed: CompletedTrick = {
    leader: trick.leader,
    plays: trick.plays,
    winner: resolution.winner,
    points: trickPoints,
  };

  const winnerTeam = teamOf(state, resolution.winner);
  const landlordTeam = teamOf(state, hand.landlord);
  const isAttackerWin = winnerTeam !== landlordTeam;
  const attackerPoints = hand.attackerPoints + (isAttackerWin ? trickPoints : 0);

  const updatedHand: HandState = {
    ...hand,
    currentTrick: null,
    tricks: [...hand.tricks, completed],
    attackerPoints,
  };

  // The hand ends when every player has emptied their hand. Counting
  // tricks doesn't work because a single trick can consume more than one
  // card per player (pair/tractor leads).
  let allEmpty = true;
  for (let s = 0; s < numPlayers; s++) {
    if ((updatedHand.hands[s] ?? []).length > 0) {
      allEmpty = false;
      break;
    }
  }
  if (allEmpty) {
    return finalizeHand(state, updatedHand, resolution, isAttackerWin);
  }

  return ok({ ...state, phase: 'play', currentHand: updatedHand });
}

// Determines the winner and points of a fully-played trick.
// Decomposes every play (single combo or multi-component throw) and
// compares structurally: a follower beats the current winner only if
// their play has the same component shape multiset and each component
// out-ranks a distinct one in the current winner.
function resolveTrick(
  trick: ActiveTrick,
  ctx: TrumpContext,
  n: number,
): { winner: Seat; winningCombo: ClassifiedCombo | null; throwCardCount: number } {
  const lead = trick.plays[0]!;
  const leadDecomp = decomposePlay(lead.cards, ctx, n);

  if (leadDecomp === null) {
    // Truly malformed lead — defensive default.
    return {
      winner: lead.seat,
      winningCombo: null,
      throwCardCount: lead.cards.length,
    };
  }

  let winnerSeat: Seat = lead.seat;
  let winningDecomp: ThrowDecomposition = leadDecomp;

  for (let i = 1; i < trick.plays.length; i++) {
    const play = trick.plays[i]!;
    const playDecomp = decomposePlay(play.cards, ctx, n);
    if (playDecomp === null) continue;
    if (decomposedBeats(playDecomp, winningDecomp)) {
      winnerSeat = play.seat;
      winningDecomp = playDecomp;
    }
  }

  if (winningDecomp.components.length === 1) {
    return {
      winner: winnerSeat,
      winningCombo: winningDecomp.components[0]!,
      throwCardCount: 0,
    };
  }
  return {
    winner: winnerSeat,
    winningCombo: null,
    throwCardCount: throwTotalCards(winningDecomp),
  };
}

function collectOpponentSeats(state: GameState, leaderSeat: Seat): Seat[] {
  // Per rules.md §7.6, any single OTHER player — opponent or teammate —
  // can break a throw. Only the leader themselves is excluded.
  const out: Seat[] = [];
  for (let s = 0; s < state.config.numPlayers; s++) {
    if (s === leaderSeat) continue;
    out.push(s);
  }
  return out;
}

// Finalizes a completed hand: applies kitty multiplier, computes scores,
// updates levels, updates landlord (rotate to partner or unset), and
// transitions to between_hands or game_over.
function finalizeHand(
  state: GameState,
  hand: HandState,
  lastTrickRes: { winner: Seat; winningCombo: ClassifiedCombo | null; throwCardCount: number },
  attackersWonLastTrick: boolean,
): ApplyResult {
  // Kitty multiplier — only if attackers won the last trick.
  let attackerPoints = hand.attackerPoints;
  if (attackersWonLastTrick && hand.kitty.length > 0) {
    let cardsInCombo: number;
    if (lastTrickRes.winningCombo !== null) {
      const c = lastTrickRes.winningCombo;
      cardsInCombo = c.kind === 'single' ? 1 : c.cards.length;
    } else {
      // Throw lead (or otherwise unclassified) — use throw total.
      cardsInCombo = lastTrickRes.throwCardCount || 1;
    }
    const m = kittyMultiplier(cardsInCombo);
    attackerPoints += m * sumPoints(hand.kitty);
  }

  const result = computeHandResult(attackerPoints);

  // Identify defending team (landlord's team) and attacking team.
  const defendingTeam: TeamId = teamOf(state, hand.landlord!);
  const attackingTeam = state.config.teamIds.find((t) => t !== defendingTeam)!;

  // Apply level change.
  const newTeamLevels = { ...state.teamLevels };
  let winnerTeam: TeamId;
  if (result.winner === 'defenders') {
    winnerTeam = defendingTeam;
    newTeamLevels[defendingTeam] = advanceLevel(
      newTeamLevels[defendingTeam]!,
      result.levelDelta,
    );
  } else {
    winnerTeam = attackingTeam;
    newTeamLevels[attackingTeam] = advanceLevel(
      newTeamLevels[attackingTeam]!,
      result.levelDelta,
    );
  }

  // Game-end check: defending team won this hand AND was at A entering the hand.
  const defendingTeamWasAtA = state.teamLevels[defendingTeam] === 'A';
  const isGameWin = result.winner === 'defenders' && defendingTeamWasAtA;

  // Landlord rotation: always passes to the next seat (clockwise from the
  // current landlord) that belongs to the team that won this hand.
  //   - Defenders win → next defender → effectively rotates to partner.
  //   - Attackers win → next attacker → first seat clockwise on attackers.
  const winningTeamId =
    result.winner === 'defenders' ? defendingTeam : attackingTeam;
  const numPlayers = state.config.numPlayers;
  const startSeat = hand.landlord!;
  let newLandlord: Seat = startSeat;
  for (let step = 1; step <= numPlayers; step++) {
    const candidate = (startSeat + step) % numPlayers;
    if (state.config.seatToTeam[candidate] === winningTeamId) {
      newLandlord = candidate;
      break;
    }
  }

  const finalHand: HandState = {
    ...hand,
    attackerPoints,
  };

  return ok({
    ...state,
    teamLevels: newTeamLevels,
    landlord: newLandlord,
    phase: isGameWin ? 'game_over' : 'between_hands',
    winner: isGameWin ? winnerTeam : null,
    handsCompleted: state.handsCompleted + 1,
    currentHand: finalHand,
  });
}

// ---------------------------------------------------------------------------
// Selectors
// ---------------------------------------------------------------------------
export function handFor(state: GameState, seat: Seat): ReadonlyArray<Card> {
  if (state.currentHand === null) return [];
  return state.currentHand.hands[seat] ?? [];
}

export function levelRankForDeclarer(state: GameState, seat: Seat): NaturalRank {
  return state.teamLevels[teamOf(state, seat)]!;
}

// Returns the seat expected to play next during the play phase, or null
// if not in play phase.
export function nextToPlay(state: GameState): Seat | null {
  if (state.phase !== 'play') return null;
  const hand = state.currentHand;
  if (hand === null || hand.landlord === null) return null;
  const trick = hand.currentTrick;
  if (trick === null) {
    return hand.tricks.length === 0
      ? hand.landlord
      : hand.tricks[hand.tricks.length - 1]!.winner;
  }
  return (trick.leader + trick.plays.length) % state.config.numPlayers;
}
