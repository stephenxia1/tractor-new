// Per-decision observation builder for the DMC play-phase agent.
//
// Returns a JSON-serializable Observation that the subprocess driver
// hands to the Python actor, which packs it into PyTorch tensors. The
// same builder is reused at inference time inside the TS forward pass.
//
// Layout summary:
//   - 6 static card-set channels (each 108-dim): own hand, plays-so-far
//     by self/left/across/right, and the "unseen pool" (everything not
//     yet revealed to me).
//   - History sequence: up to 100 prior moves, each encoded as one
//     "step" with (card-set 108 + seat one-hot 4 + trick-position 1).
//   - Scalar feature vector: role/side/score/timing/trick context.
//
// All floats are written into Float32Arrays at build time; serialization
// to JSON converts to plain number[] (lossy for performance but trivial
// to recover on the Python side).

import { sumPoints } from '../../scoring';
import { teamOf, type GameState, type Seat } from '../../state';
import { decomposePlay, decomposedBeats } from '../../throws';
import { cardKey, type Card, type Suit, type TrumpContext } from '../../types';

import {
  encodeCardSet,
  ENCODING_LEN,
  type EncodingContext,
} from './cardEncoding';
import { roleFor, type PositionFeature, type Role } from './roles';

export const HISTORY_MAX_STEPS = 100;
// 7 channels: own hand, plays by self/left/across/right, unseen pool,
// tentative-committed (cards already chosen this turn during a multi-step
// sub-action lead; zeros otherwise).
export const STATIC_CHANNELS = 7;
export const PER_STEP_DIM = ENCODING_LEN + 4 + 1; // cards + seat 1-hot + trickPos
export const SCALAR_DIM = 16;

export type Observation = {
  // Role assignment determines which Q-network handles this decision.
  readonly role: Role;
  readonly side: PositionFeature['side'];
  // 6 × 108 flattened, row-major: [own; self; left; across; right; unseen].
  readonly staticChannels: number[];
  // historySteps × PER_STEP_DIM, row-major. Length = numSteps * PER_STEP_DIM.
  // Step ordering: oldest first, newest last. Truncated to last HISTORY_MAX_STEPS.
  readonly history: number[];
  readonly historyLen: number;
  // Padded mask for the history (1 = real step, 0 = padding). Length = HISTORY_MAX_STEPS.
  readonly historyMask: number[];
  // Scalar features (length = SCALAR_DIM).
  readonly scalars: number[];
};

// Per-seat side mapping relative to a viewer seat (for the 4 play channels).
function sideOf(
  viewer: Seat,
  other: Seat,
  numPlayers: number,
): 'self' | 'left' | 'across' | 'right' {
  if (viewer === other) return 'self';
  const dist = ((other - viewer) + numPlayers) % numPlayers;
  if (dist === 1) return 'left';
  if (dist === 2) return 'across';
  return 'right';
}

const SIDE_INDEX: Record<'self' | 'left' | 'across' | 'right', number> = {
  self: 0,
  left: 1,
  across: 2,
  right: 3,
};

const ROLE_INDEX: Record<Role, number> = {
  landlord: 0,
  defender: 1,
  attacker: 2,
};

// Build per-seat "cards played so far this hand" by walking completed
// tricks + the current in-flight trick.
function playedBySeat(
  state: GameState,
  numPlayers: number,
): Map<Seat, Card[]> {
  const out = new Map<Seat, Card[]>();
  for (let s = 0; s < numPlayers; s++) out.set(s, []);
  const hand = state.currentHand;
  if (hand === null) return out;
  for (const trick of hand.tricks) {
    for (const play of trick.plays) {
      out.get(play.seat)!.push(...play.cards);
    }
  }
  if (hand.currentTrick !== null) {
    for (const play of hand.currentTrick.plays) {
      out.get(play.seat)!.push(...play.cards);
    }
  }
  return out;
}

// Compute the "unseen pool" from the deciding seat's POV: all cards
// not yet revealed = full deck − (my hand) − (everyone's plays) −
// (revealed declarations) − (revealed kitty for landlord).
function unseenPool(
  state: GameState,
  seat: Seat,
  played: Map<Seat, Card[]>,
): Card[] {
  const hand = state.currentHand;
  if (hand === null) return [];
  // Track multiset of seen card keys. The unseen pool is the
  // complement of (my hand ∪ all plays). Declarations don't leak cards
  // beyond what gets played eventually; we treat them as seen only
  // through the play channels.
  const seen = new Map<string, number>();
  const bump = (c: Card): void => {
    const k = cardKey(c);
    seen.set(k, (seen.get(k) ?? 0) + 1);
  };
  for (const c of hand.hands[seat] ?? []) bump(c);
  for (const cards of played.values()) for (const c of cards) bump(c);
  // Build the "full deck" tally and subtract.
  const fullDeck = buildFullDeckCounts(state.config.numDecks);
  const out: Card[] = [];
  for (const [k, total] of fullDeck) {
    const used = seen.get(k) ?? 0;
    const remaining = total - used;
    for (let i = 0; i < remaining; i++) {
      out.push(cardFromKey(k));
    }
  }
  return out;
}

// Cache the deck tally; cards are immutable so this is safe.
let cachedDeckCounts: { numDecks: number; counts: Map<string, number> } | null = null;
function buildFullDeckCounts(numDecks: number): Map<string, number> {
  if (cachedDeckCounts !== null && cachedDeckCounts.numDecks === numDecks) {
    return cachedDeckCounts.counts;
  }
  const counts = new Map<string, number>();
  const suits: Suit[] = ['clubs', 'diamonds', 'hearts', 'spades'];
  const ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] as const;
  for (const s of suits) {
    for (const r of ranks) {
      counts.set(`${s}:${r}`, numDecks);
    }
  }
  counts.set('joker:small', numDecks);
  counts.set('joker:big', numDecks);
  cachedDeckCounts = { numDecks, counts };
  return counts;
}

// Reverse of `cardKey`: construct a placeholder Card. deckId is left as
// 0 since the unseen-pool only matters for set encoding (count, not
// identity).
function cardFromKey(key: string): Card {
  if (key.startsWith('joker:')) {
    const rank = key.slice('joker:'.length) as 'small' | 'big';
    return { kind: 'joker', rank, deckId: 0 };
  }
  const [suitRaw, rankRaw] = key.split(':');
  return {
    kind: 'standard',
    suit: suitRaw as Suit,
    rank: rankRaw as Card extends { kind: 'standard' } ? Card['rank'] : never,
    deckId: 0,
  } as Card;
}

// Per-step history encoding: card-set + seat-side one-hot + trick-position.
function encodeHistoryStep(
  viewer: Seat,
  numPlayers: number,
  ctx: EncodingContext,
  seat: Seat,
  cards: ReadonlyArray<Card>,
  trickPos: number,
  out: Float32Array,
  offset: number,
): void {
  // Card-set portion (108 floats).
  const tmp = encodeCardSet(cards, ctx);
  for (let i = 0; i < ENCODING_LEN; i++) out[offset + i] = tmp[i]!;
  // Seat one-hot (4 floats).
  const side = sideOf(viewer, seat, numPlayers);
  out[offset + ENCODING_LEN + SIDE_INDEX[side]] = 1;
  // Trick-position (1 float, 0..3 normalized).
  out[offset + ENCODING_LEN + 4] = trickPos / 4;
}

// Compute the cumulative attacker points captured so far this hand by
// walking completed tricks. Kitty bonus is added at hand end and isn't
// in scope here.
function attackerPointsSoFar(
  state: GameState,
): number {
  return state.currentHand?.attackerPoints ?? 0;
}

// Whether `viewer`'s team is the currently-winning team of the in-flight
// trick. Mirrors `resolveTrick` in reducer.ts: walks the plays in order,
// using `decomposePlay` + `decomposedBeats` to find the standing winner.
function viewerIsWinningTrick(
  state: GameState,
  viewer: Seat,
  ctx: TrumpContext,
  numDecks: number,
): boolean {
  const trick = state.currentHand?.currentTrick;
  if (trick === null || trick === undefined || trick.plays.length === 0) {
    return false;
  }
  const lead = trick.plays[0]!;
  const leadDecomp = decomposePlay(lead.cards, ctx, numDecks);
  if (leadDecomp === null) {
    // Malformed lead — engine's resolveTrick falls back to lead seat winning.
    return teamOf(state, lead.seat) === teamOf(state, viewer);
  }
  let winnerSeat = lead.seat;
  let winningDecomp = leadDecomp;
  for (let i = 1; i < trick.plays.length; i++) {
    const play = trick.plays[i]!;
    const playDecomp = decomposePlay(play.cards, ctx, numDecks);
    if (playDecomp === null) continue;
    if (decomposedBeats(playDecomp, winningDecomp, ctx, numDecks)) {
      winnerSeat = play.seat;
      winningDecomp = playDecomp;
    }
  }
  return teamOf(state, winnerSeat) === teamOf(state, viewer);
}

export function buildObservation(
  state: GameState,
  seat: Seat,
  // For sub-action lead steps: cards already committed to the in-progress
  // multi-pattern lead. Empty for follows and for the first sub-action
  // step of a lead.
  tentativeChosen: ReadonlyArray<Card> = [],
): Observation {
  const hand = state.currentHand;
  if (hand === null || hand.trumpRank === null || hand.trumpSuit === null) {
    throw new Error('buildObservation requires play-phase state with trump fixed');
  }
  const encCtx: EncodingContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const trumpCtx: TrumpContext = {
    trumpRank: hand.trumpRank,
    trumpSuit: hand.trumpSuit,
  };
  const numPlayers = state.config.numPlayers;

  // ---- Static channels (6 × 108 = 648 floats) ----
  const staticBuf = new Float32Array(STATIC_CHANNELS * ENCODING_LEN);
  const myHand = hand.hands[seat] ?? [];
  const played = playedBySeat(state, numPlayers);
  const unseen = unseenPool(state, seat, played);

  encodeCardSet(myHand, encCtx, staticBuf.subarray(0, ENCODING_LEN));
  // Channels 1-4: plays by self/left/across/right.
  const sides: Array<'self' | 'left' | 'across' | 'right'> = ['self','left','across','right'];
  for (let i = 0; i < numPlayers; i++) {
    const otherSeat = i as Seat;
    const side = sideOf(seat, otherSeat, numPlayers);
    const channelIdx = sides.indexOf(side); // 0..3 → channel 1..4
    const off = (channelIdx + 1) * ENCODING_LEN;
    encodeCardSet(played.get(otherSeat) ?? [], encCtx, staticBuf.subarray(off, off + ENCODING_LEN));
  }
  encodeCardSet(unseen, encCtx, staticBuf.subarray(5 * ENCODING_LEN, 6 * ENCODING_LEN));
  // Channel 6: tentative-committed cards (sub-action lead context).
  encodeCardSet(tentativeChosen, encCtx, staticBuf.subarray(6 * ENCODING_LEN, 7 * ENCODING_LEN));

  // ---- History sequence (up to HISTORY_MAX_STEPS moves) ----
  // We flatten move-by-move: every play in every completed trick + plays
  // in the current in-flight trick.
  type Step = { seat: Seat; cards: ReadonlyArray<Card>; trickPos: number };
  const allSteps: Step[] = [];
  for (const trick of hand.tricks) {
    trick.plays.forEach((p, idx) => {
      allSteps.push({ seat: p.seat, cards: p.cards, trickPos: idx });
    });
  }
  if (hand.currentTrick !== null) {
    hand.currentTrick.plays.forEach((p, idx) => {
      allSteps.push({ seat: p.seat, cards: p.cards, trickPos: idx });
    });
  }
  // Take the last HISTORY_MAX_STEPS.
  const taken = allSteps.slice(-HISTORY_MAX_STEPS);
  const historyBuf = new Float32Array(HISTORY_MAX_STEPS * PER_STEP_DIM);
  const historyMask = new Float32Array(HISTORY_MAX_STEPS);
  for (let i = 0; i < taken.length; i++) {
    const step = taken[i]!;
    const offset = i * PER_STEP_DIM;
    encodeHistoryStep(seat, numPlayers, encCtx, step.seat, step.cards, step.trickPos, historyBuf, offset);
    historyMask[i] = 1;
  }

  // ---- Scalars (SCALAR_DIM = 17) ----
  const role = roleFor(state, seat);
  const scalars = new Float32Array(SCALAR_DIM);
  let off = 0;
  // role one-hot (3)
  scalars[off + ROLE_INDEX[role.role]] = 1; off += 3;
  // side one-hot (4)
  scalars[off + SIDE_INDEX[role.side]] = 1; off += 4;
  // attackerPoints / 200 (1)
  scalars[off++] = attackerPointsSoFar(state) / 200;
  // tricks_remaining / 25 (1)
  scalars[off++] = (25 - hand.tricks.length) / 25;
  // my_hand_size / 25 (1)
  scalars[off++] = myHand.length / 25;
  // trick_position normalized (1)
  const trickPos = hand.currentTrick?.plays.length ?? 0;
  scalars[off++] = trickPos / 4;
  // am_I_winning (1)
  scalars[off++] = viewerIsWinningTrick(state, seat, trumpCtx, state.config.numDecks) ? 1 : 0;
  // trick_points_so_far / 30 (1)
  let trickPts = 0;
  if (hand.currentTrick !== null) {
    for (const p of hand.currentTrick.plays) trickPts += sumPoints(p.cards);
  }
  scalars[off++] = trickPts / 30;
  // is_landlord flag (1) — redundant w/ role one-hot but useful for bias.
  scalars[off++] = role.role === 'landlord' ? 1 : 0;
  // is_my_team_attackers (1)
  const myTeam = teamOf(state, seat);
  const landlordTeam = teamOf(state, hand.landlord!);
  scalars[off++] = myTeam !== landlordTeam ? 1 : 0;
  // bias (1)
  scalars[off++] = 1;

  return {
    role: role.role,
    side: role.side,
    staticChannels: Array.from(staticBuf),
    history: Array.from(historyBuf),
    historyLen: taken.length,
    historyMask: Array.from(historyMask),
    scalars: Array.from(scalars),
  };
}
