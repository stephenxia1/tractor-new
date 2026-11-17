// The play network's observation, ported from
// research/py/tractor/nn/features.py (`encode`) and
// research/py/tractor/train/rollout.py (`_pack`).
//
// The Python side rebuilds a `PlayState` from the position the engine
// reports (research/py/tractor/eval/bots.py `state_from_wire`) and encodes
// it; this file encodes the same `ZeroPosition` directly. Every value the
// network reads must come out identical, so the arithmetic follows the
// Python line by line, down to the float16 rounding `_pack` applies to the
// history, context and card blocks before the forward pass.
// research/py/test_zero_parity.py holds the two together.
//
// Cards are the 54 slots of rl/v2/cards.ts: 13 * suit + rank for the
// standard cards (suits spades, hearts, diamonds, clubs; ranks 2..A), 52
// the small joker, 53 the big joker. Multisets are count vectors.

export const NUM_TYPES = 54;
export const NUM_RANKS = 13;
export const NUM_SEATS = 4;
export const SMALL_JOKER = 52;
export const BIG_JOKER = 53;
export const TRUMP = 4; // the effective suit of every trump card
export const NO_TRUMP = 4; // trump suit index of a joker declaration
export const NUM_TRUMP_CHOICES = 5;

const RANK_5 = 3;
const RANK_10 = 8;
const RANK_K = 11;

// --- layouts (features.py) ---------------------------------------------

export const NUM_PLANES = 10;
const PLANE_HAND = 0;
const PLANE_PLAYED = 1;
const PLANE_TRICK = 5;
const PLANE_KITTY = 9;

const NUM_EFF_SUITS = 5;
const MAX_STRENGTH = 15;
const NUM_STRENGTH = MAX_STRENGTH + 1;
const NUM_POINT_VALUES = 3;
const STATIC_SUIT = 0;
const STATIC_STRENGTH = STATIC_SUIT + NUM_EFF_SUITS;
const STATIC_POINTS = STATIC_STRENGTH + NUM_STRENGTH;
const STATIC_FLAGS = STATIC_POINTS + NUM_POINT_VALUES;
const FLAG_LEVEL = 0;
const FLAG_TRUMP_LEVEL = 1;
const FLAG_SMALL_JOKER = 2;
const FLAG_BIG_JOKER = 3;
export const D_STATIC = STATIC_FLAGS + 4; // 28
const DYN_COUNT = 0;
const DYN_TOP_UNSEEN = 1;
export const D_CARD = D_STATIC + 2; // 30

const CTX_ROLE = 0;
const CTX_ATTACKER = CTX_ROLE + NUM_SEATS;
const CTX_LEVEL = CTX_ATTACKER + 1;
const CTX_TRUMP = CTX_LEVEL + NUM_RANKS;
const CTX_ATTACKER_POINTS = CTX_TRUMP + NUM_TRUMP_CHOICES;
const CTX_UNPLAYED_POINTS = CTX_ATTACKER_POINTS + 1;
const CTX_TRICKS_PLAYED = CTX_UNPLAYED_POINTS + 1;
const CTX_CARDS_LEFT = CTX_TRICKS_PLAYED + 1;
const CTX_LEAD_SIZE = CTX_CARDS_LEFT + 1;
const CTX_LED_SUIT = CTX_LEAD_SIZE + 1;
const CTX_WINNING_SEAT = CTX_LED_SUIT + NUM_EFF_SUITS;
export const CONTEXT_DIM = CTX_WINNING_SEAT + NUM_SEATS; // 37

export const MAX_HISTORY = 128;
const HIST_SUMMARY = 0;
const HIST_SEAT = D_STATIC;
const HIST_TRICK = D_STATIC + 4;
const HIST_POSITION = D_STATIC + 5;
const HIST_IS_LEAD = D_STATIC + 6;
const HIST_WON = D_STATIC + 7;
const HIST_POINTS = D_STATIC + 8;
export const HISTORY_DIM = D_STATIC + 9; // 37

const MAX_TRICKS = 25.0;
const MAX_HAND = 25.0;
const MAX_LEAD_SIZE = 8.0;
const POINT_NORM = 200.0;
const PLAY_POINT_NORM = 10.0;
const POINTS_PER_DECK = 100;

// Distance from the landlord to the role slot: landlord, partner, the
// attacker after the landlord, the attacker before.
const ROLE_OF_OFFSET = [0, 2, 1, 3] as const;

export const POINTS: Int16Array = (() => {
  const p = new Int16Array(NUM_TYPES);
  for (let t = 0; t < 52; t++) {
    const r = t % NUM_RANKS;
    if (r === RANK_5) p[t] = 5;
    else if (r === RANK_10 || r === RANK_K) p[t] = 10;
  }
  return p;
})();
const POINT_SLOT = Array.from(POINTS, (p) => (p === 0 ? 0 : p === 5 ? 1 : 2));

// --- trump context (rules/cards.py) --------------------------------------

export type Ctx = {
  readonly levelRank: number;
  readonly trumpSuit: number; // 0..3, or NO_TRUMP
  readonly eff: Int8Array; // effective suit per type
  readonly strength: Int8Array;
};

const CTX_CACHE = new Map<number, Ctx>();

export function makeCtx(levelRank: number, trumpSuit: number): Ctx {
  if (!(levelRank >= 0 && levelRank < NUM_RANKS)) throw new Error(`level rank ${levelRank}`);
  if (!(trumpSuit >= 0 && trumpSuit < NUM_TRUMP_CHOICES)) throw new Error(`trump suit ${trumpSuit}`);
  const cid = levelRank * NUM_TRUMP_CHOICES + trumpSuit;
  const hit = CTX_CACHE.get(cid);
  if (hit !== undefined) return hit;
  const eff = new Int8Array(NUM_TYPES);
  const strength = new Int8Array(NUM_TYPES);
  const lr = levelRank;
  const ts = trumpSuit;
  for (let t = 0; t < NUM_TYPES; t++) {
    const joker = t >= SMALL_JOKER;
    const rank = t % NUM_RANKS;
    const suit = Math.floor(t / NUM_RANKS);
    const trump = joker || rank === lr || (ts !== NO_TRUMP && suit === ts);
    eff[t] = trump ? TRUMP : suit;
    if (!trump) {
      strength[t] = rank > lr ? rank - 1 : rank;
    } else if (ts === NO_TRUMP) {
      strength[t] = t === BIG_JOKER ? 2 : t === SMALL_JOKER ? 1 : 0;
    } else if (t === BIG_JOKER) {
      strength[t] = 15;
    } else if (t === SMALL_JOKER) {
      strength[t] = 14;
    } else if (rank === lr && suit === ts) {
      strength[t] = 13;
    } else if (rank === lr) {
      strength[t] = 12;
    } else {
      strength[t] = rank > lr ? rank - 1 : rank;
    }
  }
  const ctx: Ctx = { levelRank, trumpSuit, eff, strength };
  CTX_CACHE.set(cid, ctx);
  return ctx;
}

// float32[54 x D_STATIC]: what each card is in this trump context.
const STATIC_CACHE = new Map<Ctx, Float32Array>();

export function staticCardFeatures(ctx: Ctx): Float32Array {
  const hit = STATIC_CACHE.get(ctx);
  if (hit !== undefined) return hit;
  const out = new Float32Array(NUM_TYPES * D_STATIC);
  for (let t = 0; t < NUM_TYPES; t++) {
    const row = t * D_STATIC;
    out[row + STATIC_SUIT + ctx.eff[t]!] = 1;
    out[row + STATIC_STRENGTH + ctx.strength[t]!] = 1;
    out[row + STATIC_POINTS + POINT_SLOT[t]!] = 1;
    const isLevel = t < SMALL_JOKER && t % NUM_RANKS === ctx.levelRank;
    out[row + STATIC_FLAGS + FLAG_LEVEL] = isLevel ? 1 : 0;
    out[row + STATIC_FLAGS + FLAG_TRUMP_LEVEL] =
      isLevel && Math.floor(t / NUM_RANKS) === ctx.trumpSuit ? 1 : 0;
    out[row + STATIC_FLAGS + FLAG_SMALL_JOKER] = t === SMALL_JOKER ? 1 : 0;
    out[row + STATIC_FLAGS + FLAG_BIG_JOKER] = t === BIG_JOKER ? 1 : 0;
  }
  STATIC_CACHE.set(ctx, out);
  return out;
}

// --- float16 rounding ------------------------------------------------------

// `_pack` stores the history, context and card blocks as float16 and the
// network reads them back widened, so the values it sees are float16
// values. Round-to-nearest-even through float32, as numpy does.
const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);

export function toHalfAndBack(x: number): number {
  if (x === 0 || !Number.isFinite(x)) return x;
  F32[0] = x;
  const bits = U32[0]!;
  const sign = bits & 0x80000000;
  const exp = (bits >>> 23) & 0xff;
  const e = exp - 127;
  if (e > 15) return sign ? -Infinity : Infinity;
  // Quantum of float16 at this magnitude: 2^(e-10) for normals, 2^-24
  // for subnormals.
  const q = e < -14 ? Math.pow(2, -24) : Math.pow(2, e - 10);
  const a = Math.abs(x) / q;
  let r = Math.floor(a);
  const frac = a - r;
  if (frac > 0.5 || (frac === 0.5 && r % 2 === 1)) r += 1;
  const v = r * q;
  return sign ? -v : v;
}

function roundHalf(a: Float32Array): Float32Array {
  for (let i = 0; i < a.length; i++) a[i] = toHalfAndBack(a[i]!);
  return a;
}

// --- the position ----------------------------------------------------------

export type ZeroTrick = {
  readonly leader: number;
  readonly plays: ReadonlyArray<ArrayLike<number>>; // play i is seat (leader + i) % 4
  readonly winner: number;
  readonly points: number;
};

/** One seat's legal view of a play-phase position. Only `hand` and, for
 *  the landlord, `kitty` are private; everything else is on the table. */
export type ZeroPosition = {
  readonly seat: number;
  readonly levelRank: number;
  readonly trumpSuit: number;
  readonly numDecks: number;
  readonly landlord: number;
  readonly hand: ArrayLike<number>; // 54 counts
  readonly kitty: ArrayLike<number> | null; // the burial, landlord only
  readonly tricks: ReadonlyArray<ZeroTrick>;
  readonly trickPlays: ReadonlyArray<ArrayLike<number>>;
  readonly trickLeader: number;
  /** Who takes the open trick if it ends now; null between tricks. */
  readonly currentWinner: number | null;
};

const mod4 = (x: number): number => ((x % NUM_SEATS) + NUM_SEATS) % NUM_SEATS;
const teamOf = (seat: number): number => seat % 2;

function playPoints(play: ArrayLike<number>): number {
  let s = 0;
  for (let t = 0; t < NUM_TYPES; t++) s += (play[t] ?? 0) * POINTS[t]!;
  return s;
}

// The effective suit of a play, or null when it spans more than one
// (rules/combos.py `play_suit`).
function playSuit(play: ArrayLike<number>, ctx: Ctx): number | null {
  let suit = -1;
  for (let t = 0; t < NUM_TYPES; t++) {
    if ((play[t] ?? 0) === 0) continue;
    const s = ctx.eff[t]!;
    if (suit === -1) suit = s;
    else if (s !== suit) return null;
  }
  return suit === -1 ? null : suit;
}

/** The observation the network reads, candidates aside (net.ts `ZeroObs`
 *  adds them). Row-major, unpadded. */
export type ZeroFeatures = {
  history: Float32Array; // historyLen x HISTORY_DIM
  historyLen: number;
  context: Float32Array; // CONTEXT_DIM
  unseen: Float32Array; // 54, raw counts
  planes: Float32Array; // NUM_PLANES x 54, raw counts
  cards: Float32Array; // 54 x D_CARD
};

export function encodePosition(pos: ZeroPosition): ZeroFeatures {
  const { seat, numDecks } = pos;
  const ctx = makeCtx(pos.levelRank, pos.trumpSuit);
  const stat = staticCardFeatures(ctx);
  const isLandlord = seat === pos.landlord;

  // Every play of the hand, oldest first: the simulator's log.
  type Row = { counts: ArrayLike<number>; player: number; trick: number; position: number; won: number };
  const log: Row[] = [];
  const playedTotal = new Int16Array(NUM_TYPES);
  const playedBySeat = Array.from({ length: NUM_SEATS }, () => new Int16Array(NUM_TYPES));
  let trickPointsAttacker = 0;
  const defenders = teamOf(pos.landlord);
  pos.tricks.forEach((trick, index) => {
    if (teamOf(trick.winner) !== defenders) trickPointsAttacker += trick.points;
    trick.plays.forEach((play, i) => {
      const player = mod4(trick.leader + i);
      log.push({ counts: play, player, trick: index, position: i, won: player === trick.winner ? 1 : 0 });
      for (let t = 0; t < NUM_TYPES; t++) {
        const c = play[t] ?? 0;
        playedTotal[t] = playedTotal[t]! + c;
        const bySeat = playedBySeat[player]!;
        bySeat[t] = bySeat[t]! + c;
      }
    });
  });
  pos.trickPlays.forEach((play, i) => {
    const player = mod4(pos.trickLeader + i);
    log.push({ counts: play, player, trick: pos.tricks.length, position: i, won: 0 });
    for (let t = 0; t < NUM_TYPES; t++) playedTotal[t] = playedTotal[t]! + (play[t] ?? 0);
  });

  // Planes.
  const planes = new Float32Array(NUM_PLANES * NUM_TYPES);
  for (let t = 0; t < NUM_TYPES; t++) planes[PLANE_HAND * NUM_TYPES + t] = pos.hand[t] ?? 0;
  for (let rel = 0; rel < NUM_SEATS; rel++) {
    const src = playedBySeat[mod4(seat + rel)]!;
    for (let t = 0; t < NUM_TYPES; t++) planes[(PLANE_PLAYED + rel) * NUM_TYPES + t] = src[t]!;
  }
  pos.trickPlays.forEach((play, i) => {
    const rel = mod4(pos.trickLeader + i - seat);
    for (let t = 0; t < NUM_TYPES; t++) planes[(PLANE_TRICK + rel) * NUM_TYPES + t] = play[t] ?? 0;
  });
  if (isLandlord && pos.kitty !== null) {
    for (let t = 0; t < NUM_TYPES; t++) planes[PLANE_KITTY * NUM_TYPES + t] = pos.kitty[t] ?? 0;
  }

  // Unseen: the deck less my hand, every card played, and my burial.
  const unseen = new Float32Array(NUM_TYPES);
  for (let t = 0; t < NUM_TYPES; t++) {
    let seen = (pos.hand[t] ?? 0) + playedTotal[t]!;
    if (isLandlord && pos.kitty !== null) seen += pos.kitty[t] ?? 0;
    unseen[t] = Math.max(numDecks - seen, 0);
  }

  // Cards: the static block, then copies held / decks and top-of-suit.
  const cards = new Float32Array(NUM_TYPES * D_CARD);
  for (let t = 0; t < NUM_TYPES; t++) {
    const row = t * D_CARD;
    for (let j = 0; j < D_STATIC; j++) cards[row + j] = stat[t * D_STATIC + j]!;
    cards[row + D_STATIC + DYN_COUNT] = (pos.hand[t] ?? 0) / numDecks;
    let beaters = 0;
    for (let u = 0; u < NUM_TYPES; u++) {
      if (ctx.eff[u] === ctx.eff[t] && ctx.strength[u]! > ctx.strength[t]!) beaters += unseen[u]!;
    }
    cards[row + D_STATIC + DYN_TOP_UNSEEN] = beaters === 0 ? 1 : 0;
  }

  // Context.
  const context = new Float32Array(CONTEXT_DIM);
  context[CTX_ROLE + ROLE_OF_OFFSET[mod4(seat - pos.landlord)]!] = 1;
  context[CTX_ATTACKER] = teamOf(seat) !== defenders ? 1 : 0;
  context[CTX_LEVEL + ctx.levelRank] = 1;
  context[CTX_TRUMP + ctx.trumpSuit] = 1;
  let pointsPlayed = 0;
  for (let t = 0; t < NUM_TYPES; t++) pointsPlayed += playedTotal[t]! * POINTS[t]!;
  context[CTX_ATTACKER_POINTS] = trickPointsAttacker / POINT_NORM;
  context[CTX_UNPLAYED_POINTS] = (POINTS_PER_DECK * numDecks - pointsPlayed) / POINT_NORM;
  context[CTX_TRICKS_PLAYED] = pos.tricks.length / MAX_TRICKS;
  let held = 0;
  for (let t = 0; t < NUM_TYPES; t++) held += pos.hand[t] ?? 0;
  context[CTX_CARDS_LEFT] = held / MAX_HAND;
  if (pos.trickPlays.length > 0) {
    const lead = pos.trickPlays[0]!;
    let size = 0;
    for (let t = 0; t < NUM_TYPES; t++) size += lead[t] ?? 0;
    context[CTX_LEAD_SIZE] = size / MAX_LEAD_SIZE;
    const suit = playSuit(lead, ctx);
    if (suit !== null) context[CTX_LED_SUIT + suit] = 1;
    if (pos.currentWinner !== null) {
      context[CTX_WINNING_SEAT + mod4(pos.currentWinner - seat)] = 1;
    }
  }

  // History: one token per play, the most recent MAX_HISTORY.
  const total = log.length;
  const kept = Math.min(total, MAX_HISTORY);
  const history = new Float32Array(kept * HISTORY_DIM);
  for (let k = 0; k < kept; k++) {
    const row = log[total - kept + k]!;
    const at = k * HISTORY_DIM;
    for (let t = 0; t < NUM_TYPES; t++) {
      const c = row.counts[t] ?? 0;
      if (c === 0) continue;
      for (let j = 0; j < D_STATIC; j++) {
        const k = at + HIST_SUMMARY + j;
        history[k] = history[k]! + c * stat[t * D_STATIC + j]!;
      }
    }
    history[at + HIST_SEAT + mod4(row.player - seat)] = 1;
    history[at + HIST_TRICK] = row.trick / MAX_TRICKS;
    history[at + HIST_POSITION] = row.position / NUM_SEATS;
    history[at + HIST_IS_LEAD] = row.position === 0 ? 1 : 0;
    history[at + HIST_WON] = row.won;
    history[at + HIST_POINTS] = playPoints(row.counts) / PLAY_POINT_NORM;
  }

  // numpy computes the float32 results above in float32; the division
  // results here are float64 until stored, which Float32Array rounds the
  // same way numpy's float32 division does for these operands. Then the
  // float16 cast of `_pack`.
  roundHalf(history);
  roundHalf(context);
  roundHalf(cards);
  return { history, historyLen: kept, context, unseen, planes, cards };
}

/** Candidates as count vectors, row-major, for `ZeroObs`. */
export function packCandidates(moves: ReadonlyArray<ArrayLike<number>>): Float32Array {
  const out = new Float32Array(moves.length * NUM_TYPES);
  moves.forEach((m, i) => {
    for (let t = 0; t < NUM_TYPES; t++) out[i * NUM_TYPES + t] = m[t] ?? 0;
  });
  return out;
}
