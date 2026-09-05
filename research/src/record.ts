// Wire format between the TS environment (actor) and the Python learner.
//
// Design rule: this layer ships *raw* game facts, never features. Card
// multisets go out as 54-slot count planes and scalars go out as small
// integers; every modelling decision (trump-anchored permutation, plane
// stacking, normalisation) happens in numpy on the Python side, so an
// encoding experiment never requires a TypeScript edit.
//
// All integers are little-endian. spec.json (written by src/spec.ts)
// mirrors every constant here and is asserted at import time by
// py/tractor_rl/layout.py, so the two sides cannot silently drift.

import { NUM_SLOTS } from './cards';

export const RECORD_VERSION = 3;

// ---- Decision record header ------------------------------------------------

export const H_VERSION = 0; // u8
export const H_SEAT = 1; // u8  absolute seat 0..3
export const H_ROLE = 2; // u8  0 landlord, 1 defender, 2 attacker
export const H_LANDLORD_REL = 3; // u8  landlord's seat relative to me (0 = me)
export const H_TRUMP_RANK = 4; // u8  index into NATURAL_RANKS (0..12)
export const H_TRUMP_SUIT = 5; // u8  suit index 0..3, or 4 for no-trump
export const H_TRICK_INDEX = 6; // u8  completed tricks so far (0..24)
export const H_TRICK_POS = 7; // u8  plays already in the current trick (0..3)
export const H_IS_LEAD = 8; // u8
export const H_HAND_SIZE = 9; // u8  cards left in my hand
export const H_WINNER_REL = 10; // u8  relative seat currently winning, 255 if none
export const H_TABLE_POINTS = 11; // u8  point cards on the table this trick
export const H_ATTACKER_POINTS = 12; // u16 attacker points banked so far
export const H_CARDS_SEEN = 14; // u16 cards played this hand (all seats)
export const H_NUM_CANDIDATES = 16; // u8
export const H_I_AM_ATTACKER = 17; // u8
export const H_VOID_MASK = 18; // u8[4] per relative seat: bit0..3 suits, bit4 trump
export const H_GAME_ID = 22; // u16
export const H_HAND_ID = 24; // u32
export const HEADER_BYTES = 28;

// ---- Card planes -----------------------------------------------------------

export const PLANE_MY_HAND = 0;
export const PLANE_PLAYED_BASE = 1; // +0..3: cards played this hand, relative seat
export const PLANE_TRICK_BASE = 5; // +0..3: cards in the current trick, relative seat
export const PLANE_MY_KITTY = 9; // cards I buried (landlord only; zeros otherwise)
export const NUM_PLANES = 10;

export const PLANES_BYTES = NUM_PLANES * NUM_SLOTS; // 540
export const STATE_BYTES = HEADER_BYTES + PLANES_BYTES; // 568

// ---- Candidate encoding ----------------------------------------------------

// Shape tag. SHAPE_BROKEN covers follows that cannot form a single combo
// (structure-breaking discards, mixed-suit fills).
export const SHAPE_SINGLE = 0;
export const SHAPE_PAIR = 1;
export const SHAPE_TRACTOR2 = 2;
export const SHAPE_TRACTOR3PLUS = 3;
export const SHAPE_THROW = 4;
export const SHAPE_BROKEN = 5;
export const NUM_SHAPES = 6;

export const FLAG_ALL_TRUMP = 1;
export const FLAG_ALL_LED_SUIT = 2; // follows: entirely in the led suit
export const FLAG_HAS_POINTS = 4;
export const FLAG_BEATS_WINNER = 8; // would take the trick as it stands

// Per candidate: u8 nCards, u8 shape, u8 flags, u8 points, then a
// fixed-width card-slot array padded with NO_CARD. Fixed width so the
// Python side can reshape a whole batch of candidates in one numpy call
// instead of walking them in a loop.
export const CANDIDATE_HEADER_BYTES = 4;
export const MAX_CANDIDATE_CARDS = 16;
export const CANDIDATE_BYTES = CANDIDATE_HEADER_BYTES + MAX_CANDIDATE_CARDS; // 20
export const NO_CARD = 255;

// ---- Episode record --------------------------------------------------------

export const E_HAND_ID = 0; // u32
export const E_LANDLORD_SEAT = 4; // u8
export const E_NUM_TRICKS = 5; // u8
export const E_ATTACKER_POINTS = 6; // u16
export const E_LEVEL_DELTA = 8; // i8
export const E_DEFENDERS_WON = 9; // u8
export const E_TRUMP_RANK = 10; // u8
export const E_TRUMP_SUIT = 11; // u8
export const E_TRICKS_BASE = 12; // per trick: u8 points, u8 attackersWonIt
export const MAX_TRICKS = 25;
// Points buried in the kitty. Trick points sum to 200 minus this, and the
// kitty is what the last-trick multiplier (rules.md §8.2) acts on.
export const E_KITTY_POINTS = 62; // u8
// Game slot this hand belongs to. Hand ids restart per slot, so the
// learner keys decisions by (worker, game, hand).
export const E_GAME_ID = 64; // u16
// Fingerprint of the deal this hand was played from: the four dealt hands
// (seat by seat), the kitty, the landlord seat and the trump rank, hashed
// before anybody declares or buries. Duplicate-deal evaluation joins the
// two halves of a pair on (game, hand) and then *checks* this matches, so
// a broken deal-seed scheme shows up as a failed assertion rather than as
// a plausible-looking number.
export const E_DEAL_DIGEST = 66; // u32
export const EPISODE_BYTES = 72;

// ---- Frame format ----------------------------------------------------------
//
// actor to learner:  u32 payloadBytes | u16 nDecisions | u16 nEpisodes
//                    | nDecisions x (u16 recordBytes, record)
//                    | nEpisodes x EPISODE_BYTES
// learner to actor:  u32 payloadBytes | u16 nActions
//                    | nActions x (u16 gameId, u8 candidateIndex)

export const FRAME_HEADER_BYTES = 4;
export const BATCH_HEADER_BYTES = 4; // nDecisions u16 + nEpisodes u16
export const ACTION_BYTES = 3; // gameId u16 + candidateIndex u8

// A growable little-endian byte writer.
export class ByteWriter {
  private buf: Uint8Array;
  private view: DataView;
  private len = 0;

  constructor(initialBytes = 1 << 16) {
    this.buf = new Uint8Array(initialBytes);
    this.view = new DataView(this.buf.buffer);
  }

  get length(): number {
    return this.len;
  }

  reset(): void {
    this.len = 0;
  }

  private ensure(extra: number): void {
    if (this.len + extra <= this.buf.length) return;
    let next = this.buf.length * 2;
    while (next < this.len + extra) next *= 2;
    const grown = new Uint8Array(next);
    grown.set(this.buf.subarray(0, this.len));
    this.buf = grown;
    this.view = new DataView(this.buf.buffer);
  }

  u8(v: number): void {
    this.ensure(1);
    this.buf[this.len++] = v & 0xff;
  }

  i8(v: number): void {
    this.ensure(1);
    this.view.setInt8(this.len, v);
    this.len += 1;
  }

  u16(v: number): void {
    this.ensure(2);
    this.view.setUint16(this.len, v, true);
    this.len += 2;
  }

  u32(v: number): void {
    this.ensure(4);
    this.view.setUint32(this.len, v, true);
    this.len += 4;
  }

  // Reserve n zero bytes and return the offset they start at, so a length
  // prefix can be back-patched once the payload is written.
  reserve(n: number): number {
    this.ensure(n);
    const at = this.len;
    this.buf.fill(0, this.len, this.len + n);
    this.len += n;
    return at;
  }

  patchU32(at: number, v: number): void {
    this.view.setUint32(at, v, true);
  }

  patchU16(at: number, v: number): void {
    this.view.setUint16(at, v, true);
  }

  bytes(src: Uint8Array): void {
    this.ensure(src.length);
    this.buf.set(src, this.len);
    this.len += src.length;
  }

  // Zero-filled region the caller writes into directly (used for planes).
  claim(n: number): Uint8Array {
    this.ensure(n);
    const region = this.buf.subarray(this.len, this.len + n);
    region.fill(0);
    this.len += n;
    return region;
  }

  // View of everything written so far. Valid until the next write.
  taken(): Uint8Array {
    return this.buf.subarray(0, this.len);
  }
}

// Machine-readable mirror of the layout, written to spec.json and checked
// by the Python side at import.
export function layoutSpec(): Record<string, unknown> {
  return {
    version: RECORD_VERSION,
    numSlots: NUM_SLOTS,
    header: {
      bytes: HEADER_BYTES,
      version: H_VERSION,
      seat: H_SEAT,
      role: H_ROLE,
      landlordRel: H_LANDLORD_REL,
      trumpRank: H_TRUMP_RANK,
      trumpSuit: H_TRUMP_SUIT,
      trickIndex: H_TRICK_INDEX,
      trickPos: H_TRICK_POS,
      isLead: H_IS_LEAD,
      handSize: H_HAND_SIZE,
      winnerRel: H_WINNER_REL,
      tablePoints: H_TABLE_POINTS,
      attackerPoints: H_ATTACKER_POINTS,
      cardsSeen: H_CARDS_SEEN,
      numCandidates: H_NUM_CANDIDATES,
      iAmAttacker: H_I_AM_ATTACKER,
      voidMask: H_VOID_MASK,
      gameId: H_GAME_ID,
      handId: H_HAND_ID,
    },
    planes: {
      count: NUM_PLANES,
      bytes: PLANES_BYTES,
      myHand: PLANE_MY_HAND,
      playedBase: PLANE_PLAYED_BASE,
      trickBase: PLANE_TRICK_BASE,
      myKitty: PLANE_MY_KITTY,
    },
    stateBytes: STATE_BYTES,
    candidate: {
      headerBytes: CANDIDATE_HEADER_BYTES,
      bytes: CANDIDATE_BYTES,
      maxCards: MAX_CANDIDATE_CARDS,
      noCard: NO_CARD,
      shapes: {
        single: SHAPE_SINGLE,
        pair: SHAPE_PAIR,
        tractor2: SHAPE_TRACTOR2,
        tractor3plus: SHAPE_TRACTOR3PLUS,
        throwShape: SHAPE_THROW,
        broken: SHAPE_BROKEN,
        count: NUM_SHAPES,
      },
      flags: {
        allTrump: FLAG_ALL_TRUMP,
        allLedSuit: FLAG_ALL_LED_SUIT,
        hasPoints: FLAG_HAS_POINTS,
        beatsWinner: FLAG_BEATS_WINNER,
      },
    },
    episode: {
      bytes: EPISODE_BYTES,
      handId: E_HAND_ID,
      landlordSeat: E_LANDLORD_SEAT,
      numTricks: E_NUM_TRICKS,
      attackerPoints: E_ATTACKER_POINTS,
      levelDelta: E_LEVEL_DELTA,
      defendersWon: E_DEFENDERS_WON,
      trumpRank: E_TRUMP_RANK,
      trumpSuit: E_TRUMP_SUIT,
      tricksBase: E_TRICKS_BASE,
      maxTricks: MAX_TRICKS,
      kittyPoints: E_KITTY_POINTS,
      gameId: E_GAME_ID,
      dealDigest: E_DEAL_DIGEST,
    },
    frame: {
      headerBytes: FRAME_HEADER_BYTES,
      batchHeaderBytes: BATCH_HEADER_BYTES,
      actionBytes: ACTION_BYTES,
    },
  };
}
