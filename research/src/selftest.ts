// End-to-end check of the environment layer: play hands with a uniformly
// random policy over the enumerated candidates, encode every decision,
// and verify the wire records are self-consistent.
//
//   npm run selftest            (from research/)
//   HANDS=200 PREPLAY=lowball npm run selftest

import { mulberry32, sumPoints, type Card } from '@tractor/engine';

import { NUM_SLOTS } from './cards';
import { GameSlot } from './env';
import { writeDecision, writeEpisode, type EpisodeSummary } from './encode';
import {
  ByteWriter,
  CANDIDATE_BYTES,
  E_ATTACKER_POINTS,
  E_DEAL_DIGEST,
  E_DEFENDERS_WON,
  E_GAME_ID,
  E_HAND_ID,
  E_KITTY_POINTS,
  E_LANDLORD_SEAT,
  E_LEVEL_DELTA,
  E_NUM_TRICKS,
  E_TRICKS_BASE,
  E_TRUMP_RANK,
  E_TRUMP_SUIT,
  EPISODE_BYTES,
  HISTORY_BYTES,
  HISTORY_STEPS,
  H_HISTORY_LEN,
  H_HAND_SIZE,
  H_IS_LEAD,
  H_NUM_CANDIDATES,
  H_SEAT,
  H_TRICK_INDEX,
  H_VERSION,
  HEADER_BYTES,
  MAX_TRICKS,
  PLANES_BYTES,
  RECORD_VERSION,
  SHAPE_THROW,
  STATE_BYTES,
} from './record';

const HANDS = Number(process.env['HANDS'] ?? 100);
const PREPLAY = (process.env['PREPLAY'] ?? 'smart-v34') as never;
const SEED = Number(process.env['SEED'] ?? 12345);

function main(): void {
  // The episode layout has no slack at all: the per-trick array ends on the
  // byte E_KITTY_POINTS occupies. Nothing else guards that, and it would not
  // show up at runtime -- writeEpisode writes into a claim()'d subarray,
  // where an out-of-range index is a silent no-op and an in-range one
  // quietly overwrites a neighbouring field, in both cases leaving the byte
  // count at EPISODE_BYTES.
  if (E_TRICKS_BASE + MAX_TRICKS * 2 > E_KITTY_POINTS) {
    throw new Error(
      `episode tricks (${E_TRICKS_BASE} + ${MAX_TRICKS}*2) overrun E_KITTY_POINTS ${E_KITTY_POINTS}`,
    );
  }

  const rng = mulberry32(SEED);
  const slot = new GameSlot({
    gameId: 0,
    seed: SEED,
    learnerSeats: [true, true, true, true],
    prePlayBot: PREPLAY,
  });

  const writer = new ByteWriter();
  let decisions = 0;
  let leadCandidates = 0;
  let leads = 0;
  let followCandidates = 0;
  let follows = 0;
  let throwsOffered = 0;
  let recordBytes = 0;
  let handsDone = 0;
  let attackerPointsTotal = 0;
  let noTrumpHands = 0;
  let trickTotal = 0;
  let kittyPointsTotal = 0;
  const started = Date.now();

  while (handsDone < HANDS) {
    const { episodes, pending } = slot.advance();
    for (const e of episodes) {
      handsDone += 1;
      attackerPointsTotal += e.attackerPoints;
      if (e.trumpSuitIdx === 4) noTrumpHands += 1;
      // Tricks per hand vary: a pair/tractor/throw lead consumes several
      // cards per player, so a hand runs 25 tricks only if every lead is a
      // single.
      if (e.numTricks < 1 || e.numTricks > 25) {
        throw new Error(`hand ${e.handId} ended after ${e.numTricks} tricks`);
      }
      trickTotal += e.numTricks;
      const trickSum = e.trickPoints.reduce((a, b) => a + b, 0);
      if (trickSum + e.kittyPoints !== 200) {
        throw new Error(
          `hand ${e.handId}: ${trickSum} trick points + ${e.kittyPoints} kitty points != 200`,
        );
      }
      kittyPointsTotal += e.kittyPoints;
      writer.reset();
      writeEpisode(writer, e);
      checkEpisode(writer.taken(), e);
    }
    if (pending === null) continue;

    // Encode the decision exactly as the worker would, then re-read it.
    writer.reset();
    writeDecision(
      writer,
      slot.gameState,
      { gameId: slot.gameId, handId: slot.handId, seat: pending.seat, myKitty: slot.kitty },
      pending.candidates,
    );
    const buf = writer.taken();
    checkRecord(buf, pending.seat, pending.candidates.length, slot);
    recordBytes += buf.length;

    const isLead = buf[2 + H_IS_LEAD] === 1;
    if (isLead) {
      leads += 1;
      leadCandidates += pending.candidates.length;
    } else {
      follows += 1;
      followCandidates += pending.candidates.length;
    }
    for (const c of pending.candidates) {
      if (c.shape === SHAPE_THROW) throwsOffered += 1;
      if (c.points !== Math.min(sumPoints(c.cards as ReadonlyArray<Card>), 255)) {
        throw new Error('candidate point count mismatch');
      }
    }
    decisions += 1;

    slot.choose(Math.floor(rng() * pending.candidates.length));
  }

  const elapsed = (Date.now() - started) / 1000;
  console.log(`hands            ${handsDone} in ${elapsed.toFixed(2)}s = ${(handsDone / elapsed).toFixed(1)} hands/s`);
  console.log(`decisions        ${decisions} (${(decisions / handsDone).toFixed(1)}/hand)`);
  console.log(`lead candidates  mean ${(leadCandidates / Math.max(leads, 1)).toFixed(1)} over ${leads} leads`);
  console.log(`follow candidates mean ${(followCandidates / Math.max(follows, 1)).toFixed(1)} over ${follows} follows`);
  console.log(`throw candidates offered ${throwsOffered} (${(throwsOffered / handsDone).toFixed(2)}/hand)`);
  console.log(`record size      mean ${(recordBytes / decisions).toFixed(0)} bytes (state ${STATE_BYTES})`);
  console.log(`attacker points  mean ${(attackerPointsTotal / handsDone).toFixed(1)}`);
  console.log(`kitty points     mean ${(kittyPointsTotal / handsDone).toFixed(1)}`);
  console.log(`tricks per hand  mean ${(trickTotal / handsDone).toFixed(1)}`);
  console.log(`no-trump hands   ${noTrumpHands} (${((100 * noTrumpHands) / handsDone).toFixed(1)}%)`);
  console.log('selftest OK');
}

// Reads every field back out of the encoded 72 bytes. A byte-count check
// cannot catch anything here -- writeEpisode makes exactly one claim of
// EPISODE_BYTES, so the length is 72 whatever it then writes -- whereas a
// field that lands on the wrong offset, or on top of another field, shows
// up immediately as a value that does not match the summary it came from.
function checkEpisode(buf: Uint8Array, e: EpisodeSummary): void {
  if (buf.length !== EPISODE_BYTES) {
    throw new Error(`episode record is ${buf.length} bytes, expected ${EPISODE_BYTES}`);
  }
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const same = (field: string, got: number, want: number): void => {
    if (got !== want) {
      throw new Error(`episode ${field} read back as ${got}, wrote ${want}`);
    }
  };
  same('handId', view.getUint32(E_HAND_ID, true), e.handId);
  same('gameId', view.getUint16(E_GAME_ID, true), e.gameId);
  same('dealDigest', view.getUint32(E_DEAL_DIGEST, true), e.dealDigest >>> 0);
  same('landlordSeat', buf[E_LANDLORD_SEAT]!, e.landlordSeat);
  same('numTricks', buf[E_NUM_TRICKS]!, e.numTricks);
  same('attackerPoints', view.getUint16(E_ATTACKER_POINTS, true), e.attackerPoints);
  same('levelDelta', view.getInt8(E_LEVEL_DELTA), e.levelDelta);
  same('defendersWon', buf[E_DEFENDERS_WON]!, e.defendersWon ? 1 : 0);
  same('trumpRank', buf[E_TRUMP_RANK]!, e.trumpRankIdx);
  same('trumpSuit', buf[E_TRUMP_SUIT]!, e.trumpSuitIdx);
  same('kittyPoints', buf[E_KITTY_POINTS]!, e.kittyPoints);
  for (let i = 0; i < e.numTricks; i++) {
    same(`trick ${i} points`, buf[E_TRICKS_BASE + i * 2]!, e.trickPoints[i] ?? 0);
    same(
      `trick ${i} attackersWonIt`,
      buf[E_TRICKS_BASE + i * 2 + 1]!,
      e.trickAttackerWon[i] === true ? 1 : 0,
    );
  }
}

function checkRecord(
  buf: Uint8Array,
  seat: number,
  numCandidates: number,
  slot: GameSlot,
): void {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const declared = view.getUint16(0, true);
  if (declared !== buf.length - 2) {
    throw new Error(`record length prefix ${declared} != actual ${buf.length - 2}`);
  }
  const header = buf.subarray(2, 2 + HEADER_BYTES);
  if (header[H_VERSION] !== RECORD_VERSION) throw new Error('bad record version');
  if (header[H_SEAT] !== seat) throw new Error('seat mismatch in record');
  if (header[H_NUM_CANDIDATES] !== numCandidates) {
    throw new Error('candidate count mismatch in record');
  }
  const expectedBytes =
    HEADER_BYTES + PLANES_BYTES + HISTORY_BYTES + numCandidates * CANDIDATE_BYTES;
  if (declared !== expectedBytes) {
    throw new Error(`record is ${declared} bytes, layout says ${expectedBytes}`);
  }
  const handState = slot.gameState.currentHand!;
  if (header[H_TRICK_INDEX] !== handState.tricks.length) {
    throw new Error('trick index mismatch');
  }
  const handSize = (handState.hands[seat] ?? []).length;
  if (header[H_HAND_SIZE] !== handSize) throw new Error('hand size mismatch');

  // Every reader treats this byte as the sole attention mask over the
  // history block and never recomputes it, so an *under*-reported length is
  // the damaging direction: it silently hides real moves from the encoder.
  // Check the exact value rather than only the window bound, which
  // steps.slice(-HISTORY_STEPS) satisfies by construction.
  let totalPlays = 0;
  for (const t of handState.tricks) totalPlays += t.plays.length;
  if (handState.currentTrick !== null) totalPlays += handState.currentTrick.plays.length;
  const expectedHistory = Math.min(totalPlays, HISTORY_STEPS);
  if (header[H_HISTORY_LEN] !== expectedHistory) {
    throw new Error(
      `history length ${header[H_HISTORY_LEN]}, expected ${expectedHistory} from ${totalPlays} plays`,
    );
  }
  // Plane 0 must be exactly the acting seat's hand.
  const planes = buf.subarray(2 + HEADER_BYTES, 2 + HEADER_BYTES + PLANES_BYTES);
  let planeTotal = 0;
  for (let i = 0; i < NUM_SLOTS; i++) planeTotal += planes[i]!;
  if (planeTotal !== handSize) {
    throw new Error(`hand plane holds ${planeTotal} cards, hand has ${handSize}`);
  }
  // Every count must be a legal multiplicity for a 2-deck game.
  for (let i = 0; i < PLANES_BYTES; i++) {
    if (planes[i]! > 2) throw new Error(`plane slot ${i} has count ${planes[i]}`);
  }
}

main();
