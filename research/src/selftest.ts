// End-to-end check of the environment layer: play hands with a uniformly
// random policy over the enumerated candidates, encode every decision,
// and verify the wire records are self-consistent.
//
//   npm run selftest            (from research/)
//   HANDS=200 PREPLAY=lowball npm run selftest

import { mulberry32, sumPoints, type Card } from '@tractor/engine';

import { NUM_SLOTS } from './cards';
import { GameSlot } from './env';
import { writeDecision, writeEpisode } from './encode';
import {
  ByteWriter,
  CANDIDATE_BYTES,
  EPISODE_BYTES,
  HISTORY_BYTES,
  H_HISTORY_LEN,
  H_HAND_SIZE,
  H_IS_LEAD,
  H_NUM_CANDIDATES,
  H_SEAT,
  H_TRICK_INDEX,
  H_VERSION,
  HEADER_BYTES,
  PLANES_BYTES,
  RECORD_VERSION,
  SHAPE_THROW,
  STATE_BYTES,
} from './record';

const HANDS = Number(process.env['HANDS'] ?? 100);
const PREPLAY = (process.env['PREPLAY'] ?? 'smart-v34') as never;
const SEED = Number(process.env['SEED'] ?? 12345);

function main(): void {
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
      if (writer.length !== EPISODE_BYTES) {
        throw new Error(`episode record is ${writer.length} bytes, expected ${EPISODE_BYTES}`);
      }
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

  // History length must never exceed the window.
  if (header[H_HISTORY_LEN]! > 16) {
    throw new Error(`history length ${header[H_HISTORY_LEN]} exceeds the window`);
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
