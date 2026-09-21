// Does the belief prior actually move the sampler?
//
// A null A/B result is only informative if the treatment did something.
// This measures, on real decisions: how far the projected placement matrix
// sits from the same projection run on a belief that says nothing, and how
// often a world sampled with beliefs differs from one sampled from the same
// seed with that flat belief. Both comparisons are against the *projected*
// control, so what they report is the belief head's contribution rather
// than the feasibility structure every arm shares.
import { GameSlot } from './env';
import { prepareBeliefWeights } from '@tractor/engine/rl/v2/beliefSampler';
import { enumerateCandidates } from '@tractor/engine/rl/v2/candidates';
import { makeRng, sampleWorld, worldSignature, unseenCards } from '@tractor/engine/rl/v2/determinize';
import { writeDecision } from '@tractor/engine/rl/v2/encode';
import { historyFeatures, stateFeatures } from '@tractor/engine/rl/v2/features';
import { beliefs as beliefProbs, encodeState, loadWeights } from '@tractor/engine/rl/v2/net';
import { ByteWriter, H_TRUMP_RANK, H_TRUMP_SUIT, STATE_BYTES } from '@tractor/engine/rl/v2/record';
import type { TrumpContext } from '@tractor/engine';

const target = Number(process.argv[2] ?? 300);
// The control's belief value. Any constant projects to the same matrix --
// Sinkhorn normalises rows and columns, so a uniform input carries no
// information into the result -- and 0.5 is the value `buildPlacementWeights`
// already uses for a cell the head says nothing about.
const FLAT_BELIEF = 0.5;
const w = loadWeights();
const slot = new GameSlot({
  gameId: 0, seed: 31337,
  learnerSeats: [true, true, true, true],
  baselineBot: 'smart-v34', prePlayBot: 'smart-v34',
  maxCandidates: 48, randomize: true,
});

let n = 0, differed = 0, nullWeights = 0, guard = 0;
let tvSum = 0, maxRatioSum = 0, cardCount = 0;
while (n < target && guard++ < 100_000) {
  const { pending } = slot.advance();
  if (pending === null) continue;
  const state = slot.gameState;
  const hand = state.currentHand!;
  const seat = pending.seat;
  const myKitty = seat === hand.landlord ? slot.kitty : [];
  const ctx: TrumpContext = { trumpRank: hand.trumpRank!, trumpSuit: hand.trumpSuit };

  const writer = new ByteWriter(1 << 16);
  writeDecision(writer, state, { gameId: 0, handId: 0, seat, myKitty, privileged: false }, pending.candidates);
  const framed = writer.taken();
  const rec = framed.subarray(2, 2 + framed[0]! + framed[1]! * 256).subarray(0, STATE_BYTES);
  const hist = historyFeatures(rec);
  const z = encodeState(w, stateFeatures(rec), hist.tokens, hist.mask);
  const probs = beliefProbs(w, z);
  const weightOf = prepareBeliefWeights(state, seat, myKitty, ctx, probs, rec[H_TRUMP_RANK]!, rec[H_TRUMP_SUIT]!);
  // The control: the identical projection, fed a belief carrying no
  // information. Same shape as `probs` so the head's own dimensions are
  // the ones being flattened.
  const flat = new Float32Array(probs.length).fill(FLAT_BELIEF);
  const flatOf = prepareBeliefWeights(state, seat, myKitty, ctx, flat, rec[H_TRUMP_RANK]!, rec[H_TRUMP_SUIT]!);

  // `prepareBeliefWeights` bails for reasons that have nothing to do with
  // the belief (empty pool, no opponent with cards left), so the two arms
  // go null together -- but count either as unavailable rather than let a
  // half-populated decision into the denominators.
  if (weightOf === null || flatOf === null) { nullWeights += 1; slot.choose(0); continue; }

  // How far does the belief move the projected matrix, per card?
  const pool = unseenCards(state, seat, myKitty);
  const numPlayers = state.config.numPlayers;
  for (const card of pool) {
    const ws: number[] = [];
    const fs: number[] = [];
    for (let s = 0; s < numPlayers; s++) {
      if (s === seat) continue;
      ws.push(weightOf(card, s));
      fs.push(flatOf(card, s));
    }
    const total = ws.reduce((a, b) => a + b, 0);
    const flatTotal = fs.reduce((a, b) => a + b, 0);
    if (total <= 0 || flatTotal <= 0) continue;
    const p = ws.map((x) => x / total);
    const q = fs.map((x) => x / flatTotal);
    tvSum += 0.5 * p.reduce((a, x, i) => a + Math.abs(x - q[i]!), 0);
    // A zero in `q` is a void, and the void mask is the same in both arms,
    // so the matching `p` is zero too and the cell carries no ratio.
    let maxRatio = 0;
    for (let i = 0; i < p.length; i++) {
      if (q[i]! <= 0) continue;
      maxRatio = Math.max(maxRatio, p[i]! / q[i]!);
    }
    maxRatioSum += maxRatio;
    cardCount += 1;
  }

  const seed = 12345 + n;
  // Both sides must be the *weighted* sampler, differing only in the
  // belief behind the weights. Comparing against the omitted-weightOf path
  // instead compares two different algorithms -- argmax(need) versus a
  // proportional draw -- that also consume different numbers of RNG draws,
  // so the worlds diverge on essentially every seed and the line reads
  // ~100% even for a prior that does nothing. That is exactly the false
  // negative this file exists to rule out.
  //
  // A flat `() => 1` is not the right control either, and that reasoning is
  // why: it skips `prepareBeliefWeights`, and the treatment's weights are
  // the *projected* matrix, which encodes the void mask and the per-seat
  // hand sizes whatever the head says. The two arms then differ for
  // reasons that are not beliefs at all -- a head whose every output is
  // the same constant still read 63% differing against it, against 92.5%
  // for the real head, so the statistic had a false-positive floor two
  // thirds of the way up its own range. The control is a constant belief
  // pushed through the same projection: identical feasible structure, no
  // information.
  const a = worldSignature(sampleWorld(state, seat, myKitty, ctx, makeRng(seed), weightOf));
  const b = worldSignature(sampleWorld(state, seat, myKitty, ctx, makeRng(seed), flatOf));
  if (a !== b) differed += 1;
  n += 1;
  slot.choose(n % pending.candidates.length);
}

// The world comparison happens once per decision; the two distribution
// statistics accumulate once per card in the unseen pool, which runs from
// 83 cards at trick 1 down to about 11 at the last. They need different
// denominators, and only cardCount makes the per-card means readable: the
// max ratio is a per-card quantity with no fixed bound, so dividing it by
// anything but the number of cards it was summed over means nothing at all.
process.stdout.write(
  `decisions ${n}  (weights unavailable on ${nullWeights})\n` +
  `worlds differing from the flat-belief control: ${((differed / Math.max(1, n)) * 100).toFixed(1)}%\n` +
  `mean total-variation from the flat-belief control, per card: ${(tvSum / Math.max(1, cardCount)).toFixed(4)}\n` +
  `mean max weight ratio to the flat-belief control: ${(maxRatioSum / Math.max(1, cardCount)).toFixed(3)}\n`,
);
