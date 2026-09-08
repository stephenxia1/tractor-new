// Does the belief prior actually move the sampler?
//
// A null A/B result is only informative if the treatment did something.
// This measures, on real decisions: how far the projected placement matrix
// sits from uniform, and how often a world sampled with beliefs differs
// from one sampled from the same seed without them.
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

  if (weightOf === null) { nullWeights += 1; slot.choose(0); continue; }

  // How non-uniform is the projected matrix, per card?
  const pool = unseenCards(state, seat, myKitty);
  const numPlayers = state.config.numPlayers;
  for (const card of pool) {
    const ws: number[] = [];
    for (let s = 0; s < numPlayers; s++) if (s !== seat) ws.push(weightOf(card, s));
    const total = ws.reduce((a, b) => a + b, 0);
    if (total <= 0) continue;
    const p = ws.map((x) => x / total);
    const u = 1 / p.length;
    tvSum += 0.5 * p.reduce((a, x) => a + Math.abs(x - u), 0);
    maxRatioSum += Math.max(...p) / u;
    cardCount += 1;
  }

  const seed = 12345 + n;
  const a = worldSignature(sampleWorld(state, seat, myKitty, ctx, makeRng(seed), weightOf));
  const b = worldSignature(sampleWorld(state, seat, myKitty, ctx, makeRng(seed)));
  if (a !== b) differed += 1;
  n += 1;
  slot.choose(n % pending.candidates.length);
}

const cards = Math.max(1, n);
process.stdout.write(
  `decisions ${n}  (weights unavailable on ${nullWeights})\n` +
  `worlds differing from uniform: ${((differed / cards) * 100).toFixed(1)}%\n` +
  `mean total-variation from uniform, per card: ${(tvSum / Math.max(1, tvSum && cards * 20)).toFixed(4)}\n` +
  `mean max/uniform weight ratio: ${(maxRatioSum / Math.max(1, cards * 20)).toFixed(3)}\n`,
);
