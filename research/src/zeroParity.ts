// Dumps real play decisions with the TypeScript `rl-zero` featuriser's
// output (and, with --logits, the network's), so
// py/test_zero_parity.py can diff them against the Python encoder the
// weights were trained under.
//
//   npx tsx src/zeroParity.ts --decisions 400 > zero-parity.jsonl
//   npx tsx src/zeroParity.ts --decisions 400 --logits --weights DIR > zero-parity.jsonl
//
// Each line carries the position in exactly the form evalhost.ts sends a
// `pending` message (the Python side rebuilds it with `state_from_wire`),
// the candidates, and the TS observation. Moves are chosen at random from
// the candidates so the positions spread over whole hands.

import { NATURAL_RANKS } from '@tractor/engine';
import { encodePosition, packCandidates } from '@tractor/engine/rl/zero/features';
import { plane, positionFor } from '@tractor/engine/rl/zero/position';
import { trumpSuitIndex } from '@tractor/engine/rl/v2/encode';

import { GameSlot } from './env';

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? (process.argv[i + 1] ?? fallback) : fallback;
}

const target = Number(arg('decisions', '400'));
const seed = Number(arg('seed', '20261117'));
const withLogits = process.argv.includes('--logits');
const weightsDir = arg('weights', '');

let rng = seed >>> 0;
function nextRandom(): number {
  // mulberry32
  rng = (rng + 0x6d2b79f5) >>> 0;
  let t = rng;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

async function main(): Promise<void> {
  let logitsOf: ((obs: any) => Float32Array) | null = null;
  if (withLogits) {
    const { loadZeroWeights } = await import('@tractor/engine/rl/zero/weights');
    const { actorLogits } = await import('@tractor/engine/rl/zero/net');
    const w = loadZeroWeights(weightsDir || undefined);
    logitsOf = (obs) => actorLogits(w, obs);
  }

  const slot = new GameSlot({
    gameId: 0,
    seed,
    learnerSeats: [true, true, true, true],
    baselineBot: 'smart-v34',
    prePlayBot: 'smart-v34',
    maxCandidates: 128,
    randomize: true,
  });

  let emitted = 0;
  let guard = 0;
  while (emitted < target && guard++ < 500_000) {
    const { pending } = slot.advance();
    if (pending === null) continue;
    const state = slot.gameState;
    const hand = state.currentHand!;
    const seat = pending.seat;
    const myKitty = seat === hand.landlord ? hand.kitty : null;
    const pos = positionFor(state, seat, myKitty);
    const moves = pending.candidates.map((c) => plane(c.cards));
    const obs = {
      ...encodePosition(pos),
      candidates: packCandidates(moves),
      numCandidates: moves.length,
    };
    const open = hand.currentTrick;
    const msg = {
      seat,
      level_rank: NATURAL_RANKS.indexOf(hand.trumpRank!),
      trump_suit: trumpSuitIndex(hand),
      num_decks: state.config.numDecks,
      landlord: hand.landlord,
      to_act: seat,
      trick_leader: open === null || open.plays.length === 0 ? seat : open.leader,
      hands: [0, 1, 2, 3].map((s) => Array.from(plane(hand.hands[s as 0 | 1 | 2 | 3] ?? []))),
      kitty: Array.from(plane(hand.kitty)),
      tricks: hand.tricks.map((t) => ({
        leader: t.leader,
        winner: t.winner,
        points: t.points,
        plays: t.plays.map((p) => Array.from(plane(p.cards))),
      })),
      trick_plays: (open?.plays ?? []).map((p) => Array.from(plane(p.cards))),
      candidates: moves.map((m) => Array.from(m)),
    };
    const ts: Record<string, unknown> = {
      history: Array.from(obs.history),
      history_len: obs.historyLen,
      context: Array.from(obs.context),
      unseen: Array.from(obs.unseen),
      planes: Array.from(obs.planes),
      cards: Array.from(obs.cards),
      current_winner: pos.currentWinner,
    };
    if (logitsOf !== null) {
      const t0 = performance.now();
      ts.logits = Array.from(logitsOf(obs));
      ts.ms = performance.now() - t0;
    }
    process.stdout.write(JSON.stringify({ msg, ts }) + '\n');
    emitted += 1;
    slot.choose(Math.floor(nextRandom() * pending.candidates.length));
  }
  if (emitted < target) throw new Error(`only ${emitted} decisions in the guard`);
}

main().catch((err) => {
  process.stderr.write(String(err?.stack ?? err) + '\n');
  process.exit(1);
});
