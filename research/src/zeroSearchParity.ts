// Dumps the `rl-zero-search` bot's workings on real engine positions so
// py/test_zero_search_parity.py can check them against the Python search
// (research/py/tractor/eval/search.py) and the PyTorch critic:
//
//   npx tsx src/zeroSearchParity.ts --decisions 300 --searches 40 --weights DIR > search.jsonl
//
// Every decision carries the critic's value of the real position. For
// the first --searches decisions that search, the line also carries the
// sampled worlds, each (world, candidate) leaf as a position the Python
// side can rebuild, every simulated move with the engine's candidate list,
// and the per-candidate means and choice. Moves are chosen at random so
// positions spread over whole hands.

import { NATURAL_RANKS, type GameState, type Seat } from '@tractor/engine';
import { trumpSuitIndex } from '@tractor/engine/rl/v2/encode';
import { criticValue, privilegedView } from '@tractor/engine/rl/zero/critic';
import { encodePosition, packCandidates } from '@tractor/engine/rl/zero/features';
import { actorLogits } from '@tractor/engine/rl/zero/net';
import { plane, positionFor } from '@tractor/engine/rl/zero/position';
import { DEFAULT_SEARCH, mulberry32, positionSeed, searchMove, settleOpenTrick } from '@tractor/engine/rl/zero/search';
import { loadZeroWeights } from '@tractor/engine/rl/zero/weights';

import { GameSlot } from './env';

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? (process.argv[i + 1] ?? fallback) : fallback;
}

const target = Number(arg('decisions', '300'));
const maxSearches = Number(arg('searches', '40'));
const seed = Number(arg('seed', '20261118'));
const weightsDir = arg('weights', '');
const confident = Number(arg('confident', String(DEFAULT_SEARCH.confident)));
// Also dump each engine GameState, for the vitest fixture.
const withStates = process.argv.includes('--states');

// A position as evalhost.ts sends it; `closeOpen` turns a full open trick
// into a closed one (a simulation that ended the hand).
function wire(state: GameState, seat: number, closeOpen = false): Record<string, unknown> {
  const hand = state.currentHand!;
  const open = hand.currentTrick;
  const tricks = hand.tricks.map((t) => ({
    leader: t.leader,
    winner: t.winner,
    points: t.points,
    plays: t.plays.map((p) => Array.from(plane(p.cards))),
  }));
  let trickPlays = (open?.plays ?? []).map((p) => Array.from(plane(p.cards)));
  let leader = open === null || open.plays.length === 0 ? seat : open.leader;
  if (closeOpen && open !== null && open.plays.length === 4) {
    const res = settleOpenTrick(state);
    tricks.push({ leader: open.leader, winner: res.winner, points: res.points, plays: trickPlays });
    trickPlays = [];
    leader = res.winner;
  }
  return {
    seat,
    level_rank: NATURAL_RANKS.indexOf(hand.trumpRank!),
    trump_suit: trumpSuitIndex(hand),
    num_decks: state.config.numDecks,
    landlord: hand.landlord,
    to_act: closeOpen ? leader : seat,
    trick_leader: leader,
    hands: [0, 1, 2, 3].map((s) => Array.from(plane(hand.hands[s as Seat] ?? []))),
    kitty: Array.from(plane(hand.kitty)),
    tricks,
    trick_plays: trickPlays,
  };
}

let rng = seed >>> 0;
function nextRandom(): number {
  rng = (rng + 0x6d2b79f5) >>> 0;
  let t = rng;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

const w = loadZeroWeights(weightsDir || undefined);
const config = { ...DEFAULT_SEARCH, confident };
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
let searches = 0;
let multi = 0;
const searchMs: number[] = [];
let guard = 0;
while (emitted < target && guard++ < 500_000) {
  const { pending } = slot.advance();
  if (pending === null) continue;
  const state = slot.gameState;
  const hand = state.currentHand!;
  const seat = pending.seat as Seat;
  const myKitty = seat === hand.landlord ? hand.kitty : null;
  const pos = positionFor(state, seat, myKitty);
  const feats = encodePosition(pos);
  const moves = pending.candidates.map((c) => plane(c.cards));
  const logits = actorLogits(w, { ...feats, candidates: packCandidates(moves), numCandidates: moves.length });
  const realHands = [0, 1, 2, 3].map((s) => plane(hand.hands[s as Seat] ?? []));
  const critic = criticValue(w, feats.history, feats.historyLen, feats.context, privilegedView(realHands, plane(hand.kitty), seat));
  const rec: Record<string, unknown> = {
    msg: { ...wire(state, seat), candidates: moves.map((m) => Array.from(m)) },
    logits: Array.from(logits),
    critic,
  };
  if (withStates) rec.state = state;
  if (moves.length > 1) multi += 1;
  if (moves.length > 1 && searches < maxSearches) {
    const trace: Record<string, any> = {};
    const t0 = performance.now();
    const res = searchMove(w, state, seat, pending.candidates.map((c) => c.cards), logits, {
      config,
      rng: mulberry32(positionSeed(seed, pos)),
      trace,
    });
    const ms = performance.now() - t0;
    if (res.searched) {
      searches += 1;
      searchMs.push(ms);
      rec.search = {
        ms,
        choice: res.index,
        top: trace.top,
        means: trace.means,
        scores: trace.scores,
        worlds: trace.worlds.map((wd: any) => ({
          hands: wd.hands.map((h: Int8Array) => Array.from(h)),
          kitty: Array.from(wd.kitty as Int8Array),
          relaxed: wd.relaxed,
        })),
        leaves: trace.leaves.map((l: any) => ({
          world: l.world,
          cand: l.cand,
          settled: l.settled,
          future: l.future,
          value: l.value,
          done: l.done,
          msg: l.done ? wire(l.state, seat, true) : wire(l.state, l.seat),
        })),
        playouts: trace.playouts.map((p: any) => ({
          msg: wire(p.state, p.seat),
          candidates: p.candidates.map((c: Int8Array) => Array.from(c)),
          chosen: p.chosen,
        })),
      };
    }
  }
  process.stdout.write(JSON.stringify(rec) + '\n');
  emitted += 1;
  slot.choose(Math.floor(nextRandom() * pending.candidates.length));
}
searchMs.sort((a, b) => a - b);
process.stderr.write(
  `${emitted} decisions, ${multi} with a choice, ${searches} searched` +
    (searchMs.length
      ? `; ms per search median ${searchMs[Math.floor(searchMs.length / 2)]!.toFixed(0)}, max ${searchMs[searchMs.length - 1]!.toFixed(0)}`
      : '') +
    '\n',
);
