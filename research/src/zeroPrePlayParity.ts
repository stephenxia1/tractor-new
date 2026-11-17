// Dumps the TypeScript zero bot's declaration and burial decisions on real
// engine deals, for research/py/test_zero_preplay_parity.py to recompute
// with the Python networks:
//
//   npx tsx research/src/zeroPrePlayParity.ts --weights DIR --hands 120 > /tmp/preplay.jsonl
//
// Deals are driven as the ws server drives them (apps/ws/src/index.ts):
// one `reveal_card` per card to seat `i % 4`, every seat asked after each
// card, then `finish_dealing`, a closing round of asks in the lockout,
// `release_lockout`, the landlord's `pick_up_kitty`, and the burial. All
// four seats are the zero bot, so its own declarations set the standing
// declarations later asks see. Half the hands start with a landlord set,
// half without, at random team levels.
//
// Each line is one decision: the view the network read (the fields the
// evaluation host's `declare` / `bury` messages carry), the features and
// logits the TS port computed, and its choice.

import { applyEvent, createGame, defaultV1Config, NATURAL_RANKS, type GameEvent, type GameState } from '@tractor/engine';
import { buryCardTokens, buryChoice, buryContext, buryLogits, buryView, zeroBury } from '@tractor/engine/rl/zero/bury';
import { declareChoice, declareFeatures, declareLogits, declareView, zeroDeclare } from '@tractor/engine/rl/zero/declare';
import { loadZeroWeights } from '@tractor/engine/rl/zero/weights';

const argOf = (name: string, fallback: string): string => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? (process.argv[i + 1] ?? fallback) : fallback;
};

const weights = loadZeroWeights(argOf('weights', ''));
const hands = Number(argOf('hands', '120'));
let seed = Number(argOf('seed', '1'));

function rand(): number {
  // mulberry32
  seed = (seed + 0x6d2b79f5) | 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function apply(state: GameState, event: GameEvent): GameState {
  const r = applyEvent(state, event);
  if (!r.ok) throw new Error(`${event.kind} refused: ${r.error}`);
  return r.state;
}

const f32 = (a: Float32Array | Float64Array): number[] => Array.from(a, (x) => Number(x));
const b64 = (a: Float32Array): string => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString('base64');
const out = (obj: unknown): void => {
  process.stdout.write(JSON.stringify(obj) + '\n');
};

let declares = 0;
let declared = 0;
let buries = 0;

function ask(state: GameState, seat: number, hand: number): GameState {
  const view = declareView(state, seat);
  if (view === null) return state;
  const feats = declareFeatures(view);
  const logits = declareLogits(weights, feats);
  const choice = declareChoice(weights, view);
  declares += 1;
  out({
    t: 'declare', hand, phase: state.phase,
    seat: view.seat, level_rank: view.levelRank, landlord: view.landlord,
    dealt: view.dealt, total: view.total, closing: view.closing,
    standing_tier: view.standingTier, standing_suit: view.standingSuit,
    standing_seat: view.standingSeat, hand_counts: view.hand, mask: view.mask,
    features: b64(feats), logits: f32(logits), choice,
  });
  const action = zeroDeclare(weights, state, seat);
  if ((action === null) !== (choice === 0)) throw new Error('zeroDeclare disagrees with declareChoice');
  if (action === null || action.kind !== 'declare') return state;
  declared += 1;
  return apply(state, { kind: 'declare', seat, cards: action.cards });
}

for (let h = 0; h < hands; h++) {
  let state = createGame({ ...defaultV1Config(), incrementalDeal: true });
  const levelA = NATURAL_RANKS[Math.floor(rand() * 13)]!;
  const levelB = NATURAL_RANKS[Math.floor(rand() * 13)]!;
  const landlord = rand() < 0.5 ? Math.floor(rand() * 4) : null;
  state = { ...state, landlord, teamLevels: { A: levelA, B: levelB } };
  state = apply(state, { kind: 'start_hand', seed: Math.floor(rand() * 0x7fffffff) });
  const total = state.config.numDecks * 54 - state.config.kittySize;
  for (let i = 0; i < total; i++) {
    state = apply(state, { kind: 'reveal_card', seat: i % 4 });
    for (let s = 0; s < 4; s++) state = ask(state, s, h);
  }
  state = apply(state, { kind: 'finish_dealing' });
  // The closing round: every seat in turn until four in a row pass.
  let passes = 0;
  let cursor = state.currentHand?.landlord ?? 0;
  while (passes < 4) {
    const before = state.currentHand?.standingDeclaration ?? null;
    state = ask(state, cursor, h);
    passes = (state.currentHand?.standingDeclaration ?? null) !== before ? 0 : passes + 1;
    cursor = (cursor + 1) % 4;
  }
  state = apply(state, { kind: 'release_lockout', seed: Math.floor(rand() * 0x7fffffff) });
  if (state.phase === 'declare_open') {
    state = apply(state, { kind: 'pick_up_kitty', seat: state.currentHand!.landlord! });
  }
  if (state.phase !== 'kitty_replace') throw new Error(`expected kitty_replace, got ${state.phase}`);
  const ll = state.currentHand!.landlord!;
  const view = buryView(state, ll)!;
  const steps: unknown[] = [];
  const buried = new Array<number>(54).fill(0);
  const { counts, picks } = buryChoice(weights, view);
  for (let step = 0; step < 8; step++) {
    const { cards, mask } = buryCardTokens(view, buried);
    const ctx = buryContext(view, step);
    const logits = buryLogits(weights, cards, ctx);
    steps.push({ cards: b64(cards), ctx: b64(ctx), mask, logits: f32(logits), pick: picks[step] });
    buried[picks[step]!] = buried[picks[step]!]! + 1;
  }
  const cards = zeroBury(weights, state, ll);
  state = apply(state, { kind: 'replace_kitty', seat: ll, cards });
  buries += 1;
  const st = state.currentHand!.standingDeclaration;
  out({
    t: 'bury', hand: h, seat: ll, level_rank: view.levelRank, trump_suit: view.trumpSuit,
    standing_tier: view.standingTier, standing_seat: st === null ? -1 : st.seat,
    declarer_rel: view.declarerRel, pool: view.pool, counts, steps,
  });
}
process.stderr.write(`zeroPrePlayParity: ${declares} declaration asks (${declared} declared), ${buries} burials\n`);
