// Plays whole hands the way the ws server runs a live table, with
// `rl-zero` on one team and a heuristic bot on the other, to check the bot
// end to end: it declares while the cards come in, buries, plays every
// trick, and never stalls or makes a move the engine refuses.
//
//   npx tsx src/zeroMatch.ts --hands 20 --opponent smart-v34 [--weights DIR]
//
// Not a strength measurement -- there is no duplicate pairing here; that
// is research/py/tractor/eval (modal_eval.py::smart). It reports what
// happened and how long the bot took per decision.

import {
  applyEvent,
  createGame,
  defaultV1Config,
  pickBotAction,
  type BotAction,
  type BotType,
  type GameState,
  type Seat,
} from '@tractor/engine';

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? (process.argv[i + 1] ?? fallback) : fallback;
}

const hands = Number(arg('hands', '10'));
const opponent = arg('opponent', 'smart-v34') as BotType;
const weightsDir = arg('weights', '');
const seed0 = Number(arg('seed', '1'));

type Event = Parameters<typeof applyEvent>[1];

function toEvent(action: BotAction, seat: Seat): Event | null {
  switch (action.kind) {
    case 'declare':
      return { kind: 'declare', seat, cards: action.cards };
    case 'pick_up_kitty':
      return { kind: 'pick_up_kitty', seat };
    case 'replace_kitty':
      return { kind: 'replace_kitty', seat, cards: action.cards };
    case 'play_cards':
      return { kind: 'play_cards', seat, cards: action.cards };
    case 'wait':
      return null;
  }
}

async function main(): Promise<void> {
  if (weightsDir) process.env.TRACTOR_ZERO_WEIGHTS = weightsDir;
  await import('@tractor/engine/rl/zero/register');

  const timing: Record<string, number[]> = { declare: [], bury: [], play: [] };
  const typeOf = (seat: number): BotType => (seat % 2 === 0 ? 'rl-zero' : opponent);
  // The ws server asks every bot at every broadcast; so does this.
  const ask = (state: GameState, seat: Seat): BotAction => {
    const t0 = performance.now();
    const a = pickBotAction(state, seat, typeOf(seat));
    const ms = performance.now() - t0;
    if (typeOf(seat) === 'rl-zero') {
      if (state.phase === 'play' && a.kind === 'play_cards') timing.play!.push(ms);
      else if (state.phase === 'kitty_replace' && a.kind === 'replace_kitty') timing.bury!.push(ms);
      else if (state.phase === 'dealing' || state.phase === 'declare_lockout') timing.declare!.push(ms);
    }
    return a;
  };
  const apply = (state: GameState, ev: Event): GameState => {
    const r = applyEvent(state, ev);
    if (!r.ok) throw new Error(`engine refused ${JSON.stringify(ev).slice(0, 200)}: ${r.error}`);
    return r.state;
  };

  let state = createGame({ ...defaultV1Config(), incrementalDeal: true });
  let zeroDeclared = 0;
  let zeroLandlordHands = 0;
  let pointsAgainstZero = 0;
  let pointsForZero = 0;
  for (let h = 0; h < hands; h++) {
    if (state.phase === 'game_over') state = createGame({ ...defaultV1Config(), incrementalDeal: true });
    state = apply(state, { kind: 'start_hand', seed: seed0 * 1000 + h });
    // Deal: one card at a time, every bot asked after each.
    const total = state.config.numDecks * 54 - state.config.kittySize;
    for (let dealt = 0; dealt < total; dealt++) {
      state = apply(state, { kind: 'reveal_card', seat: (dealt % 4) as Seat });
      for (let s = 0; s < 4; s++) {
        const ev = toEvent(ask(state, s as Seat), s as Seat);
        if (ev === null) continue;
        const r = applyEvent(state, ev);
        if (r.ok) {
          state = r.state;
          if (s % 2 === 0) zeroDeclared += 1;
        }
      }
    }
    state = apply(state, { kind: 'finish_dealing' });
    // Lockout: ask until a full round passes.
    for (let quiet = 0, s = 0; quiet < 4; s = (s + 1) % 4) {
      const ev = toEvent(ask(state, s as Seat), s as Seat);
      const r = ev === null ? null : applyEvent(state, ev);
      if (r !== null && r.ok) {
        state = r.state;
        quiet = 0;
        if (s % 2 === 0) zeroDeclared += 1;
      } else quiet += 1;
    }
    state = apply(state, { kind: 'release_lockout', seed: seed0 * 7919 + h });
    let guard = 0;
    let last: { landlord: number; attackerPoints: number } | null = null;
    while (state.phase !== 'between_hands' && state.phase !== 'game_over') {
      if (guard++ > 2000) throw new Error(`stalled in ${state.phase}`);
      if (state.phase === 'trick_review') {
        const live = state.currentHand!;
        last = { landlord: live.landlord!, attackerPoints: live.attackerPoints };
        state = apply(state, { kind: 'advance_trick' });
        continue;
      }
      let acted = false;
      for (let s = 0; s < 4 && !acted; s++) {
        const ev = toEvent(ask(state, s as Seat), s as Seat);
        if (ev === null) continue;
        state = apply(state, ev);
        acted = true;
      }
      if (!acted) throw new Error(`no bot acts in ${state.phase}`);
    }
    if (last !== null) {
      // Trick points before the kitty bonus.
      const zeroDefends = last.landlord % 2 === 0;
      if (zeroDefends) zeroLandlordHands += 1;
      if (zeroDefends) pointsAgainstZero += last.attackerPoints;
      else pointsForZero += last.attackerPoints;
    }
    process.stderr.write(`hand ${h + 1}/${hands} done (${state.phase}), levels ${JSON.stringify(state.teamLevels)}\n`);
  }
  const stat = (xs: number[]) =>
    xs.length === 0
      ? 'none'
      : `${xs.length} decisions, median ${xs.sort((a, b) => a - b)[Math.floor(xs.length / 2)]!.toFixed(1)} ms, max ${Math.max(...xs).toFixed(1)} ms`;
  console.log(`hands ${hands} vs ${opponent}; zero declarations ${zeroDeclared}; zero landlord hands ${zeroLandlordHands}`);
  console.log(`points scored attacking ${pointsForZero}, conceded defending ${pointsAgainstZero}`);
  console.log(`declare: ${stat(timing.declare!)}`);
  console.log(`bury:    ${stat(timing.bury!)}`);
  console.log(`play:    ${stat(timing.play!)}`);
  console.log(`final levels ${JSON.stringify(state.teamLevels)} (zero is team of seats 0 and 2)`);
  const mem = process.memoryUsage();
  console.log(`memory: rss ${(mem.rss / 1e6).toFixed(0)} MB, heap ${(mem.heapUsed / 1e6).toFixed(0)} MB, external+arrays ${((mem.external + mem.arrayBuffers) / 1e6).toFixed(0)} MB`);
}

main().catch((err) => {
  process.stderr.write(String(err?.stack ?? err) + '\n');
  process.exit(1);
});
