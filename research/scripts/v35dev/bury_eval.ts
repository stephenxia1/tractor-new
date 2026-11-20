// smart-v35 development: burial A/B. Each deal is declared by smart-v34,
// then the landlord buries with bot X and (separately) with bot Y, and
// both are played out by the same play policy at every seat. Paired per
// deal; bury_summary.py reads the records. Records of the runs:
// evals/2026-11-19/v35-dev/bury_*.jsonl.
//   npx tsx research/scripts/v35dev/bury_eval.ts --x smart-v35 --y smart-v34 --deals 200 --play smart
import { appendFileSync } from 'node:fs';
import {
  applyEvent,
  createGame,
  defaultV1Config,
  makeSmartV35Agent,
  pickBotAction,
  sumPoints,
  type BotAction,
  type BotType,
  type GameState,
  type Seat,
} from '@tractor/engine';

const arg = (k: string, d: string): string => {
  const i = process.argv.indexOf(`--${k}`);
  return i >= 0 ? process.argv[i + 1] ?? d : d;
};
const X = arg('x', 'smart-v35') as BotType;
const Y = arg('y', 'smart-v34') as BotType;
const deals = Number(arg('deals', '50'));
const shards = Number(arg('shards', '1'));
const shard = Number(arg('shard', '0'));
const seed0 = Number(arg('seed', '777'));
const playSpec = arg('play', 'smart');
const out = arg('out', '');
const play: (s: GameState, seat: Seat) => BotAction = playSpec.startsWith('{')
  ? makeSmartV35Agent(JSON.parse(playSpec))
  : (s, seat) => pickBotAction(s, seat, playSpec as BotType);

function toKitty(seed: number): GameState | null {
  let s = createGame(defaultV1Config());
  s = { ...s, landlord: seed % 4, teamLevels: { A: '2', B: '2' } };
  const levels = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] as const;
  const lv = levels[Math.floor(seed / 4) % 13]!;
  s = { ...s, teamLevels: { A: lv, B: lv } };
  s = applyEvent(s, { kind: 'start_hand', seed }).state;
  for (let i = 0; i < 60 && s.phase !== 'kitty_replace'; i++) {
    if (s.phase === 'declare_lockout') {
      let r = applyEvent(s, { kind: 'release_lockout', seed });
      for (const seat of [0, 1, 2, 3]) {
        const a = pickBotAction(s, seat, 'smart-v34');
        if (a.kind !== 'declare') continue;
        const d = applyEvent(s, { kind: 'declare', seat, cards: a.cards });
        if (d.ok) r = d;
        break;
      }
      if (!r.ok) return null;
      s = r.state;
    } else if (s.phase === 'declare_open') {
      s = applyEvent(s, { kind: 'pick_up_kitty', seat: s.currentHand!.landlord! }).state;
    } else return null;
  }
  return s.phase === 'kitty_replace' ? s : null;
}

function playOut(s0: GameState): number {
  let s = s0;
  for (let g = 0; g < 2000; g++) {
    if (s.phase === 'between_hands' || s.phase === 'game_over') break;
    if (s.phase === 'trick_review') {
      s = applyEvent(s, { kind: 'advance_trick' }).state;
      continue;
    }
    const hand = s.currentHand!;
    const trick = hand.currentTrick;
    const seat = trick === null
      ? (hand.tricks.length === 0 ? hand.landlord! : hand.tricks[hand.tricks.length - 1]!.winner)
      : (trick.leader + trick.plays.length) % 4;
    const a = pickBotAction(s, seat, 'lowball');
    const b = play(s, seat);
    const r = applyEvent(s, { kind: 'play_cards', seat, cards: b.kind === 'play_cards' ? b.cards : (a as any).cards });
    if (!r.ok) throw new Error(r.error);
    s = r.state;
  }
  return s.currentHand!.attackerPoints;
}

for (let d = shard; d < deals; d += shards) {
  const s = toKitty(seed0 * 100003 + d);
  if (s === null) continue;
  const ll = s.currentHand!.landlord!;
  const res: number[] = [];
  const kp: number[] = [];
  for (const bot of [X, Y]) {
    const a = pickBotAction(s, ll, bot);
    if (a.kind !== 'replace_kitty') throw new Error('no burial');
    kp.push(sumPoints(a.cards));
    const after = applyEvent(s, { kind: 'replace_kitty', seat: ll, cards: a.cards }).state;
    res.push(playOut(after));
  }
  const line = JSON.stringify({ d, x_att: res[0], y_att: res[1], x_kitty: kp[0], y_kitty: kp[1] });
  if (out) appendFileSync(out, line + '\n');
  else console.log(line);
}
