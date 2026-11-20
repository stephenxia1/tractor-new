// smart-v35 development: a smart-v35 with some knobs changed
// (makeSmartV35Agent's overrides, as JSON) against an engine bot on
// duplicate deals, the seeding research/src/botMatch.ts uses. One shard
// per process; summary.py / paired.py read the records. The runs it made
// are in evals/2026-11-19/v35-dev/ (e1, e2, e3).
//   npx tsx research/scripts/v35dev/match.ts --a '{"budget":6000}' --b smart-v34 --deals 64 \
//     --seed 20261119 --shards 8 --shard 0 --out e1.s0.jsonl
import { appendFileSync } from 'node:fs';
import {
  makeSmartV35Agent,
  pickBotAction,
  type BotAction,
  type BotType,
  type GameState,
  type Seat,
} from '@tractor/engine';
import { cardSlot, NUM_SLOTS } from '../../src/cards';
import { GameSlot } from '../../src/env';

const arg = (k: string, d: string): string => {
  const i = process.argv.indexOf(`--${k}`);
  return i >= 0 ? process.argv[i + 1] ?? d : d;
};
const aSpec = arg('a', '{}');
const bBot = arg('b', 'smart-v34') as BotType;
const deals = Number(arg('deals', '8'));
const seedBase = Number(arg('seed', '20261119'));
const shards = Number(arg('shards', '1'));
const shard = Number(arg('shard', '0'));
const prePlay = arg('pre-play', 'random');
const out = arg('out', '');
// Who declares and buries for A's seats under --pre-play bot.
const aPre = arg('a-pre', 'smart-v35');

const agentA: (s: GameState, seat: Seat) => BotAction = aSpec.startsWith('bot:')
  ? (s, seat) => pickBotAction(s, seat, aSpec.slice(4) as BotType)
  : makeSmartV35Agent(JSON.parse(aSpec));

const halves = [
  [true, false, true, false],
  [false, true, false, true],
];

for (let game = shard; game < deals; game += shards) {
  const rec: any = { game, halves: [] };
  for (let half = 0; half < 2; half++) {
    const slot = new GameSlot({
      gameId: game,
      seed: seedBase * 1_000_003 + game * 7919 + half,
      learnerSeats: halves[half]!,
      baselineBot: bBot,
      prePlayBot: bBot,
      prePlayBotBySeat: halves[half]!.map((isA) =>
        isA ? ((aSpec.startsWith('bot:') ? aSpec.slice(4) : aPre) as BotType) : undefined,
      ),
      randomPrePlay: prePlay === 'random',
      maxCandidates: 128,
      randomize: false,
      dealSeedBase: seedBase,
    });
    let aMs = 0;
    let bMs = 0;
    let aDec = 0;
    for (;;) {
      const t0 = performance.now();
      const { episodes, pending } = slot.advance();
      bMs += performance.now() - t0;
      if (episodes.length > 0) {
        const e = episodes[0]!;
        rec.halves.push({
          landlord: e.landlordSeat,
          attacker_points: e.attackerPoints,
          digest: e.dealDigest,
          a_ms: aMs,
          b_ms: bMs,
          a_dec: aDec,
        });
        break;
      }
      if (pending === null) throw new Error('stalled');
      const t1 = performance.now();
      const a = agentA(slot.gameState, pending.seat);
      aMs += performance.now() - t1;
      aDec++;
      if (a.kind !== 'play_cards') throw new Error(`A returned ${a.kind}`);
      const counts = new Array<number>(NUM_SLOTS).fill(0);
      for (const c of a.cards) counts[cardSlot(c)]! += 1;
      slot.playCounts(counts);
    }
  }
  if (rec.halves[0].digest !== rec.halves[1].digest) throw new Error('digest mismatch');
  const line = JSON.stringify(rec);
  if (out) appendFileSync(out, line + '\n');
  else console.log(line);
}
