// Sanity + cost check for the searching agents, and a look at how often
// belief weighting actually changes the sampled worlds.
import { GameSlot } from './env';
import type { BotType } from '@tractor/engine';

const bot = (process.argv[2] ?? 'rl-v2-search') as BotType;
const hands = Number(process.argv[3] ?? 2);
const slot = new GameSlot({
  gameId: 0, seed: 4242,
  learnerSeats: [false, false, false, false],
  baselineBot: bot, prePlayBot: 'smart-v34',
  maxCandidates: 48, randomize: true,
});
const t0 = Date.now();
let done = 0, guard = 0;
while (done < hands && guard++ < 100_000) {
  const { episodes } = slot.advance();
  done += episodes.length;
  for (const e of episodes) process.stdout.write(`hand: attacker ${e.attackerPoints} pts\n`);
}
const ms = Date.now() - t0;
process.stdout.write(`${bot}: ${done} hands in ${(ms/1000).toFixed(1)}s (${(ms/done).toFixed(0)} ms/hand)\n`);
