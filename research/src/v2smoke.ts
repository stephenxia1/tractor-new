// Plays whole hands with `rl-v2` in every seat through the ordinary
// pickBotAction path, which is exactly how the ws server drives it.
import { GameSlot } from './env';

const hands = Number(process.argv[2] ?? 3);
const slot = new GameSlot({
  gameId: 0,
  seed: 4242,
  learnerSeats: [false, false, false, false], // all seats driven by the baseline bot
  baselineBot: 'rl-v2',
  prePlayBot: 'smart-v34',
  maxCandidates: 48,
  randomize: true,
});

const t0 = Date.now();
let done = 0;
let decisions = 0;
let guard = 0;
while (done < hands && guard++ < 100_000) {
  const { episodes } = slot.advance();
  done += episodes.length;
  for (const e of episodes) {
    decisions += 1;
    process.stdout.write(
      `hand ${decisions}: attacker ${e.attackerPoints} pts, kitty ${e.kittyPoints}\n`,
    );
  }
}
const ms = Date.now() - t0;
process.stdout.write(`\n${done} hands in ${(ms / 1000).toFixed(1)}s (${(ms / done).toFixed(0)} ms/hand)\n`);
