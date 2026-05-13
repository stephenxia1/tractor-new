// Evaluation harness for the trained RL bot.
//
// Loads the saved weights and runs a series of matches:
//   1. Trained policy (team A) vs random (team B)
//   2. Random (team A) vs trained policy (team B)
//   3. Lowball baseline vs random (both placements) as a reference
//
// Reports wins, attacker-points avg, hands-per-game, and stalemates.
//
// Run with:
//   ../../../node_modules/.bin/tsx packages/engine/src/rl/mlp/eval.ts

import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runMatch, type Agent } from '../../eval';
import { makePolicyAgent } from './agent';
import { deserialize, type Policy } from './policy';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const NUM_GAMES = parseEnvInt('NUM_GAMES', 100);
const MAX_HANDS = parseEnvInt('MAX_HANDS', 50);
const SEED = parseEnvInt('SEED', 9001);
const WEIGHTS_PATH =
  process.env.WEIGHTS_PATH ?? resolve(__dirname, 'weights.json');

function parseEnvInt(name: string, fallback: number): number {
  const v = process.env[name];
  return v === undefined ? fallback : parseInt(v, 10);
}

function main(): void {
  if (!existsSync(WEIGHTS_PATH)) {
    console.error(`No weights at ${WEIGHTS_PATH}. Run train.ts first.`);
    process.exit(1);
  }
  const policy: Policy = deserialize(
    JSON.parse(readFileSync(WEIGHTS_PATH, 'utf8')),
  );
  const rl = makePolicyAgent({ policy, mode: 'eval' });

  console.log(`Eval: ${NUM_GAMES} games, ${MAX_HANDS} hands max, seed=${SEED}`);
  console.log(`Weights: ${WEIGHTS_PATH}\n`);

  // 1. RL on team A
  const rlVsRandomA = run('rl-mlp (A) vs random (B)', [
    rl,
    'random',
    rl,
    'random',
  ]);

  // 2. RL on team B
  const rlVsRandomB = run('random (A) vs rl-mlp (B)', [
    'random',
    rl,
    'random',
    rl,
  ]);

  // 3. Reference: lowball vs random
  const lowballA = run('lowball (A) vs random (B)', [
    'lowball',
    'random',
    'lowball',
    'random',
  ]);

  const lowballB = run('random (A) vs lowball (B)', [
    'random',
    'lowball',
    'random',
    'lowball',
  ]);

  // 4. Reference: random vs random (sanity)
  const rr = run('random (A) vs random (B)', [
    'random',
    'random',
    'random',
    'random',
  ]);

  // Summary table
  console.log('\n=== Summary ===');
  printRow('placement', 'team-A wins', 'team-B wins', 'ties/cap', 'avg atk pts', 'avg hands/game');
  printRow('---', '---', '---', '---', '---', '---');
  printResult('rl-mlp (A) vs random (B)', rlVsRandomA);
  printResult('random (A) vs rl-mlp (B)', rlVsRandomB);
  printResult('lowball (A) vs random (B)', lowballA);
  printResult('random (A) vs lowball (B)', lowballB);
  printResult('random vs random', rr);

  // Headline numbers
  const rlWins = (rlVsRandomA.winsByTeam['A'] ?? 0) + (rlVsRandomB.winsByTeam['B'] ?? 0);
  const randomWinsVsRl = (rlVsRandomA.winsByTeam['B'] ?? 0) + (rlVsRandomB.winsByTeam['A'] ?? 0);
  const totalGamesVsRandom = NUM_GAMES * 2;
  const ties = totalGamesVsRandom - rlWins - randomWinsVsRl;
  console.log(
    `\nrl-mlp vs random (combined): ${rlWins}/${totalGamesVsRandom} wins ` +
    `(${((rlWins / totalGamesVsRandom) * 100).toFixed(1)}%), ` +
    `${randomWinsVsRl} losses, ${ties} ties/cap`,
  );

  const lbWins = (lowballA.winsByTeam['A'] ?? 0) + (lowballB.winsByTeam['B'] ?? 0);
  console.log(
    `lowball vs random (combined, reference): ${lbWins}/${totalGamesVsRandom} wins ` +
    `(${((lbWins / totalGamesVsRandom) * 100).toFixed(1)}%)`,
  );
}

function run(
  label: string,
  agents: Agent[],
): ReturnType<typeof runMatch> {
  const start = Date.now();
  const m = runMatch({
    agents,
    numGames: NUM_GAMES,
    seed: SEED,
    maxHandsPerGame: MAX_HANDS,
  });
  const dt = ((Date.now() - start) / 1000).toFixed(1);
  console.log(
    `[${dt}s] ${label}: A=${m.winsByTeam['A'] ?? 0}, B=${m.winsByTeam['B'] ?? 0}, ` +
    `stalemates=${m.stalemates}, avgAtkPts=${m.avgAttackerPoints.toFixed(1)}, ` +
    `avgHandsPerGame=${m.avgHandsPerGame.toFixed(1)}`,
  );
  return m;
}

function printResult(label: string, m: ReturnType<typeof runMatch>): void {
  printRow(
    label,
    String(m.winsByTeam['A'] ?? 0),
    String(m.winsByTeam['B'] ?? 0),
    String(m.stalemates),
    m.avgAttackerPoints.toFixed(1),
    m.avgHandsPerGame.toFixed(1),
  );
}

function printRow(...cols: string[]): void {
  const widths = [30, 12, 12, 9, 12, 16];
  console.log(
    cols.map((c, i) => c.padEnd(widths[i] ?? 12)).join(' | '),
  );
}

main();
