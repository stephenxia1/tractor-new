// Integration eval: exercises the 'rl-mlp' BotType through pickBotAction
// (rather than constructing the policy agent directly). This is the path
// the WS server / UI will use, so it's the test that matters for shipping.

import { runMatch } from '../../eval';

const NUM_GAMES = parseEnvInt('NUM_GAMES', 100);
const MAX_HANDS = parseEnvInt('MAX_HANDS', 50);
const SEED = parseEnvInt('SEED', 9001);

function parseEnvInt(name: string, fallback: number): number {
  const v = process.env[name];
  return v === undefined ? fallback : parseInt(v, 10);
}

const matchups: Array<[string, ['lowball' | 'aggressive' | 'point-hoarder' | 'random' | 'smart' | 'smart-v2' | 'rl-mlp', 'lowball' | 'aggressive' | 'point-hoarder' | 'random' | 'smart' | 'smart-v2' | 'rl-mlp', 'lowball' | 'aggressive' | 'point-hoarder' | 'random' | 'smart' | 'smart-v2' | 'rl-mlp', 'lowball' | 'aggressive' | 'point-hoarder' | 'random' | 'smart' | 'smart-v2' | 'rl-mlp']]> = [
  ['rl-mlp(A) vs random(B)',     ['rl-mlp', 'random', 'rl-mlp', 'random']],
  ['random(A) vs rl-mlp(B)',     ['random', 'rl-mlp', 'random', 'rl-mlp']],
  ['rl-mlp(A) vs lowball(B)',    ['rl-mlp', 'lowball', 'rl-mlp', 'lowball']],
  ['lowball(A) vs rl-mlp(B)',    ['lowball', 'rl-mlp', 'lowball', 'rl-mlp']],
  ['rl-mlp(A) vs smart(B)',      ['rl-mlp', 'smart', 'rl-mlp', 'smart']],
  ['smart(A) vs rl-mlp(B)',      ['smart', 'rl-mlp', 'smart', 'rl-mlp']],
  ['rl-mlp(A) vs smart-v2(B)',   ['rl-mlp', 'smart-v2', 'rl-mlp', 'smart-v2']],
  ['smart-v2(A) vs rl-mlp(B)',   ['smart-v2', 'rl-mlp', 'smart-v2', 'rl-mlp']],
];

console.log(`Integration eval: ${NUM_GAMES} games, ${MAX_HANDS} hands max, seed=${SEED}`);
console.log('All matchups dispatched through pickBotAction (BotType strings).\n');

const results: Array<{
  label: string;
  aWins: number;
  bWins: number;
  stalemates: number;
  avgAtkPts: number;
  avgHands: number;
  ms: number;
}> = [];

for (const [label, agents] of matchups) {
  const start = Date.now();
  const m = runMatch({
    agents,
    numGames: NUM_GAMES,
    seed: SEED,
    maxHandsPerGame: MAX_HANDS,
  });
  const ms = Date.now() - start;
  results.push({
    label,
    aWins: m.winsByTeam['A'] ?? 0,
    bWins: m.winsByTeam['B'] ?? 0,
    stalemates: m.stalemates,
    avgAtkPts: m.avgAttackerPoints,
    avgHands: m.avgHandsPerGame,
    ms,
  });
  console.log(
    `[${(ms / 1000).toFixed(1)}s] ${label.padEnd(32)} ` +
    `A=${(m.winsByTeam['A'] ?? 0).toString().padStart(3)} ` +
    `B=${(m.winsByTeam['B'] ?? 0).toString().padStart(3)} ` +
    `cap=${m.stalemates.toString().padStart(2)} ` +
    `avgAtkPts=${m.avgAttackerPoints.toFixed(1).padStart(5)} ` +
    `avgHands=${m.avgHandsPerGame.toFixed(1).padStart(4)}`,
  );
}

console.log('\n=== rl-mlp combined win rates ===');
function pairTotal(label1: string, label2: string): { rlWins: number; rlLosses: number; cap: number } {
  const a = results.find((r) => r.label === label1)!;
  const b = results.find((r) => r.label === label2)!;
  // In `*(A) vs X(B)`: rl is on team A in entry a, on team B in entry b.
  return {
    rlWins: a.aWins + b.bWins,
    rlLosses: a.bWins + b.aWins,
    cap: a.stalemates + b.stalemates,
  };
}

function row(label: string, t: { rlWins: number; rlLosses: number; cap: number }): void {
  const total = t.rlWins + t.rlLosses + t.cap;
  console.log(
    `  vs ${label.padEnd(10)} : ${t.rlWins.toString().padStart(3)}/${total} wins ` +
    `(${((t.rlWins / total) * 100).toFixed(1)}%), ` +
    `${t.rlLosses} losses, ${t.cap} stalemates`,
  );
}

row('random',   pairTotal('rl-mlp(A) vs random(B)',   'random(A) vs rl-mlp(B)'));
row('lowball',  pairTotal('rl-mlp(A) vs lowball(B)',  'lowball(A) vs rl-mlp(B)'));
row('smart',    pairTotal('rl-mlp(A) vs smart(B)',    'smart(A) vs rl-mlp(B)'));
row('smart-v2', pairTotal('rl-mlp(A) vs smart-v2(B)', 'smart-v2(A) vs rl-mlp(B)'));
