import { runHand, type Agent } from '../src/eval';

const agents: Agent[] = ['rl-dmc', 'random', 'random', 'random'];
let totalSeconds = 0;
const N = 5;
for (let i = 0; i < N; i++) {
  const t0 = Date.now();
  const { summary } = runHand({ agents, seed: 1000 + i });
  const dt = (Date.now() - t0) / 1000;
  totalSeconds += dt;
  console.log(`hand ${i}: ${dt.toFixed(2)}s atkPts=${summary.attackerPoints} winner=${summary.winner}`);
}
console.log(`avg ${(totalSeconds / N).toFixed(2)}s per hand`);
