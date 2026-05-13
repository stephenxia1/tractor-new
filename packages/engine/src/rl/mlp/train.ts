// REINFORCE self-play trainer for the small-MLP Tractor agent.
//
// One training episode = one hand. For each episode:
//   1. Pick which seats use the current policy. Pick which seats use
//      opponents (drawn from a pool of past snapshots + heuristics).
//   2. Run the hand via `runHand` while the current-policy agent records
//      its trajectory of (per-candidate caches, sampled idx).
//   3. Compute the per-team reward from the hand summary. All decisions
//      taken by current-policy seats on team T get reward = team_T_reward.
//   4. Subtract a baseline (EMA of recent rewards) and accumulate the
//      REINFORCE gradient across all decisions.
//   5. Apply Adam.
//   6. Snapshot the policy every K episodes; periodically eval against
//      the random baseline and log progress.
//
// Run with:
//   ../../../node_modules/.bin/tsx packages/engine/src/rl/mlp/train.ts
// from the repo root, or use the included npm-script wrapper.

import { writeFileSync, existsSync, readFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runHand, runMatch, type Agent } from '../../eval';
import { mulberry32 } from '../../prng';
import { defaultV1Config } from '../../state';

import { makePolicyAgent, type Trajectory } from './agent';
import {
  accumulateReinforceGradient,
  adamStep,
  clonePolicy,
  createAdam,
  createPolicy,
  PARAM_COUNT,
  serialize,
  deserialize,
  type Policy,
} from './policy';

// ---------------------------------------------------------------- config --

const CONFIG = {
  episodes: parseEnvInt('EPISODES', 8000),
  lr: parseEnvFloat('LR', 5e-4),
  baselineEMA: parseEnvFloat('BASELINE_EMA', 0.99),
  trainTemperature: parseEnvFloat('TRAIN_TEMP', 1.0),
  evalTemperature: parseEnvFloat('EVAL_TEMP', 0.05),
  snapshotEvery: parseEnvInt('SNAPSHOT_EVERY', 250),
  snapshotPoolMax: parseEnvInt('SNAPSHOT_POOL_MAX', 5),
  evalEvery: parseEnvInt('EVAL_EVERY', 1000),
  evalGames: parseEnvInt('EVAL_GAMES', 30),
  evalMaxHandsPerGame: parseEnvInt('EVAL_MAX_HANDS', 30),
  gradClip: parseEnvFloat('GRAD_CLIP', 5.0),
  seed: parseEnvInt('SEED', 17),
  weightsOutPath: process.env.WEIGHTS_OUT,
  resumeFrom: process.env.RESUME_FROM,
};

function parseEnvInt(name: string, fallback: number): number {
  const v = process.env[name];
  return v === undefined ? fallback : parseInt(v, 10);
}
function parseEnvFloat(name: string, fallback: number): number {
  const v = process.env[name];
  return v === undefined ? fallback : parseFloat(v);
}

// ------------------------------------------------------------- main loop --

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const weightsPath = CONFIG.weightsOutPath ?? resolve(__dirname, 'weights.json');

function main(): void {
  const config = defaultV1Config();
  const rng = mulberry32(CONFIG.seed);

  let policy: Policy;
  if (CONFIG.resumeFrom !== undefined && existsSync(CONFIG.resumeFrom)) {
    policy = deserialize(JSON.parse(readFileSync(CONFIG.resumeFrom, 'utf8')));
    console.log(`Resumed from ${CONFIG.resumeFrom}`);
  } else {
    policy = createPolicy(rng);
  }
  const adam = createAdam({ lr: CONFIG.lr });

  const snapshots: Policy[] = [];
  let baseline = 0;
  let baselineInitialized = false;

  const grad = new Float64Array(PARAM_COUNT);
  const heuristicAgents: Array<Agent> = [
    'lowball',
    'aggressive',
    'point-hoarder',
    'random',
  ];

  const startTime = Date.now();
  let lastEvalLog = 0;

  for (let ep = 1; ep <= CONFIG.episodes; ep++) {
    // Decide which team is "ours" this episode. Both seats of that team
    // use the current policy; the other team uses opponents drawn from
    // the snapshot pool + heuristic baselines.
    const ourTeamIdx = Math.floor(rng() * 2); // 0 = team A, 1 = team B
    const ourSeats: number[] = [];
    const oppSeats: number[] = [];
    for (let s = 0; s < config.numPlayers; s++) {
      const team = config.seatToTeam[s]!;
      const isOurs =
        (ourTeamIdx === 0 && team === 'A') ||
        (ourTeamIdx === 1 && team === 'B');
      if (isOurs) ourSeats.push(s);
      else oppSeats.push(s);
    }

    // Per-seat trajectory storage for our policy seats.
    const trajectoriesBySeat = new Map<number, Trajectory>();
    const seatRng = mulberry32(((CONFIG.seed * 1009) ^ ep) >>> 0);
    const agents: Agent[] = new Array(config.numPlayers);
    for (const s of ourSeats) {
      const traj: Trajectory = [];
      trajectoriesBySeat.set(s, traj);
      agents[s] = makePolicyAgent({
        policy,
        mode: 'train',
        temperature: CONFIG.trainTemperature,
        rng: seatRng,
        trajectory: traj,
      });
    }
    for (const s of oppSeats) {
      agents[s] = chooseOpponent(snapshots, heuristicAgents, rng, seatRng);
    }

    let summary;
    try {
      const result = runHand({
        agents,
        seed: Math.floor(rng() * 0x7fffffff),
      });
      summary = result.summary;
    } catch (err) {
      console.error(`Episode ${ep} threw: ${(err as Error).message}`);
      continue;
    }

    // Reward per team: from attackers' POV, score range 0..200.
    // Map to [-1, 1] via (attackerPoints - 80) / 80 then clip.
    const attackerSigned =
      Math.max(-1, Math.min(1, (summary.attackerPoints - 80) / 80));

    // Defending team = landlord's team; attackers = the other team.
    const landlordSeat = summary.landlord;
    const defendingTeamId = config.seatToTeam[landlordSeat]!;
    const attackingTeamId = defendingTeamId === 'A' ? 'B' : 'A';

    // Reward per seat (constant within a hand, based on its team).
    const rewardForTeam: Record<string, number> = {};
    rewardForTeam[attackingTeamId] = attackerSigned;
    rewardForTeam[defendingTeamId] = -attackerSigned;

    // Update baseline (single global EMA over per-hand "advantage").
    // We use the average reward across the seats whose decisions we're
    // counting (equivalent within a single team since both seats share
    // the same per-team reward).
    const ourTeamId = config.seatToTeam[ourSeats[0]!]!;
    const ourReward = rewardForTeam[ourTeamId]!;
    if (!baselineInitialized) {
      baseline = ourReward;
      baselineInitialized = true;
    } else {
      baseline = CONFIG.baselineEMA * baseline + (1 - CONFIG.baselineEMA) * ourReward;
    }
    const advantage = ourReward - baseline;

    // Accumulate gradient across all our-policy decisions in this episode.
    // Same advantage applied to every decision (no per-step credit assignment).
    for (let i = 0; i < grad.length; i++) grad[i] = 0;
    let totalDecisions = 0;
    for (const traj of trajectoriesBySeat.values()) {
      for (const d of traj) {
        accumulateReinforceGradient(
          policy,
          d.caches,
          d.probs,
          d.sampledIdx,
          d.temperature,
          advantage,
          grad,
        );
        totalDecisions++;
      }
    }

    if (totalDecisions > 0) {
      // Average over decisions to keep the per-step gradient magnitude
      // independent of hand length.
      const scale = 1 / totalDecisions;
      for (let i = 0; i < grad.length; i++) grad[i] = grad[i]! * scale;
      adamStep(policy, grad, adam, CONFIG.gradClip);
    }

    // Snapshot the current policy for the opponent pool.
    if (ep % CONFIG.snapshotEvery === 0) {
      snapshots.push(clonePolicy(policy));
      while (snapshots.length > CONFIG.snapshotPoolMax) snapshots.shift();
    }

    // Periodic eval & progress log.
    if (ep % CONFIG.evalEvery === 0 || ep === CONFIG.episodes) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const epPerSec = (ep / Math.max(1, Date.now() - startTime) * 1000).toFixed(1);
      const evalRes = quickEval(policy, CONFIG.evalGames, CONFIG.evalMaxHandsPerGame, ep + 1);
      console.log(
        `ep=${ep.toString().padStart(5)} ` +
        `t=${elapsed}s ` +
        `(${epPerSec}eps/s) ` +
        `baseline=${baseline.toFixed(3)} ` +
        `lastReward=${ourReward.toFixed(2)} ` +
        `evalA(win/atk_pts)=${evalRes.teamAWins}/${CONFIG.evalGames}, ${evalRes.avgAttackerPoints.toFixed(1)} ` +
        `lastDecisions=${totalDecisions}`,
      );
      lastEvalLog = ep;
    }
  }

  // Final save: JSON file (for tooling/inspection) and a TS module
  // (so the engine can import the weights without doing filesystem I/O —
  // important because the engine is consumed by both Node and the web
  // bundle).
  ensureDir(weightsPath);
  const serialized = serialize(policy);
  writeFileSync(weightsPath, JSON.stringify(serialized));
  const weightsTsPath = resolve(__dirname, 'embeddedWeights.ts');
  const tsContent =
    `// AUTO-GENERATED by train.ts — do not edit by hand.\n` +
    `// Trained ${new Date().toISOString()}.\n` +
    `import type { SerializedPolicy } from './policy';\n\n` +
    `export const EMBEDDED_WEIGHTS: SerializedPolicy = ${JSON.stringify(serialized)};\n`;
  writeFileSync(weightsTsPath, tsContent);
  console.log(`\nFinal weights written to ${weightsPath}`);
  console.log(`Embedded weights module: ${weightsTsPath}`);
  console.log(`Total wall time: ${((Date.now() - startTime) / 1000).toFixed(1)}s`);
  console.log(`Last eval at episode ${lastEvalLog}.`);
}

function chooseOpponent(
  snapshots: ReadonlyArray<Policy>,
  heuristics: ReadonlyArray<Agent>,
  rng: () => number,
  seatRng: () => number,
): Agent {
  // Mix snapshots with heuristics. If no snapshots yet, only heuristics.
  const useSnapshot = snapshots.length > 0 && rng() < 0.5;
  if (useSnapshot) {
    const snap = snapshots[Math.floor(rng() * snapshots.length)]!;
    return makePolicyAgent({
      policy: snap,
      mode: 'eval',
      temperature: 0.5, // small noise so the snapshots aren't deterministic
      rng: seatRng,
    });
  }
  return heuristics[Math.floor(rng() * heuristics.length)]!;
}

// Quick eval: trained policy on team A vs `random` on team B.
// Returns wins for team A and avg attacker points.
function quickEval(
  policy: Policy,
  numGames: number,
  maxHands: number,
  baseSeed: number,
): { teamAWins: number; avgAttackerPoints: number } {
  const evalAgent = makePolicyAgent({
    policy,
    mode: 'eval',
  });
  const m = runMatch({
    agents: [evalAgent, 'random', evalAgent, 'random'],
    numGames,
    seed: baseSeed,
    maxHandsPerGame: maxHands,
  });
  return {
    teamAWins: m.winsByTeam['A'] ?? 0,
    avgAttackerPoints: m.avgAttackerPoints,
  };
}

function ensureDir(filePath: string): void {
  const dir = dirname(filePath);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

main();
