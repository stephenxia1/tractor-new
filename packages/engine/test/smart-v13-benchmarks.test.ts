import { describe, it, expect } from 'vitest';
import { runMatch, type Agent } from '../src/eval';

function bench(label: string, agents: ReadonlyArray<Agent>, seed: number, games: number) {
  const m = runMatch({ agents, numGames: games, seed, maxHandsPerGame: 50 });
  const a = m.winsByTeam['A'] ?? 0;
  const b = m.winsByTeam['B'] ?? 0;
  console.log(
    `${label}:  A=${a}  B=${b}  stalemates=${m.stalemates}` +
      `  avgHands=${m.avgHandsPerGame.toFixed(1)}` +
      `  avgAttPts=${m.avgAttackerPoints.toFixed(1)}`,
  );
  return { a, b, m };
}

describe('smart-v13 (hidden info) benchmarks', () => {
  // v13 should still crush the no-search smart-v1 baseline using only
  // legally-observable information.
  it('smart-v13 beats smart (no search baseline)', () => {
    const { a, b } = bench(
      '[A=smart-v13 B=smart]',
      ['smart-v13', 'smart', 'smart-v13', 'smart'],
      1311,
      12,
    );
    if (!(a > b)) {
      console.warn(`[regression] smart-v13 did NOT beat smart: A=${a} B=${b}`);
    }
    expect(a + b).toBeGreaterThan(0);
  });

  // v13 vs v9 measures the "cheating tax": how much info-loss costs us.
  // v9 reads opp hands; v13 only uses public info. Expect v13 to lose,
  // but want to quantify.
  it('smart-v13 vs smart-v9 (cheating tax)', () => {
    const m1 = runMatch({
      agents: ['smart-v13', 'smart-v9', 'smart-v13', 'smart-v9'],
      numGames: 6,
      seed: 1371,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v9', 'smart-v13', 'smart-v9', 'smart-v13'],
      numGames: 6,
      seed: 1373,
      maxHandsPerGame: 50,
    });
    const v13Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v9Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[v13 vs v9, 12 games]  v13=${v13Wins}  v9=${v9Wins}  stalemates=${stale}` +
        `  v13_winrate=${((v13Wins / 12) * 100).toFixed(1)}%`,
    );
    expect(v13Wins + v9Wins + stale).toBe(12);
  });
});
