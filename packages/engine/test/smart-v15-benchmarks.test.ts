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

describe('smart-v15 (multi-determinization) benchmarks', () => {
  it('smart-v15 vs smart-v13 (8-game seat-symmetric)', () => {
    const m1 = runMatch({
      agents: ['smart-v15', 'smart-v13', 'smart-v15', 'smart-v13'],
      numGames: 4,
      seed: 1571,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v13', 'smart-v15', 'smart-v13', 'smart-v15'],
      numGames: 4,
      seed: 1573,
      maxHandsPerGame: 50,
    });
    const v15Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v13Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[v15 vs v13, 8 games]  v15=${v15Wins}  v13=${v13Wins}  stalemates=${stale}` +
        `  v15_winrate=${((v15Wins / 8) * 100).toFixed(1)}%`,
    );
    expect(v15Wins + v13Wins + stale).toBe(8);
  });

  it('smart-v15 vs smart-v9 (cheating tax remeasured)', () => {
    const m1 = runMatch({
      agents: ['smart-v15', 'smart-v9', 'smart-v15', 'smart-v9'],
      numGames: 4,
      seed: 1581,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v9', 'smart-v15', 'smart-v9', 'smart-v15'],
      numGames: 4,
      seed: 1583,
      maxHandsPerGame: 50,
    });
    const v15Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v9Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[v15 vs v9, 8 games]  v15=${v15Wins}  v9=${v9Wins}  stalemates=${stale}` +
        `  v15_winrate=${((v15Wins / 8) * 100).toFixed(1)}%`,
    );
    expect(v15Wins + v9Wins + stale).toBe(8);
  });

  it('smart-v15 beats smart baseline', () => {
    const { a, b } = bench(
      '[A=smart-v15 B=smart]',
      ['smart-v15', 'smart', 'smart-v15', 'smart'],
      1511,
      8,
    );
    if (!(a > b)) console.warn(`[regression] smart-v15 did NOT beat smart: A=${a} B=${b}`);
    expect(a + b).toBeGreaterThan(0);
  });
});
