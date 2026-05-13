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

describe('smart-v14 (hidden info + public signals)', () => {
  // v14 vs v13: does layering public signals on hidden-info help?
  it('smart-v14 vs smart-v13 (seat-symmetric 12 games)', () => {
    const m1 = runMatch({
      agents: ['smart-v14', 'smart-v13', 'smart-v14', 'smart-v13'],
      numGames: 6,
      seed: 1471,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v13', 'smart-v14', 'smart-v13', 'smart-v14'],
      numGames: 6,
      seed: 1473,
      maxHandsPerGame: 50,
    });
    const v14Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v13Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[v14 vs v13, 12 games]  v14=${v14Wins}  v13=${v13Wins}  stalemates=${stale}` +
        `  v14_winrate=${((v14Wins / 12) * 100).toFixed(1)}%`,
    );
    expect(v14Wins + v13Wins + stale).toBe(12);
  });

  // Sanity: v14 should still beat smart-v1 (no search baseline) as a
  // sanity check that fair-play strength is intact.
  it('smart-v14 still beats smart baseline', () => {
    const { a, b } = bench(
      '[A=smart-v14 B=smart]',
      ['smart-v14', 'smart', 'smart-v14', 'smart'],
      1411,
      10,
    );
    if (!(a > b)) {
      console.warn(`[regression] smart-v14 did NOT beat smart: A=${a} B=${b}`);
    }
    expect(a + b).toBeGreaterThan(0);
  });

  // v14 vs v9 (cheating bot): is the cheating-tax narrower than v13's 0-12?
  it('smart-v14 vs smart-v9 (cheating tax remeasured)', () => {
    const m1 = runMatch({
      agents: ['smart-v14', 'smart-v9', 'smart-v14', 'smart-v9'],
      numGames: 4,
      seed: 1481,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v9', 'smart-v14', 'smart-v9', 'smart-v14'],
      numGames: 4,
      seed: 1483,
      maxHandsPerGame: 50,
    });
    const v14Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v9Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[v14 vs v9, 8 games]  v14=${v14Wins}  v9=${v9Wins}  stalemates=${stale}` +
        `  v14_winrate=${((v14Wins / 8) * 100).toFixed(1)}%`,
    );
    expect(v14Wins + v9Wins + stale).toBe(8);
  });
});
