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

describe('smart-v12 vs smart-v9 head-to-head', () => {
  it('smart-v12 (A) vs smart-v9 (B): quick check', () => {
    const { a, b } = bench('[A=smart-v12 B=smart-v9]', ['smart-v12', 'smart-v9', 'smart-v12', 'smart-v9'], 911, 6);
    if (!(a > b)) console.warn(`[regression] smart-v12 did NOT outperform smart-v9: A=${a} B=${b}`);
    expect(a + b).toBeGreaterThan(0);
  });
  it('smart-v9 (A) vs smart-v12 (B) — flipped seats', () => {
    const { a, b } = bench('[A=smart-v9  B=smart-v12]', ['smart-v9', 'smart-v12', 'smart-v9', 'smart-v12'], 921, 6);
    if (!(b > a)) console.warn(`[regression] smart-v12 did NOT outperform smart-v9 in flipped seats: A(v9)=${a} B(v12)=${b}`);
    expect(a + b).toBeGreaterThan(0);
  });
  it('combined seat-symmetric tally over 12 games', () => {
    const m1 = runMatch({ agents: ['smart-v12', 'smart-v9', 'smart-v12', 'smart-v9'], numGames: 6, seed: 971, maxHandsPerGame: 50 });
    const m2 = runMatch({ agents: ['smart-v9', 'smart-v12', 'smart-v9', 'smart-v12'], numGames: 6, seed: 973, maxHandsPerGame: 50 });
    const v12Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v9Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(`[symmetric 12 games]  v12_wins=${v12Wins}  v9_wins=${v9Wins}  stalemates=${stale}  v12_winrate=${((v12Wins / 12) * 100).toFixed(1)}%`);
    expect(v12Wins + v9Wins + stale).toBe(12);
  });
});
