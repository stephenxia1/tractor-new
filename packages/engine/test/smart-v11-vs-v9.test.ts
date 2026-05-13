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

describe('smart-v11 vs smart-v9 head-to-head', () => {
  it('smart-v11 (A) vs smart-v9 (B): quick check', () => {
    const { a, b } = bench('[A=smart-v11 B=smart-v9]', ['smart-v11', 'smart-v9', 'smart-v11', 'smart-v9'], 811, 6);
    if (!(a > b)) console.warn(`[regression] smart-v11 did NOT outperform smart-v9: A=${a} B=${b}`);
    expect(a + b).toBeGreaterThan(0);
  });
  it('smart-v9 (A) vs smart-v11 (B) — flipped seats', () => {
    const { a, b } = bench('[A=smart-v9  B=smart-v11]', ['smart-v9', 'smart-v11', 'smart-v9', 'smart-v11'], 821, 6);
    if (!(b > a)) console.warn(`[regression] smart-v11 did NOT outperform smart-v9 in flipped seats: A(v9)=${a} B(v11)=${b}`);
    expect(a + b).toBeGreaterThan(0);
  });
  it('combined seat-symmetric tally over 12 games', () => {
    const m1 = runMatch({ agents: ['smart-v11', 'smart-v9', 'smart-v11', 'smart-v9'], numGames: 6, seed: 871, maxHandsPerGame: 50 });
    const m2 = runMatch({ agents: ['smart-v9', 'smart-v11', 'smart-v9', 'smart-v11'], numGames: 6, seed: 873, maxHandsPerGame: 50 });
    const v11Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v9Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(`[symmetric 12 games]  v11_wins=${v11Wins}  v9_wins=${v9Wins}  stalemates=${stale}  v11_winrate=${((v11Wins / 12) * 100).toFixed(1)}%`);
    expect(v11Wins + v9Wins + stale).toBe(12);
  });
});
