import { describe, it, expect } from 'vitest';
import { runMatch, type Agent } from '../src/eval';

function bench(label: string, agents: ReadonlyArray<Agent>, seed: number, games: number) {
  const m = runMatch({
    agents,
    numGames: games,
    seed,
    maxHandsPerGame: 50,
  });
  const a = m.winsByTeam['A'] ?? 0;
  const b = m.winsByTeam['B'] ?? 0;
  console.log(
    `${label}:  A=${a}  B=${b}  stalemates=${m.stalemates}` +
      `  avgHands=${m.avgHandsPerGame.toFixed(1)}` +
      `  avgAttPts=${m.avgAttackerPoints.toFixed(1)}`,
  );
  return { a, b, m };
}

describe('smart-v5 vs smart-v4 head-to-head', () => {
  it('smart-v5 (A) vs smart-v4 (B): quick check', () => {
    const { a, b } = bench(
      '[A=smart-v5  B=smart-v4]',
      ['smart-v5', 'smart-v4', 'smart-v5', 'smart-v4'],
      211,
      10,
    );
    if (!(a > b)) {
      console.warn(`[regression] smart-v5 did NOT outperform smart-v4: A=${a} B=${b}`);
    }
    expect(a + b).toBeGreaterThan(0);
  });

  it('smart-v4 (A) vs smart-v5 (B) — flipped seats', () => {
    const { a, b } = bench(
      '[A=smart-v4  B=smart-v5]',
      ['smart-v4', 'smart-v5', 'smart-v4', 'smart-v5'],
      221,
      10,
    );
    if (!(b > a)) {
      console.warn(`[regression] smart-v5 did NOT outperform smart-v4 in flipped seats: A(v4)=${a} B(v5)=${b}`);
    }
    expect(a + b).toBeGreaterThan(0);
  });

  it('combined seat-symmetric tally over 20 games', () => {
    const m1 = runMatch({
      agents: ['smart-v5', 'smart-v4', 'smart-v5', 'smart-v4'],
      numGames: 10,
      seed: 271,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v4', 'smart-v5', 'smart-v4', 'smart-v5'],
      numGames: 10,
      seed: 273,
      maxHandsPerGame: 50,
    });
    const v5Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v4Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[symmetric 20 games]  v5_wins=${v5Wins}  v4_wins=${v4Wins}` +
        `  stalemates=${stale}` +
        `  v5_winrate=${((v5Wins / 20) * 100).toFixed(1)}%`,
    );
    expect(v5Wins + v4Wins + stale).toBe(20);
  });
});
