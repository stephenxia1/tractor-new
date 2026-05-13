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

describe('smart-v6 vs smart-v5 head-to-head', () => {
  it('smart-v6 (A) vs smart-v5 (B): quick check', () => {
    const { a, b } = bench(
      '[A=smart-v6  B=smart-v5]',
      ['smart-v6', 'smart-v5', 'smart-v6', 'smart-v5'],
      311,
      6,
    );
    if (!(a > b)) {
      console.warn(`[regression] smart-v6 did NOT outperform smart-v5: A=${a} B=${b}`);
    }
    expect(a + b).toBeGreaterThan(0);
  });

  it('smart-v5 (A) vs smart-v6 (B) — flipped seats', () => {
    const { a, b } = bench(
      '[A=smart-v5  B=smart-v6]',
      ['smart-v5', 'smart-v6', 'smart-v5', 'smart-v6'],
      321,
      6,
    );
    if (!(b > a)) {
      console.warn(`[regression] smart-v6 did NOT outperform smart-v5 in flipped seats: A(v5)=${a} B(v6)=${b}`);
    }
    expect(a + b).toBeGreaterThan(0);
  });

  it('combined seat-symmetric tally over 12 games', () => {
    const m1 = runMatch({
      agents: ['smart-v6', 'smart-v5', 'smart-v6', 'smart-v5'],
      numGames: 6,
      seed: 371,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v5', 'smart-v6', 'smart-v5', 'smart-v6'],
      numGames: 6,
      seed: 373,
      maxHandsPerGame: 50,
    });
    const v6Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v5Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[symmetric 12 games]  v6_wins=${v6Wins}  v5_wins=${v5Wins}` +
        `  stalemates=${stale}` +
        `  v6_winrate=${((v6Wins / 12) * 100).toFixed(1)}%`,
    );
    expect(v6Wins + v5Wins + stale).toBe(12);
  });
});
