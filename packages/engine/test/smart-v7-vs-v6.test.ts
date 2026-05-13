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

describe('smart-v7 vs smart-v6 head-to-head', () => {
  it('smart-v7 (A) vs smart-v6 (B): quick check', () => {
    const { a, b } = bench(
      '[A=smart-v7  B=smart-v6]',
      ['smart-v7', 'smart-v6', 'smart-v7', 'smart-v6'],
      411,
      6,
    );
    if (!(a > b)) {
      console.warn(`[regression] smart-v7 did NOT outperform smart-v6: A=${a} B=${b}`);
    }
    expect(a + b).toBeGreaterThan(0);
  });

  it('smart-v6 (A) vs smart-v7 (B) — flipped seats', () => {
    const { a, b } = bench(
      '[A=smart-v6  B=smart-v7]',
      ['smart-v6', 'smart-v7', 'smart-v6', 'smart-v7'],
      421,
      6,
    );
    if (!(b > a)) {
      console.warn(`[regression] smart-v7 did NOT outperform smart-v6 in flipped seats: A(v6)=${a} B(v7)=${b}`);
    }
    expect(a + b).toBeGreaterThan(0);
  });

  it('combined seat-symmetric tally over 12 games', () => {
    const m1 = runMatch({
      agents: ['smart-v7', 'smart-v6', 'smart-v7', 'smart-v6'],
      numGames: 6,
      seed: 471,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v6', 'smart-v7', 'smart-v6', 'smart-v7'],
      numGames: 6,
      seed: 473,
      maxHandsPerGame: 50,
    });
    const v7Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v6Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[symmetric 12 games]  v7_wins=${v7Wins}  v6_wins=${v6Wins}` +
        `  stalemates=${stale}` +
        `  v7_winrate=${((v7Wins / 12) * 100).toFixed(1)}%`,
    );
    expect(v7Wins + v6Wins + stale).toBe(12);
  });
});
