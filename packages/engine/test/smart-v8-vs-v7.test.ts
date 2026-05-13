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

describe('smart-v8 vs smart-v7 head-to-head', () => {
  it('smart-v8 (A) vs smart-v7 (B): quick check', () => {
    const { a, b } = bench(
      '[A=smart-v8  B=smart-v7]',
      ['smart-v8', 'smart-v7', 'smart-v8', 'smart-v7'],
      511,
      6,
    );
    if (!(a > b)) {
      console.warn(`[regression] smart-v8 did NOT outperform smart-v7: A=${a} B=${b}`);
    }
    expect(a + b).toBeGreaterThan(0);
  });

  it('smart-v7 (A) vs smart-v8 (B) — flipped seats', () => {
    const { a, b } = bench(
      '[A=smart-v7  B=smart-v8]',
      ['smart-v7', 'smart-v8', 'smart-v7', 'smart-v8'],
      521,
      6,
    );
    if (!(b > a)) {
      console.warn(`[regression] smart-v8 did NOT outperform smart-v7 in flipped seats: A(v7)=${a} B(v8)=${b}`);
    }
    expect(a + b).toBeGreaterThan(0);
  });

  it('combined seat-symmetric tally over 12 games', () => {
    const m1 = runMatch({
      agents: ['smart-v8', 'smart-v7', 'smart-v8', 'smart-v7'],
      numGames: 6,
      seed: 571,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v7', 'smart-v8', 'smart-v7', 'smart-v8'],
      numGames: 6,
      seed: 573,
      maxHandsPerGame: 50,
    });
    const v8Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v7Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[symmetric 12 games]  v8_wins=${v8Wins}  v7_wins=${v7Wins}` +
        `  stalemates=${stale}` +
        `  v8_winrate=${((v8Wins / 12) * 100).toFixed(1)}%`,
    );
    expect(v8Wins + v7Wins + stale).toBe(12);
  });
});
