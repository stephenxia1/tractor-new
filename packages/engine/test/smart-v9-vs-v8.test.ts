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

describe('smart-v9 vs smart-v8 head-to-head', () => {
  it('smart-v9 (A) vs smart-v8 (B): quick check', () => {
    const { a, b } = bench(
      '[A=smart-v9  B=smart-v8]',
      ['smart-v9', 'smart-v8', 'smart-v9', 'smart-v8'],
      611,
      6,
    );
    if (!(a > b)) {
      console.warn(`[regression] smart-v9 did NOT outperform smart-v8: A=${a} B=${b}`);
    }
    expect(a + b).toBeGreaterThan(0);
  });

  it('smart-v8 (A) vs smart-v9 (B) — flipped seats', () => {
    const { a, b } = bench(
      '[A=smart-v8  B=smart-v9]',
      ['smart-v8', 'smart-v9', 'smart-v8', 'smart-v9'],
      621,
      6,
    );
    if (!(b > a)) {
      console.warn(`[regression] smart-v9 did NOT outperform smart-v8 in flipped seats: A(v8)=${a} B(v9)=${b}`);
    }
    expect(a + b).toBeGreaterThan(0);
  });

  it('combined seat-symmetric tally over 12 games', () => {
    const m1 = runMatch({
      agents: ['smart-v9', 'smart-v8', 'smart-v9', 'smart-v8'],
      numGames: 6,
      seed: 671,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v8', 'smart-v9', 'smart-v8', 'smart-v9'],
      numGames: 6,
      seed: 673,
      maxHandsPerGame: 50,
    });
    const v9Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v8Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[symmetric 12 games]  v9_wins=${v9Wins}  v8_wins=${v8Wins}` +
        `  stalemates=${stale}` +
        `  v9_winrate=${((v9Wins / 12) * 100).toFixed(1)}%`,
    );
    expect(v9Wins + v8Wins + stale).toBe(12);
  });
});
