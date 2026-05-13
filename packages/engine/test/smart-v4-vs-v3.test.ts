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

describe('smart-v4 vs smart-v3 head-to-head', () => {
  it('smart-v4 (A) vs smart-v3 (B)', () => {
    const { a, b } = bench(
      '[A=smart-v4  B=smart-v3]',
      ['smart-v4', 'smart-v3', 'smart-v4', 'smart-v3'],
      131,
      30,
    );
    expect(a + b).toBeGreaterThan(0);
    expect(a).toBeGreaterThanOrEqual(b);
  });

  it('smart-v3 (A) vs smart-v4 (B) — flipped seats', () => {
    const { a, b } = bench(
      '[A=smart-v3  B=smart-v4]',
      ['smart-v3', 'smart-v4', 'smart-v3', 'smart-v4'],
      141,
      30,
    );
    expect(a + b).toBeGreaterThan(0);
    expect(b).toBeGreaterThanOrEqual(a);
  });

  it('combined seat-symmetric tally over 60 games', () => {
    const m1 = runMatch({
      agents: ['smart-v4', 'smart-v3', 'smart-v4', 'smart-v3'],
      numGames: 30,
      seed: 171,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v3', 'smart-v4', 'smart-v3', 'smart-v4'],
      numGames: 30,
      seed: 173,
      maxHandsPerGame: 50,
    });
    const v4Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v3Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[symmetric 60 games]  v4_wins=${v4Wins}  v3_wins=${v3Wins}` +
        `  stalemates=${stale}` +
        `  v4_winrate=${((v4Wins / 60) * 100).toFixed(1)}%`,
    );
    expect(v4Wins + v3Wins + stale).toBe(60);
    expect(v4Wins).toBeGreaterThan(v3Wins);
  });

  // Regression: smart-v4 should still crush weaker bots.
  it('smart-v4 also crushes random', () => {
    const { a, b } = bench(
      '[A=smart-v4 vs B=random]',
      ['smart-v4', 'random', 'smart-v4', 'random'],
      505,
      20,
    );
    expect(a).toBeGreaterThanOrEqual(17);
    expect(a).toBeGreaterThan(b);
  });
});
