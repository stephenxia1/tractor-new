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

describe('smart-v3 vs smart-v2 head-to-head', () => {
  it('smart-v3 (A) vs smart-v2 (B): dominates', () => {
    const { a, b } = bench(
      '[A=smart-v3  B=smart-v2]',
      ['smart-v3', 'smart-v2', 'smart-v3', 'smart-v2'],
      31,
      40,
    );
    expect(a).toBeGreaterThan(b * 4);
    expect(a).toBeGreaterThanOrEqual(30);
  });

  it('smart-v2 (A) vs smart-v3 (B) — flipped seats', () => {
    const { a, b } = bench(
      '[A=smart-v2  B=smart-v3]',
      ['smart-v2', 'smart-v3', 'smart-v2', 'smart-v3'],
      41,
      40,
    );
    expect(b).toBeGreaterThan(a * 4);
    expect(b).toBeGreaterThanOrEqual(30);
  });

  it('combined seat-symmetric tally over 80 games', () => {
    const m1 = runMatch({
      agents: ['smart-v3', 'smart-v2', 'smart-v3', 'smart-v2'],
      numGames: 40,
      seed: 71,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v2', 'smart-v3', 'smart-v2', 'smart-v3'],
      numGames: 40,
      seed: 73,
      maxHandsPerGame: 50,
    });
    const v3Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v2Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[symmetric 80 games]  v3_wins=${v3Wins}  v2_wins=${v2Wins}` +
        `  stalemates=${stale}` +
        `  v3_winrate=${((v3Wins / 80) * 100).toFixed(1)}%`,
    );
    expect(v3Wins + v2Wins + stale).toBe(80);
    // Expect strong dominance — > 80% win rate.
    expect(v3Wins).toBeGreaterThanOrEqual(64);
  });

  // Regressions: smart-v3 should still crush weaker bots.
  it('smart-v3 also crushes random', () => {
    const { a, b } = bench(
      '[A=smart-v3 vs B=random]',
      ['smart-v3', 'random', 'smart-v3', 'random'],
      505,
      20,
    );
    expect(a).toBeGreaterThanOrEqual(17);
    expect(a).toBeGreaterThan(b);
  });

  it('smart-v3 beats smart-v1 too', () => {
    const { a, b } = bench(
      '[A=smart-v3 vs B=smart-v1]',
      ['smart-v3', 'smart', 'smart-v3', 'smart'],
      606,
      30,
    );
    expect(a).toBeGreaterThan(b * 3);
  });
});
