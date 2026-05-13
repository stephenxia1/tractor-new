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
  it('smart-v4 (A) vs smart-v3 (B): reports head-to-head', () => {
    const { a, b } = bench(
      '[A=smart-v4  B=smart-v3]',
      ['smart-v4', 'smart-v3', 'smart-v4', 'smart-v3'],
      131,
      20,
    );
    if (!(a > b)) {
      console.warn(
        `[regression] smart-v4 did NOT outperform smart-v3 in this seat config: A=${a} B=${b}`,
      );
    }
    expect(a + b).toBeGreaterThan(0);
  });

  it('smart-v3 (A) vs smart-v4 (B) — flipped seats', () => {
    const { a, b } = bench(
      '[A=smart-v3  B=smart-v4]',
      ['smart-v3', 'smart-v4', 'smart-v3', 'smart-v4'],
      141,
      20,
    );
    if (!(b > a)) {
      console.warn(
        `[regression] smart-v4 did NOT outperform smart-v3 in flipped seats: A(v3)=${a} B(v4)=${b}`,
      );
    }
    expect(a + b).toBeGreaterThan(0);
  });

  it('combined seat-symmetric tally over 40 games', () => {
    const m1 = runMatch({
      agents: ['smart-v4', 'smart-v3', 'smart-v4', 'smart-v3'],
      numGames: 20,
      seed: 171,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v3', 'smart-v4', 'smart-v3', 'smart-v4'],
      numGames: 20,
      seed: 173,
      maxHandsPerGame: 50,
    });
    const v4Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v3Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[symmetric 40 games]  v4_wins=${v4Wins}  v3_wins=${v3Wins}` +
        `  stalemates=${stale}` +
        `  v4_winrate=${((v4Wins / 40) * 100).toFixed(1)}%`,
    );
    if (v4Wins <= v3Wins) {
      console.warn(
        `[regression] smart-v4 did NOT improve over smart-v3 over 40 symmetric games: v4=${v4Wins} v3=${v3Wins}`,
      );
    }
    expect(v4Wins + v3Wins + stale).toBe(40);
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
