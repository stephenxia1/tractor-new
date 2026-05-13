import { describe, it, expect } from 'vitest';
import { runMatch, type Agent } from '../src/eval';

// Head-to-head benchmark: `smart-v2` vs the v1 `smart` baseline. We run
// both seatings to control for first-mover advantage (defenders rotate
// landlord on a successful defense, so seat 0 starting the game gets the
// kitty for the first hand).

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

describe('smart-v2 vs smart head-to-head', () => {
  it('smart-v2 (A) vs smart (B): dominates', () => {
    const { a, b } = bench(
      '[A=smart-v2  B=smart]',
      ['smart-v2', 'smart', 'smart-v2', 'smart'],
      1001,
      60,
    );
    // Smart-v2 should win the vast majority. Lower bound conservative.
    expect(a).toBeGreaterThan(b * 5);
    expect(a).toBeGreaterThanOrEqual(40);
  });

  it('smart (A) vs smart-v2 (B) — flipped seats', () => {
    const { a, b } = bench(
      '[A=smart      B=smart-v2]',
      ['smart', 'smart-v2', 'smart', 'smart-v2'],
      2002,
      60,
    );
    expect(b).toBeGreaterThan(a * 5);
    expect(b).toBeGreaterThanOrEqual(40);
  });

  it('combined seat-symmetric tally over 120 games', () => {
    const m1 = runMatch({
      agents: ['smart-v2', 'smart', 'smart-v2', 'smart'],
      numGames: 60,
      seed: 17,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart', 'smart-v2', 'smart', 'smart-v2'],
      numGames: 60,
      seed: 19,
      maxHandsPerGame: 50,
    });
    const v2Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v1Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[symmetric 120 games]  v2_wins=${v2Wins}  v1_wins=${v1Wins}` +
        `  stalemates=${stale}` +
        `  v2_winrate=${((v2Wins / 120) * 100).toFixed(1)}%`,
    );
    expect(v2Wins + v1Wins + stale).toBe(120);
    // Expect strong dominance — > 80% win rate.
    expect(v2Wins).toBeGreaterThanOrEqual(96);
  });

  // Sanity: smart-v2 should still crush random.
  it('smart-v2 also crushes random (regression)', () => {
    const { a, b } = bench(
      '[A=smart-v2 vs B=random]',
      ['smart-v2', 'random', 'smart-v2', 'random'],
      303,
      30,
    );
    expect(a).toBeGreaterThanOrEqual(25);
    expect(a).toBeGreaterThan(b);
  });
});
