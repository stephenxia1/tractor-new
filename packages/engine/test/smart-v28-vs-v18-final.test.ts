import { describe, it, expect } from 'vitest';
import { runMatch } from '../src/eval';

// Final confirmation benchmark with fresh seeds (2841/2843, 2861/2863).
// Initial: 9-7 (seeds 2801/2803/2821/2823). Combined 32-game total
// will give CI95 of roughly ±17%, enough to distinguish a real ~58%
// improvement from a true ~50% tie.
describe('smart-v28 vs smart-v18 (final 16-game confirmation)', () => {
  it('seat-symmetric 16 games', () => {
    const m1 = runMatch({
      agents: ['smart-v28', 'smart-v18', 'smart-v28', 'smart-v18'],
      numGames: 4,
      seed: 2841,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v18', 'smart-v28', 'smart-v18', 'smart-v28'],
      numGames: 4,
      seed: 2843,
      maxHandsPerGame: 50,
    });
    const m3 = runMatch({
      agents: ['smart-v28', 'smart-v18', 'smart-v28', 'smart-v18'],
      numGames: 4,
      seed: 2861,
      maxHandsPerGame: 50,
    });
    const m4 = runMatch({
      agents: ['smart-v18', 'smart-v28', 'smart-v18', 'smart-v28'],
      numGames: 4,
      seed: 2863,
      maxHandsPerGame: 50,
    });
    const v28 =
      (m1.winsByTeam['A'] ?? 0) +
      (m2.winsByTeam['B'] ?? 0) +
      (m3.winsByTeam['A'] ?? 0) +
      (m4.winsByTeam['B'] ?? 0);
    const v18 =
      (m1.winsByTeam['B'] ?? 0) +
      (m2.winsByTeam['A'] ?? 0) +
      (m3.winsByTeam['B'] ?? 0) +
      (m4.winsByTeam['A'] ?? 0);
    const stale =
      m1.stalemates + m2.stalemates + m3.stalemates + m4.stalemates;
    console.log(
      `[v28 vs v18 FINAL, 16 games]  v28=${v28}  v18=${v18}  stalemates=${stale}` +
        `  v28_winrate=${((v28 / 16) * 100).toFixed(1)}%`,
    );
    expect(v28 + v18 + stale).toBe(16);
  }, 8 * 60 * 60 * 1000);
});
