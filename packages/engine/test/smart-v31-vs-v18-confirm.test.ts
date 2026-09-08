import { describe, it, expect } from 'vitest';
import { runMatch } from '../src/eval';

// 32-game confirmation with fresh seeds (3141/3143/3161/3163).
// Combined with initial 32 games (17-15 = 53.1%), total 64 games will
// give CI95 of roughly ±12%, enough to distinguish a real ~58% from
// a true ~50%.
describe('smart-v31 vs smart-v18 (confirmation 32 games)', () => {
  it('seat-symmetric', () => {
    const m1 = runMatch({ agents: ['smart-v31','smart-v18','smart-v31','smart-v18'], numGames: 8, seed: 3141, maxHandsPerGame: 50 });
    const m2 = runMatch({ agents: ['smart-v18','smart-v31','smart-v18','smart-v31'], numGames: 8, seed: 3143, maxHandsPerGame: 50 });
    const m3 = runMatch({ agents: ['smart-v31','smart-v18','smart-v31','smart-v18'], numGames: 8, seed: 3161, maxHandsPerGame: 50 });
    const m4 = runMatch({ agents: ['smart-v18','smart-v31','smart-v18','smart-v31'], numGames: 8, seed: 3163, maxHandsPerGame: 50 });
    const v31 =
      (m1.winsByTeam['A'] ?? 0) +
      (m2.winsByTeam['B'] ?? 0) +
      (m3.winsByTeam['A'] ?? 0) +
      (m4.winsByTeam['B'] ?? 0);
    const v18 =
      (m1.winsByTeam['B'] ?? 0) +
      (m2.winsByTeam['A'] ?? 0) +
      (m3.winsByTeam['B'] ?? 0) +
      (m4.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates + m3.stalemates + m4.stalemates;
    console.log(
      `[v31 vs v18 CONFIRM, 32 games]  v31=${v31}  v18=${v18}  stale=${stale}` +
        `  v31_winrate=${((v31 / 32) * 100).toFixed(1)}%`,
    );
    expect(v31 + v18 + stale).toBe(32);
  }, 8 * 60 * 60 * 1000);
});
