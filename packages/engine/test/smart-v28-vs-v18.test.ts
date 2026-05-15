import { describe, it, expect } from 'vitest';
import { runMatch } from '../src/eval';

describe('smart-v28 vs smart-v18', () => {
  it(
    'seat-symmetric 8 games',
    () => {
      const m1 = runMatch({
        agents: ['smart-v28', 'smart-v18', 'smart-v28', 'smart-v18'],
        numGames: 4,
        seed: 2801,
        maxHandsPerGame: 50,
      });
      const m2 = runMatch({
        agents: ['smart-v18', 'smart-v28', 'smart-v18', 'smart-v28'],
        numGames: 4,
        seed: 2803,
        maxHandsPerGame: 50,
      });
      const v28 = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
      const v18 = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
      const stale = m1.stalemates + m2.stalemates;
      console.log(
        `[v28 vs v18, 8 games]  v28=${v28}  v18=${v18}  stalemates=${stale}` +
          `  v28_winrate=${((v28 / 8) * 100).toFixed(1)}%`,
      );
      expect(v28 + v18 + stale).toBe(8);
    },
    4 * 60 * 60 * 1000,
  );
});
