import { describe, it, expect } from 'vitest';
import { runMatch } from '../src/eval';
import {
  getSmartV27Diagnostics,
  resetSmartV27Diagnostics,
} from '../src/bot';

describe('smart-v27 vs smart-v18', () => {
  it(
    'seat-symmetric 8 games',
    () => {
      resetSmartV27Diagnostics();
      const m1 = runMatch({
        agents: ['smart-v27', 'smart-v18', 'smart-v27', 'smart-v18'],
        numGames: 4,
        seed: 2701,
        maxHandsPerGame: 50,
      });
      const m2 = runMatch({
        agents: ['smart-v18', 'smart-v27', 'smart-v18', 'smart-v27'],
        numGames: 4,
        seed: 2703,
        maxHandsPerGame: 50,
      });
      const v27 = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
      const v18 = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
      const stale = m1.stalemates + m2.stalemates;
      console.log(
        `[v27 vs v18, 8 games]  v27=${v27}  v18=${v18}  stalemates=${stale}` +
          `  v27_winrate=${((v27 / 8) * 100).toFixed(1)}%`,
      );
      const diag = getSmartV27Diagnostics();
      console.log('[v27 diagnostics]', JSON.stringify(diag));
      expect(v27 + v18 + stale).toBe(8);
    },
    4 * 60 * 60 * 1000,
  );
});
