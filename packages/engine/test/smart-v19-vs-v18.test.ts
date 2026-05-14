import { describe, it, expect } from 'vitest';
import { runMatch } from '../src/eval';

describe('smart-v19 vs smart-v18', () => {
  it('seat-symmetric 8 games', () => {
    const m1 = runMatch({
      agents: ['smart-v19', 'smart-v18', 'smart-v19', 'smart-v18'],
      numGames: 4,
      seed: 1971,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v18', 'smart-v19', 'smart-v18', 'smart-v19'],
      numGames: 4,
      seed: 1973,
      maxHandsPerGame: 50,
    });
    const v19 = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v18 = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[v19 vs v18, 8 games]  v19=${v19}  v18=${v18}  stalemates=${stale}` +
        `  v19_winrate=${((v19 / 8) * 100).toFixed(1)}%`,
    );
    expect(v19 + v18 + stale).toBe(8);
  });
});
