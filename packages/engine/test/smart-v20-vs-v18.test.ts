import { describe, it, expect } from 'vitest';
import { runMatch } from '../src/eval';

describe('smart-v20 vs smart-v18', () => {
  it('seat-symmetric 8 games', () => {
    const m1 = runMatch({
      agents: ['smart-v20', 'smart-v18', 'smart-v20', 'smart-v18'],
      numGames: 4,
      seed: 2001,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v18', 'smart-v20', 'smart-v18', 'smart-v20'],
      numGames: 4,
      seed: 2003,
      maxHandsPerGame: 50,
    });
    const v20 = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v18 = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[v20 vs v18, 8 games]  v20=${v20}  v18=${v18}  stalemates=${stale}` +
        `  v20_winrate=${((v20 / 8) * 100).toFixed(1)}%`,
    );
    expect(v20 + v18 + stale).toBe(8);
  });
});
