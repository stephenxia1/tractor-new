import { describe, it, expect } from 'vitest';
import { runMatch } from '../src/eval';

describe('smart-v18 vs smart-v9 (cheating tax)', () => {
  it('seat-symmetric 8 games', () => {
    const m1 = runMatch({
      agents: ['smart-v18', 'smart-v9', 'smart-v18', 'smart-v9'],
      numGames: 4,
      seed: 1891,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v9', 'smart-v18', 'smart-v9', 'smart-v18'],
      numGames: 4,
      seed: 1893,
      maxHandsPerGame: 50,
    });
    const v18 = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v9 = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[v18 vs v9, 8 games]  v18=${v18}  v9=${v9}  stalemates=${stale}` +
        `  v18_winrate=${((v18 / 8) * 100).toFixed(1)}%`,
    );
    expect(v18 + v9 + stale).toBe(8);
  });
});
