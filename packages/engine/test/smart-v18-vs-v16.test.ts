import { describe, it, expect } from 'vitest';
import { runMatch } from '../src/eval';

describe('smart-v18 vs smart-v16', () => {
  it('seat-symmetric 8 games', () => {
    const m1 = runMatch({
      agents: ['smart-v18', 'smart-v16', 'smart-v18', 'smart-v16'],
      numGames: 4,
      seed: 1871,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v16', 'smart-v18', 'smart-v16', 'smart-v18'],
      numGames: 4,
      seed: 1873,
      maxHandsPerGame: 50,
    });
    const v18 = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v16 = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[v18 vs v16, 8 games]  v18=${v18}  v16=${v16}  stalemates=${stale}` +
        `  v18_winrate=${((v18 / 8) * 100).toFixed(1)}%`,
    );
    expect(v18 + v16 + stale).toBe(8);
  });
});
