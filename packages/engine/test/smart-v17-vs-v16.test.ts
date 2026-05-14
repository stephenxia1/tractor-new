import { describe, it, expect } from 'vitest';
import { runMatch } from '../src/eval';

describe('smart-v17 vs smart-v16', () => {
  it('seat-symmetric 8 games (N=10 vs N=5 determinization)', () => {
    const m1 = runMatch({
      agents: ['smart-v17', 'smart-v16', 'smart-v17', 'smart-v16'],
      numGames: 4,
      seed: 1771,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v16', 'smart-v17', 'smart-v16', 'smart-v17'],
      numGames: 4,
      seed: 1773,
      maxHandsPerGame: 50,
    });
    const v17Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v16Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[v17 vs v16, 8 games]  v17=${v17Wins}  v16=${v16Wins}  stalemates=${stale}` +
        `  v17_winrate=${((v17Wins / 8) * 100).toFixed(1)}%`,
    );
    expect(v17Wins + v16Wins + stale).toBe(8);
  });
});
