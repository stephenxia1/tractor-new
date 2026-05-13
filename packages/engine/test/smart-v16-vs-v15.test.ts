import { describe, it, expect } from 'vitest';
import { runMatch } from '../src/eval';

describe('smart-v16 vs smart-v15', () => {
  it('seat-symmetric 8 games (N=5 vs N=3 determinization)', () => {
    const m1 = runMatch({
      agents: ['smart-v16', 'smart-v15', 'smart-v16', 'smart-v15'],
      numGames: 4,
      seed: 1671,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v15', 'smart-v16', 'smart-v15', 'smart-v16'],
      numGames: 4,
      seed: 1673,
      maxHandsPerGame: 50,
    });
    const v16Wins = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v15Wins = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    console.log(
      `[v16 vs v15, 8 games]  v16=${v16Wins}  v15=${v15Wins}  stalemates=${stale}` +
        `  v16_winrate=${((v16Wins / 8) * 100).toFixed(1)}%`,
    );
    expect(v16Wins + v15Wins + stale).toBe(8);
  });
});
