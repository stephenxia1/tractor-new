import { describe, it, expect } from 'vitest';
import { runMatch } from '../src/eval';

describe('smart-v34 vs smart-v33 (32 games)', () => {
  it('seat-symmetric', () => {
    const m1 = runMatch({ agents: ['smart-v34','smart-v33','smart-v34','smart-v33'], numGames: 8, seed: 3401, maxHandsPerGame: 50 });
    const m2 = runMatch({ agents: ['smart-v33','smart-v34','smart-v33','smart-v34'], numGames: 8, seed: 3403, maxHandsPerGame: 50 });
    const m3 = runMatch({ agents: ['smart-v34','smart-v33','smart-v34','smart-v33'], numGames: 8, seed: 3421, maxHandsPerGame: 50 });
    const m4 = runMatch({ agents: ['smart-v33','smart-v34','smart-v33','smart-v34'], numGames: 8, seed: 3423, maxHandsPerGame: 50 });
    const v34 = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0) + (m3.winsByTeam['A'] ?? 0) + (m4.winsByTeam['B'] ?? 0);
    const v33 = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0) + (m3.winsByTeam['B'] ?? 0) + (m4.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates + m3.stalemates + m4.stalemates;
    console.log(`[v34 vs v33, 32 games]  v34=${v34}  v33=${v33}  stale=${stale}  v34_winrate=${((v34 / 32) * 100).toFixed(1)}%`);
    expect(v34 + v33 + stale).toBe(32);
  }, 8 * 60 * 60 * 1000);
});
