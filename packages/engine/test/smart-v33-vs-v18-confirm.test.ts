import { describe, it, expect } from 'vitest';
import { runMatch } from '../src/eval';

// Confirmation with fresh seeds (3341/3343/3361/3363).
// Initial 32-game: v33=17 v18=15 (53.1%). Combined 64-game test
// reveals whether v33 truly improves over v18 (lead bonus 0.10 vs 0.15).
describe('smart-v33 vs smart-v18 (confirmation 32 games)', () => {
  it('seat-symmetric', () => {
    const m1 = runMatch({ agents: ['smart-v33','smart-v18','smart-v33','smart-v18'], numGames: 8, seed: 3341, maxHandsPerGame: 50 });
    const m2 = runMatch({ agents: ['smart-v18','smart-v33','smart-v18','smart-v33'], numGames: 8, seed: 3343, maxHandsPerGame: 50 });
    const m3 = runMatch({ agents: ['smart-v33','smart-v18','smart-v33','smart-v18'], numGames: 8, seed: 3361, maxHandsPerGame: 50 });
    const m4 = runMatch({ agents: ['smart-v18','smart-v33','smart-v18','smart-v33'], numGames: 8, seed: 3363, maxHandsPerGame: 50 });
    const v33 =
      (m1.winsByTeam['A'] ?? 0) +
      (m2.winsByTeam['B'] ?? 0) +
      (m3.winsByTeam['A'] ?? 0) +
      (m4.winsByTeam['B'] ?? 0);
    const v18 =
      (m1.winsByTeam['B'] ?? 0) +
      (m2.winsByTeam['A'] ?? 0) +
      (m3.winsByTeam['B'] ?? 0) +
      (m4.winsByTeam['A'] ?? 0);
    const stale =
      m1.stalemates + m2.stalemates + m3.stalemates + m4.stalemates;
    console.log(
      `[v33 vs v18 CONFIRM, 32 games]  v33=${v33}  v18=${v18}  stale=${stale}` +
        `  v33_winrate=${((v33 / 32) * 100).toFixed(1)}%`,
    );
    expect(v33 + v18 + stale).toBe(32);
  }, 8 * 60 * 60 * 1000);
});
