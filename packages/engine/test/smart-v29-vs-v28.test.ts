import { describe, it, expect } from 'vitest';
import { runMatch } from '../src/eval';
import { getSmartV29Diagnostics, resetSmartV29Diagnostics } from '../src/bot';

describe('smart-v29 vs smart-v28', () => {
  it('seat-symmetric 8 games', () => {
    resetSmartV29Diagnostics();
    const m1 = runMatch({ agents: ['smart-v29','smart-v28','smart-v29','smart-v28'], numGames: 4, seed: 2901, maxHandsPerGame: 50 });
    const m2 = runMatch({ agents: ['smart-v28','smart-v29','smart-v28','smart-v29'], numGames: 4, seed: 2903, maxHandsPerGame: 50 });
    const v29 = (m1.winsByTeam['A'] ?? 0) + (m2.winsByTeam['B'] ?? 0);
    const v28 = (m1.winsByTeam['B'] ?? 0) + (m2.winsByTeam['A'] ?? 0);
    const stale = m1.stalemates + m2.stalemates;
    const diag = getSmartV29Diagnostics();
    const sel = diag.offered > 0 ? ((diag.selected / diag.offered) * 100).toFixed(1) : 'n/a';
    console.log(`[v29 vs v28, 8 games]  v29=${v29}  v28=${v28}  stalemates=${stale}  v29_winrate=${((v29 / 8) * 100).toFixed(1)}%`);
    console.log(`[v29 diag]  v27-candidates offered=${diag.offered}  selected=${diag.selected}  selection_rate=${sel}%`);
    expect(v29 + v28 + stale).toBe(8);
  }, 4 * 60 * 60 * 1000);
});
