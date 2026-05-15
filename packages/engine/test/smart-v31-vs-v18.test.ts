import { describe, it, expect } from 'vitest';
import { runMatch } from '../src/eval';
import { SMART_V31_DIAG } from '../src/bot';

describe('smart-v31 vs smart-v18 (32 games)', () => {
  it('seat-symmetric', () => {
    // Reset diag counters in case other tests touched them.
    SMART_V31_DIAG.declareCalls = 0;
    SMART_V31_DIAG.declareSameAsV1 = 0;
    SMART_V31_DIAG.declareDiffSuit = 0;
    SMART_V31_DIAG.declareDiffKind = 0;
    SMART_V31_DIAG.declareWait = 0;
    SMART_V31_DIAG.buryCalls = 0;

    const m1 = runMatch({
      agents: ['smart-v31', 'smart-v18', 'smart-v31', 'smart-v18'],
      numGames: 8,
      seed: 3101,
      maxHandsPerGame: 50,
    });
    const m2 = runMatch({
      agents: ['smart-v18', 'smart-v31', 'smart-v18', 'smart-v31'],
      numGames: 8,
      seed: 3103,
      maxHandsPerGame: 50,
    });
    const m3 = runMatch({
      agents: ['smart-v31', 'smart-v18', 'smart-v31', 'smart-v18'],
      numGames: 8,
      seed: 3121,
      maxHandsPerGame: 50,
    });
    const m4 = runMatch({
      agents: ['smart-v18', 'smart-v31', 'smart-v18', 'smart-v31'],
      numGames: 8,
      seed: 3123,
      maxHandsPerGame: 50,
    });

    const v31 =
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
      `[v31 vs v18, 32 games]  v31=${v31}  v18=${v18}  stale=${stale}` +
        `  v31_winrate=${((v31 / 32) * 100).toFixed(1)}%`,
    );
    console.log(
      `[v31 diag]  declareCalls=${SMART_V31_DIAG.declareCalls}` +
        `  same=${SMART_V31_DIAG.declareSameAsV1}` +
        `  diffSuit=${SMART_V31_DIAG.declareDiffSuit}` +
        `  diffKind=${SMART_V31_DIAG.declareDiffKind}` +
        `  waits=${SMART_V31_DIAG.declareWait}` +
        `  buryCalls=${SMART_V31_DIAG.buryCalls}`,
    );
    expect(v31 + v18 + stale).toBe(32);
  }, 8 * 60 * 60 * 1000);
});
