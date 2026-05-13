import { describe, it, expect } from 'vitest';
import { runMatch, type Agent } from '../src/eval';

// Benchmarks the `smart` bot against each baseline. Reports win rates,
// stalemates, average hands-per-game, and average attacker points per hand
// to the test console. Win-rate floors are conservative to avoid flakiness.

function bench(label: string, agents: ReadonlyArray<Agent>, seed: number, games: number) {
  const m = runMatch({
    agents,
    numGames: games,
    seed,
    maxHandsPerGame: 40,
  });
  const a = m.winsByTeam['A'] ?? 0;
  const b = m.winsByTeam['B'] ?? 0;
  console.log(
    `${label}:  A=${a}  B=${b}  stalemates=${m.stalemates}` +
      `  avgHands=${m.avgHandsPerGame.toFixed(1)}` +
      `  avgAttPts=${m.avgAttackerPoints.toFixed(1)}`,
  );
  return { a, b, m };
}

describe('smart bot benchmarks', () => {
  it('crushes random when sat on Team A', () => {
    const { a, b } = bench(
      '[A=smart vs B=random]',
      ['smart', 'random', 'smart', 'random'],
      17,
      50,
    );
    expect(a).toBeGreaterThanOrEqual(40);
    expect(a).toBeGreaterThan(b);
  });

  it('crushes random when sat on Team B (no seat bias)', () => {
    const { a, b } = bench(
      '[A=random vs B=smart]',
      ['random', 'smart', 'random', 'smart'],
      23,
      50,
    );
    expect(b).toBeGreaterThanOrEqual(40);
    expect(b).toBeGreaterThan(a);
  });

  it('beats lowball decisively', () => {
    const { a, b } = bench(
      '[A=smart vs B=lowball]',
      ['smart', 'lowball', 'smart', 'lowball'],
      5,
      30,
    );
    expect(a).toBeGreaterThan(b);
  });

  it('beats aggressive', () => {
    const { a, b } = bench(
      '[A=smart vs B=aggressive]',
      ['smart', 'aggressive', 'smart', 'aggressive'],
      11,
      30,
    );
    expect(a).toBeGreaterThanOrEqual(b);
  });

  it('beats point-hoarder', () => {
    const { a, b } = bench(
      '[A=smart vs B=point-hoarder]',
      ['smart', 'point-hoarder', 'smart', 'point-hoarder'],
      31,
      30,
    );
    expect(a).toBeGreaterThanOrEqual(b);
  });
});
