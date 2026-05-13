import { describe, it, expect } from 'vitest';
import { runHand, runGame, runMatch } from '../src/eval';
import { pickBotAction } from '../src/bot';

describe('eval driver', () => {
  it('plays a single hand to completion with all-lowball bots', () => {
    const { state, summary } = runHand({
      agents: ['lowball', 'lowball', 'lowball', 'lowball'],
      seed: 42,
    });
    expect(state.phase).toMatch(/between_hands|game_over/);
    expect(summary.tricksPlayed).toBeGreaterThan(0);
    expect(summary.attackerPoints).toBeGreaterThanOrEqual(0);
    expect(['attackers', 'defenders']).toContain(summary.winner);
  });

  it('runs a full game and returns a winner or hits the hand cap', () => {
    const { summary } = runGame({
      agents: ['lowball', 'aggressive', 'lowball', 'aggressive'],
      seed: 7,
      maxHands: 30,
    });
    expect(summary.handsPlayed).toBeGreaterThan(0);
    expect(summary.handsPlayed).toBeLessThanOrEqual(30);
    expect(['game_over', 'max_hands_reached']).toContain(summary.stoppedReason);
  });

  it('runMatch tallies wins / stalemates across multiple games', () => {
    const m = runMatch({
      agents: ['lowball', 'random', 'lowball', 'random'],
      numGames: 3,
      seed: 100,
      maxHandsPerGame: 20,
    });
    expect(m.games.length).toBe(3);
    const totalOutcomes =
      (m.winsByTeam['A'] ?? 0) + (m.winsByTeam['B'] ?? 0) + m.stalemates;
    expect(totalOutcomes).toBe(3);
  });

  it('accepts a custom agent function as well as a built-in type-string', () => {
    const { state } = runHand({
      agents: [
        'lowball',
        (s, seat) => pickBotAction(s, seat, 'lowball'),
        'lowball',
        'lowball',
      ],
      seed: 11,
    });
    expect(['between_hands', 'game_over']).toContain(state.phase);
  });
});
