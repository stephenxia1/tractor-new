import { describe, it, expect } from 'vitest';
import {
  applyEvent,
  createGame,
  defaultV1Config,
  handFor,
  mulberry32,
  shuffle,
  type GameConfig,
} from '../src';

describe('createGame', () => {
  it('initializes a v1 game in between_hands with both teams at level 2', () => {
    const game = createGame();
    expect(game.phase).toBe('between_hands');
    expect(game.landlord).toBeNull();
    expect(game.currentHand).toBeNull();
    expect(game.winner).toBeNull();
    expect(game.handsCompleted).toBe(0);
    expect(game.teamLevels).toEqual({ A: '2', B: '2' });
  });

  it('rejects invalid configs', () => {
    expect(() =>
      createGame({ ...defaultV1Config(), numPlayers: 0 }),
    ).toThrow();
    expect(() =>
      createGame({ ...defaultV1Config(), numDecks: 0 }),
    ).toThrow();
    // (108 - 9) / 4 = 24.75 — not an integer.
    const bad: GameConfig = {
      ...defaultV1Config(),
      kittySize: 9,
    };
    expect(() => createGame(bad)).toThrow();
  });

  it('rejects seatToTeam referring to unknown team', () => {
    const bad: GameConfig = {
      ...defaultV1Config(),
      seatToTeam: { 0: 'A', 1: 'B', 2: 'A', 3: 'C' as 'A' },
    };
    expect(() => createGame(bad)).toThrow();
  });
});

describe('start_hand', () => {
  it('transitions between_hands → declare_lockout and deals all cards', () => {
    const game = createGame();
    const res = applyEvent(game, { kind: 'start_hand', seed: 42 });
    expect(res.ok).toBe(true);
    if (!res.ok) return;
    expect(res.state.phase).toBe('declare_lockout');
    expect(res.state.currentHand).not.toBeNull();

    const hand = res.state.currentHand!;
    // 4 seats × 25 cards = 100; kitty = 8; total = 108
    expect(hand.kitty).toHaveLength(8);
    for (let s = 0; s < 4; s++) {
      expect(handFor(res.state, s)).toHaveLength(25);
    }
    // Cards from all 4 hands + kitty equals the full deck.
    const allCards = [
      ...handFor(res.state, 0),
      ...handFor(res.state, 1),
      ...handFor(res.state, 2),
      ...handFor(res.state, 3),
      ...hand.kitty,
    ];
    expect(allCards).toHaveLength(108);
  });

  it('produces identical hands for identical seeds (deterministic)', () => {
    const a = applyEvent(createGame(), { kind: 'start_hand', seed: 7 });
    const b = applyEvent(createGame(), { kind: 'start_hand', seed: 7 });
    expect(a.ok && b.ok).toBe(true);
    if (!a.ok || !b.ok) return;
    for (let s = 0; s < 4; s++) {
      expect(handFor(a.state, s)).toEqual(handFor(b.state, s));
    }
    expect(a.state.currentHand!.kitty).toEqual(b.state.currentHand!.kitty);
  });

  it('produces different hands for different seeds', () => {
    const a = applyEvent(createGame(), { kind: 'start_hand', seed: 1 });
    const b = applyEvent(createGame(), { kind: 'start_hand', seed: 2 });
    expect(a.ok && b.ok).toBe(true);
    if (!a.ok || !b.ok) return;
    expect(handFor(a.state, 0)).not.toEqual(handFor(b.state, 0));
  });

  it('trumpRank is null when landlord is unset', () => {
    const res = applyEvent(createGame(), { kind: 'start_hand', seed: 1 });
    expect(res.ok).toBe(true);
    if (!res.ok) return;
    expect(res.state.currentHand!.trumpRank).toBeNull();
  });

  it('trumpRank inherits from landlord team level when landlord is set', () => {
    const game = createGame();
    const withLandlord = {
      ...game,
      landlord: 0,
      teamLevels: { ...game.teamLevels, A: '7' as const },
    };
    const res = applyEvent(withLandlord, { kind: 'start_hand', seed: 1 });
    expect(res.ok).toBe(true);
    if (!res.ok) return;
    expect(res.state.currentHand!.trumpRank).toBe('7');
    expect(res.state.currentHand!.landlord).toBe(0);
  });

  it('rejects start_hand if not in between_hands', () => {
    const game = createGame();
    const inHand = applyEvent(game, { kind: 'start_hand', seed: 1 });
    expect(inHand.ok).toBe(true);
    if (!inHand.ok) return;
    const second = applyEvent(inHand.state, { kind: 'start_hand', seed: 2 });
    expect(second.ok).toBe(false);
  });
});

describe('shuffle (mulberry32 + Fisher–Yates)', () => {
  it('returns a permutation of the input', () => {
    const input = [1, 2, 3, 4, 5];
    const out = shuffle(input, mulberry32(1));
    expect(out).toHaveLength(5);
    expect([...out].sort()).toEqual([1, 2, 3, 4, 5]);
  });

  it('does not mutate the input', () => {
    const input = [1, 2, 3, 4, 5];
    shuffle(input, mulberry32(1));
    expect(input).toEqual([1, 2, 3, 4, 5]);
  });

  it('is deterministic for the same seed', () => {
    const a = shuffle([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], mulberry32(123));
    const b = shuffle([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], mulberry32(123));
    expect(a).toEqual(b);
  });
});
