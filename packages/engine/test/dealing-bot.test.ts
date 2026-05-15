import { describe, it, expect } from 'vitest';
import { applyEvent } from '../src/reducer';
import { createGame, defaultV1Config, type GameConfig, type Seat } from '../src/state';
import { pickBotAction } from '../src/bot';
import type { Card } from '../src/types';

function incrementalConfig(): GameConfig {
  return { ...defaultV1Config(), incrementalDeal: true };
}

describe('bot dealing-phase declare', () => {
  it('non-smart bot waits during dealing', () => {
    const s0 = createGame(incrementalConfig());
    const r1 = applyEvent(s0, { kind: 'start_hand', seed: 1 });
    if (!r1.ok) throw new Error('start_hand failed');
    const action = pickBotAction(r1.state, 0, 'lowball');
    expect(action.kind).toBe('wait');
  });

  it('smart bot waits with no cards revealed', () => {
    const s0 = createGame(incrementalConfig());
    const r1 = applyEvent(s0, { kind: 'start_hand', seed: 1 });
    if (!r1.ok) throw new Error('start_hand failed');
    const action = pickBotAction(r1.state, 0, 'smart-v18');
    expect(action.kind).toBe('wait');
  });

  it('smart bot waits on a revealed single when <half hand dealt and no one else declared', () => {
    // Reveal cards to seat 0 until a level-rank card appears. Verify the
    // bot doesn't declare (waits) because we haven't seen half the hand.
    const s0 = createGame(incrementalConfig());
    const r1 = applyEvent(s0, { kind: 'start_hand', seed: 1 });
    if (!r1.ok) throw new Error('start_hand failed');

    const teamLevel = r1.state.teamLevels.A!;
    const seatHand = r1.state.currentHand!.hands[0]!;
    const levelIdx = seatHand.findIndex(
      (c) => c.kind === 'standard' && c.rank === teamLevel,
    );
    if (levelIdx === -1) return; // skip seeds where seat 0 has no level card

    const halfThreshold = Math.ceil(seatHand.length / 2);
    if (levelIdx >= halfThreshold) return; // level card is past the half point — fine

    let s = r1.state;
    for (let i = 0; i <= levelIdx; i++) {
      const r = applyEvent(s, { kind: 'reveal_card', seat: 0 });
      if (!r.ok) throw new Error('reveal failed');
      s = r.state;
    }
    // At this point, seat 0 has level-rank card visible but is well
    // under half its hand revealed. The bot should wait.
    const action = pickBotAction(s, 0, 'smart-v18');
    expect(action.kind).toBe('wait');
  });

  it('smart bot declares immediately on a revealed rank-R suit pair', () => {
    // Find a seed where seat 0 has a rank-R suit pair somewhere.
    const teamLevel = defaultV1Config().startingLevel;
    let seedFound = -1;
    let pairCards: ReadonlyArray<Card> = [];
    let pairLastIdx = -1;
    for (let seed = 1; seed < 200 && seedFound === -1; seed++) {
      const s0 = createGame(incrementalConfig());
      const r1 = applyEvent(s0, { kind: 'start_hand', seed });
      if (!r1.ok) continue;
      const seatHand = r1.state.currentHand!.hands[0]!;
      // Look for 2 same-suit standard cards of rank R.
      const candidates = seatHand
        .map((c, i) => ({ c, i }))
        .filter(
          (x) =>
            x.c.kind === 'standard' &&
            x.c.rank === teamLevel,
        );
      // Group by suit
      const bySuit = new Map<string, typeof candidates>();
      for (const x of candidates) {
        if (x.c.kind !== 'standard') continue;
        const arr = bySuit.get(x.c.suit) ?? [];
        arr.push(x);
        bySuit.set(x.c.suit, arr);
      }
      for (const [, arr] of bySuit) {
        if (arr.length >= 2) {
          seedFound = seed;
          pairCards = [arr[0]!.c, arr[1]!.c];
          pairLastIdx = Math.max(arr[0]!.i, arr[1]!.i);
          break;
        }
      }
    }
    if (seedFound === -1) {
      console.log('no seed with a rank-R suit pair in seat 0 found; skipping');
      return;
    }

    const s0 = createGame(incrementalConfig());
    let s = applyEvent(s0, { kind: 'start_hand', seed: seedFound }).state;
    // Reveal up to and including the last pair card.
    for (let i = 0; i <= pairLastIdx; i++) {
      const r = applyEvent(s, { kind: 'reveal_card', seat: 0 });
      if (!r.ok) throw new Error('reveal failed');
      s = r.state;
    }

    const action = pickBotAction(s, 0, 'smart-v18');
    expect(action.kind).toBe('declare');
    if (action.kind === 'declare') {
      expect(action.cards.length).toBe(2);
    }
  });
});
