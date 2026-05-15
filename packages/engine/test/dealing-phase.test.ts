import { describe, it, expect } from 'vitest';
import { applyEvent } from '../src/reducer';
import { createGame, defaultV1Config, type GameConfig } from '../src/state';
import type { Card } from '../src/types';

function incrementalConfig(): GameConfig {
  return { ...defaultV1Config(), incrementalDeal: true };
}

describe('dealing phase', () => {
  it('default config: start_hand goes straight to declare_lockout (backward compat)', () => {
    const s0 = createGame(defaultV1Config());
    const r = applyEvent(s0, { kind: 'start_hand', seed: 1 });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.state.phase).toBe('declare_lockout');
    const counts = r.state.currentHand!.cardsDealtBySeat;
    for (let s = 0; s < 4; s++) {
      expect(counts[s]).toBe(r.state.currentHand!.hands[s]!.length);
    }
  });

  it('incremental config: start_hand goes to dealing with zero counts', () => {
    const s0 = createGame(incrementalConfig());
    const r = applyEvent(s0, { kind: 'start_hand', seed: 1 });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.state.phase).toBe('dealing');
    const counts = r.state.currentHand!.cardsDealtBySeat;
    expect(counts[0]).toBe(0);
    expect(counts[1]).toBe(0);
    expect(counts[2]).toBe(0);
    expect(counts[3]).toBe(0);
  });

  it('reveal_card advances one seat by 1', () => {
    const s0 = createGame(incrementalConfig());
    const r1 = applyEvent(s0, { kind: 'start_hand', seed: 1 });
    if (!r1.ok) throw new Error('start_hand failed');
    const r2 = applyEvent(r1.state, { kind: 'reveal_card', seat: 0 });
    expect(r2.ok).toBe(true);
    if (!r2.ok) return;
    expect(r2.state.currentHand!.cardsDealtBySeat[0]).toBe(1);
    expect(r2.state.currentHand!.cardsDealtBySeat[1]).toBe(0);
  });

  it('declare during dealing must use a revealed card', () => {
    const s0 = createGame(incrementalConfig());
    const r1 = applyEvent(s0, { kind: 'start_hand', seed: 1 });
    if (!r1.ok) throw new Error('start_hand failed');

    // Pick a level-rank card from any seat's hand.
    const teamLevel = r1.state.teamLevels.A!;
    let declSeat = -1;
    let declCard: Card | null = null;
    for (let s = 0; s < 4; s++) {
      const card = (r1.state.currentHand!.hands[s] ?? []).find(
        (c) => c.kind === 'standard' && c.rank === teamLevel,
      );
      if (card !== undefined) {
        declSeat = s;
        declCard = card;
        break;
      }
    }
    expect(declSeat).not.toBe(-1);
    expect(declCard).not.toBeNull();

    // BEFORE revealing the card to declSeat: declare must fail.
    const before = applyEvent(r1.state, {
      kind: 'declare',
      seat: declSeat,
      cards: [declCard!],
    });
    expect(before.ok).toBe(false);

    // Reveal cards to declSeat until the level-rank card is in the revealed prefix.
    let s: typeof r1.state = r1.state;
    const seatHand = r1.state.currentHand!.hands[declSeat]!;
    const idx = seatHand.findIndex(
      (c) => c.kind === 'standard' && c.rank === teamLevel,
    );
    for (let i = 0; i <= idx; i++) {
      const rev = applyEvent(s, { kind: 'reveal_card', seat: declSeat });
      if (!rev.ok) throw new Error('reveal failed');
      s = rev.state;
    }

    // Now declare should succeed.
    const after = applyEvent(s, {
      kind: 'declare',
      seat: declSeat,
      cards: [declCard!],
    });
    expect(after.ok).toBe(true);
    if (after.ok) {
      expect(after.state.currentHand!.standingDeclaration).not.toBeNull();
      // Phase stays 'dealing' — declarations don't shortcut dealing.
      expect(after.state.phase).toBe('dealing');
    }
  });

  it('finish_dealing transitions to declare_lockout with full counts', () => {
    const s0 = createGame(incrementalConfig());
    const r1 = applyEvent(s0, { kind: 'start_hand', seed: 1 });
    if (!r1.ok) throw new Error('start_hand failed');
    const r2 = applyEvent(r1.state, { kind: 'finish_dealing' });
    expect(r2.ok).toBe(true);
    if (!r2.ok) return;
    expect(r2.state.phase).toBe('declare_lockout');
    const counts = r2.state.currentHand!.cardsDealtBySeat;
    for (let s = 0; s < 4; s++) {
      expect(counts[s]).toBe(r2.state.currentHand!.hands[s]!.length);
    }
  });

  it('reveal_card past hand size is rejected', () => {
    const s0 = createGame(incrementalConfig());
    const r1 = applyEvent(s0, { kind: 'start_hand', seed: 1 });
    if (!r1.ok) throw new Error('start_hand failed');
    let s = r1.state;
    const total = s.currentHand!.hands[0]!.length;
    for (let i = 0; i < total; i++) {
      const r = applyEvent(s, { kind: 'reveal_card', seat: 0 });
      if (!r.ok) throw new Error(`reveal ${i} failed`);
      s = r.state;
    }
    // One more should fail.
    const extra = applyEvent(s, { kind: 'reveal_card', seat: 0 });
    expect(extra.ok).toBe(false);
  });

  it('reveal_card outside dealing phase rejected', () => {
    const s0 = createGame(defaultV1Config()); // non-incremental
    const r1 = applyEvent(s0, { kind: 'start_hand', seed: 1 });
    if (!r1.ok) throw new Error('start_hand failed');
    expect(r1.state.phase).toBe('declare_lockout');
    const r2 = applyEvent(r1.state, { kind: 'reveal_card', seat: 0 });
    expect(r2.ok).toBe(false);
  });
});
