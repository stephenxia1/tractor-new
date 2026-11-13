// Every follow a bot returns must be one the engine accepts.
//
// The hand-written strategies predate the order-independent follow
// obligations, and on positions they had never met -- a shorter tractor
// held against a longer one led, a throw carrying a tractor -- `smart`
// returned follows the reducer refuses. In an evaluation that stops the
// game: pop8c's agent 3 against `smart` failed on every attempt with
// "must include tractors of length 2 in trump". `pickBotAction` now
// repairs such a follow on its way out and leaves legal ones untouched.

import { describe, expect, it } from 'vitest';
import { pickBotAction, type BotType } from '../src/bot';
import { applyEvent, nextToPlay } from '../src/reducer';
import { createGame, defaultV1Config, type GameState, type Seat } from '../src/state';
import { mulberry32 } from '../src/prng';
import { effectiveSuit } from '../src/ordering';
import { NATURAL_RANKS, SUITS, cardKey, type Card, type NaturalRank, type Suit } from '../src/types';

const DECKS = 2;

function cardOfSlot(slot: number, deckId: number): Card {
  if (slot === 52) return { kind: 'joker', rank: 'small', deckId };
  if (slot === 53) return { kind: 'joker', rank: 'big', deckId };
  return { kind: 'standard', suit: SUITS[Math.floor(slot / 13)]!, rank: NATURAL_RANKS[slot % 13]!, deckId };
}

function slot(name: string): number {
  if (name === 'SJ') return 52;
  if (name === 'BJ') return 53;
  const suit = 'shdc'.indexOf(name.slice(-1));
  const rank = NATURAL_RANKS.indexOf(name.slice(0, -1) as NaturalRank);
  return suit * 13 + rank;
}

function cards(names: string): Card[] {
  const seen = new Map<number, number>();
  return names.split(' ').map((n) => {
    const s = slot(n);
    const d = seen.get(s) ?? 0;
    seen.set(s, d + 1);
    return cardOfSlot(s, d);
  });
}

// A real game brought to its first play decision, to graft positions onto.
function playPhaseTemplate(): GameState {
  for (let seed = 1; seed < 50; seed++) {
    let s = createGame(defaultV1Config());
    s = applyEvent(s, { kind: 'start_hand', seed }).state;
    for (let i = 0; i < 400 && s.phase !== 'play'; i++) {
      if (s.phase === 'declare_lockout') {
        s = applyEvent(s, { kind: 'release_lockout', seed }).state;
        continue;
      }
      let acted = false;
      for (let seat = 0; seat < 4 && !acted; seat++) {
        const a = pickBotAction(s, seat, 'lowball');
        if (a.kind === 'wait') continue;
        const ev = a.kind === 'pick_up_kitty' ? { kind: a.kind, seat } : { ...a, seat };
        const r = applyEvent(s, ev as never);
        if (r.ok) {
          s = r.state;
          acted = true;
        }
      }
      if (!acted) break;
    }
    if (s.phase === 'play') return s;
  }
  throw new Error('no play-phase template');
}

const TEMPLATE = playPhaseTemplate();

// The follower to act on `hand`, after the leader played `lead`.
function position(
  trumpRank: NaturalRank,
  trumpSuit: Suit | null,
  lead: Card[],
  hand: Card[],
): { state: GameState; follower: Seat } {
  const h = TEMPLATE.currentHand!;
  const leader = nextToPlay(TEMPLATE)!;
  const follower = (leader + 1) % 4;
  const state = {
    ...TEMPLATE,
    currentHand: {
      ...h,
      trumpRank,
      trumpSuit,
      hands: { ...h.hands, [follower]: hand },
      currentTrick: { leader, plays: [{ seat: leader, cards: lead }] },
      tricks: [],
    },
  } as GameState;
  return { state, follower };
}

function accepted(state: GameState, seat: Seat, cardsPlayed: ReadonlyArray<Card>): boolean {
  return applyEvent(state, { kind: 'play_cards', seat, cards: cardsPlayed }).ok;
}

describe('bot follows are always legal', () => {
  it('smart plays the tractor a throw lead obliges, not three loose pairs', () => {
    // Diamonds plain (no-trump hand at level 3): K-A diamonds is a tractor.
    const { state, follower } = position(
      '3',
      null,
      cards('Kd Kd Ad Ad 4d 4d'),
      cards('Kd Kd Ad Ad 6d 6d 9d 9d 4d Jd Jd 2s 4s Js As'),
    );
    const a = pickBotAction(state, follower, 'smart');
    expect(a.kind).toBe('play_cards');
    if (a.kind !== 'play_cards') return;
    expect(accepted(state, follower, a.cards)).toBe(true);
    const keys = a.cards.map(cardKey);
    expect(keys.filter((k) => k === 'diamonds:K')).toHaveLength(2);
    expect(keys.filter((k) => k === 'diamonds:A')).toHaveLength(2);
  });

  it('smart plays a shorter trump tractor against a longer one', () => {
    // Hearts trump at level 3: K-A hearts tractor led with a pair of 3s;
    // the follower's jokers form a trump tractor it must commit.
    const { state, follower } = position(
      '3',
      'hearts',
      cards('Kh Kh Ah Ah 3d 3d'),
      cards('BJ BJ 5h 5h SJ SJ Kh Kh 3c 3c 4s As 2d 4d'),
    );
    const a = pickBotAction(state, follower, 'smart');
    expect(a.kind).toBe('play_cards');
    if (a.kind !== 'play_cards') return;
    expect(accepted(state, follower, a.cards)).toBe(true);
  });

  it('every cheap bot follows legally on structure-rich positions', () => {
    const rand = mulberry32(0x6a7d);
    const bots: BotType[] = ['smart', 'smart-v2', 'aggressive', 'point-hoarder', 'lowball', 'random'];
    let positions = 0;
    for (let attempt = 0; attempt < 4000 && positions < 300; attempt++) {
      const trumpRank = NATURAL_RANKS[Math.floor(rand() * NATURAL_RANKS.length)]!;
      const ts = Math.floor(rand() * 5);
      const trumpSuit: Suit | null = ts === 4 ? null : SUITS[ts]!;
      const ctx = { trumpRank, trumpSuit };
      // Led suit: trump half the time, where the tier boundaries make
      // tractors that ordinary deals rarely produce.
      const ledIdx = rand() < 0.5 ? 4 : Math.floor(rand() * 4);
      const pool: number[] = [];
      for (let s = 0; s < 54; s++) {
        const es = effectiveSuit(cardOfSlot(s, 0), ctx);
        if ((es === 'trump' ? 4 : SUITS.indexOf(es as Suit)) === ledIdx) pool.push(s);
      }
      if (pool.length < 6) continue;
      const base = Math.floor(rand() * pool.length);
      const at = (i: number): number => pool[(base + i) % pool.length]!;
      const shape = Math.floor(rand() * 4);
      const leadSlots =
        shape === 0 ? [at(0), at(0), at(1), at(1)]
        : shape === 1 ? [at(0), at(0), at(1), at(1), at(2), at(2)]
        : shape === 2 ? [at(0), at(0), at(1), at(1), at(3), at(3)]
        : [at(0), at(0), at(1), at(1), at(3)];
      const handCounts = new Map<number, number>();
      for (let i = 0; i < 3 + Math.floor(rand() * 6); i++) {
        const s = pool[Math.floor(rand() * pool.length)]!;
        handCounts.set(s, Math.min(DECKS, (handCounts.get(s) ?? 0) + (rand() < 0.6 ? 2 : 1)));
      }
      for (let s = 0; s < 54 && handCounts.size < 14; s++) {
        if (!pool.includes(s) && rand() < 0.15) handCounts.set(s, 1);
      }
      const hand: Card[] = [];
      for (const [s, k] of handCounts) for (let d = 0; d < k; d++) hand.push(cardOfSlot(s, d));
      if (hand.length < leadSlots.length) continue;
      const leadSeen = new Map<number, number>();
      const lead = leadSlots.map((s) => {
        const d = leadSeen.get(s) ?? 0;
        leadSeen.set(s, d + 1);
        return cardOfSlot(s, d);
      });
      const { state, follower } = position(trumpRank, trumpSuit, lead, hand);
      if (nextToPlay(state) !== follower) continue;
      positions += 1;
      for (const bot of bots) {
        const a = pickBotAction(state, follower, bot);
        if (a.kind !== 'play_cards') continue;
        expect(accepted(state, follower, a.cards), `${bot} at position ${positions}`).toBe(true);
      }
    }
    expect(positions).toBeGreaterThan(250);
  });
});

describe('follow repairs are counted', () => {
  it('counts a repaired follow under the bot that made it', async () => {
    const { botFollowRepairs, resetBotFollowRepairs } = await import('../src/bot');
    resetBotFollowRepairs();
    const { state, follower } = position(
      '3',
      null,
      cards('Kd Kd Ad Ad 4d 4d'),
      cards('Kd Kd Ad Ad 6d 6d 9d 9d 4d Jd Jd 2s 4s Js As'),
    );
    pickBotAction(state, follower, 'smart');
    pickBotAction(state, follower, 'lowball');
    expect(botFollowRepairs().get('smart')).toBe(1);
    expect(botFollowRepairs().get('lowball')).toBeUndefined();
  });
});
