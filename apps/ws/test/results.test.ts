import { mkdtempSync, readFileSync, appendFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  applyEvent,
  computeHandResult,
  createGame,
  defaultV1Config,
  pickBotAction,
  type GameState,
} from '@tractor/engine';
import {
  ResultsStore,
  buildHandRecord,
  isHandEnd,
  isHandStart,
  matchupOf,
  playerId,
  seatingKey,
  summarize,
  type HandRecord,
  type SeatRecord,
} from '../src/results';

// Plays one hand with lowball bots, returning every state it passed through.
function playHand(seed: number, from?: GameState): GameState[] {
  let s = from ?? createGame(defaultV1Config());
  const states = [s];
  const step = (ev: Parameters<typeof applyEvent>[1]): boolean => {
    const r = applyEvent(s, ev);
    if (!r.ok) return false;
    s = r.state;
    states.push(s);
    return true;
  };
  step({ kind: 'start_hand', seed });
  for (let i = 0; i < 2000 && s.phase !== 'between_hands' && s.phase !== 'game_over'; i++) {
    if (s.phase === 'declare_lockout') {
      step({ kind: 'release_lockout', seed });
      continue;
    }
    if (s.phase === 'trick_review') {
      step({ kind: 'advance_trick' });
      continue;
    }
    let acted = false;
    for (let seat = 0; seat < 4 && !acted; seat++) {
      const a = pickBotAction(s, seat, 'lowball');
      if (a.kind === 'wait') continue;
      const ev = a.kind === 'pick_up_kitty' ? { kind: a.kind, seat } : { ...a, seat };
      acted = step(ev as never);
    }
    if (!acted) throw new Error(`stuck in ${s.phase}`);
  }
  return states;
}

const HUMAN: SeatRecord = { kind: 'human', player: 'p1' };
const bot = (botType: string): SeatRecord => ({ kind: 'bot', botType });

// A record with the human team on team A (seats 0 and 2).
function rec(over: Partial<HandRecord>): HandRecord {
  return {
    v: 1,
    ts: 1,
    room: 'R',
    hand: 1,
    seats: [HUMAN, bot('smart-v35'), bot('smart-v35'), bot('smart-v35')],
    seatToTeam: { 0: 'A', 1: 'B', 2: 'A', 3: 'B' },
    landlord: 0,
    defenders: 'A',
    attackerPoints: 60,
    winner: 'defenders',
    levelDelta: 1,
    levelsBefore: { A: '2', B: '2' },
    levelsAfter: { A: '3', B: '2' },
    gameOver: false,
    gameWinner: null,
    seatingChanged: false,
    ...over,
  };
}

describe('hand boundaries on a real game', () => {
  it('finds exactly one start and one end per hand, and records it as the engine scored it', () => {
    const states = playHand(7);
    let starts = 0;
    let ends: Array<[GameState, GameState]> = [];
    for (let i = 1; i < states.length; i++) {
      if (isHandStart(states[i - 1]!, states[i]!)) starts++;
      if (isHandEnd(states[i - 1]!, states[i]!)) ends.push([states[i - 1]!, states[i]!]);
    }
    expect(starts).toBe(1);
    expect(ends).toHaveLength(1);
    const [prev, next] = ends[0]!;
    const seats = [HUMAN, bot('lowball'), bot('lowball'), bot('lowball')];
    const r = buildHandRecord({ room: 'T', prev, next, seats, seatingChanged: false, now: 5 });
    const ch = next.currentHand!;
    expect(r.attackerPoints).toBe(ch.attackerPoints);
    expect(r.winner).toBe(computeHandResult(ch.attackerPoints).winner);
    expect(r.defenders).toBe(next.config.seatToTeam[ch.landlord!]);
    expect(r.levelsBefore).toEqual({ A: '2', B: '2' });
    expect(r.levelsAfter).toEqual(next.teamLevels);
    expect(r.hand).toBe(1);
    expect(r.gameOver).toBe(false);
  });

  it('marks the second hand of a game as a new start and end', () => {
    const first = playHand(3);
    const second = playHand(4, first[first.length - 1]);
    const ends = second.filter((s, i) => i > 0 && isHandEnd(second[i - 1]!, s));
    expect(ends).toHaveLength(1);
    expect(ends[0]!.currentHand!.handNumber).toBe(2);
  });
});

describe('matchups', () => {
  it('keys a solo human with a bot partner against two bots', () => {
    expect(matchupOf(rec({}))).toMatchObject({
      humanSide: 'human + smart-v35',
      botSide: 'smart-v35',
      humanTeam: 'A',
    });
  });

  it('keys two humans, and mixed opposing bots', () => {
    const r = rec({ seats: [HUMAN, bot('rl-zero'), { kind: 'human', player: 'p2' }, bot('smart-v34')] });
    expect(matchupOf(r)).toMatchObject({ humanSide: '2 humans', botSide: 'rl-zero + smart-v34' });
  });

  it('counts no matchup with humans on both teams, none at all, or changed seats', () => {
    expect(matchupOf(rec({ seats: [HUMAN, HUMAN, bot('x'), bot('x')] }))).toBeNull();
    expect(matchupOf(rec({ seats: [bot('x'), bot('x'), bot('x'), bot('x')] }))).toBeNull();
    expect(matchupOf(rec({ seatingChanged: true }))).toBeNull();
  });
});

describe('summarize', () => {
  it('scores from the human side as defenders and as attackers', () => {
    const records = [
      // Humans defend and hold the attackers to 60: win, +20, +1 level.
      rec({ attackerPoints: 60, winner: 'defenders', levelDelta: 1 }),
      // Humans attack with 125: win, +45, +1 level.
      rec({ landlord: 1, defenders: 'B', attackerPoints: 125, winner: 'attackers', levelDelta: 1 }),
      // Humans defend, attackers take 140: loss, -60, -1 level.
      rec({ attackerPoints: 140, winner: 'attackers', levelDelta: 1 }),
      // Humans attack with 0: loss, -80, -3 levels; the game ends, bots win.
      rec({ landlord: 1, defenders: 'B', attackerPoints: 0, winner: 'defenders', levelDelta: 3, gameOver: true, gameWinner: 'B' }),
      // Not counted in a matchup.
      rec({ seatingChanged: true }),
    ];
    const s = summarize(records, 99);
    expect(s.totalHands).toBe(5);
    expect(s.matchupHands).toBe(4);
    expect(s.players).toBe(1);
    expect(s.since).toBe(1);
    expect(s.matchups).toHaveLength(1);
    const m = s.matchups[0]!;
    expect(m.key).toBe('human + smart-v35 vs smart-v35');
    expect(m.hands).toBe(4);
    expect(m.wins).toBe(2);
    expect(m.winRate).toBe(0.5);
    expect(m.winRateLow).toBeLessThan(0.5);
    expect(m.winRateHigh).toBeGreaterThan(0.5);
    expect(m.pointMargin).toBeCloseTo((20 + 45 - 60 - 80) / 4);
    expect(m.levels).toBeCloseTo((1 + 1 - 1 - 3) / 4);
    expect(m.pointMarginHalfWidth).not.toBeNull();
    expect(m.defending).toEqual({ hands: 2, wins: 1 });
    expect(m.attacking).toEqual({ hands: 2, wins: 1 });
    expect(m.games).toBe(1);
    expect(m.gamesWon).toBe(0);
  });

  it('gives a Wilson interval that stays inside [0, 1] and no half-width for one hand', () => {
    const m = summarize([rec({})], 0).matchups[0]!;
    expect(m.winRate).toBe(1);
    expect(m.winRateHigh).toBe(1);
    expect(m.winRateLow).toBeGreaterThan(0);
    expect(m.pointMarginHalfWidth).toBeNull();
  });

  it('is empty before any hand', () => {
    expect(summarize([], 0)).toMatchObject({ totalHands: 0, since: null, matchups: [] });
  });
});

describe('ResultsStore', () => {
  it('appends one line a hand and reads them back, skipping a torn line', () => {
    const path = join(mkdtempSync(join(tmpdir(), 'tractor-results-')), 'sub', 'results.jsonl');
    const a = new ResultsStore(path);
    a.append(rec({ hand: 1 }));
    a.append(rec({ hand: 2 }));
    expect(readFileSync(path, 'utf8').trim().split('\n')).toHaveLength(2);
    appendFileSync(path, '{"v":1,"ts":');
    const b = new ResultsStore(path);
    expect(b.size).toBe(2);
    expect(b.summary().matchups[0]!.hands).toBe(2);
  });

  it('keeps records in memory without a path, and refreshes its summary', () => {
    const st = new ResultsStore(null);
    expect(st.summary().totalHands).toBe(0);
    st.append(rec({}));
    expect(st.summary().totalHands).toBe(1);
  });
});

describe('player ids and seating keys', () => {
  it('hashes the session token stably, without containing it', () => {
    expect(playerId('abc')).toBe(playerId('abc'));
    expect(playerId('abc')).not.toBe(playerId('abd'));
    expect(playerId('secret-token')).not.toContain('secret');
    expect(playerId('abc')).toHaveLength(12);
  });

  it('tells seatings apart', () => {
    expect(seatingKey([HUMAN, bot('a'), null])).toBe('h:p1|b:a|-');
    expect(seatingKey([HUMAN, bot('a')])).not.toBe(seatingKey([HUMAN, bot('b')]));
  });
});
