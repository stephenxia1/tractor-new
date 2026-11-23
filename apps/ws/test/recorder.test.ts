// The server's recording path end to end: rooms played through
// applyToRoom (an incremental deal, then every seat moved by a bot, as
// the server drives them) with handRecorder installed as the observer.
import { afterEach, describe, expect, it } from 'vitest';
import { pickBotAction } from '@tractor/engine';
import { applyToRoom, newRoom, setApplyObserver, type Room } from '../src/rooms';
import { ResultsStore, handRecorder, playerId, type SeatRecord } from '../src/results';

function seatsOf(room: Room): Array<SeatRecord | null> {
  const out: Array<SeatRecord | null> = [];
  for (let s = 0; s < room.game.config.numPlayers; s++) {
    const occ = room.seating[s];
    if (occ === null || occ === undefined) out.push(null);
    else if (occ.kind === 'bot') out.push({ kind: 'bot', botType: occ.botType });
    else out.push({ kind: 'human', player: playerId(occ.clientId) });
  }
  return out;
}

function apply(room: Room, ev: Parameters<typeof applyToRoom>[1]): Room {
  const r = applyToRoom(room, ev);
  if (!r.ok) throw new Error(r.error);
  return r.room;
}

const ended = (room: Room): boolean =>
  room.game.phase === 'between_hands' || room.game.phase === 'game_over';

function playHand(room: Room, seed: number, midway?: (room: Room) => void): Room {
  room = apply(room, { kind: 'start_hand', seed });
  const total = room.game.config.numDecks * 54 - room.game.config.kittySize;
  for (let i = 0; i < total; i++) {
    room = apply(room, { kind: 'reveal_card', seat: i % room.game.config.numPlayers });
  }
  room = apply(room, { kind: 'finish_dealing' });
  let calledMidway = false;
  for (let i = 0; i < 3000 && !ended(room); i++) {
    if (room.game.phase === 'declare_lockout') {
      room = apply(room, { kind: 'release_lockout', seed });
      continue;
    }
    if (room.game.phase === 'trick_review') {
      room = apply(room, { kind: 'advance_trick' });
      if (midway !== undefined && !calledMidway && !ended(room)) {
        midway(room);
        calledMidway = true;
      }
      continue;
    }
    let acted = false;
    for (let seat = 0; seat < room.game.config.numPlayers && !acted; seat++) {
      const a = pickBotAction(room.game, seat, 'lowball');
      if (a.kind === 'wait') continue;
      const ev = a.kind === 'pick_up_kitty' ? { kind: a.kind, seat } : { ...a, seat };
      const r = applyToRoom(room, ev as never);
      if (r.ok) {
        room = r.room;
        acted = true;
      }
    }
    if (!acted) throw new Error(`stuck in ${room.game.phase}`);
  }
  if (!ended(room)) throw new Error('hand did not finish');
  return room;
}

function soloRoom(): Room {
  const room = newRoom('TEST');
  room.seating[0] = { kind: 'human', clientId: 'browser-1' };
  for (const s of [1, 2, 3]) room.seating[s] = { kind: 'bot', name: `Bot ${s}`, botType: 'lowball' };
  return room;
}

afterEach(() => setApplyObserver(null));

describe('handRecorder through applyToRoom', () => {
  it('records each finished hand at a human table, from the engine state', () => {
    const store = new ResultsStore(null);
    setApplyObserver(handRecorder(store, seatsOf, () => 42));
    let room = playHand(soloRoom(), 11);
    expect(store.size).toBe(1);
    room = playHand(room, 12);
    expect(store.size).toBe(2);
    const s = store.summary();
    expect(s.totalHands).toBe(2);
    expect(s.matchupHands).toBe(2);
    expect(s.players).toBe(1);
    expect(s.matchups.map((m) => m.key)).toEqual(['human + lowball vs lowball']);
    expect(s.matchups[0]!.hands).toBe(2);
    expect(s.since).toBe(42);
    expect(room.game.handsCompleted).toBe(2);
  });

  it('marks a hand whose seating changed midway and leaves it out of the matchups', () => {
    const store = new ResultsStore(null);
    setApplyObserver(handRecorder(store, seatsOf));
    playHand(soloRoom(), 21, (room) => {
      room.seating[3] = { kind: 'bot', name: 'Bot 3', botType: 'random' };
    });
    expect(store.size).toBe(1);
    expect(store.summary().matchupHands).toBe(0);
  });

  it('records nothing at an all-bot table', () => {
    const store = new ResultsStore(null);
    setApplyObserver(handRecorder(store, seatsOf));
    const room = newRoom('BOTS');
    for (const s of [0, 1, 2, 3]) room.seating[s] = { kind: 'bot', name: `Bot ${s}`, botType: 'lowball' };
    playHand(room, 31);
    expect(store.size).toBe(0);
  });
});
