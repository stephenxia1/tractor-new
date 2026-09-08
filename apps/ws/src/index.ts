import { createServer } from 'node:http';
import { WebSocketServer, type WebSocket } from 'ws';
// Installs the `rl-v2` bot. Side-effect import: it reads its weights
// from disk, so it is deliberately not part of the engine barrel.
import '@tractor/engine/rl/v2/register';

import {
  decodeClient,
  encode,
  DEAL_INTERVAL_MS,
  POST_DEAL_LOCKOUT_MS,
  type ClientMessage,
  type ServerMessage,
} from '@tractor/protocol';
import type {
  ApplyNotice,
  BotAction,
  Card,
  GameState as EngineGameState,
  Seat,
} from '@tractor/engine';
import {
  appendLog,
  applyToRoom,
  botTypeForSeat,
  findBotSeats,
  findClientSeat,
  newRoom,
  privateStateFor,
  roomView,
  type Client,
  type Room,
} from './rooms';
import {
  applyEvent,
  createGame,
  defaultV1Config,
  pickBotAction,
  type GameState,
} from '@tractor/engine';

const PORT = Number(process.env.PORT ?? 8787);

const clients = new Map<string, Client>();
const rooms = new Map<string, Room>();
const sockets = new Map<string, WebSocket>();
// Map sessionToken → clientId. Lets a reconnecting client reclaim their seat.
const sessions = new Map<string, string>();
// Map clientId → disconnect-cleanup timer. Cleared on reconnect.
const disconnectTimers = new Map<string, ReturnType<typeof setTimeout>>();
const DISCONNECT_GRACE_MS = 60_000;
// Delay before the server auto-advances a completed trick when every
// seat is a bot. Gives spectators a beat to read the trick before it
// clears.
const ALL_BOT_TRICK_REVIEW_MS = 1200;

function nextSeed(): number {
  return Math.floor(Math.random() * 0x7fffffff);
}

const SUIT_SYMBOL: Record<string, string> = {
  spades: '♠',
  hearts: '♥',
  diamonds: '♦',
  clubs: '♣',
};

function cardText(c: Card): string {
  if (c.kind === 'joker') return c.rank === 'big' ? 'BJ' : 'SJ';
  return `${c.rank}${SUIT_SYMBOL[c.suit] ?? c.suit}`;
}

function cardsText(cards: ReadonlyArray<Card>): string {
  return cards.map(cardText).join(' ');
}

function seatName(room: Room, seat: Seat): string {
  const occ = room.seating[seat];
  if (occ === null || occ === undefined) return `seat ${seat}`;
  if (occ.kind === 'bot') return occ.name;
  const c = clients.get(occ.clientId);
  return c?.nickname ?? `seat ${seat}`;
}

// Describes the diff between two game states in a player-facing log line.
function logEventEffect(
  room: Room,
  prev: EngineGameState,
  next: EngineGameState,
  cause: { kind: string; seat?: Seat; cards?: ReadonlyArray<Card> },
  notice?: ApplyNotice,
): void {
  if (notice !== undefined && notice.kind === 'forced_lead') {
    appendLog(
      room,
      `⚠ ${seatName(room, notice.beaterSeat)} can beat ${seatName(room, notice.leaderSeat)}'s throw — only ${cardsText(notice.playedCards)} stays as the lead; ${cardsText(notice.returnedCards)} returned to hand.`,
    );
  }
  const ch = next.currentHand;
  const prevCh = prev.currentHand;

  if (cause.kind === 'start_hand') {
    appendLog(room, `🃏 New hand begins (hand ${ch?.handNumber ?? '?'}).`);
    return;
  }

  if (cause.kind === 'finish_dealing') {
    appendLog(room, `All cards dealt. Declaration window closing soon.`);
    return;
  }

  if (cause.kind === 'declare' && ch?.standingDeclaration !== null) {
    const d = ch!.standingDeclaration!;
    const seat = d.seat;
    const name = seatName(room, seat);
    if (d.kind === 'single') {
      appendLog(room, `${name} declares ${SUIT_SYMBOL[d.suit] ?? d.suit} (${cardText(d.cards[0])}).`);
    } else if (d.kind === 'pair') {
      appendLog(room, `${name} declares ${SUIT_SYMBOL[d.suit] ?? d.suit} pair.`);
    } else {
      appendLog(room, `${name} declares no-trump (${d.jokerRank} jokers).`);
    }
    return;
  }

  if (cause.kind === 'release_lockout') {
    if (prev.phase === 'declare_lockout' && next.phase === 'kitty_replace') {
      const seat = ch?.landlord;
      const name = seat === null || seat === undefined ? '(random)' : seatName(room, seat);
      const suit = ch?.trumpSuit ?? 'no-trump';
      appendLog(
        room,
        `No declarations → kitty flipped. Trump ${suit}; ${name} is landlord.`,
      );
    } else if (prev.phase === 'declare_lockout' && next.phase === 'declare_open') {
      appendLog(room, `Declaration window opens (landlord may take the kitty).`);
    }
    return;
  }

  if (cause.kind === 'pick_up_kitty') {
    const seat = ch?.landlord;
    if (seat !== null && seat !== undefined) {
      appendLog(room, `${seatName(room, seat)} picks up the kitty.`);
    }
    return;
  }

  if (cause.kind === 'replace_kitty') {
    const seat = ch?.landlord;
    if (seat !== null && seat !== undefined) {
      appendLog(
        room,
        `${seatName(room, seat)} buries 8 cards. Trump is ${ch?.trumpSuit ?? 'no-trump'} (R=${ch?.trumpRank ?? '?'}).`,
      );
    }
    return;
  }

  if (cause.kind === 'play_cards' && cause.seat !== undefined && cause.cards) {
    appendLog(
      room,
      `${seatName(room, cause.seat)} plays ${cardsText(cause.cards)}.`,
    );
    return;
  }

  if (cause.kind === 'advance_trick') {
    // The trick that just finalized is the last entry of next.tricks.
    const nextTricks = ch?.tricks.length ?? 0;
    if (nextTricks > (prevCh?.tricks.length ?? 0) && ch !== null) {
      const t = ch.tricks[nextTricks - 1]!;
      appendLog(
        room,
        `→ ${seatName(room, t.winner)} wins trick${t.points > 0 ? ` (+${t.points} pts)` : ''}.`,
      );
    }
    if (next.phase !== 'trick_review' && next.phase !== 'play') {
      appendLog(
        room,
        `Hand complete: attackers ${ch?.attackerPoints ?? 0} pts.`,
      );
      if (next.phase === 'game_over' && next.winner !== null) {
        appendLog(room, `🏆 Game over — Team ${next.winner} wins!`);
      }
    }
  }
}

// Pending bot timers. Keyed by `${roomCode}:${seat}` so we don't
// double-schedule when multiple broadcasts arrive in quick succession.
const pendingBots = new Set<string>();

// Track which rooms have an active deal-driver running so we never
// double-start the loop.
const dealDriving = new Set<string>();

// Drives the incremental deal for a room. Fires `reveal_card` events
// round-robin across seats at `DEAL_INTERVAL_MS` intervals, then
// `finish_dealing`, then schedules `release_lockout`
// `POST_DEAL_LOCKOUT_MS` later. Each reveal broadcasts state so
// clients animate the new card and bots get a chance to declare
// based on their updated revealed slice.
function driveIncrementalDeal(roomCode: string): void {
  if (dealDriving.has(roomCode)) return;
  const initial = rooms.get(roomCode);
  if (initial === undefined) return;
  if (initial.game.phase !== 'dealing') return;
  dealDriving.add(roomCode);

  const totalCards =
    initial.game.config.numDecks * 54 - initial.game.config.kittySize;
  let dealt = 0;

  const tick = (): void => {
    const current = rooms.get(roomCode);
    if (current === undefined) {
      dealDriving.delete(roomCode);
      return;
    }
    if (current.game.phase !== 'dealing') {
      dealDriving.delete(roomCode);
      return;
    }
    if (dealt >= totalCards) {
      // Finish dealing → declare_lockout.
      const prevGame = current.game;
      const r = applyToRoom(current, { kind: 'finish_dealing' });
      if (!r.ok) {
        console.error('finish_dealing failed:', r.error);
        dealDriving.delete(roomCode);
        return;
      }
      logEventEffect(r.room, prevGame, r.room.game, {
        kind: 'finish_dealing',
      });
      rooms.set(roomCode, r.room);
      broadcastRoom(r.room);
      // After a brief lockout grace, release.
      setTimeout(() => {
        dealDriving.delete(roomCode);
        const cur2 = rooms.get(roomCode);
        if (cur2 === undefined) return;
        if (cur2.game.phase !== 'declare_lockout') return;
        const prevGame2 = cur2.game;
        const r2 = applyToRoom(cur2, {
          kind: 'release_lockout',
          seed: nextSeed(),
        });
        if (!r2.ok) return;
        logEventEffect(r2.room, prevGame2, r2.room.game, {
          kind: 'release_lockout',
        });
        rooms.set(roomCode, r2.room);
        broadcastRoom(r2.room);
      }, POST_DEAL_LOCKOUT_MS);
      return;
    }
    const seat = dealt % current.game.config.numPlayers;
    dealt++;
    const r = applyToRoom(current, { kind: 'reveal_card', seat });
    if (!r.ok) {
      console.error('reveal_card failed:', r.error);
      dealDriving.delete(roomCode);
      return;
    }
    rooms.set(roomCode, r.room);
    broadcastRoom(r.room);
    setTimeout(tick, DEAL_INTERVAL_MS);
  };

  setTimeout(tick, DEAL_INTERVAL_MS);
}

function allSeatsAreBots(room: Room): boolean {
  for (let s = 0; s < room.game.config.numPlayers; s++) {
    const occ = room.seating[s];
    if (occ === null || occ === undefined) return false;
    if (occ.kind !== 'bot') return false;
  }
  return true;
}

function scheduleBotActions(room: Room): void {
  if (room.game.phase === 'between_hands') {
    // All-bot tables auto-start the next hand after a short delay so a
    // spectator doesn't have to click "Start next hand" between rounds.
    if (allSeatsAreBots(room)) {
      const key = `${room.code}:autostart`;
      if (pendingBots.has(key)) return;
      pendingBots.add(key);
      setTimeout(() => {
        pendingBots.delete(key);
        const current = rooms.get(room.code);
        if (current === undefined) return;
        if (current.game.phase !== 'between_hands') return;
        if (!allSeatsAreBots(current)) return;
        const seed = nextSeed();
        const prevGame = current.game;
        const r = applyToRoom(current, { kind: 'start_hand', seed });
        if (!r.ok) return;
        r.room.log = current.log;
        logEventEffect(r.room, prevGame, r.room.game, { kind: 'start_hand' });
        rooms.set(current.code, r.room);
        broadcastRoom(r.room);
        // Drive the incremental deal: round-robin reveal_card events
        // at DEAL_INTERVAL_MS, then finish_dealing, then release_lockout.
        driveIncrementalDeal(current.code);
      }, 2000);
    }
    return;
  }
  if (room.game.phase === 'game_over') {
    return;
  }
  if (process.env.BOT_DEBUG === '1') {
    console.log(
      `[scheduleBotActions] phase=${room.game.phase} ` +
        `trick=${JSON.stringify(room.game.currentHand?.currentTrick)} ` +
        `landlord=${room.game.currentHand?.landlord} ` +
        `tricks=${room.game.currentHand?.tricks.length ?? 0}`,
    );
  }
  // Trick review only advances when a human clicks (or has auto-advance
  // toggled on). When every seat is a bot, spectators have no UI to
  // click — auto-advance after a short delay so all-bot games keep
  // playing on their own.
  if (room.game.phase === 'trick_review') {
    if (!allSeatsAreBots(room)) return;
    const key = `${room.code}:trick-advance`;
    if (pendingBots.has(key)) return;
    pendingBots.add(key);
    setTimeout(() => {
      pendingBots.delete(key);
      const current = rooms.get(room.code);
      if (current === undefined) return;
      if (current.game.phase !== 'trick_review') return;
      if (!allSeatsAreBots(current)) return;
      const prevGame = current.game;
      const r = applyToRoom(current, { kind: 'advance_trick' });
      if (!r.ok) return;
      r.room.log = current.log;
      logEventEffect(r.room, prevGame, r.room.game, { kind: 'advance_trick' });
      rooms.set(current.code, r.room);
      broadcastRoom(r.room);
    }, ALL_BOT_TRICK_REVIEW_MS);
    return;
  }
  // During the 'dealing' phase, bots act on their revealed-prefix
  // hand — strong shapes declare immediately, singles wait — and the
  // deal-driver loop emits `reveal_card` events on a timer. No
  // suppression needed.
  const botSeats = findBotSeats(room);
  for (const s of botSeats) {
    const key = `${room.code}:${s}`;
    if (pendingBots.has(key)) {
      console.log(`[bot ${s}] skipped: already pending`);
      continue;
    }
    const action = pickBotAction(room.game, s, botTypeForSeat(room, s));
    console.log(`[bot ${s}] action=${action.kind}`);
    if (action.kind === 'wait') continue;
    pendingBots.add(key);
    setTimeout(
      () => {
        pendingBots.delete(key);
        const current = rooms.get(room.code);
        if (current === undefined) return;
        const act = pickBotAction(current.game, s, botTypeForSeat(current, s));
        console.log(`[bot ${s}] timer fired, action=${act.kind}`);
        if (act.kind === 'wait') {
          // State moved on between scheduling and firing. Make sure any
          // *other* bot whose turn it is now gets scheduled.
          scheduleBotActions(current);
          return;
        }
        const event = botActionToEvent(act, s);
        if (event === null) return;
        const prevGame = current.game;
        const r = applyToRoom(current, event);
        if (!r.ok) {
          console.error(
            `[bot ${s}] play rejected: ${r.error}`,
            'event=', JSON.stringify(event),
          );
          return;
        }
        logEventEffect(
          r.room,
          prevGame,
          r.room.game,
          {
            kind: event.kind,
            seat: s,
            cards: 'cards' in event ? event.cards : undefined,
          },
          r.notice,
        );
        rooms.set(current.code, r.room);
        broadcastRoom(r.room);
      },
      500 + Math.random() * 500,
    );
  }
}

function botActionToEvent(
  action: BotAction,
  seat: Seat,
): Parameters<typeof applyEvent>[1] | null {
  switch (action.kind) {
    case 'declare':
      return { kind: 'declare', seat, cards: action.cards };
    case 'pick_up_kitty':
      return { kind: 'pick_up_kitty', seat };
    case 'replace_kitty':
      return { kind: 'replace_kitty', seat, cards: action.cards };
    case 'play_cards':
      return { kind: 'play_cards', seat, cards: action.cards };
    case 'wait':
      return null;
  }
}

function send(clientId: string, msg: ServerMessage): void {
  const ws = sockets.get(clientId);
  if (ws === undefined) return;
  if (ws.readyState !== ws.OPEN) return;
  ws.send(encode(msg));
}

function broadcastRoom(room: Room): void {
  const view = roomView(room, clients);
  const spectateHands = collectAllHands(room);
  for (let s = 0; s < room.game.config.numPlayers; s++) {
    const occ = room.seating[s];
    if (occ === null || occ === undefined) continue;
    if (occ.kind !== 'human') continue;
    send(occ.clientId, { kind: 'room_state', room: view });
    const client = clients.get(occ.clientId);
    if (client !== undefined) {
      send(occ.clientId, { kind: 'private_state', state: privateStateFor(room, client) });
      if (client.spectating) {
        send(occ.clientId, { kind: 'spectate_state', hands: spectateHands });
      }
    }
  }
  // Also notify spectators (clients in the room but not seated).
  for (const c of clients.values()) {
    if (c.roomCode !== room.code) continue;
    if (c.seat !== null) continue;
    send(c.id, { kind: 'room_state', room: view });
    send(c.id, { kind: 'private_state', state: null });
    if (c.spectating) {
      send(c.id, { kind: 'spectate_state', hands: spectateHands });
    }
  }
  // After every state broadcast, schedule any pending bot actions.
  scheduleBotActions(room);
}

// Snapshot of every seat's cards. Used to power spectate mode. During
// the 'dealing' phase, only reveal as many cards per seat as the engine
// has dealt so far (cardsDealtBySeat), so spectators see the deal
// animate one card at a time just like regular players.
function collectAllHands(
  room: Room,
): Readonly<Record<Seat, ReadonlyArray<Card>>> {
  const ch = room.game.currentHand;
  const out: Record<Seat, ReadonlyArray<Card>> = {};
  const dealing = room.game.phase === 'dealing';
  for (let s = 0; s < room.game.config.numPlayers; s++) {
    if (ch === null) {
      out[s] = [];
      continue;
    }
    const full = ch.hands[s] ?? [];
    if (dealing) {
      const dealt = ch.cardsDealtBySeat[s] ?? 0;
      out[s] = full.slice(0, dealt);
    } else {
      out[s] = [...full];
    }
  }
  return out;
}

function getOrCreateRoom(code: string): Room {
  let room = rooms.get(code);
  if (room === undefined) {
    room = newRoom(code);
    rooms.set(code, room);
  }
  return room;
}

function handleClientMessage(clientId: string, msg: ClientMessage): void {
  const client = clients.get(clientId);
  if (client === undefined) return;

  switch (msg.kind) {
    case 'join': {
      const token = msg.sessionToken;
      if (token !== undefined && token.length > 0) {
        if (!sessions.has(token)) {
          sessions.set(token, clientId);
        }
        client.sessionToken = token;
      }
      // Only vacate the previous room if we're switching rooms. Rejoining
      // the same room (e.g., after a refresh that already migrated us via
      // session token or nickname match) must preserve the existing seat.
      if (client.roomCode !== null && client.roomCode !== msg.roomCode) {
        leaveRoom(clientId);
      }
      client.nickname = msg.nickname.slice(0, 32) || 'guest';
      client.roomCode = msg.roomCode;
      const room = getOrCreateRoom(msg.roomCode);
      send(clientId, { kind: 'welcome', clientId });
      broadcastRoom(room);
      return;
    }
    case 'leave': {
      leaveRoom(clientId);
      return;
    }
    case 'take_seat': {
      if (client.roomCode === null) {
        send(clientId, { kind: 'error', error: 'not in a room' });
        return;
      }
      const room = rooms.get(client.roomCode);
      if (room === undefined) return;
      if (room.seating[msg.seat] !== null) {
        send(clientId, { kind: 'error', error: 'seat taken' });
        return;
      }
      const prev = findClientSeat(room, clientId);
      if (prev !== null) room.seating[prev] = null;
      room.seating[msg.seat] = { kind: 'human', clientId };
      client.seat = msg.seat;
      broadcastRoom(room);
      return;
    }
    case 'leave_seat': {
      if (client.roomCode === null) return;
      const room = rooms.get(client.roomCode);
      if (room === undefined) return;
      const prev = findClientSeat(room, clientId);
      if (prev !== null) {
        room.seating[prev] = null;
        client.seat = null;
        broadcastRoom(room);
      }
      return;
    }
    case 'add_bot': {
      const room = mustRoom(client);
      if (room === null) return;
      if (room.seating[msg.seat] !== null) {
        send(clientId, { kind: 'error', error: 'seat taken' });
        return;
      }
      const botType = msg.botType ?? 'lowball';
      room.seating[msg.seat] = {
        kind: 'bot',
        name: `Bot ${msg.seat} (${botType})`,
        botType,
      };
      broadcastRoom(room);
      return;
    }
    case 'remove_bot': {
      const room = mustRoom(client);
      if (room === null) return;
      const occ = room.seating[msg.seat];
      if (occ !== null && occ !== undefined && occ.kind === 'bot') {
        room.seating[msg.seat] = null;
        broadcastRoom(room);
      }
      return;
    }
    case 'start_hand': {
      const room = mustRoom(client);
      if (room === null) return;
      for (let s = 0; s < room.game.config.numPlayers; s++) {
        if (room.seating[s] === null) {
          send(clientId, {
            kind: 'error',
            error: `seat ${s} is empty — fill with a player or bot first`,
          });
          return;
        }
      }
      const seed = nextSeed();
      const prevGame = room.game;
      const r = applyToRoom(room, { kind: 'start_hand', seed });
      if (!r.ok) {
        send(clientId, { kind: 'error', error: r.error });
        return;
      }
      // Carry log forward to the new room object.
      r.room.log = room.log;
      logEventEffect(r.room, prevGame, r.room.game, { kind: 'start_hand' });
      rooms.set(room.code, r.room);
      broadcastRoom(r.room);
      // Drive incremental deal → finish_dealing → release_lockout.
      driveIncrementalDeal(room.code);
      return;
    }
    case 'declare': {
      forwardSeatEvent(clientId, (seat) => ({
        kind: 'declare',
        seat,
        cards: msg.cards,
      }));
      return;
    }
    case 'release_lockout': {
      const room = mustRoom(client);
      if (room === null) return;
      const prevGame = room.game;
      const r = applyToRoom(room, { kind: 'release_lockout', seed: nextSeed() });
      if (!r.ok) {
        send(clientId, { kind: 'error', error: r.error });
        return;
      }
      logEventEffect(r.room, prevGame, r.room.game, { kind: 'release_lockout' });
      rooms.set(room.code, r.room);
      broadcastRoom(r.room);
      return;
    }
    case 'pick_up_kitty': {
      forwardSeatEvent(clientId, (seat) => ({ kind: 'pick_up_kitty', seat }));
      return;
    }
    case 'replace_kitty': {
      forwardSeatEvent(clientId, (seat) => ({
        kind: 'replace_kitty',
        seat,
        cards: msg.cards,
      }));
      return;
    }
    case 'play_cards': {
      forwardSeatEvent(clientId, (seat) => ({
        kind: 'play_cards',
        seat,
        cards: msg.cards,
      }));
      return;
    }
    case 'set_spectate': {
      client.spectating = msg.on;
      const room = mustRoom(client);
      if (room !== null) {
        send(clientId, {
          kind: 'spectate_state',
          hands: msg.on ? collectAllHands(room) : null,
        });
      } else {
        send(clientId, { kind: 'spectate_state', hands: null });
      }
      return;
    }
    case 'advance_trick': {
      const room = mustRoom(client);
      if (room === null) return;
      const prevGame = room.game;
      const r = applyToRoom(room, { kind: 'advance_trick' });
      if (!r.ok) {
        send(clientId, { kind: 'error', error: r.error });
        return;
      }
      logEventEffect(r.room, prevGame, r.room.game, { kind: 'advance_trick' });
      rooms.set(room.code, r.room);
      broadcastRoom(r.room);
      return;
    }
    case 'reset_game': {
      const room = mustRoom(client);
      if (room === null) return;
      const fresh = {
        ...room,
        game: createGame({ ...defaultV1Config(), incrementalDeal: true }),
        log: room.log,
      };
      appendLog(fresh, '🔄 Game reset — starting fresh.');
      rooms.set(room.code, fresh);
      broadcastRoom(fresh);
      return;
    }
    case 'chat_message': {
      const room = mustRoom(client);
      if (room === null) return;
      const text = msg.text.slice(0, 300).trim();
      if (text === '') return;
      appendLog(room, text, client.nickname);
      broadcastRoom(room);
      return;
    }
  }
}

function mustRoom(client: Client): Room | null {
  if (client.roomCode === null) return null;
  return rooms.get(client.roomCode) ?? null;
}

function forwardSeatEvent(
  clientId: string,
  build: (seat: Seat) => Parameters<typeof applyToRoom>[1],
): void {
  const client = clients.get(clientId);
  if (client === undefined) return;
  if (client.roomCode === null) {
    send(clientId, { kind: 'error', error: 'not in a room' });
    return;
  }
  const room = rooms.get(client.roomCode);
  if (room === undefined) return;
  const seat = findClientSeat(room, clientId);
  if (seat === null) {
    send(clientId, { kind: 'error', error: 'not seated' });
    return;
  }
  const ev = build(seat);
  const prevGame = room.game;
  const r = applyToRoom(room, ev);
  if (!r.ok) {
    send(clientId, { kind: 'error', error: r.error });
    return;
  }
  logEventEffect(
    r.room,
    prevGame,
    r.room.game,
    {
      kind: ev.kind,
      seat,
      cards: 'cards' in ev ? ev.cards : undefined,
    },
    r.notice,
  );
  rooms.set(room.code, r.room);
  broadcastRoom(r.room);
}

function leaveRoom(clientId: string): void {
  const client = clients.get(clientId);
  if (client === undefined) return;
  if (client.roomCode === null) return;
  const room = rooms.get(client.roomCode);
  if (room !== undefined) {
    const seat = findClientSeat(room, clientId);
    if (seat !== null) room.seating[seat] = null;
    broadcastRoom(room);
  }
  client.roomCode = null;
  client.seat = null;
}

const http = createServer((req, res) => {
  if (req.url === '/health') {
    res.writeHead(200, { 'content-type': 'text/plain' });
    res.end('ok');
    return;
  }
  res.writeHead(404);
  res.end();
});

const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS ?? '')
  .split(',')
  .map((s) => s.trim())
  .filter((s) => s.length > 0);

const wss = new WebSocketServer({
  server: http,
  path: '/ws',
  verifyClient: ({ origin }, cb) => {
    if (ALLOWED_ORIGINS.length === 0) return cb(true);
    if (origin === undefined || origin === '') return cb(true);
    if (ALLOWED_ORIGINS.includes(origin)) return cb(true);
    cb(false, 403, 'origin not allowed');
  },
});

wss.on('connection', (socket) => {
  // ctx.clientId can be reassigned if the first 'join' message includes a
  // known sessionToken — the socket then "becomes" the existing client.
  const ctx = { clientId: generateId() };
  clients.set(ctx.clientId, {
    id: ctx.clientId,
    nickname: 'guest',
    roomCode: null,
    seat: null,
    connected: true,
    sessionToken: null,
    spectating: false,
  });
  sockets.set(ctx.clientId, socket);

  socket.on('message', (raw) => {
    let msg;
    try {
      msg = decodeClient(String(raw));
    } catch (e) {
      send(ctx.clientId, { kind: 'error', error: String(e) });
      return;
    }

    // On the first 'join', attempt to migrate this socket onto an
    // existing client identity in priority order:
    //   1. Matching sessionToken (same browser/tab reconnect).
    //   2. Matching nickname in the same room (cross-browser rejoin —
    //      "usernames are unique accounts").
    if (msg.kind === 'join') {
      let existingId: string | undefined;

      const token = msg.sessionToken;
      if (token !== undefined && token.length > 0) {
        const candidate = sessions.get(token);
        if (candidate !== undefined && candidate !== ctx.clientId) {
          existingId = candidate;
        }
      }

      if (existingId === undefined) {
        const wantedName = msg.nickname.trim().toLowerCase();
        if (wantedName.length > 0) {
          for (const c of clients.values()) {
            if (c.id === ctx.clientId) continue;
            if (c.roomCode !== msg.roomCode) continue;
            if (c.nickname.trim().toLowerCase() === wantedName) {
              existingId = c.id;
              break;
            }
          }
        }
      }

      if (existingId !== undefined) {
        // Kick the prior socket (if any) so it doesn't keep sending events.
        const oldSocket = sockets.get(existingId);
        if (oldSocket !== undefined && oldSocket !== socket) {
          try {
            oldSocket.close(1000, 'replaced');
          } catch {
            // ignore
          }
        }
        const tempId = ctx.clientId;
        clients.delete(tempId);
        sockets.delete(tempId);
        sockets.set(existingId, socket);
        ctx.clientId = existingId;
        const existing = clients.get(existingId);
        if (existing !== undefined) {
          existing.connected = true;
        }
        const t = disconnectTimers.get(existingId);
        if (t !== undefined) {
          clearTimeout(t);
          disconnectTimers.delete(existingId);
        }
      }
    }

    try {
      handleClientMessage(ctx.clientId, msg);
    } catch (e) {
      send(ctx.clientId, { kind: 'error', error: String(e) });
    }
  });

  socket.on('close', () => {
    const c = clients.get(ctx.clientId);
    if (c === undefined) return;
    // If a newer socket has already taken over this client identity, the
    // identity is still in use — don't mark disconnected or schedule cleanup.
    if (sockets.get(ctx.clientId) !== socket) return;
    c.connected = false;
    sockets.delete(ctx.clientId);
    if (c.roomCode !== null) {
      const room = rooms.get(c.roomCode);
      if (room !== undefined) broadcastRoom(room);
    }
    // Schedule full cleanup after the grace period.
    const id = ctx.clientId;
    const timer = setTimeout(() => {
      const stillThere = clients.get(id);
      if (stillThere === undefined) return;
      if (stillThere.connected) return;
      // Free the seat if still held.
      if (stillThere.roomCode !== null) {
        const room = rooms.get(stillThere.roomCode);
        if (room !== undefined) {
          const seat = findClientSeat(room, id);
          if (seat !== null) room.seating[seat] = null;
          broadcastRoom(room);
        }
      }
      if (stillThere.sessionToken !== null) {
        sessions.delete(stillThere.sessionToken);
      }
      clients.delete(id);
      disconnectTimers.delete(id);
    }, DISCONNECT_GRACE_MS);
    disconnectTimers.set(ctx.clientId, timer);
  });
});

http.listen(PORT, () => {
  console.log(`tractor ws server listening on :${PORT}`);
});

function generateId(): string {
  return (
    Math.random().toString(36).slice(2, 10) +
    Math.random().toString(36).slice(2, 10)
  );
}
