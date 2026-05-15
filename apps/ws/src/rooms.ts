import {
  applyEvent,
  createGame,
  defaultV1Config,
  type ApplyNotice,
  type BotType,
  type GameState,
  type Seat,
} from '@tractor/engine';
import type {
  GameStateView,
  HandStateView,
  PlayerInfo,
  PrivateState,
  RoomView,
  SeatInfo,
} from '@tractor/protocol';

export type Client = {
  readonly id: string;
  nickname: string;
  roomCode: string | null;
  seat: Seat | null;
  connected: boolean;
  sessionToken: string | null;
  // Spectate mode: when true, the server also pushes `spectate_state`
  // with every player's hand to this client on each broadcast.
  spectating: boolean;
};

// A seat can be occupied by a human (clientId), a bot, or empty.
export type SeatOccupant =
  | { readonly kind: 'human'; readonly clientId: string }
  | { readonly kind: 'bot'; readonly name: string; readonly botType: BotType }
  | null;

export type Room = {
  readonly code: string;
  game: GameState;
  seating: Record<Seat, SeatOccupant>;
  log: Array<{ ts: number; text: string; from?: string }>;
  // Timestamp until which bots should NOT act — used to keep them from
  // declaring before the client-side deal animation has revealed all cards.
  dealingEndsAt?: number;
};

export function newRoom(code: string): Room {
  const game = createGame({ ...defaultV1Config(), incrementalDeal: true });
  const seating: Record<Seat, SeatOccupant> = {};
  for (let s = 0; s < game.config.numPlayers; s++) seating[s] = null;
  return { code, game, seating, log: [] };
}

const LOG_MAX = 250;

export function appendLog(room: Room, text: string, from?: string): void {
  const entry: { ts: number; text: string; from?: string } = {
    ts: Date.now(),
    text,
  };
  if (from !== undefined) entry.from = from;
  room.log.push(entry);
  if (room.log.length > LOG_MAX) {
    room.log.splice(0, room.log.length - LOG_MAX);
  }
}

export function roomView(room: Room, clients: ReadonlyMap<string, Client>): RoomView {
  const seats: SeatInfo[] = [];
  for (let s = 0; s < room.game.config.numPlayers; s++) {
    const occ = room.seating[s];
    let occupant: PlayerInfo | null = null;
    let isBot = false;
    if (occ !== null && occ !== undefined) {
      if (occ.kind === 'human') {
        const c = clients.get(occ.clientId);
        if (c !== undefined) {
          occupant = { clientId: c.id, nickname: c.nickname, connected: c.connected };
        }
      } else {
        occupant = { clientId: `bot-${s}`, nickname: occ.name, connected: true };
        isBot = true;
      }
    }
    const info: SeatInfo =
      occ !== null && occ !== undefined && occ.kind === 'bot'
        ? { seat: s, occupant, isBot, botType: occ.botType }
        : { seat: s, occupant, isBot };
    seats.push(info);
  }
  return {
    roomCode: room.code,
    seats,
    gameState: redactGameState(room.game),
    log: room.log,
  };
}

// Strips private hand contents from the game state. Each client gets a
// `private_state` message with their own cards separately.
export function redactGameState(game: GameState): GameStateView {
  if (game.currentHand === null) {
    return { ...game, currentHand: null };
  }
  const ch = game.currentHand;
  // During the 'dealing' phase, opponents' hand sizes grow as cards are
  // revealed. Everywhere else, `cardsDealtBySeat` equals `hands[s].length`
  // and the two are interchangeable.
  const isDealing = game.phase === 'dealing';
  const handSizes: Record<Seat, number> = {};
  for (const [seatStr, cards] of Object.entries(ch.hands)) {
    const seat = Number(seatStr);
    handSizes[seat] = isDealing
      ? (ch.cardsDealtBySeat[seat] ?? 0)
      : cards.length;
  }
  const view: HandStateView = {
    handNumber: ch.handNumber,
    trumpRank: ch.trumpRank,
    trumpSuit: ch.trumpSuit,
    trumpFinalized: ch.trumpFinalized,
    landlord: ch.landlord,
    handSizes,
    kittySize: ch.kitty.length,
    standingDeclaration: ch.standingDeclaration,
    tricks: ch.tricks,
    currentTrick: ch.currentTrick,
    attackerPoints: ch.attackerPoints,
  };
  // Reveal kitty contents once the hand is over so the points panel can
  // attribute the kitty bonus visibly.
  if (game.phase === 'between_hands' || game.phase === 'game_over') {
    return { ...game, currentHand: { ...view, kittyCards: [...ch.kitty] } };
  }
  return { ...game, currentHand: view };
}

export function privateStateFor(
  room: Room,
  client: Client,
): PrivateState | null {
  if (client.seat === null) return null;
  if (room.game.currentHand === null) {
    return { seat: client.seat, cards: [] };
  }
  const ch = room.game.currentHand;
  // During the 'dealing' phase, only show the revealed prefix of the
  // seat's hand. Past dealing, the full hand is the seat's own cards.
  const isDealing = room.game.phase === 'dealing';
  const fullCards = ch.hands[client.seat] ?? [];
  const cards = isDealing
    ? fullCards.slice(0, ch.cardsDealtBySeat[client.seat] ?? 0)
    : fullCards;
  const out: PrivateState = { seat: client.seat, cards: [...cards] };
  // Show the buried kitty to the landlord once they've committed the
  // burial — i.e., past kitty_replace, while the hand is still active.
  const phase = room.game.phase;
  const past = phase === 'play' || phase === 'trick_review';
  if (past && ch.landlord === client.seat && ch.kitty.length > 0) {
    return { ...out, buriedKitty: [...ch.kitty] };
  }
  return out;
}

export function applyToRoom(
  room: Room,
  event: Parameters<typeof applyEvent>[1],
):
  | { ok: true; room: Room; notice?: ApplyNotice }
  | { ok: false; error: string } {
  const res = applyEvent(room.game, event);
  if (!res.ok) return { ok: false, error: res.error };
  const out: { ok: true; room: Room; notice?: ApplyNotice } = {
    ok: true,
    room: { ...room, game: res.state },
  };
  if (res.notice !== undefined) out.notice = res.notice;
  return out;
}

export function findClientSeat(room: Room, clientId: string): Seat | null {
  for (let s = 0; s < room.game.config.numPlayers; s++) {
    const occ = room.seating[s];
    if (occ !== null && occ !== undefined && occ.kind === 'human' && occ.clientId === clientId) {
      return s;
    }
  }
  return null;
}

export function findBotSeats(room: Room): Seat[] {
  const out: Seat[] = [];
  for (let s = 0; s < room.game.config.numPlayers; s++) {
    const occ = room.seating[s];
    if (occ !== null && occ !== undefined && occ.kind === 'bot') out.push(s);
  }
  return out;
}

export function botTypeForSeat(room: Room, seat: Seat): BotType {
  const occ = room.seating[seat];
  if (occ !== null && occ !== undefined && occ.kind === 'bot') return occ.botType;
  return 'lowball';
}
