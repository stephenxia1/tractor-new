import type { BotType, Card, GameState, Seat } from '@tractor/engine';

// Timing constants shared by client and server.
// Cards are dealt one at a time, one player at a time (seat 0 fully then
// seat 1 fully, etc.). Each card arrival is `DEAL_INTERVAL_MS` apart.
// The server uses the same constants to delay its `release_lockout` timer
// until the dealing animation finishes on the client.
export const DEAL_INTERVAL_MS = 150;
export const POST_DEAL_LOCKOUT_MS = 3000;

// ---------------------------------------------------------------------------
// Messages from client → server
// ---------------------------------------------------------------------------
export type ClientMessage =
  | {
      kind: 'join';
      nickname: string;
      roomCode: string;
      sessionToken?: string;
    }
  | { kind: 'leave' }
  | { kind: 'chat_message'; text: string }
  | { kind: 'take_seat'; seat: Seat }
  | { kind: 'leave_seat' }
  | { kind: 'add_bot'; seat: Seat; botType?: BotType }
  | { kind: 'remove_bot'; seat: Seat }
  | { kind: 'start_hand' }
  | { kind: 'declare'; cards: ReadonlyArray<Card> }
  | { kind: 'release_lockout' } // dev/debug; normally the server times this
  | { kind: 'pick_up_kitty' }
  | { kind: 'replace_kitty'; cards: ReadonlyArray<Card> }
  | { kind: 'play_cards'; cards: ReadonlyArray<Card> }
  | { kind: 'advance_trick' }
  | { kind: 'reset_game' }
  | { kind: 'set_spectate'; on: boolean };

// ---------------------------------------------------------------------------
// Messages from server → client
// ---------------------------------------------------------------------------

// Public room view sent to all clients. Each client also gets their private
// hand (cards in their seat) separately via `PrivateState`.
export type RoomView = {
  readonly roomCode: string;
  readonly seats: ReadonlyArray<SeatInfo>;
  readonly gameState: GameStateView;
  readonly log: ReadonlyArray<LogEntry>;
};

export type LogEntry = {
  readonly ts: number;
  readonly text: string;
  // Author nickname for player chat. Absent for system/game events.
  readonly from?: string;
};

export type SeatInfo = {
  readonly seat: Seat;
  readonly occupant: PlayerInfo | null;
  readonly isBot: boolean;
  // Strategy this seat's bot uses. Absent for human seats / empty seats.
  readonly botType?: BotType;
};

export type PlayerInfo = {
  readonly clientId: string;
  readonly nickname: string;
  readonly connected: boolean;
};

// Server-side state minus other players' hands (which are redacted client-side).
// The engine's GameState is sent in full to the server but the server redacts
// before sending to clients.
export type GameStateView = Omit<GameState, 'currentHand'> & {
  readonly currentHand: HandStateView | null;
};

export type HandStateView = {
  readonly handNumber: number;
  readonly trumpRank: GameState['currentHand'] extends infer H
    ? H extends { trumpRank: infer R }
      ? R
      : never
    : never;
  readonly trumpSuit: GameState['currentHand'] extends infer H
    ? H extends { trumpSuit: infer S }
      ? S
      : never
    : never;
  readonly trumpFinalized: boolean;
  readonly landlord: Seat | null;
  // Per-seat card counts only (not the cards themselves).
  readonly handSizes: Readonly<Record<Seat, number>>;
  readonly kittySize: number;
  readonly standingDeclaration: NonNullable<
    GameState['currentHand']
  >['standingDeclaration'];
  readonly tricks: NonNullable<GameState['currentHand']>['tricks'];
  readonly currentTrick: NonNullable<GameState['currentHand']>['currentTrick'];
  readonly attackerPoints: number;
  // Revealed publicly only after the hand has ended (between_hands /
  // game_over) so the kitty bonus can be displayed to everyone.
  readonly kittyCards?: ReadonlyArray<Card>;
};

// Private state sent only to the client at a given seat.
export type PrivateState = {
  readonly seat: Seat;
  readonly cards: ReadonlyArray<Card>;
  // The 8 cards this client buried in the kitty, sent only to the landlord
  // once their burial is committed. Null/absent otherwise.
  readonly buriedKitty?: ReadonlyArray<Card>;
};

export type ServerMessage =
  | { kind: 'welcome'; clientId: string }
  | { kind: 'room_state'; room: RoomView }
  | { kind: 'private_state'; state: PrivateState | null }
  | { kind: 'spectate_state'; hands: Readonly<Record<Seat, ReadonlyArray<Card>>> | null }
  | { kind: 'chat'; from: string; text: string }
  | { kind: 'error'; error: string };

// ---------------------------------------------------------------------------
// Wire helpers
// ---------------------------------------------------------------------------
export function encode(msg: ClientMessage | ServerMessage): string {
  return JSON.stringify(msg);
}

export function decodeClient(s: string): ClientMessage {
  return JSON.parse(s) as ClientMessage;
}

export function decodeServer(s: string): ServerMessage {
  return JSON.parse(s) as ServerMessage;
}
