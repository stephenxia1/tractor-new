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
// Cumulative results (GET /stats on the ws server, shown at /stats)
// ---------------------------------------------------------------------------

// One role's hands in a matchup (the human team defending, or attacking),
// from the human team's side. The two roles have different baselines --
// equal teams need not win half their hands in either -- so a role's
// figures say how the humans fare in it, not who is better.
export type RoleStats = {
  readonly hands: number;
  readonly wins: number;
  readonly winRate: number;
  // 95% Wilson interval on the win rate.
  readonly winRateLow: number;
  readonly winRateHigh: number;
  // Mean point margin a hand (80 - attacker points defending, attacker
  // points - 80 attacking) and the half-width of its 95% interval, null
  // below two hands.
  readonly pointMargin: number;
  readonly pointMarginHalfWidth: number | null;
  readonly levels: number;
  readonly levelsHalfWidth: number | null;
};

// The defending and attacking figures averaged with equal weight, so the
// share of hands a team spent defending (the winners keep the deal) does
// not move them: the research evaluations' points margin, (points taken -
// points conceded) / 2. Equal teams score 50% and 0 here. Half-widths are
// 95%, from the two roles' standard errors.
export type BalancedStats = {
  readonly winRate: number;
  readonly winRateHalfWidth: number;
  readonly pointMargin: number;
  readonly pointMarginHalfWidth: number;
  readonly levels: number;
  readonly levelsHalfWidth: number;
};

// Every finished hand at a table with a human is recorded. A matchup is
// one human team against one all-bot team, keyed by who sat where: for
// example "human + smart-v35" against "smart-v35". Figures are from the
// human team's side.
export type MatchupStats = {
  readonly key: string;
  // "2 humans", or "human + <partner bot type>".
  readonly humanSide: string;
  // The opposing bots: "<type>" when both are the same, else "<a> + <b>".
  readonly botSide: string;
  readonly hands: number;
  // Distinct human players (by browser) who played these hands.
  readonly players: number;
  // Hands the human team won: held the attackers under 80 as defenders,
  // or reached 80 as attackers.
  readonly wins: number;
  readonly winRate: number;
  // 95% Wilson interval on the win rate.
  readonly winRateLow: number;
  readonly winRateHigh: number;
  // Mean point margin a hand over all hands: attacker points - 80 when the
  // human team attacks, 80 - attacker points when it defends. It depends
  // on the share of hands spent defending; `balanced` does not.
  readonly pointMargin: number;
  // Half-width of its 95% interval; null below two hands.
  readonly pointMarginHalfWidth: number | null;
  // Mean levels a hand: the bracket's levels, + when the human team wins
  // the hand, - when the bots do.
  readonly levels: number;
  readonly levelsHalfWidth: number | null;
  // The same, split by the human team's role in the hand.
  readonly defending: RoleStats;
  readonly attacking: RoleStats;
  // Both roles weighted equally; null until each role has two hands.
  readonly balanced: BalancedStats | null;
  // Games played to the end (a team defended at A) and won by the humans.
  readonly games: number;
  readonly gamesWon: number;
};

export type StatsSummary = {
  readonly generatedAt: number;
  // Time of the first recorded hand, null before any.
  readonly since: number | null;
  // Every recorded hand (any table with a human).
  readonly totalHands: number;
  // Hands counted in a matchup: one team all human or human + bot, the
  // other all bots, seating unchanged through the hand.
  readonly matchupHands: number;
  readonly players: number;
  readonly matchups: ReadonlyArray<MatchupStats>;
};

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
