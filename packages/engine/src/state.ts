import type {
  Card,
  JokerRank,
  NaturalRank,
  Suit,
} from './types';

export type Seat = number; // 0..numPlayers-1
export type TeamId = string; // 'A' | 'B' for v1

export type GameConfig = {
  readonly numPlayers: number;
  readonly numDecks: number;
  readonly kittySize: number;
  readonly teamIds: ReadonlyArray<TeamId>;
  // Initial seat → team mapping. Treated as the "fixed partnership" mapping
  // in v1. For future Finding Friends, this is the *initial* mapping and
  // may change per-hand as friend cards are played.
  readonly seatToTeam: Readonly<Record<Seat, TeamId>>;
  readonly startingLevel: NaturalRank;
};

// Standard v1 config: 4 players, 2 decks, 8 kitty, fixed N+S vs E+W partnership.
export function defaultV1Config(): GameConfig {
  return {
    numPlayers: 4,
    numDecks: 2,
    kittySize: 8,
    teamIds: ['A', 'B'],
    seatToTeam: { 0: 'A', 1: 'B', 2: 'A', 3: 'B' },
    startingLevel: '2',
  };
}

// A "declaration" of trump during the declaration window (rules.md §5).
export type Declaration =
  | {
      readonly kind: 'single';
      readonly seat: Seat;
      readonly suit: Suit;
      readonly cards: readonly [Card];
    }
  | {
      readonly kind: 'pair';
      readonly seat: Seat;
      readonly suit: Suit;
      readonly cards: readonly [Card, Card];
    }
  | {
      readonly kind: 'joker-pair';
      readonly seat: Seat;
      readonly jokerRank: JokerRank;
      readonly cards: readonly [Card, Card];
    };

// Tier of a declaration, weakest → strongest. A higher tier overrides any
// lower tier (rules.md §5.1).
export function declarationTier(d: Declaration): number {
  if (d.kind === 'single') return 1;
  if (d.kind === 'pair') return 2;
  return d.jokerRank === 'small' ? 3 : 4;
}

// One play within a trick: the seat that played and the cards they played
// (in any order — the engine doesn't impose order on cards within a combo).
export type TrickPlay = {
  readonly seat: Seat;
  readonly cards: ReadonlyArray<Card>;
};

// A trick still being played: leader and partial plays. The leader's play
// is always `plays[0]`.
export type ActiveTrick = {
  readonly leader: Seat;
  readonly plays: ReadonlyArray<TrickPlay>;
};

export type CompletedTrick = {
  readonly leader: Seat;
  readonly plays: ReadonlyArray<TrickPlay>;
  readonly winner: Seat;
  readonly points: number; // attacker-relevant points (only used by scoring)
};

// Phase of the overall game.
export type GamePhase =
  // No active hand. Initial state and the gap between hands.
  | 'between_hands'
  // Declarations open; landlord cannot pick up kitty yet (rules.md §5.3 lockout).
  | 'declare_lockout'
  // Declarations open; landlord may pick up the kitty at any time.
  | 'declare_open'
  // Landlord chooses 8 cards from their hand (33 cards) to set aside.
  | 'kitty_replace'
  // Tricks are being played.
  | 'play'
  // A trick has just completed; cards stay visible on the table until
  // someone advances to the next trick.
  | 'trick_review'
  // Game is over.
  | 'game_over';

// State of the current hand. Null when game is between hands.
export type HandState = {
  readonly handNumber: number;
  // The trump rank `R` for this hand. Null while landlord is unset entering
  // the hand and before any declaration; finalized when landlord is set.
  readonly trumpRank: NaturalRank | null;
  // The trump suit. Null can mean "no-trump (joker) declaration" — disambiguate
  // by checking phase or `trumpFinalized`.
  readonly trumpSuit: Suit | null;
  readonly trumpFinalized: boolean;
  // Landlord for THIS hand. Mirrors `GameState.landlord` once it's set, but
  // tracked separately because the landlord may be set mid-hand (during
  // declaration) when it was unset at hand start.
  readonly landlord: Seat | null;
  readonly hands: Readonly<Record<Seat, ReadonlyArray<Card>>>;
  readonly kitty: ReadonlyArray<Card>;
  readonly declarationLog: ReadonlyArray<Declaration>;
  readonly standingDeclaration: Declaration | null;
  readonly tricks: ReadonlyArray<CompletedTrick>;
  readonly currentTrick: ActiveTrick | null;
  readonly attackerPoints: number;
};

export type GameState = {
  readonly config: GameConfig;
  // Each team's current level.
  readonly teamLevels: Readonly<Record<TeamId, NaturalRank>>;
  // Landlord at the game level, persistent across hands. Null when unset
  // (initial state, or after attackers win a hand — rules.md §4.1, §8.3).
  readonly landlord: Seat | null;
  readonly phase: GamePhase;
  readonly currentHand: HandState | null;
  readonly winner: TeamId | null;
  readonly handsCompleted: number;
};

export function createGame(config: GameConfig = defaultV1Config()): GameState {
  validateConfig(config);
  const teamLevels: Record<TeamId, NaturalRank> = {};
  for (const team of config.teamIds) {
    teamLevels[team] = config.startingLevel;
  }
  return {
    config,
    teamLevels,
    landlord: null,
    phase: 'between_hands',
    currentHand: null,
    winner: null,
    handsCompleted: 0,
  };
}

function validateConfig(config: GameConfig): void {
  if (config.numPlayers < 2 || !Number.isInteger(config.numPlayers)) {
    throw new Error(`numPlayers must be an integer >= 2`);
  }
  if (config.numDecks < 1 || !Number.isInteger(config.numDecks)) {
    throw new Error(`numDecks must be an integer >= 1`);
  }
  if (config.kittySize < 0 || !Number.isInteger(config.kittySize)) {
    throw new Error(`kittySize must be a non-negative integer`);
  }
  const totalCards = config.numDecks * 54;
  const dealable = totalCards - config.kittySize;
  if (dealable <= 0 || dealable % config.numPlayers !== 0) {
    throw new Error(
      `(numDecks * 54 - kittySize) must be a positive multiple of numPlayers; ` +
        `got ${dealable} / ${config.numPlayers}`,
    );
  }
  for (let s = 0; s < config.numPlayers; s++) {
    if (!(s in config.seatToTeam)) {
      throw new Error(`seatToTeam missing seat ${s}`);
    }
    const team = config.seatToTeam[s]!;
    if (!config.teamIds.includes(team)) {
      throw new Error(`seatToTeam[${s}] = ${team} not in teamIds`);
    }
  }
}

export function teamOf(state: GameState, seat: Seat): TeamId {
  const team = state.config.seatToTeam[seat];
  if (team === undefined) throw new Error(`seat ${seat} has no team`);
  return team;
}

export function partnerSeats(state: GameState, seat: Seat): Seat[] {
  const team = teamOf(state, seat);
  const out: Seat[] = [];
  for (let s = 0; s < state.config.numPlayers; s++) {
    if (s !== seat && state.config.seatToTeam[s] === team) out.push(s);
  }
  return out;
}
