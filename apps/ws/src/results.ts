// Cumulative results: every finished hand at a table with a human is
// appended, one JSON line a hand, to RESULTS_PATH (a file on the Fly
// volume in production), and summarised per matchup for GET /stats.
//
// A hand's record names who sat where (a human by a hash of their
// browser's session token, never the token or nickname; a bot by its
// type), the landlord, the attackers' final points, the engine's hand
// result, and the levels before and after.
import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname } from 'node:path';
import { computeHandResult, type GameState, type Seat, type TeamId } from '@tractor/engine';
import type { MatchupStats, StatsSummary } from '@tractor/protocol';
import type { Room } from './rooms';

export type SeatRecord =
  | { readonly kind: 'human'; readonly player: string }
  | { readonly kind: 'bot'; readonly botType: string };

export type HandRecord = {
  readonly v: 1;
  readonly ts: number;
  readonly room: string;
  readonly hand: number;
  readonly seats: ReadonlyArray<SeatRecord>;
  readonly seatToTeam: Readonly<Record<Seat, TeamId>>;
  readonly landlord: Seat;
  readonly defenders: TeamId;
  readonly attackerPoints: number;
  readonly winner: 'defenders' | 'attackers';
  // The bracket's levels for the hand's winner (computeHandResult), before
  // any cap at A.
  readonly levelDelta: number;
  readonly levelsBefore: Readonly<Record<TeamId, string>>;
  readonly levelsAfter: Readonly<Record<TeamId, string>>;
  readonly gameOver: boolean;
  readonly gameWinner: TeamId | null;
  // True when anyone sat down or got up between the deal and the end.
  readonly seatingChanged: boolean;
};

// The attackers' threshold (rules.md §8.3; computeHandResult's 80).
const THRESHOLD = 80;

export function isHandStart(prev: GameState, next: GameState): boolean {
  return (
    (prev.phase === 'between_hands' || prev.phase === 'game_over') &&
    next.phase !== 'between_hands' &&
    next.phase !== 'game_over'
  );
}

export function isHandEnd(prev: GameState, next: GameState): boolean {
  return (
    prev.phase !== 'between_hands' &&
    prev.phase !== 'game_over' &&
    (next.phase === 'between_hands' || next.phase === 'game_over') &&
    next.currentHand !== null &&
    next.currentHand.landlord !== null
  );
}

// A human's player id: a short hash of their browser's session token, so
// the same browser counts once across visits without storing the token.
export function playerId(token: string): string {
  return createHash('sha256').update(`tractor-player:${token}`).digest('hex').slice(0, 12);
}

export function seatingKey(seats: ReadonlyArray<SeatRecord | null>): string {
  return seats
    .map((s) => (s === null ? '-' : s.kind === 'human' ? `h:${s.player}` : `b:${s.botType}`))
    .join('|');
}

export function buildHandRecord(args: {
  readonly room: string;
  readonly prev: GameState;
  readonly next: GameState;
  readonly seats: ReadonlyArray<SeatRecord>;
  readonly seatingChanged: boolean;
  readonly now: number;
}): HandRecord {
  const { prev, next } = args;
  const hand = next.currentHand!;
  const landlord = hand.landlord!;
  const defenders = next.config.seatToTeam[landlord]!;
  const result = computeHandResult(hand.attackerPoints);
  return {
    v: 1,
    ts: args.now,
    room: args.room,
    hand: hand.handNumber,
    seats: args.seats,
    seatToTeam: next.config.seatToTeam,
    landlord,
    defenders,
    attackerPoints: hand.attackerPoints,
    winner: result.winner,
    levelDelta: result.levelDelta,
    levelsBefore: prev.teamLevels,
    levelsAfter: next.teamLevels,
    gameOver: next.phase === 'game_over',
    gameWinner: next.winner,
    seatingChanged: args.seatingChanged,
  };
}

// The observer for rooms.setApplyObserver: notes who sat where at each
// deal, and appends a record when a hand at a table with a human ends
// (all-bot tables, and hands ending with an empty seat, are skipped).
export function handRecorder(
  store: ResultsStore,
  seatsOf: (room: Room) => Array<SeatRecord | null>,
  now: () => number = Date.now,
): (before: Room, after: Room) => void {
  return (before, after) => {
    if (isHandStart(before.game, after.game)) {
      after.handStartSeating = seatingKey(seatsOf(after));
      return;
    }
    if (!isHandEnd(before.game, after.game)) return;
    const seats = seatsOf(after);
    if (!seats.some((s) => s?.kind === 'human')) return;
    if (seats.some((s) => s === null)) return;
    store.append(
      buildHandRecord({
        room: after.code,
        prev: before.game,
        next: after.game,
        seats: seats as SeatRecord[],
        seatingChanged: after.handStartSeating !== seatingKey(seats),
        now: now(),
      }),
    );
  };
}

// --- summary ---------------------------------------------------------------

type Side = { readonly team: TeamId; readonly seats: ReadonlyArray<SeatRecord> };

function sides(r: HandRecord): Side[] {
  const byTeam = new Map<TeamId, SeatRecord[]>();
  r.seats.forEach((s, seat) => {
    const team = r.seatToTeam[seat];
    if (team === undefined) return;
    const list = byTeam.get(team) ?? [];
    list.push(s);
    byTeam.set(team, list);
  });
  return [...byTeam.entries()].map(([team, seats]) => ({ team, seats }));
}

function hasHuman(side: Side): boolean {
  return side.seats.some((s) => s.kind === 'human');
}

function humanSideLabel(side: Side): string {
  const humans = side.seats.filter((s) => s.kind === 'human').length;
  const bots = side.seats.flatMap((s) => (s.kind === 'bot' ? [s.botType] : [])).sort();
  if (bots.length === 0) return humans === 1 ? 'human' : `${humans} humans`;
  return `${humans === 1 ? 'human' : `${humans} humans`} + ${bots.join(' + ')}`;
}

function botSideLabel(side: Side): string {
  const bots = side.seats.flatMap((s) => (s.kind === 'bot' ? [s.botType] : [])).sort();
  const distinct = [...new Set(bots)];
  return distinct.length === 1 ? distinct[0]! : bots.join(' + ');
}

// The matchup a record counts in, from the human team's side; null when
// it counts in none (humans on both teams or neither, or seats changed).
export function matchupOf(
  r: HandRecord,
): { key: string; humanSide: string; botSide: string; humanTeam: TeamId } | null {
  if (r.seatingChanged) return null;
  const ss = sides(r);
  const human = ss.filter(hasHuman);
  const bots = ss.filter((s) => !hasHuman(s));
  if (human.length !== 1 || bots.length !== 1) return null;
  const h = human[0]!;
  const b = bots[0]!;
  const humanSide = humanSideLabel(h);
  const botSide = botSideLabel(b);
  return { key: `${humanSide} vs ${botSide}`, humanSide, botSide, humanTeam: h.team };
}

const Z = 1.96;

function wilson(wins: number, n: number): [number, number] {
  if (n === 0) return [0, 1];
  const p = wins / n;
  const z2 = Z * Z;
  const centre = (p + z2 / (2 * n)) / (1 + z2 / n);
  const half = (Z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / (1 + z2 / n);
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

function meanAndHalfWidth(xs: ReadonlyArray<number>): [number, number | null] {
  const n = xs.length;
  if (n === 0) return [0, null];
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  if (n < 2) return [mean, null];
  const variance = xs.reduce((a, x) => a + (x - mean) ** 2, 0) / (n - 1);
  return [mean, Z * Math.sqrt(variance / n)];
}

type Acc = {
  humanSide: string;
  botSide: string;
  players: Set<string>;
  wins: number;
  margins: number[];
  levels: number[];
  defending: { hands: number; wins: number };
  attacking: { hands: number; wins: number };
  games: number;
  gamesWon: number;
};

export function summarize(records: ReadonlyArray<HandRecord>, now: number): StatsSummary {
  const acc = new Map<string, Acc>();
  const allPlayers = new Set<string>();
  let matchupHands = 0;
  for (const r of records) {
    for (const s of r.seats) if (s.kind === 'human') allPlayers.add(s.player);
    const m = matchupOf(r);
    if (m === null) continue;
    matchupHands++;
    let a = acc.get(m.key);
    if (a === undefined) {
      a = {
        humanSide: m.humanSide,
        botSide: m.botSide,
        players: new Set(),
        wins: 0,
        margins: [],
        levels: [],
        defending: { hands: 0, wins: 0 },
        attacking: { hands: 0, wins: 0 },
        games: 0,
        gamesWon: 0,
      };
      acc.set(m.key, a);
    }
    r.seats.forEach((s, seat) => {
      if (s.kind === 'human' && r.seatToTeam[seat] === m.humanTeam) a!.players.add(s.player);
    });
    const defending = r.defenders === m.humanTeam;
    const winningTeam = r.winner === 'defenders' ? defending : !defending;
    if (winningTeam) a.wins++;
    const role = defending ? a.defending : a.attacking;
    role.hands++;
    if (winningTeam) role.wins++;
    a.margins.push(defending ? THRESHOLD - r.attackerPoints : r.attackerPoints - THRESHOLD);
    a.levels.push(winningTeam ? r.levelDelta : -r.levelDelta);
    if (r.gameOver) {
      a.games++;
      if (r.gameWinner === m.humanTeam) a.gamesWon++;
    }
  }
  const matchups: MatchupStats[] = [...acc.entries()].map(([key, a]) => {
    const hands = a.margins.length;
    const [low, high] = wilson(a.wins, hands);
    const [pointMargin, pointMarginHalfWidth] = meanAndHalfWidth(a.margins);
    const [levels, levelsHalfWidth] = meanAndHalfWidth(a.levels);
    return {
      key,
      humanSide: a.humanSide,
      botSide: a.botSide,
      hands,
      players: a.players.size,
      wins: a.wins,
      winRate: hands === 0 ? 0 : a.wins / hands,
      winRateLow: low,
      winRateHigh: high,
      pointMargin,
      pointMarginHalfWidth,
      levels,
      levelsHalfWidth,
      defending: a.defending,
      attacking: a.attacking,
      games: a.games,
      gamesWon: a.gamesWon,
    };
  });
  matchups.sort((x, y) => y.hands - x.hands || x.key.localeCompare(y.key));
  return {
    generatedAt: now,
    since: records.length > 0 ? records[0]!.ts : null,
    totalHands: records.length,
    matchupHands,
    players: allPlayers.size,
    matchups,
  };
}

// --- store -----------------------------------------------------------------

// Holds every record in memory and, given a path, appends each to it and
// reads the earlier ones back at startup. A write that fails is logged
// and the record kept in memory: results must never stop a game.
export class ResultsStore {
  private readonly records: HandRecord[] = [];
  private cached: StatsSummary | null = null;

  constructor(private readonly path: string | null) {
    if (path === null) return;
    try {
      mkdirSync(dirname(path), { recursive: true });
      if (existsSync(path)) {
        for (const line of readFileSync(path, 'utf8').split('\n')) {
          if (line.trim() === '') continue;
          try {
            const r = JSON.parse(line) as HandRecord;
            if (r.v === 1) this.records.push(r);
          } catch {
            // A torn last line (a crash mid-write) is skipped.
          }
        }
      }
    } catch (e) {
      console.error(`[results] could not read ${path}:`, e);
    }
  }

  get size(): number {
    return this.records.length;
  }

  append(record: HandRecord): void {
    this.records.push(record);
    this.cached = null;
    if (this.path === null) return;
    try {
      appendFileSync(this.path, `${JSON.stringify(record)}\n`);
    } catch (e) {
      console.error(`[results] could not append to ${this.path}:`, e);
    }
  }

  summary(now: number = Date.now()): StatsSummary {
    if (this.cached === null) this.cached = summarize(this.records, now);
    return this.cached;
  }
}
