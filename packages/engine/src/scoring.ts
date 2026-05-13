import { NATURAL_RANKS, type Card, type NaturalRank } from './types';

// Point value of a card. Per rules.md §8.1: 5 = 5 pts, 10 = 10 pts, K = 10 pts.
// All other cards (jokers, other ranks) are 0 points.
export function pointsForCard(c: Card): number {
  if (c.kind === 'joker') return 0;
  if (c.rank === '5') return 5;
  if (c.rank === '10' || c.rank === 'K') return 10;
  return 0;
}

export function sumPoints(cards: ReadonlyArray<Card>): number {
  let total = 0;
  for (const c of cards) total += pointsForCard(c);
  return total;
}

// Kitty multiplier per rules.md §8.2: m = 2 * (cards in winning combo).
// `cardsInWinningCombo` is the total card count of the LAST trick's
// winning combo (e.g., 1 for a single, 2 for a pair, 4 for a length-2
// tractor, etc.).
export function kittyMultiplier(cardsInWinningCombo: number): number {
  return 2 * cardsInWinningCombo;
}

// Bracket-based level change per rules.md §8.3.
// `attackerPoints` is the attackers' final score (including any kitty
// bonus). Returns the side that "won" the hand (with the meaning that
// defenders win means they hold attackers under 80) and how many levels
// they gain.
export type HandResult = {
  readonly winner: 'defenders' | 'attackers';
  readonly levelDelta: number;
};

export function computeHandResult(attackerPoints: number): HandResult {
  if (attackerPoints === 0) return { winner: 'defenders', levelDelta: 3 };
  if (attackerPoints < 40) return { winner: 'defenders', levelDelta: 2 };
  if (attackerPoints < 80) return { winner: 'defenders', levelDelta: 1 };
  if (attackerPoints < 120) return { winner: 'attackers', levelDelta: 0 };
  if (attackerPoints < 160) return { winner: 'attackers', levelDelta: 1 };
  if (attackerPoints < 200) return { winner: 'attackers', levelDelta: 2 };
  return { winner: 'attackers', levelDelta: 3 };
}

// Advance a team's level by `delta` steps, capped at A.
export function advanceLevel(current: NaturalRank, delta: number): NaturalRank {
  const idx = NATURAL_RANKS.indexOf(current);
  const newIdx = Math.min(idx + delta, NATURAL_RANKS.length - 1);
  return NATURAL_RANKS[newIdx]!;
}
