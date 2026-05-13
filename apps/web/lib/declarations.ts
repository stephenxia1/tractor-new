import type {
  Card,
  Declaration,
  JokerRank,
  NaturalRank,
  Seat,
  Suit,
} from '@tractor/engine';

const SUIT_SYMBOL: Record<Suit, string> = {
  spades: '♠',
  hearts: '♥',
  diamonds: '♦',
  clubs: '♣',
};

const RED_SUITS = new Set<Suit>(['hearts', 'diamonds']);

// A potential declaration a player can make from their hand. Same shape
// as engine `Declaration` but without `seat` (the server sets seat from
// the connection).
export type AvailableDeclaration =
  | { kind: 'single'; suit: Suit; cards: [Card] }
  | { kind: 'pair'; suit: Suit; cards: [Card, Card] }
  | { kind: 'joker-pair'; jokerRank: JokerRank; cards: [Card, Card] };

function tier(d: AvailableDeclaration): number {
  if (d.kind === 'single') return 1;
  if (d.kind === 'pair') return 2;
  return d.jokerRank === 'small' ? 3 : 4;
}

function standingTier(d: Declaration | null): number {
  if (d === null) return 0;
  if (d.kind === 'single') return 1;
  if (d.kind === 'pair') return 2;
  return d.jokerRank === 'small' ? 3 : 4;
}

// Returns every declaration the player can legally make from their hand,
// given the trump rank R and the currently-standing declaration. Tiers
// at or below the standing declaration are filtered out (rules.md §5.1).
export function availableDeclarations(
  myCards: ReadonlyArray<Card>,
  R: NaturalRank,
  standing: Declaration | null,
): AvailableDeclaration[] {
  const out: AvailableDeclaration[] = [];

  // Bucket rank-R standard cards by suit.
  const bySuit = new Map<Suit, Card[]>();
  for (const c of myCards) {
    if (c.kind === 'standard' && c.rank === R) {
      const list = bySuit.get(c.suit) ?? [];
      list.push(c);
      bySuit.set(c.suit, list);
    }
  }

  // Singles — one button per suit you hold rank-R in.
  for (const [suit, cards] of bySuit) {
    out.push({ kind: 'single', suit, cards: [cards[0]!] });
  }

  // Pairs — one button per suit where you hold both copies.
  for (const [suit, cards] of bySuit) {
    if (cards.length >= 2 && cards[0]!.deckId !== cards[1]!.deckId) {
      out.push({
        kind: 'pair',
        suit,
        cards: [cards[0]!, cards[1]!],
      });
    }
  }

  // Joker pairs.
  const smallJokers = myCards.filter(
    (c) => c.kind === 'joker' && c.rank === 'small',
  );
  if (smallJokers.length >= 2 && smallJokers[0]!.deckId !== smallJokers[1]!.deckId) {
    out.push({
      kind: 'joker-pair',
      jokerRank: 'small',
      cards: [smallJokers[0]!, smallJokers[1]!],
    });
  }
  const bigJokers = myCards.filter((c) => c.kind === 'joker' && c.rank === 'big');
  if (bigJokers.length >= 2 && bigJokers[0]!.deckId !== bigJokers[1]!.deckId) {
    out.push({
      kind: 'joker-pair',
      jokerRank: 'big',
      cards: [bigJokers[0]!, bigJokers[1]!],
    });
  }

  // Keep only declarations that would override the standing one.
  const minTier = standingTier(standing);
  return out.filter((d) => tier(d) > minTier);
}

// Trump rank R that this seat would declare under. If a landlord is set
// at hand level, R is fixed to the landlord's team's level. If unset,
// each player declares using their *own* team's level.
export function trumpRankForSeat(
  seat: Seat,
  game: {
    landlord: Seat | null;
    teamLevels: Readonly<Record<string, NaturalRank>>;
    config: { seatToTeam: Readonly<Record<Seat, string>> };
    currentHand: { trumpRank: NaturalRank | null } | null;
  },
): NaturalRank | null {
  if (game.landlord !== null && game.currentHand?.trumpRank) {
    return game.currentHand.trumpRank;
  }
  const team = game.config.seatToTeam[seat];
  if (team === undefined) return null;
  return game.teamLevels[team] ?? null;
}

export function declarationLabel(d: AvailableDeclaration): string {
  if (d.kind === 'single') {
    const c = d.cards[0];
    if (c.kind === 'standard') {
      return `Declare ${SUIT_SYMBOL[d.suit]} (${c.rank}${SUIT_SYMBOL[d.suit]})`;
    }
    return 'Declare';
  }
  if (d.kind === 'pair') {
    const c = d.cards[0];
    if (c.kind === 'standard') {
      return `Declare ${SUIT_SYMBOL[d.suit]} pair (${c.rank}${SUIT_SYMBOL[d.suit]} × 2)`;
    }
    return 'Declare pair';
  }
  return `Declare no-trump (${d.jokerRank} jokers ★)`;
}

export function declarationSuitClass(d: AvailableDeclaration): string {
  if (d.kind === 'joker-pair') return d.jokerRank === 'big' ? 'jbig' : 'jsm';
  return RED_SUITS.has(d.suit) ? 'red' : 'black';
}
