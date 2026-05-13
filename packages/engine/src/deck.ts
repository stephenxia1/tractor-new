import {
  JOKER_RANKS,
  NATURAL_RANKS,
  SUITS,
  type Card,
} from './types';

// Builds `numDecks * 54` cards (52 standard + 2 jokers per deck) in a
// deterministic order. Callers should shuffle before dealing.
export function buildDeck(numDecks: number): Card[] {
  if (numDecks < 1 || !Number.isInteger(numDecks)) {
    throw new Error(`numDecks must be a positive integer, got ${numDecks}`);
  }
  const cards: Card[] = [];
  for (let d = 0; d < numDecks; d++) {
    for (const suit of SUITS) {
      for (const rank of NATURAL_RANKS) {
        cards.push({ kind: 'standard', suit, rank, deckId: d });
      }
    }
    for (const rank of JOKER_RANKS) {
      cards.push({ kind: 'joker', rank, deckId: d });
    }
  }
  return cards;
}
