import type { Card } from '@tractor/engine';

const SUIT_SYMBOL = {
  spades: '♠',
  hearts: '♥',
  diamonds: '♦',
  clubs: '♣',
} as const;

export function cardLabel(c: Card): string {
  if (c.kind === 'joker') return c.rank === 'big' ? 'BJ' : 'SJ';
  return `${c.rank}${SUIT_SYMBOL[c.suit]}`;
}

export function cardClassNames(c: Card): string {
  const classes = ['card'];
  if (c.kind === 'joker') {
    classes.push('joker');
    if (c.rank === 'big') classes.push('big');
  } else if (c.suit === 'hearts' || c.suit === 'diamonds') {
    classes.push('red');
  }
  return classes.join(' ');
}

// Stable key for a card (used for React `key`).
export function cardId(c: Card): string {
  if (c.kind === 'joker') return `joker-${c.rank}-${c.deckId}`;
  return `${c.suit}-${c.rank}-${c.deckId}`;
}
