import { describe, expect, it } from 'vitest';
import {
  buildDeck,
  cardKey,
  classifyCombo,
  decomposeThrow,
  effectiveSuit,
  mulberry32,
  shuffle,
  validateFollow,
  validateFollowThrow,
  type Card,
  type NaturalRank,
  type Suit,
  type TrumpContext,
} from '@tractor/engine';
import { onlyLegalPlay } from './forcedPlay';

// Trump: hearts, level 2. Spades, clubs and diamonds are plain suits.
const CTX: TrumpContext = { trumpRank: '2', trumpSuit: 'hearts' };
const N = 2;

function c(suit: Suit, rank: NaturalRank, deckId = 0): Card {
  return { kind: 'standard', suit, rank, deckId };
}

function keys(cards: ReadonlyArray<Card> | null): string[] | null {
  return cards === null
    ? null
    : cards.map((x) => (x.kind === 'joker' ? `${x.rank}J` : `${x.rank}${x.suit[0]}`)).sort();
}

describe('onlyLegalPlay: leading', () => {
  it('forces only a last card', () => {
    const last = c('spades', 'K');
    expect(onlyLegalPlay([last], null, CTX, N)).toEqual([last]);
    expect(onlyLegalPlay([c('spades', 'K'), c('clubs', '3')], null, CTX, N)).toBeNull();
    // A pair can lead as a pair or as a single: two plays.
    expect(
      onlyLegalPlay([c('spades', 'K', 0), c('spades', 'K', 1)], null, CTX, N),
    ).toBeNull();
  });
});

describe('onlyLegalPlay: following a single', () => {
  const lead = [c('spades', '5')];
  it('forces the one led-suit card', () => {
    const hand = [c('spades', '9'), c('clubs', '3'), c('diamonds', 'A')];
    expect(keys(onlyLegalPlay(hand, lead, CTX, N))).toEqual(['9s']);
  });
  it('is not forced with two different led-suit cards', () => {
    const hand = [c('spades', '9'), c('spades', 'J'), c('clubs', '3')];
    expect(onlyLegalPlay(hand, lead, CTX, N)).toBeNull();
  });
  it('treats two copies of one card as one play', () => {
    const hand = [c('spades', '9', 0), c('spades', '9', 1), c('clubs', '3')];
    expect(keys(onlyLegalPlay(hand, lead, CTX, N))).toEqual(['9s']);
  });
  it('is not forced when void with a choice of discards', () => {
    const hand = [c('clubs', '3'), c('diamonds', 'A')];
    expect(onlyLegalPlay(hand, lead, CTX, N)).toBeNull();
  });
  it('is forced when void and every other card is the same', () => {
    const hand = [c('clubs', '3', 0), c('clubs', '3', 1)];
    expect(keys(onlyLegalPlay(hand, lead, CTX, N))).toEqual(['3c']);
  });
  it('counts trumps as their own suit, not as spades', () => {
    // The 2 of spades is a level card: trump, so not a spade.
    const hand = [c('spades', '2'), c('spades', '9')];
    expect(keys(onlyLegalPlay(hand, lead, CTX, N))).toEqual(['9s']);
  });
});

describe('onlyLegalPlay: following a pair', () => {
  const lead = [c('spades', '5', 0), c('spades', '5', 1)];
  it('forces the one pair held in the led suit', () => {
    const hand = [c('spades', '9', 0), c('spades', '9', 1), c('spades', 'K'), c('clubs', '3')];
    expect(keys(onlyLegalPlay(hand, lead, CTX, N))).toEqual(['9s', '9s']);
  });
  it('is not forced with two pairs to choose from', () => {
    const hand = [
      c('spades', '9', 0), c('spades', '9', 1), c('spades', 'J', 0), c('spades', 'J', 1),
    ];
    expect(onlyLegalPlay(hand, lead, CTX, N)).toBeNull();
  });
  it('forces exactly as many led-suit singles as the lead has cards', () => {
    const hand = [c('spades', 'K'), c('spades', '3'), c('clubs', '4')];
    expect(keys(onlyLegalPlay(hand, lead, CTX, N))).toEqual(['3s', 'Ks']);
  });
  it('is not forced when short of the suit with a choice of filler', () => {
    const hand = [c('spades', 'K'), c('diamonds', '7'), c('clubs', '8')];
    expect(onlyLegalPlay(hand, lead, CTX, N)).toBeNull();
  });
  it('is forced when short of the suit and the filler is all one card', () => {
    const hand = [c('spades', 'K'), c('diamonds', '7', 0), c('diamonds', '7', 1)];
    expect(keys(onlyLegalPlay(hand, lead, CTX, N))).toEqual(['7d', 'Ks']);
  });
  it('is forced when the hand is exactly the lead size', () => {
    const hand = [c('clubs', '3'), c('diamonds', 'A')];
    expect(keys(onlyLegalPlay(hand, lead, CTX, N))).toEqual(['3c', 'Ad']);
  });
});

describe('onlyLegalPlay: what it hands back', () => {
  it('returns the physical cards from the hand', () => {
    const hand = [c('spades', '9', 1), c('clubs', '3')];
    const got = onlyLegalPlay(hand, [c('spades', '5')], CTX, N);
    expect(got).not.toBeNull();
    for (const card of got!) expect(hand).toContain(card);
  });
  it('gives up rather than guess past its budget', () => {
    const hand = [c('spades', '9'), c('clubs', '3')];
    expect(onlyLegalPlay(hand, [c('spades', '5')], CTX, N, 0)).toBeNull();
  });
});

// Every subset of the hand's physical cards, judged by the engine, and the
// distinct legal plays among them by suit and rank.
function bruteForceLegal(
  hand: ReadonlyArray<Card>,
  lead: ReadonlyArray<Card>,
  ctx: TrumpContext,
): Set<string> {
  const combo = classifyCombo(lead, ctx, N);
  const thr = combo.ok ? null : decomposeThrow(lead, ctx, N);
  const legal = (cand: Card[]): boolean =>
    combo.ok
      ? validateFollow(combo.combo, cand, hand, ctx, N).ok
      : thr !== null && thr.ok && validateFollowThrow(thr.throw, cand, hand, ctx, N).ok;
  const out = new Set<string>();
  const k = lead.length;
  const rec = (start: number, chosen: Card[]): void => {
    if (chosen.length === k) {
      if (legal(chosen)) out.add(chosen.map(cardKey).sort().join(','));
      return;
    }
    for (let i = start; i <= hand.length - (k - chosen.length); i++) {
      chosen.push(hand[i]!);
      rec(i + 1, chosen);
      chosen.pop();
    }
  };
  rec(0, []);
  return out;
}

describe('onlyLegalPlay against brute force', () => {
  it('agrees on random positions', () => {
    const rng = mulberry32(20261124);
    const deck = buildDeck(N);
    const ranks: NaturalRank[] = ['2', '5', '10', 'A'];
    const suits: Array<Suit | null> = ['hearts', 'spades', null];
    let forced = 0;
    let checked = 0;
    for (let trial = 0; trial < 1500; trial++) {
      const ctx: TrumpContext = {
        trumpRank: ranks[trial % ranks.length]!,
        trumpSuit: suits[trial % suits.length]!,
      };
      // Small hands, rich in one suit, so forced follows come up often.
      const shuffled = shuffle(deck, rng);
      const focus = effectiveSuit(shuffled[0]!, ctx);
      const inFocus = shuffled.filter((x) => effectiveSuit(x, ctx) === focus);
      const other = shuffled.filter((x) => effectiveSuit(x, ctx) !== focus);
      const leadCards = inFocus.slice(0, 1 + Math.floor(rng() * 4));
      const lead = classifyCombo(leadCards, ctx, N).ok
        ? leadCards
        : decomposeThrow(leadCards, ctx, N).ok
          ? leadCards
          : leadCards.slice(0, 1);
      const nFocus = Math.floor(rng() * 7);
      const nOther = Math.max(lead.length - nFocus, 0) + Math.floor(rng() * 4);
      const hand = [
        ...inFocus.slice(lead.length, lead.length + nFocus),
        ...other.slice(0, nOther),
      ];
      if (hand.length < lead.length) continue;
      const truth = bruteForceLegal(hand, lead, ctx);
      const got = onlyLegalPlay(hand, lead, ctx, N);
      checked++;
      if (truth.size === 1) {
        forced++;
        expect(got).not.toBeNull();
        expect(got!.map(cardKey).sort().join(',')).toEqual([...truth][0]);
      } else {
        expect(got).toBeNull();
      }
    }
    // The positions exercise both answers.
    expect(checked).toBeGreaterThan(1000);
    expect(forced).toBeGreaterThan(100);
    expect(checked - forced).toBeGreaterThan(100);
  });
});
