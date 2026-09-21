// Exhaustive check of the follow rule (rules.md §7.3).
//
// For randomly generated led-suit holdings the test enumerates every
// n-subset by brute force, filters it through `validateFollow`, and
// compares the accepted set against the obligations computed a second
// time from first principles — grouping cards by key for pairs and by
// chain position for tractor runs, without going through `findNKinds`
// or `findTractors`. A bug that is shared by the implementation and the
// oracle would have to be made twice, in two different shapes.
//
// The positions are synthetic rather than dealt: only the follower's
// led-suit holding and the lead's structure enter the predicate, so
// generating those directly reaches hands (a tractor alongside isolated
// pairs, several tractors of different lengths) that random deals
// produce far too rarely to cover.

import { describe, expect, it } from 'vitest';
import { classifyCombo, type ClassifiedCombo } from '../src/classifier';
import { chainPosition } from '../src/combos';
import { validateFollow } from '../src/follow';
import { mulberry32 } from '../src/prng';
import {
  cardKey,
  NATURAL_RANKS,
  SUITS,
  type Card,
  type NaturalRank,
  type Suit,
  type TrumpContext,
} from '../src/types';

const DECKS = 2;

function card(suit: Suit, rank: NaturalRank, deckId: number): Card {
  return { kind: 'standard', suit, rank, deckId };
}

// --- oracle: obligations computed without the combo helpers -------------

// Number of disjoint pairs, i.e. types held in duplicate.
function oraclePairs(cards: ReadonlyArray<Card>): number {
  const counts = new Map<string, number>();
  for (const c of cards) counts.set(cardKey(c), (counts.get(cardKey(c)) ?? 0) + 1);
  let pairs = 0;
  for (const v of counts.values()) pairs += Math.floor(v / DECKS);
  return pairs;
}

// Longest run of consecutive chain positions that are each held as a
// pair. Two off-suit level cards share a chain position, so they occupy
// one slot and cannot chain with each other — which is the behaviour the
// engine must also have.
function oracleLongestTractor(
  cards: ReadonlyArray<Card>,
  ctx: TrumpContext,
): number {
  const counts = new Map<string, { n: number; pos: number }>();
  for (const c of cards) {
    const k = cardKey(c);
    const e = counts.get(k);
    if (e === undefined) counts.set(k, { n: 1, pos: chainPosition(c, ctx) });
    else e.n += 1;
  }
  const paired = new Set<number>();
  for (const { n, pos } of counts.values()) if (n >= DECKS) paired.add(pos);
  if (paired.size === 0) return 0;
  const sorted = [...paired].sort((a, b) => a - b);
  let best = 1;
  let run = 1;
  for (let i = 1; i < sorted.length; i++) {
    run = sorted[i] === sorted[i - 1]! + 1 ? run + 1 : 1;
    if (run > best) best = run;
  }
  return best >= 2 ? best : 0;
}

// --- subset enumeration --------------------------------------------------

function subsetsOfSize(cards: ReadonlyArray<Card>, k: number): Card[][] {
  const out: Card[][] = [];
  const cur: Card[] = [];
  const recurse = (start: number): void => {
    if (cur.length === k) {
      out.push([...cur]);
      return;
    }
    for (let i = start; i < cards.length; i++) {
      cur.push(cards[i]!);
      recurse(i + 1);
      cur.pop();
    }
  };
  recurse(0);
  return out;
}

// --- generation ----------------------------------------------------------

type Position = {
  readonly ctx: TrumpContext;
  readonly lead: ClassifiedCombo;
  readonly hand: Card[];
};

// A holding in `suit` with between `lo` and `hi` distinct ranks, each
// present once or twice.
function holding(
  rand: () => number,
  suit: Suit,
  ranks: ReadonlyArray<NaturalRank>,
  lo: number,
  hi: number,
): Card[] {
  const pool = [...ranks];
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [pool[i], pool[j]] = [pool[j]!, pool[i]!];
  }
  const count = lo + Math.floor(rand() * (hi - lo + 1));
  const out: Card[] = [];
  for (const r of pool.slice(0, count)) {
    out.push(card(suit, r, 0));
    if (rand() < 0.5) out.push(card(suit, r, 1));
  }
  return out;
}

function makePosition(rand: () => number): Position | null {
  const trumpRank = NATURAL_RANKS[Math.floor(rand() * NATURAL_RANKS.length)]!;
  const trumpSuit = SUITS[Math.floor(rand() * SUITS.length)]!;
  const ctx: TrumpContext = { trumpRank, trumpSuit };

  // Lead and follow in a non-trump suit so the level card is excluded
  // from the chain and the run arithmetic is exercised across its gap.
  const ledSuit = SUITS.filter((s) => s !== trumpSuit)[
    Math.floor(rand() * 3)
  ]!;
  const ranks = NATURAL_RANKS.filter((r) => r !== trumpRank);

  const hand = holding(rand, ledSuit, ranks, 4, 7);
  if (hand.length < 3) return null;

  // Lead structure. Its cards only have to be a valid combo of the led
  // suit; the predicate reads the lead's shape, not its provenance.
  const roll = rand();
  let leadCards: Card[];
  if (roll < 0.3) {
    leadCards = [card(ledSuit, ranks[0]!, 0)];
  } else if (roll < 0.6) {
    leadCards = [card(ledSuit, ranks[0]!, 0), card(ledSuit, ranks[0]!, 1)];
  } else {
    // A tractor of length 2 or 3 built from consecutive non-level ranks.
    const len = roll < 0.85 ? 2 : 3;
    const start = Math.floor(rand() * (ranks.length - len));
    leadCards = [];
    for (let i = 0; i < len; i++) {
      leadCards.push(card(ledSuit, ranks[start + i]!, 0));
      leadCards.push(card(ledSuit, ranks[start + i]!, 1));
    }
  }
  const cls = classifyCombo(leadCards, ctx, DECKS);
  if (!cls.ok) return null;
  // Only the c > n branch carries structural obligations.
  if (hand.length <= leadCards.length) return null;
  return { ctx, lead: cls.combo, hand };
}

// --- the properties ------------------------------------------------------

describe('validateFollow against brute-force enumeration', () => {
  it('accepts exactly the maximal-structure selections', () => {
    const rand = mulberry32(0x5eed);
    let positions = 0;
    let checked = 0;
    let skippedTooLarge = 0;

    for (let iter = 0; positions < 2000 && iter < 60000; iter++) {
      const pos = makePosition(rand);
      if (pos === null) continue;
      const { ctx, lead, hand } = pos;
      const n = lead.kind === 'single' ? 1 : lead.cards.length;

      const subsets = subsetsOfSize(hand, n);
      if (subsets.length > 20000) {
        skippedTooLarge += 1;
        continue;
      }
      positions += 1;

      // Obligations, computed independently of the implementation.
      const leadTractor = lead.kind === 'tractor' ? lead.length : 0;
      const leadPairs =
        lead.kind === 'tractor'
          ? lead.length
          : lead.kind === 'nkind' && lead.n >= DECKS
            ? 1
            : 0;
      const needTractor = Math.min(leadTractor, oracleLongestTractor(hand, ctx));
      const needPairs = Math.min(leadPairs, oraclePairs(hand));

      const accepted: Card[][] = [];
      for (const s of subsets) {
        const v = validateFollow(lead, s, hand, ctx, DECKS);
        const oracleOk =
          Math.min(oracleLongestTractor(s, ctx), leadTractor) >= needTractor &&
          oraclePairs(s) >= needPairs;
        checked += 1;
        expect(
          v.ok,
          `lead=${lead.kind}/${n} trump=${ctx.trumpSuit}/${ctx.trumpRank} ` +
            `hand=[${hand.map(cardKey).join(' ')}] play=[${s.map(cardKey).join(' ')}] ` +
            `needTractor=${needTractor} needPairs=${needPairs}`,
        ).toBe(oracleOk);
        if (v.ok) accepted.push(s);
      }

      // The rule must always leave the follower something to play.
      expect(
        accepted.length,
        `no legal follow: hand=[${hand.map(cardKey).join(' ')}] lead=${lead.kind}/${n}`,
      ).toBeGreaterThan(0);
    }

    expect(positions).toBeGreaterThan(1500);
    expect(checked).toBeGreaterThan(50000);
    expect(skippedTooLarge).toBeLessThan(positions);
  });

  it('rejects a broken tractor when an intact shorter one is held', () => {
    // Trump hearts, level 2. Spades: A-A K-K (a 2-tractor) plus 5-5 and
    // 3-3 as isolated pairs. Against a 3-tractor lead rules.md §7.3
    // requires the longest available tractor first, so A-A K-K must be
    // in the play and A-A 5-5 3-3 is illegal even though it is 3 pairs.
    const ctx: TrumpContext = { trumpRank: '2', trumpSuit: 'hearts' };
    const hand = [
      card('spades', 'A', 0), card('spades', 'A', 1),
      card('spades', 'K', 0), card('spades', 'K', 1),
      card('spades', '5', 0), card('spades', '5', 1),
      card('spades', '3', 0), card('spades', '3', 1),
    ];
    const leadCards = [
      card('spades', '10', 0), card('spades', '10', 1),
      card('spades', '9', 0), card('spades', '9', 1),
      card('spades', '8', 0), card('spades', '8', 1),
    ];
    const cls = classifyCombo(leadCards, ctx, DECKS);
    expect(cls.ok).toBe(true);
    if (!cls.ok) return;

    const intact = [hand[0]!, hand[1]!, hand[2]!, hand[3]!, hand[4]!, hand[5]!];
    const broken = [hand[0]!, hand[1]!, hand[4]!, hand[5]!, hand[6]!, hand[7]!];
    expect(validateFollow(cls.combo, intact, hand, ctx, DECKS).ok).toBe(true);
    expect(validateFollow(cls.combo, broken, hand, ctx, DECKS).ok).toBe(false);
  });
});
