// Fast mode: is there exactly one legal play? If so, which cards.
//
// The engine's enumerators (`legalActions`, `allLegalPlayActions`) are
// capped, not exhaustive, so they cannot prove a play is the only one. This
// checks exactly, with the engine's own follow rules as the judge:
//
// * Leading, the only forced lead is a last card.
// * Following, every distinct set of cards that could be legal is tried
//   against `validateFollow` / `validateFollowThrow` (rules.md §7.3, §7.6).
//   A follower spends as many led-suit cards as they hold, up to the lead's
//   count, so the candidates are: with at least that many led-suit cards,
//   every choice of that many of them; with fewer, all of them plus every
//   choice of the rest from the other cards. Cards of one suit and rank are
//   interchangeable (two decks hold two of each), so choices are counted by
//   suit and rank, not by physical card.
//
// The search stops at the second legal play. If it would try more than
// `budget` candidates (a long throw against a long holding) it gives up and
// reports no forced play: fast mode never plays for you unless it is sure.

import {
  cardKey,
  classifyCombo,
  decomposeThrow,
  effectiveSuit,
  validateFollow,
  validateFollowThrow,
  type Card,
  type TrumpContext,
} from '@tractor/engine';

export const FORCED_PLAY_BUDGET = 5000;

type Group = { readonly key: string; readonly cards: ReadonlyArray<Card> };

function groups(cards: ReadonlyArray<Card>): Group[] {
  const by = new Map<string, Card[]>();
  for (const c of cards) {
    const k = cardKey(c);
    const g = by.get(k);
    if (g === undefined) by.set(k, [c]);
    else g.push(c);
  }
  return [...by.entries()].map(([key, cs]) => ({ key, cards: cs }));
}

// Every way to take `k` cards from `gs`, as a count per group, in a fixed
// order. Returns false if `visit` asked to stop.
function chooseFrom(
  gs: ReadonlyArray<Group>,
  k: number,
  visit: (counts: ReadonlyArray<number>) => boolean,
): boolean {
  const counts = new Array<number>(gs.length).fill(0);
  // Cards still available from group i onward, to prune dead branches.
  const after = new Array<number>(gs.length + 1).fill(0);
  for (let i = gs.length - 1; i >= 0; i--) after[i] = after[i + 1]! + gs[i]!.cards.length;
  const rec = (i: number, left: number): boolean => {
    if (left === 0) return visit(counts);
    if (i === gs.length || after[i]! < left) return true;
    const most = Math.min(gs[i]!.cards.length, left);
    for (let take = most; take >= 0; take--) {
      counts[i] = take;
      if (!rec(i + 1, left - take)) return false;
    }
    counts[i] = 0;
    return true;
  };
  return rec(0, k);
}

function pick(gs: ReadonlyArray<Group>, counts: ReadonlyArray<number>): Card[] {
  const out: Card[] = [];
  gs.forEach((g, i) => out.push(...g.cards.slice(0, counts[i])));
  return out;
}

/**
 * The cards of the only legal play for a player holding `hand`, or null if
 * there is more than one (or the check gave up). `lead` is the trick's led
 * play, or null when this player leads.
 */
export function onlyLegalPlay(
  hand: ReadonlyArray<Card>,
  lead: ReadonlyArray<Card> | null,
  ctx: TrumpContext,
  numDecks: number,
  budget: number = FORCED_PLAY_BUDGET,
): Card[] | null {
  if (hand.length === 0) return null;
  if (lead === null) return hand.length === 1 ? [...hand] : null;

  const count = lead.length;
  if (count === 0 || hand.length < count) return null;
  let legal: (cand: ReadonlyArray<Card>) => boolean;
  const asCombo = classifyCombo(lead, ctx, numDecks);
  if (asCombo.ok) {
    const combo = asCombo.combo;
    legal = (cand) => validateFollow(combo, cand, hand, ctx, numDecks).ok;
  } else {
    const asThrow = decomposeThrow(lead, ctx, numDecks);
    if (!asThrow.ok) return null;
    const thr = asThrow.throw;
    legal = (cand) => validateFollowThrow(thr, cand, hand, ctx, numDecks).ok;
  }

  const ledSuit = effectiveSuit(lead[0]!, ctx);
  const inLed = hand.filter((c) => effectiveSuit(c, ctx) === ledSuit);
  const rest = hand.filter((c) => effectiveSuit(c, ctx) !== ledSuit);
  // The led-suit cards every legal play contains, and where the others
  // come from.
  const fixed: Card[] = inLed.length >= count ? [] : inLed;
  const pool = groups(inLed.length >= count ? inLed : rest);
  const free = count - fixed.length;

  let found: Card[] | null = null;
  let tried = 0;
  let ambiguous = false;
  const finished = chooseFrom(pool, free, (counts) => {
    if (++tried > budget) {
      ambiguous = true;
      return false;
    }
    const cand = [...fixed, ...pick(pool, counts)];
    if (!legal(cand)) return true;
    if (found !== null) {
      ambiguous = true;
      return false;
    }
    found = cand;
    return true;
  });
  if (!finished || ambiguous) return null;
  return found;
}
