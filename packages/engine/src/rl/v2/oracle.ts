// Brute-force reference for the fair-play legal set: every distinct play a
// seat may make, by card key, using only information that seat has.
//
// This is a test and diagnostics oracle, not a production path. It
// enumerates instead of reasoning, so it can be trusted where it finishes
// and is allowed to give up (`complete: false`) where the space is too
// large. The sequential action space is verified against it: every play
// it lists must be reachable, and nothing else may be.
//
// "Fair-play legal" differs from "accepted by the reducer" in one place.
// The reducer accepts any well-formed throw and then applies the
// forced-lead penalty if an opponent can beat it, which it decides from the
// opponents' actual hands. A fair-play agent may only offer a throw it can
// prove safe from public information, so a throw is included here only if
// no component can be beaten even by a single opponent holding *every*
// unseen card. Beating ability only grows with holdings, so that pooled
// opponent is the worst case.

import { classifyCombo } from '../../classifier';
import { effectiveSuit } from '../../ordering';
import { type GameState, type Seat } from '../../state';
import { decomposeThrow } from '../../throws';
import { validateFollow, validateFollowThrow } from '../../follow';
import { type Card, type EffectiveSuit, type TrumpContext } from '../../types';

import { cardSlot } from './cards';
import { MAX_CANDIDATE_CARDS } from './record';
import { isFairLead, pooledUnseenHand } from './sequential';

export type OracleResult = {
  // Signature (sorted slot list, comma-joined) -> one physical realisation.
  readonly plays: Map<string, ReadonlyArray<Card>>;
  // False when the search gave up; `plays` is then a subset.
  readonly complete: boolean;
};

export type OracleOptions = {
  readonly myKitty?: ReadonlyArray<Card>;
  // Give up beyond this many raw multisets examined. Default 200k.
  readonly limit?: number;
  // Largest play considered, matching the wire format. Default 16.
  readonly maxCards?: number;
};

export function playSignature(cards: ReadonlyArray<Card>): string {
  return cards
    .map(cardSlot)
    .sort((a, b) => a - b)
    .join(',');
}

// Groups a hand into distinct keys: slot -> the physical cards holding it.
function bySlot(cards: ReadonlyArray<Card>): Map<number, Card[]> {
  const m = new Map<number, Card[]>();
  for (const c of cards) {
    const s = cardSlot(c);
    const list = m.get(s);
    if (list === undefined) m.set(s, [c]);
    else list.push(c);
  }
  return m;
}

// Every multiset of exactly `k` cards drawn from `groups` (each group
// contributing 0..group.length copies). Calls `emit` with physical cards.
// Returns false if it stopped at `budget`.
function eachMultiset(
  groups: ReadonlyArray<ReadonlyArray<Card>>,
  k: number,
  budget: { left: number },
  emit: (cards: Card[]) => void,
): boolean {
  const cur: Card[] = [];
  const suffixCap: number[] = new Array(groups.length + 1).fill(0);
  for (let i = groups.length - 1; i >= 0; i--) {
    suffixCap[i] = suffixCap[i + 1]! + groups[i]!.length;
  }
  let ok = true;
  const rec = (i: number): void => {
    if (!ok) return;
    const need = k - cur.length;
    if (need === 0) {
      if (budget.left-- <= 0) {
        ok = false;
        return;
      }
      emit([...cur]);
      return;
    }
    if (i >= groups.length || suffixCap[i]! < need) return;
    const g = groups[i]!;
    for (let take = Math.min(g.length, need); take >= 0; take--) {
      for (let t = 0; t < take; t++) cur.push(g[t]!);
      rec(i + 1);
      for (let t = 0; t < take; t++) cur.pop();
      if (!ok) return;
    }
  };
  rec(0);
  return ok;
}

export function fairLegalPlays(
  state: GameState,
  seat: Seat,
  opts: OracleOptions = {},
): OracleResult {
  const hand = state.currentHand;
  const plays = new Map<string, ReadonlyArray<Card>>();
  if (hand === null || hand.trumpRank === null || state.phase !== 'play') {
    return { plays, complete: true };
  }
  const ctx: TrumpContext = { trumpRank: hand.trumpRank, trumpSuit: hand.trumpSuit };
  const n = state.config.numDecks;
  const myCards = hand.hands[seat] ?? [];
  const limit = opts.limit ?? 200_000;
  const maxCards = opts.maxCards ?? MAX_CANDIDATE_CARDS;
  const budget = { left: limit };
  const trick = hand.currentTrick;

  if (trick !== null && trick.plays.length > 0) {
    const lead = trick.plays[0]!.cards;
    const k = lead.length;
    if (k > maxCards) return { plays, complete: true };
    const ledSuit = effectiveSuit(lead[0]!, ctx);
    const inLed = myCards.filter((c) => effectiveSuit(c, ctx) === ledSuit);
    const rest = myCards.filter((c) => effectiveSuit(c, ctx) !== ledSuit);
    const req = Math.min(inLed.length, k);
    const leadCls = classifyCombo(lead, ctx, n);
    const leadThrow = leadCls.ok ? null : decomposeThrow(lead, ctx, n);
    const valid = (cards: ReadonlyArray<Card>): boolean => {
      if (leadCls.ok) return validateFollow(leadCls.combo, cards, myCards, ctx, n).ok;
      if (leadThrow !== null && leadThrow.ok) {
        return validateFollowThrow(leadThrow.throw, cards, myCards, ctx).ok;
      }
      return false;
    };
    const ledGroups = [...bySlot(inLed).values()];
    const restGroups = [...bySlot(rest).values()];
    let complete = eachMultiset(ledGroups, req, budget, (ledPart) => {
      const ok2 = eachMultiset(restGroups, k - req, budget, (fill) => {
        const cards = [...ledPart, ...fill];
        if (valid(cards)) plays.set(playSignature(cards), cards);
      });
      if (!ok2) budget.left = 0;
    });
    if (budget.left <= 0) complete = false;
    return { plays, complete };
  }

  // Lead: every fair lead, suit by suit.
  const pooled = pooledUnseenHand(state, seat, opts.myKitty);
  const bySuit = new Map<EffectiveSuit, Card[]>();
  for (const c of myCards) {
    const es = effectiveSuit(c, ctx);
    const list = bySuit.get(es);
    if (list === undefined) bySuit.set(es, [c]);
    else list.push(c);
  }
  let complete = true;
  for (const suitCards of bySuit.values()) {
    const groups = [...bySlot(suitCards).values()];
    const top = Math.min(maxCards, suitCards.length);
    for (let size = 1; size <= top; size++) {
      const ok = eachMultiset(groups, size, budget, (cards) => {
        if (isFairLead(cards, pooled, ctx, n)) plays.set(playSignature(cards), cards);
      });
      if (!ok) {
        complete = false;
        break;
      }
    }
    if (!complete) break;
  }
  return { plays, complete };
}
