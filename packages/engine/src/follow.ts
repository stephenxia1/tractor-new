import type { ClassifiedCombo } from './classifier';
import { findNKinds, findTractors } from './combos';
import { effectiveSuit } from './ordering';
import type { ThrowDecomposition } from './throws';
import { throwTotalCards } from './throws';
import type { Card, TrumpContext } from './types';

export type FollowValidation =
  | { readonly ok: true }
  | { readonly ok: false; readonly error: string };

// Validates a follower's play against the lead per rules.md §7.3.
// Returns ok if the candidate cards obey the follow-suit obligation and
// the structural obligation (pairs/tractors of led suit must be played
// when available).
//
// Not yet enforced (deferred to a stricter pass):
//   - "Longest available tractor first" greedy fill for tractor leads.
//   - Choice-among-equivalents within the led-suit set (e.g., which pair).
// For v1 these gaps allow some legal flexibility but no cheating — the
// count + suit + pair-obligation checks below cover the common cases.
export function validateFollow(
  lead: ClassifiedCombo,
  candidate: ReadonlyArray<Card>,
  hand: ReadonlyArray<Card>,
  ctx: TrumpContext,
  n: number,
): FollowValidation {
  const leadCount = leadCardCount(lead);
  if (candidate.length !== leadCount) {
    return { ok: false, error: `must play ${leadCount} cards` };
  }

  const ledSuit = lead.effectiveSuit;

  const handInLed = hand.filter((c) => effectiveSuit(c, ctx) === ledSuit);
  const candInLed = candidate.filter((c) => effectiveSuit(c, ctx) === ledSuit);

  // 1. Follow suit: use as many led-suit cards as you have, up to the
  // required count.
  const required = Math.min(handInLed.length, leadCount);
  if (candInLed.length < required) {
    return {
      ok: false,
      error: `must follow ${ledSuit}: ${required} required, ${candInLed.length} given`,
    };
  }

  // 2. Pair obligation (rules.md §7.3): if lead is a pair (or contains
  // pairs as part of a tractor) and the player has pairs in led suit,
  // the candidate must include at least that many pairs in led suit.
  const leadPairs = countPairsInLead(lead, n);
  if (leadPairs > 0) {
    const handPairsInLed = findNKinds(handInLed, n).length;
    const candPairsInLed = findNKinds(candInLed, n).length;
    const required2 = Math.min(handPairsInLed, leadPairs);
    if (candPairsInLed < required2) {
      return {
        ok: false,
        error: `must include ${required2} pair(s) of ${ledSuit}: ${candPairsInLed} given`,
      };
    }
  }

  // 3. Tractor obligation (rules.md §7.3): if lead is a tractor of length
  // k and the player has a tractor of length k in led suit, must include
  // one. Shorter-tractor fallback isn't enforced yet.
  if (lead.kind === 'tractor') {
    const handTractors = findTractors(handInLed, ctx, n).filter(
      (t) => t.length === lead.length,
    );
    if (handTractors.length > 0) {
      const candTractors = findTractors(candInLed, ctx, n).filter(
        (t) => t.length === lead.length,
      );
      if (candTractors.length === 0) {
        return {
          ok: false,
          error: `must include a tractor of length ${lead.length} in ${ledSuit}`,
        };
      }
    }
  }

  return { ok: true };
}

function leadCardCount(c: ClassifiedCombo): number {
  if (c.kind === 'single') return 1;
  return c.cards.length;
}

// Lighter follow validation for throw leads — count + follow-suit only.
// Structural matching for compound leads is deferred for v1.
export function validateFollowThrow(
  throwLead: ThrowDecomposition,
  candidate: ReadonlyArray<Card>,
  hand: ReadonlyArray<Card>,
  ctx: TrumpContext,
): FollowValidation {
  const total = throwTotalCards(throwLead);
  if (candidate.length !== total) {
    return { ok: false, error: `must play ${total} cards` };
  }
  const ledSuit = throwLead.effectiveSuit;
  const handInLed = hand.filter((c) => effectiveSuit(c, ctx) === ledSuit);
  const candInLed = candidate.filter((c) => effectiveSuit(c, ctx) === ledSuit);
  const required = Math.min(handInLed.length, total);
  if (candInLed.length < required) {
    return {
      ok: false,
      error: `must follow ${ledSuit}: ${required} required, ${candInLed.length} given`,
    };
  }
  return { ok: true };
}

function countPairsInLead(c: ClassifiedCombo, n: number): number {
  if (c.kind === 'single') return 0;
  if (c.kind === 'nkind') return c.n >= n ? 1 : 0;
  // tractor
  return c.length;
}
